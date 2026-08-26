"""Pipeline position: VOICE REGISTRY — create / list / get / delete cloned voices.

Role in pipeline:
  Owns every voice the server can speak in. The synthesis path calls `get()` and
  receives a `VoicePrompt` whose mel is already on the GPU; nothing else about a
  voice is computed at request time.

On-disk layout (`voice.store_dir`):
    <voice_id>.npz    mel (float16) + token_ids (int32)
    <voice_id>.json   metadata (transcript, language, durations, checksum)

Writes are atomic (tmp file + os.replace) so a crash mid-upload cannot leave a
half-written voice that fails to load on the next boot.

Why the transcript is mandatory
-------------------------------
DhVaani has no duration predictor. Generated length comes straight from

    prompt_frames / prompt_tokens  x  target_tokens

so the reference transcript's LENGTH sets the speaking rate of every utterance
in that voice. A transcript that is half the true length makes the voice speak
at half speed. We validate the implied characters-per-second and warn loudly
when it is implausible, because the symptom (a voice that drawls) looks like a
model problem rather than a data-entry problem.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import structlog

from flowtts.dhvaani.config import dhv_settings
from flowtts.dhvaani.text.chunker import add_punctuation
from flowtts.dhvaani.types import (
    VoiceAlreadyExists,
    VoiceNotFound,
    VoicePrompt,
    InvalidReferenceAudio,
)
from flowtts.dhvaani.voices.clone import prepare_prompt

logger = structlog.get_logger(__name__)

_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")

# Plausible speaking rates, in characters of transcript per second of audio.
# Devanagari and Dravidian scripts pack more phonemes per character than Latin,
# hence the wide band; this only catches gross mismatches.
_MIN_CPS, _MAX_CPS = 2.0, 45.0


def validate_voice_id(voice_id: str) -> str:
    vid = (voice_id or "").strip()
    if not _ID_RE.match(vid):
        raise InvalidReferenceAudio(
            f"invalid voice_id {voice_id!r}: must match {_ID_RE.pattern} "
            "(letters, digits, underscore, hyphen; 1-64 chars)"
        )
    return vid


class VoiceStore:
    """Disk-backed voice registry with an LRU of GPU-resident prompts."""

    def __init__(self, loaded, settings=None):
        self._s = settings or dhv_settings
        self._m = loaded
        self._dir = Path(self._s.voice.store_dir).expanduser()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._cache: OrderedDict[str, VoicePrompt] = OrderedDict()
        self._meta: dict[str, dict] = {}
        self._hits = 0
        self._misses = 0
        self.reload()

    # -- persistence ---------------------------------------------------------
    def _npz(self, vid: str) -> Path:
        return self._dir / f"{vid}.npz"

    def _json(self, vid: str) -> Path:
        return self._dir / f"{vid}.json"

    def reload(self) -> int:
        """Rescan the store directory. Returns the number of voices found."""
        with self._lock:
            self._meta.clear()
            for jf in sorted(self._dir.glob("*.json")):
                vid = jf.stem
                try:
                    meta = json.loads(jf.read_text())
                except Exception as e:
                    logger.warning("voice_metadata_unreadable", voice_id=vid, error=str(e))
                    continue
                if not self._npz(vid).exists():
                    logger.warning("voice_tensors_missing", voice_id=vid)
                    continue
                self._meta[vid] = meta
            logger.info("voice_store_loaded", count=len(self._meta), dir=str(self._dir))
            return len(self._meta)

    def _persist(self, prompt: VoicePrompt) -> None:
        import torch

        vid = prompt.voice_id
        mel = prompt.mel
        arr = (mel.detach().to("cpu", dtype=torch.float16).numpy()
               if isinstance(mel, torch.Tensor) else np.asarray(mel, dtype=np.float16))

        # Write through a file object: np.savez silently appends ".npz" when the
        # *path* does not already end in it, so np.savez("v.npz.tmp") would
        # produce "v.npz.tmp.npz" and the os.replace below would miss it.
        # Handing it an open handle bypasses that naming logic entirely.
        tmp_npz = Path(str(self._npz(vid)) + ".tmp")
        with open(tmp_npz, "wb") as fh:
            np.savez(
                fh,
                mel=arr,
                token_ids=np.asarray(prompt.token_ids, dtype=np.int32),
            )
        os.replace(tmp_npz, self._npz(vid))

        meta = prompt.to_metadata()
        tmp_json = Path(str(self._json(vid)) + ".tmp")
        tmp_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        os.replace(tmp_json, self._json(vid))
        self._meta[vid] = meta

    def _load_from_disk(self, vid: str) -> VoicePrompt:
        import torch

        meta = self._meta.get(vid)
        if meta is None:
            raise VoiceNotFound(f"voice {vid!r} not found")
        with np.load(self._npz(vid)) as z:
            mel_np = z["mel"]
            tokens = z["token_ids"].tolist()
        mel = torch.from_numpy(np.ascontiguousarray(mel_np)).to(
            self._m.device, dtype=self._m.dtype
        )
        token_ids = [int(t) for t in tokens]
        # Recomputed rather than read back: to_metadata() rounds it for
        # human-readable JSON, and it is exactly derivable from the two values
        # that ARE stored losslessly. Keeping it exact matters because it is what
        # the chunker uses to size spans.
        frames_per_token = int(mel.shape[0]) / max(len(token_ids), 1)
        return VoicePrompt(
            voice_id=vid,
            mel=mel,
            mel_frames=int(mel.shape[0]),
            token_ids=token_ids,
            prompt_rms=float(meta.get("prompt_rms", 0.1)),
            frames_per_token=frames_per_token,
            name=meta.get("name", ""),
            description=meta.get("description", ""),
            language=meta.get("language", ""),
            transcript=meta.get("transcript", ""),
            sample_rate=int(meta.get("sample_rate", 24000)),
            duration_s=float(meta.get("duration_s", 0.0)),
            created_at=float(meta.get("created_at", time.time())),
            source_filename=meta.get("source_filename", ""),
            checksum=meta.get("checksum", ""),
        )

    # -- cache ---------------------------------------------------------------
    def _cache_put(self, prompt: VoicePrompt) -> None:
        self._cache[prompt.voice_id] = prompt
        self._cache.move_to_end(prompt.voice_id)
        while len(self._cache) > self._s.voice.gpu_cache_size:
            vid, evicted = self._cache.popitem(last=False)
            evicted.mel = None  # drop the GPU reference; reloaded on next get()
            logger.debug("voice_evicted", voice_id=vid)

    # -- public API ----------------------------------------------------------
    def create(
        self,
        voice_id: str,
        audio,
        transcript: str,
        name: str = "",
        description: str = "",
        language: str = "",
        overwrite: bool = False,
        source_filename: str = "",
    ) -> VoicePrompt:
        vid = validate_voice_id(voice_id)
        transcript = (transcript or "").strip()
        if not transcript:
            raise InvalidReferenceAudio(
                "a transcript of the reference audio is required: DhVaani derives "
                "the speaking rate from prompt_frames / prompt_tokens, so an "
                "absent or wrong-length transcript changes how fast the voice speaks"
            )

        with self._lock:
            if vid in self._meta and not overwrite:
                raise VoiceAlreadyExists(
                    f"voice {vid!r} already exists; pass overwrite=true to replace it"
                )

            prepared = prepare_prompt(audio, self._m, self._s)

            from flowtts.dhvaani.text import lang as langmod

            lang_code = langmod.resolve(language, transcript, self._s.text.default_language)
            token_ids = self._m.token_ids(add_punctuation(transcript))
            if not token_ids:
                raise InvalidReferenceAudio(
                    "the transcript produced zero in-vocabulary tokens -- check it "
                    "is written in a script DhVaani supports"
                )

            cps = len(transcript) / max(prepared.duration_s, 1e-6)
            if not (_MIN_CPS <= cps <= _MAX_CPS):
                logger.warning(
                    "voice_transcript_rate_implausible",
                    voice_id=vid,
                    chars_per_second=round(cps, 1),
                    duration_s=round(prepared.duration_s, 2),
                    transcript_chars=len(transcript),
                    hint="the clone will speak at the wrong rate; verify the "
                         "transcript matches the audio exactly",
                )

            checksum = hashlib.sha256(
                prepared.wav_24k.tobytes() + transcript.encode("utf-8")
            ).hexdigest()[:32]

            prompt = VoicePrompt(
                voice_id=vid,
                mel=prepared.mel,
                mel_frames=prepared.frames,
                token_ids=token_ids,
                prompt_rms=prepared.prompt_rms,
                frames_per_token=prepared.frames / max(len(token_ids), 1),
                name=name or vid,
                description=description,
                language=lang_code,
                transcript=transcript,
                sample_rate=24000,
                duration_s=prepared.duration_s,
                source_filename=source_filename,
                checksum=checksum,
            )
            self._persist(prompt)
            self._cache_put(prompt)
            logger.info(
                "voice_created",
                voice_id=vid,
                language=lang_code,
                duration_s=round(prepared.duration_s, 2),
                mel_frames=prepared.frames,
                tokens=len(token_ids),
                frames_per_token=round(prompt.frames_per_token, 3),
            )
            return prompt

    def get(self, voice_id: str) -> VoicePrompt:
        vid = (voice_id or "").strip()
        with self._lock:
            cached = self._cache.get(vid)
            if cached is not None and cached.mel is not None:
                self._cache.move_to_end(vid)
                self._hits += 1
                return cached
            self._misses += 1
            prompt = self._load_from_disk(vid)
            self._cache_put(prompt)
            return prompt

    def exists(self, voice_id: str) -> bool:
        return (voice_id or "").strip() in self._meta

    def list(self) -> list[dict]:
        with self._lock:
            return sorted(self._meta.values(), key=lambda m: m.get("created_at", 0))

    def delete(self, voice_id: str) -> None:
        vid = (voice_id or "").strip()
        with self._lock:
            if vid not in self._meta:
                raise VoiceNotFound(f"voice {vid!r} not found")
            self._cache.pop(vid, None)
            self._meta.pop(vid, None)
            for p in (self._npz(vid), self._json(vid)):
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
            logger.info("voice_deleted", voice_id=vid)

    def default(self) -> VoicePrompt:
        configured = self._s.voice.default_voice
        if configured and self.exists(configured):
            return self.get(configured)
        listing = self.list()
        if not listing:
            raise VoiceNotFound(
                "no voices exist yet. Create one with "
                "POST /v1/voices (multipart: file + transcript + voice_id) or "
                "`python -m flowtts.dhvaani.setup.seed_voices`."
            )
        return self.get(listing[0]["voice_id"])

    def resolve(self, voice_id: str | None) -> VoicePrompt:
        """Named voice, or the default when unspecified."""
        if voice_id:
            return self.get(voice_id)
        return self.default()

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "voices": len(self._meta),
            "gpu_cached": len(self._cache),
            "gpu_cache_size": self._s.voice.gpu_cache_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total else 0.0,
            "store_dir": str(self._dir),
        }
