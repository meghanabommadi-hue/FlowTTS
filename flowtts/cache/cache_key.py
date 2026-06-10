"""Cache key generation for TTS audio.

Produces a stable SHA256 digest that encodes every parameter that affects
the generated audio waveform.  A cache hit is only valid when ALL of these
parameters match — stale entries from an old model version or different voice
are automatically missed.

Key ingredients
---------------
  text          — Unicode-normalised (NFC), whitespace-collapsed, lowercased
  voice_id      — canonical voice name (empty string if not set)
  model_version — semver/git-hash string identifying the TTS checkpoint
  speaking_rate — float rounded to 3 decimal places (avoids float noise)
  language      — BCP-47 language tag (empty string if not set)
  extra_params  — sorted JSON of any additional synthesis parameters

Design notes
------------
- Text normalisation is intentionally conservative: we only collapse whitespace
  and apply Unicode NFC.  We do NOT strip punctuation or case-fold aggressively
  because punctuation and casing influence prosody in the TTS model.
- The digest is 64 hex characters (256 bits).  Collision probability for
  realistic TTS corpora (millions of unique phrases) is negligible.
- `CacheKey` is a plain frozen dataclass — cheap to create, hashable, and
  serialisable to JSON for Redis metadata storage.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

# Collapse runs of any whitespace (space, tab, newline, NBSP, etc.) to one space.
_WS_RE = re.compile(r"\s+")


def _normalise_text(text: str) -> str:
    """NFC-normalise + collapse internal whitespace + strip outer whitespace."""
    t = unicodedata.normalize("NFC", text)
    t = _WS_RE.sub(" ", t).strip()
    return t


def _round_rate(rate: float) -> str:
    """Represent speaking_rate as a stable 3-dp string, e.g. '1.000', '0.850'."""
    return f"{rate:.3f}"


@dataclass(frozen=True)
class CacheKey:
    """Immutable descriptor for one unique audio generation request.

    Attributes
    ----------
    digest        : hex SHA256 of all fields — used as the filename stem.
    normalised_text: text after normalisation (stored for debugging).
    voice_id      : canonical voice identifier.
    model_version : TTS model checkpoint identifier.
    speaking_rate : synthesis speed multiplier.
    language      : BCP-47 language tag.
    extra_params  : arbitrary sorted-JSON synthesis parameters.
    """

    digest: str
    normalised_text: str
    voice_id: str
    model_version: str
    speaking_rate: float
    language: str
    extra_params: str  # sorted JSON string

    # Convenience: the Redis metadata key for this cache entry.
    @property
    def meta_key(self) -> str:
        return f"audio:{self.digest}"

    # Convenience: the distributed lock key.
    @property
    def lock_key(self) -> str:
        return f"lock:{self.digest}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "normalised_text": self.normalised_text,
            "voice_id": self.voice_id,
            "model_version": self.model_version,
            "speaking_rate": self.speaking_rate,
            "language": self.language,
            "extra_params": self.extra_params,
        }


def make_cache_key(
    text: str,
    *,
    voice_id: str = "",
    model_version: str = "",
    speaking_rate: float = 1.0,
    language: str = "",
    extra_params: dict[str, Any] | None = None,
) -> CacheKey:
    """Build a :class:`CacheKey` from synthesis request parameters.

    Parameters
    ----------
    text          : raw text to synthesise (will be normalised).
    voice_id      : voice name, e.g. "simran", "tara".
    model_version : TTS checkpoint identifier, e.g. "v2.1.0" or a git hash.
                    If empty, the key will still be unique per text+voice but
                    will NOT invalidate when the model changes — set this!
    speaking_rate : synthesis speed multiplier (default 1.0 = normal speed).
    language      : BCP-47 tag, e.g. "hi-IN", "en-US".
    extra_params  : any additional synthesis parameters that affect audio output.

    Returns
    -------
    CacheKey with a stable digest derived from all fields.
    """
    norm_text = _normalise_text(text)
    rate_str = _round_rate(speaking_rate)
    extras_str = json.dumps(extra_params or {}, sort_keys=True, separators=(",", ":"))

    # Canonical string: NUL-delimited fields so that no field can bleed into
    # another (e.g. voice_id="ab\x00" text="c" ≠ voice_id="ab" text="\x00c").
    canonical = "\x00".join([
        norm_text,
        voice_id,
        model_version,
        rate_str,
        language,
        extras_str,
    ])

    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    return CacheKey(
        digest=digest,
        normalised_text=norm_text,
        voice_id=voice_id,
        model_version=model_version,
        speaking_rate=speaking_rate,
        language=language,
        extra_params=extras_str,
    )


def legacy_cache_key(text: str) -> str:
    """Reproduce the old bare-text SHA256 used by the existing cache dirs.

    This lets CacheManager fall back to the legacy filename format so that
    the existing 22GB+ per-voice WAV caches remain usable without migration.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
