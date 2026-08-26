"""Pipeline position: MODEL LOAD — weights, tokenizer, mel front end, vocoder.

Role in pipeline:
  Runs once at process start, before any port is bound. Everything downstream
  (text encoder, scheduler, backends, vocoder) takes a `LoadedModel` and never
  touches the filesystem or the Hub again.

      HF snapshot (gated)  ->  ZipVoice nn.Module   (fm_decoder + text_encoder + embed)
                           ->  SimpleTokenizer      (1058-char Indic vocab)
                           ->  VocosFbank           (24 kHz / hop 256 / 100 mel)
                           ->  Vocos                (mel -> waveform)

The DhVaani repo vendors a trimmed copy of upstream ZipVoice under `_backend/`.
Rather than committing a fork of that tree into FlowTTS we put the downloaded
snapshot's `_backend/` on `sys.path`, which is exactly what the model's own
`modeling_dhvaani.py` does. That keeps us byte-identical to whatever the model
authors shipped alongside the weights.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from flowtts.dhvaani.config import dhv_settings
from flowtts.dhvaani.types import DhvaaniError, EngineNotReady

logger = structlog.get_logger(__name__)

_TORCH_DTYPES = {
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float32": "float32",
}

# Files that must exist in the snapshot before we consider it usable.
_REQUIRED = ("model.safetensors", "tokens.txt", "model.json", "_backend")

_GATED_HELP = (
    "ARTPARK-IISc/DhVaani-0.5 is a gated repository. To download it:\n"
    "  1. Open https://huggingface.co/ARTPARK-IISc/DhVaani-0.5 while signed in "
    "and accept the model terms.\n"
    "  2. Create a token at https://huggingface.co/settings/tokens and export it:\n"
    "       export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx\n"
    "  3. Re-run, or pre-fetch with:\n"
    "       python -m flowtts.dhvaani.setup.fetch_model\n"
    "Alternatively point DHVAANI_MODEL__LOCAL_DIR at an existing snapshot."
)


class ModelDownloadError(DhvaaniError):
    status_code = 503
    code = "model_download_failed"


@dataclass
class LoadedModel:
    """Everything the engine needs from disk, loaded exactly once per process."""

    zipvoice: Any            # ZipVoice nn.Module, eval, on device
    tokenizer: Any           # SimpleTokenizer
    feature_extractor: Any   # VocosFbank
    vocoder: Any             # Vocos nn.Module, eval, on device
    device: Any              # torch.device
    dtype: Any               # torch.dtype
    repo_dir: str
    vocab_size: int
    pad_id: int
    sampling_rate: int

    def token_ids(self, text: str) -> list[int]:
        """Character-level tokenise, dropping out-of-vocabulary characters.

        `SimpleTokenizer` silently skips OOV; we keep that behaviour (it is what
        the model was trained to expect) but callers that care can diff the
        length against `len(text)` to count drops.
        """
        return self.tokenizer.texts_to_token_ids([text])[0]

    def n_oov(self, text: str) -> int:
        t2i = self.tokenizer.token2id
        return sum(1 for ch in text if ch not in t2i)


# ---------------------------------------------------------------------------
# Snapshot acquisition
# ---------------------------------------------------------------------------
def ensure_backend_on_path(repo_dir: str) -> None:
    """Put `<repo_dir>/_backend` on sys.path so `import zipvoice...` resolves."""
    backend = str(Path(repo_dir) / "_backend")
    if not Path(backend).is_dir():
        raise ModelDownloadError(
            f"{backend} not found -- the snapshot at {repo_dir} is incomplete. "
            "Re-run: python -m flowtts.dhvaani.setup.fetch_model --force"
        )
    if backend not in sys.path:
        sys.path.insert(0, backend)


def _snapshot_complete(d: Path) -> bool:
    return d.is_dir() and all((d / f).exists() for f in _REQUIRED)


def download_model(settings=None, force: bool = False) -> str:
    """Fetch (or locate) the DhVaani snapshot. Returns the local directory."""
    s = (settings or dhv_settings).model
    local = Path(s.local_dir).expanduser() if s.local_dir else None

    if local is not None and _snapshot_complete(local) and not force:
        logger.info("dhvaani_snapshot_present", path=str(local))
        return str(local)

    token = os.environ.get(s.hf_token_env) or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # pragma: no cover
        raise ModelDownloadError(f"huggingface_hub is required to fetch the model: {e}")

    logger.info("dhvaani_snapshot_download", repo=s.repo_id, dest=str(local))
    try:
        path = snapshot_download(
            s.repo_id,
            local_dir=str(local) if local else None,
            token=token,
            # The safetensors file is 491 MB; resume so a flaky link does not
            # restart the whole download.
            resume_download=True,
        )
    except Exception as e:
        msg = str(e)
        if "401" in msg or "403" in msg or "gated" in msg.lower() or "Unauthorized" in msg:
            raise ModelDownloadError(f"{msg}\n\n{_GATED_HELP}") from e
        raise ModelDownloadError(f"failed to download {s.repo_id}: {msg}") from e

    if not _snapshot_complete(Path(path)):
        missing = [f for f in _REQUIRED if not (Path(path) / f).exists()]
        raise ModelDownloadError(f"snapshot at {path} is missing: {missing}")
    return str(path)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
_loaded: LoadedModel | None = None
_load_lock = threading.Lock()


def _resolve_dtype(name: str):
    import torch

    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def _read_arch_config(repo_dir: Path) -> tuple[dict, int]:
    """ZipVoice kwargs + sampling rate, preferring model.json (the standalone
    path) and falling back to config.json's `zipvoice_config` (the AutoModel
    path). Both ship identical values; reading rather than hardcoding means a
    future checkpoint with a different width still loads."""
    mj = repo_dir / "model.json"
    if mj.exists():
        cfg = json.loads(mj.read_text())
        return dict(cfg["model"]), int(cfg["feature"]["sampling_rate"])
    cj = json.loads((repo_dir / "config.json").read_text())
    return dict(cj["zipvoice_config"]), int(cj.get("sampling_rate", 24000))


def _make_feature_extractor():
    """Vocos mel front end, preferring the vendored implementation.

    ``zipvoice.utils.feature`` imports torchaudio at module scope. On NVIDIA NGC
    containers torchaudio cannot be installed without replacing the container's
    custom torch build, so fall back to our own extractor -- which is asserted
    bit-identical in ``test_audio_compat.py``.
    """
    from flowtts.dhvaani.model.audio_compat import HAVE_TORCHAUDIO, VocosFbankCompat

    if HAVE_TORCHAUDIO:
        try:
            from zipvoice.utils.feature import VocosFbank

            return VocosFbank()
        except Exception as e:  # pragma: no cover
            logger.warning("vendored_vocosfbank_unavailable", error=str(e))
    logger.info("using_torchaudio_free_mel_frontend")
    return VocosFbankCompat()


def _load_vocoder(s, device):
    """Load the Vocos 24 kHz mel vocoder.

    We import `vocos` directly rather than going through
    `zipvoice.bin.infer_zipvoice.get_vocoder`: that module imports pydub, k2
    helpers and builds an argparse parser at import time, none of which we want
    in a server process. The construction below is what `get_vocoder` does.
    """
    # Vocos imports torchaudio and encodec at module scope for a feature
    # extractor the mel-24kHz vocoder never uses. Where those are unusable
    # (NGC containers), install stand-ins first so the import can complete.
    from flowtts.dhvaani.model.audio_compat import install_compat_shims

    shimmed = install_compat_shims()
    if shimmed:
        logger.info("installed_compat_shims", modules=shimmed)

    try:
        from vocos import Vocos
    except ImportError as e:  # pragma: no cover
        raise ModelDownloadError(
            "the `vocos` package is required for the 24 kHz vocoder: pip install vocos"
        ) from e

    if s.vocoder_local_dir:
        import torch

        d = Path(s.vocoder_local_dir)
        voc = Vocos.from_hparams(str(d / "config.yaml"))
        sd = torch.load(str(d / "pytorch_model.bin"), weights_only=True, map_location="cpu")
        voc.load_state_dict(sd)
    else:
        voc = Vocos.from_pretrained(s.vocoder_repo)
    return voc.to(device).eval()


def load_model(settings=None) -> LoadedModel:
    """Idempotent process-wide load. Safe to call from multiple threads."""
    global _loaded
    if _loaded is not None:
        return _loaded

    with _load_lock:
        if _loaded is not None:
            return _loaded

        import torch

        st = settings or dhv_settings
        s = st.model
        t0 = time.perf_counter()

        repo_dir = Path(download_model(st))
        ensure_backend_on_path(str(repo_dir))

        from zipvoice.models.zipvoice import ZipVoice
        from zipvoice.tokenizer.tokenizer import SimpleTokenizer

        device = torch.device(s.device if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            logger.warning("dhvaani_cpu_fallback", requested=s.device)
        dtype = _resolve_dtype(s.dtype) if device.type == "cuda" else torch.float32

        arch, sampling_rate = _read_arch_config(repo_dir)
        tokenizer = SimpleTokenizer(token_file=str(repo_dir / "tokens.txt"))

        model = ZipVoice(**arch, vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id)

        from safetensors.torch import load_file

        sd = load_file(str(repo_dir / "model.safetensors"))
        # The checkpoint namespaces everything under "model." because it was
        # saved through the PreTrainedModel wrapper, which holds the ZipVoice as
        # `self.model`.
        sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
        model.load_state_dict(sd, strict=True)

        model = model.to(device).eval()
        model.requires_grad_(False)

        # Swap the k2-fallback activations for capture-safe, dtype-preserving
        # equivalents. Without this, CUDA graphs cannot be captured at all and
        # every activation silently upcasts to fp32. See model/runtime_patch.py.
        from flowtts.dhvaani.model.runtime_patch import make_inference_safe

        make_inference_safe(model)

        if dtype != torch.float32:
            # The flow decoder is ~95% of the FLOPs and is what the TRT engines
            # are built in, so it carries the low-precision cast. The text
            # encoder is 4 layers at width 192 -- keeping it fp32 costs well
            # under a millisecond and its output is reused by every ODE step, so
            # any error there is amplified num_step times.
            model.fm_decoder.to(dtype)
            model.embed.to(dtype if not s.text_encoder_fp32 else torch.float32)
            if not s.text_encoder_fp32:
                model.text_encoder.to(dtype)

        vocoder = _load_vocoder(s, device)
        vocoder.requires_grad_(False)

        feature_extractor = _make_feature_extractor()

        _loaded = LoadedModel(
            zipvoice=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            vocoder=vocoder,
            device=device,
            dtype=dtype,
            repo_dir=str(repo_dir),
            vocab_size=tokenizer.vocab_size,
            pad_id=tokenizer.pad_id,
            sampling_rate=sampling_rate,
        )

        n_params = sum(p.numel() for p in model.parameters())
        vram = None
        if device.type == "cuda":
            vram = round(torch.cuda.memory_allocated(device) / 2**20, 1)
        logger.info(
            "dhvaani_model_loaded",
            repo_dir=str(repo_dir),
            params_m=round(n_params / 1e6, 1),
            device=str(device),
            dtype=str(dtype),
            text_encoder_fp32=s.text_encoder_fp32,
            vocab_size=tokenizer.vocab_size,
            sampling_rate=sampling_rate,
            vram_mib=vram,
            load_s=round(time.perf_counter() - t0, 2),
        )
        return _loaded


def get_loaded_model() -> LoadedModel:
    if _loaded is None:
        raise EngineNotReady("model not loaded -- call load_model() during startup")
    return _loaded


def is_loaded() -> bool:
    return _loaded is not None
