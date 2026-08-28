#!/usr/bin/env python3
"""NaijaVox (Whisper) forced-alignment service.

Runs on the GPU box. Given long-form audio plus its ground-truth transcript, it
returns word-level timings so the caller can cut the audio into TTS-sized
chunks that carry the CORRECT text.

Why ASR-then-align rather than raw ASR text: the corpus already has human
transcripts. Whisper is used only to locate words in time; the text written to
the dataset stays the ground truth, so transcription errors cannot leak into
training data.

POST /align  {audio_b64, transcript, language}  -> {words:[{w,start,end}], ...}
GET  /health
"""
from __future__ import annotations

import base64, io, os, threading, time

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from transformers import (GenerationConfig, WhisperForConditionalGeneration,
                          WhisperProcessor)

MODEL_ID = os.environ.get("NAIJAVOX_MODEL", "Axiveri/NaijaVox-2.0")
DEVICE = os.environ.get("ALIGN_DEVICE", "cuda")
DTYPE = torch.float16 if DEVICE.startswith("cuda") else torch.float32
SR = 16_000
# NaijaVox language tags -> whisper language codes it was trained with
# NaijaVox defines <|ig|> and <|pcm|> natively, so use them directly.
LANG_MAP = {"ig": "ig", "yo": "yo", "ha": "ha", "pcm": "pcm", "en": "en"}

app = FastAPI(title="naijavox-align")
_model = None
_proc = None
_lock = threading.Lock()


BASE_ID = os.environ.get("WHISPER_BASE", "openai/whisper-large-v3")


def _fix_generation_config(model, proc):
    """NaijaVox ships a generation_config with every timestamp field missing.

    It is a whisper-large-v3 fine-tune with identical architecture and matching
    special-token ids (verified: notimestamps=50364, transcribe=50360,
    ha=50354, yo=50325), so the base config's timestamp fields and DTW
    alignment heads transfer directly. It additionally defines two NEW language
    tokens the base model has no concept of - <|ig|>=51866 and <|pcm|>=51867 -
    so the language map is extended rather than copied wholesale.
    """
    gc = model.generation_config
    if getattr(gc, "no_timestamps_token_id", None) is not None:
        return gc
    base = GenerationConfig.from_pretrained(BASE_ID)
    for key in ("no_timestamps_token_id", "alignment_heads",
                "max_initial_timestamp_index", "task_to_id",
                "begin_suppress_tokens", "suppress_tokens",
                "prev_sot_token_id", "is_multilingual"):
        val = getattr(base, key, None)
        if val is not None:
            setattr(gc, key, val)

    lang_to_id = dict(getattr(base, "lang_to_id", {}) or {})
    tk = proc.tokenizer
    for tag in ("<|ig|>", "<|pcm|>"):
        tid = tk.convert_tokens_to_ids(tag)
        if tid is not None and tid != tk.convert_tokens_to_ids("<|endoftext|>"):
            lang_to_id[tag] = tid
    gc.lang_to_id = lang_to_id
    if getattr(gc, "temperature", None) is None:
        gc.temperature = 0.0
    model.generation_config = gc
    print(f"patched generation_config from {BASE_ID}; "
          f"{len(lang_to_id)} languages incl. ig/pcm", flush=True)
    return gc


def load():
    global _model, _proc
    if _model is None:
        t0 = time.time()
        _proc = WhisperProcessor.from_pretrained(MODEL_ID)
        _model = WhisperForConditionalGeneration.from_pretrained(
            MODEL_ID, dtype=DTYPE).to(DEVICE).eval()
        _fix_generation_config(_model, _proc)
        print(f"loaded {MODEL_ID} on {DEVICE} in {time.time()-t0:.0f}s", flush=True)
    return _model, _proc


_pipe = None


def _pipeline():
    """Long-form ASR pipeline with word timestamps (chunked internally)."""
    global _pipe
    if _pipe is None:
        from transformers import pipeline as hf_pipeline
        model, proc = load()
        _pipe = hf_pipeline(
            "automatic-speech-recognition", model=model,
            tokenizer=proc.tokenizer, feature_extractor=proc.feature_extractor,
            chunk_length_s=30, stride_length_s=5,
            device=0 if DEVICE.startswith("cuda") else -1,
            dtype=DTYPE,
        )
    return _pipe


class AlignReq(BaseModel):
    audio_b64: str
    transcript: str = ""
    language: str = "en"
    sample_rate: int = SR


@app.get("/health")
def health():
    m, _ = load()
    free, total = (torch.cuda.mem_get_info() if DEVICE.startswith("cuda")
                   else (0, 0))
    return {"ok": True, "model": MODEL_ID, "device": DEVICE,
            "vram_free_gb": round(free / 1e9, 1), "vram_total_gb": round(total / 1e9, 1)}


def _decode_audio(b64: str, sr: int):
    raw = base64.b64decode(b64)
    wav, got_sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if got_sr != SR:
        import librosa
        wav = librosa.resample(wav, orig_sr=got_sr, target_sr=SR, res_type="soxr_hq")
    return wav.astype(np.float32)


@app.post("/align")
def align(req: AlignReq):
    """Word-level timings for the whole (possibly very long) recording."""
    model, proc = load()
    try:
        wav = _decode_audio(req.audio_b64, req.sample_rate)
    except Exception as e:
        raise HTTPException(400, f"audio decode failed: {e!r}")
    if wav.size < SR // 2:
        raise HTTPException(400, "audio shorter than 0.5s")

    lang = LANG_MAP.get(req.language, "en")
    pipe = _pipeline()
    with _lock:
        out = pipe(
            wav,
            return_timestamps="word",
            generate_kwargs={"language": lang, "task": "transcribe",
                             "max_new_tokens": 440},
        )

    words = []
    for ch in out.get("chunks", []) or []:
        ts = ch.get("timestamp") or (None, None)
        txt = (ch.get("text") or "").strip()
        if not txt or ts[0] is None:
            continue
        start = float(ts[0])
        end = float(ts[1]) if ts[1] is not None else start
        words.append({"w": txt, "start": round(start, 3), "end": round(end, 3)})

    return {"duration": len(wav) / SR, "language": lang,
            "n_words": len(words), "words": words,
            "asr_text": (out.get("text") or "").strip()}


# ---- OpenAI-compatible transcription -------------------------------------- #
# tts-bench's `vllm`/`openai` ASR backend posts multipart audio to
# /v1/audio/transcriptions. Serving that here means WER for Igbo/Yoruba/Hausa/
# Pidgin is scored by a model that actually knows those languages, instead of
# stock Whisper which has no Igbo or Pidgin token at all.
@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(default=MODEL_ID),
    language: str = Form(default="en"),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
):
    raw = await file.read()
    try:
        wav, got_sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    except Exception as e:
        raise HTTPException(400, f"audio decode failed: {e!r}")
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1)
    if got_sr != SR:
        import librosa
        wav = librosa.resample(wav, orig_sr=got_sr, target_sr=SR, res_type="soxr_hq")
    wav = np.asarray(wav, dtype=np.float32)
    if wav.size < SR // 20:
        return {"text": ""}

    lang = LANG_MAP.get((language or "en").lower().split("-")[0], "en")
    pipe = _pipeline()
    with _lock:
        out = pipe(wav, return_timestamps=False,
                   generate_kwargs={"language": lang, "task": "transcribe",
                                    "max_new_tokens": 440})
    return {"text": (out.get("text") or "").strip(), "language": lang}


@app.get("/v1/models")
def list_models():
    return {"object": "list",
            "data": [{"id": MODEL_ID, "object": "model", "owned_by": "axiveri"}]}


if __name__ == "__main__":
    import uvicorn
    load()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("ALIGN_PORT", "8899")),
                log_level="warning")
