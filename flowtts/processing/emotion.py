"""Pipeline position: TEXT PREPROCESSING (called from server.py's HTTP /tts handler).

Role in pipeline:
  Auto-detects an emotion voice tag for text that arrives without one.
  Wraps tabularisai/multilingual-emotion-classification (XLM-RoBERTa-base,
  multi-label, 11 emotions, 23 languages incl. Hindi/English) and maps its
  top-scoring emotion onto one of the four voice tags server.py already
  understands: angry, angrier, sad, happy (or "" for neutral/no tag).

  Client text (no [tag])
    │
    ▼
  detect_tag(text)  [this module — sigmoid classifier, CPU or GPU]
    │  → "angry" | "angrier" | "sad" | "happy" | ""
    ▼
  server.py:_http_tts  prepends "[tag] " before existing tag-parsing logic

Model card: https://huggingface.co/tabularisai/multilingual-emotion-classification

Loaded lazily and cached — first call pays model-load cost, every call after
is just a forward pass. Runs on CPU by default so it doesn't compete with the
TTS model for GPU memory; set FLOWTTS_EMOTION_DEVICE=cuda to change that.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "tabularisai/multilingual-emotion-classification"
DEFAULT_THRESHOLD = 0.5

# Map the classifier's 11 labels onto FlowTTS's 4 preloaded voice tags.
# Labels not listed (e.g. "neutral") fall through to "" — the default voice.
_LABEL_TO_TAG = {
    "anger": "angrier",
    "contempt": "angry",
    "disgust": "angry",
    "frustration": "angry",
    "fear": "sad",
    "sadness": "sad",
    "joy": "happy",
    "love": "happy",
    "gratitude": "happy",
    "surprise": "happy",
    "neutral": "",
}


class _EmotionTagger:
    def __init__(self) -> None:
        device_pref = os.environ.get("FLOWTTS_EMOTION_DEVICE", "cpu")
        self.device = device_pref if (device_pref == "cpu" or torch.cuda.is_available()) else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    @torch.inference_mode()
    def top_emotion(self, text: str) -> tuple[str, float]:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        logits = self.model(**inputs).logits[0]
        probs = torch.sigmoid(logits)
        idx = int(torch.argmax(probs))
        return self.id2label[idx], float(probs[idx])


@lru_cache(maxsize=1)
def _get_tagger() -> _EmotionTagger:
    return _EmotionTagger()


def detect_tag(text: str, threshold: float = DEFAULT_THRESHOLD) -> str:
    """Return the FlowTTS voice tag ("angry"/"angrier"/"sad"/"happy"/"") for *text*.

    Below `threshold` confidence, returns "" (default/neutral voice) rather
    than guessing — a low-confidence top score isn't a reliable signal.
    """
    if not text or not text.strip():
        return ""
    tagger = _get_tagger()
    label, score = tagger.top_emotion(text)
    if score < threshold:
        return ""
    return _LABEL_TO_TAG.get(label, "")
