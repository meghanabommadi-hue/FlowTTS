"""Pipeline position: TEXT NORMALISATION — written form -> spoken form.

Role in pipeline:
  First stage of every request. Turns "आपकी EMI Rs.2,500 है, 15/08 तक जमा करें"
  into something a character-level TTS can actually pronounce, then hands the
  result to the chunker.

      raw text -> [normalise] -> [chunk] -> [tokenise] -> scheduler

Backed by `indic_tts_normalizer` (github.com/Ajaj-Ali/text_preprocessor_for_TTS).
Two deliberate deviations from its `normalize_text()` entry point:

  1. We call its stage functions directly rather than its pipeline, because
     `normalize_text` unconditionally lowercases. DhVaani's vocabulary contains
     both ASCII cases, so destroying capitalisation throws away information the
     model can use. `text.lowercase` controls it here.
  2. It has profiles for 14 languages; DhVaani speaks 27. The other 13 get a
     local partial pass (native digits -> ASCII, symbols, whitespace) instead of
     a `KeyError`.

Latency
-------
Normalisation is pure Python regex and runs in tens of microseconds for typical
IVR text -- but it is on the critical path to first audio, and pathological
input (a wall of digits) can cost milliseconds. Two defences: an LRU cache
keyed by (text, language, config), which IVR traffic hits constantly because it
replays the same prompts; and an optional thread pool so the event loop is never
blocked. The cache is the reason this stage does not appear in the p99 budget.
"""

from __future__ import annotations

import asyncio
import re
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import structlog

from flowtts.dhvaani.config import dhv_settings
from flowtts.dhvaani.text import lang as langmod

logger = structlog.get_logger(__name__)

_WS = re.compile(r"\s+")

# Codepoint ranges stripped before tokenisation: characters DhVaani has no token
# for and that carry no prosody. The tokenizer would drop them silently;
# removing them here keeps the OOV counter meaningful.
#
# Deliberately ABSENT: U+200C ZWNJ and U+200D ZWJ. Those are meaningful inside
# Indic conjuncts -- stripping them changes how a word is rendered and spoken.
_STRIP_RANGES = (
    (0x0000, 0x0008), (0x000B, 0x000C), (0x000E, 0x001F),   # C0 control
    (0x007F, 0x009F),                                        # DEL + C1 control
    (0x200B, 0x200B),                                        # zero-width space
    (0x200E, 0x200F), (0x202A, 0x202E),                      # bidi overrides
    (0x2060, 0x206F),                                        # word joiner, invisibles
    (0xFEFF, 0xFEFF),                                        # BOM / ZWNBSP
    (0x2600, 0x27BF),                                        # misc symbols, dingbats
    (0xFE0F, 0xFE0F),                                        # emoji variation selector
    (0x1F000, 0x1FAFF),                                      # emoji planes
)
_STRIP = re.compile(
    "[" + "".join("%s-%s" % (chr(a), chr(b)) for a, b in _STRIP_RANGES) + "]"
)

# Currency and operator symbols expanded locally for the "partial" language tier
# (the 13 languages indic_tts_normalizer has no profile for).
_SYMBOL_WORDS = {
    "₹": {   # RUPEE SIGN
        "hi": "रुपये",
        "mr": "रुपये",
        "ne": "रुपैयाँ",
        "bn": "টাকা",
        "as": "টকা",
        "ta": "ரூபாய்",
        "te": "రూపాయలు",
        "kn": "ರೂಪಾಯಿ",
        "ml": "രൂപ",
        "gu": "રૂપિયા",
        "pa": "ਰੁਪਏ",
        "or": "ଟଙ୍କା",
        "ur": "روپے",
        "en": "rupees",
    },
    "$": {"en": "dollars", "hi": "डॉलर"},
    "%": {
        "en": "percent",
        "hi": "प्रतिशत",
        "mr": "टक्के",
        "bn": "শতাংশ",
        "ta": "சதவீதம்",
        "te": "శాతం",
        "kn": "ಶೇಕಡಾ",
        "ml": "ശതമാനം",
        "gu": "ટકા",
        "pa": "ਫ਼ੀਸਦੀ",
        "or": "ପ୍ରତିଶତ",
        "ur": "فیصد",
    },
    "&": {"en": "and", "hi": "और"},
    "@": {"en": "at", "hi": "एट"},
}


class _LRU:
    """Small locked LRU. `functools.lru_cache` cannot report stats per instance
    or be resized from config, and this is hot enough to want both."""

    def __init__(self, maxsize: int):
        self.maxsize = max(1, maxsize)
        self._d: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                self.hits += 1
                return self._d[key]
            self.misses += 1
            return None

    def put(self, key, value) -> None:
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "size": len(self._d),
            "maxsize": self.maxsize,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }


class TextNormalizer:
    """Normalises text for DhVaani, with a cache in front of it."""

    def __init__(self, settings=None):
        self._s = settings or dhv_settings
        c = self._s.text
        self._cache = _LRU(c.cache_size)
        self._pool = (
            ThreadPoolExecutor(
                max_workers=c.normalize_threads, thread_name_prefix="dhv_norm"
            )
            if c.normalize_threads > 0
            else None
        )
        self._fingerprint = ":".join(
            str(int(getattr(c, k)))
            for k in (
                "contractions", "datetime", "urls_emails", "otp_digit_splitting",
                "numbers", "abbreviations", "symbols", "lowercase",
            )
        )
        self._lib = self._load_library()
        self._warned_partial: set = set()
        self._register_extras()

    # -- library -------------------------------------------------------------
    def _load_library(self):
        try:
            from indic_tts_normalizer import abbreviations as _abbr
            from indic_tts_normalizer import contact_info as _contact
            from indic_tts_normalizer import contractions as _contr
            from indic_tts_normalizer import datetime_expand as _dt
            from indic_tts_normalizer import numbers as _num
            from indic_tts_normalizer import sanitize as _san
            from indic_tts_normalizer import symbols as _sym

            return {
                "sanitize": _san.sanitize,
                "collapse": _san.collapse_whitespace,
                "contractions": _contr.expand_contractions,
                "datetime": _dt.expand_datetime,
                "urls": _contact.expand_urls_and_emails,
                "digits": _contact.split_long_digit_runs,
                "numbers": _num.expand_numbers,
                "abbreviations": _abbr.expand_abbreviations,
                "symbols": _sym.expand_symbols,
            }
        except ImportError as e:
            logger.warning(
                "indic_tts_normalizer_unavailable",
                error=str(e),
                hint=(
                    "pip install "
                    "'git+https://github.com/Ajaj-Ali/text_preprocessor_for_TTS.git' "
                    "-- falling back to sanitise + digit mapping only"
                ),
            )
            return None

    # Abbreviations that dominate Indian voice-bot traffic but are not in the
    # library's starter dictionaries. Registered once at construction so the
    # library's own expansion stage picks them up.
    _EXTRA_ABBREVIATIONS = {
        "en": {
            "rs.": "rupees", "rs": "rupees", "emi": "E M I", "otp": "O T P",
            "kyc": "K Y C", "ivr": "I V R", "neft": "N E F T", "imps": "I M P S",
            "upi": "U P I", "rtgs": "R T G S", "nach": "NACH", "a/c": "account",
            "ltd.": "limited", "pvt.": "private", "no.": "number",
            "cr.": "crore", "lac": "lakh", "lacs": "lakhs", "&": "and",
        },
        "hi": {
            "rs.": "रुपये", "rs": "रुपये", "emi": "ई एम आई", "otp": "ओ टी पी",
            "kyc": "के वाई सी", "upi": "यू पी आई", "a/c": "अकाउंट",
        },
    }

    def _register_extras(self) -> None:
        if not self._lib:
            return
        try:
            from indic_tts_normalizer import register_abbreviation
        except ImportError:
            return
        for lang_code, mapping in self._EXTRA_ABBREVIATIONS.items():
            for k, v in mapping.items():
                try:
                    register_abbreviation(lang_code, k, v)
                except Exception:
                    # A rejected registration is not worth failing startup over.
                    pass

    @property
    def available(self) -> bool:
        return self._lib is not None

    # -- normalisation -------------------------------------------------------
    def normalize(self, text: str, language: str) -> str:
        if not text:
            return text
        code = langmod.resolve(language, text, self._s.text.default_language)
        if not self._s.text.normalize:
            return _WS.sub(" ", _STRIP.sub("", text)).strip()

        key = (text, code, self._fingerprint)
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        try:
            out = self._normalize_uncached(text, code)
        except Exception as e:
            # A normalisation bug must never fail a synthesis request; the
            # unnormalised text is still perfectly speakable, just less natural.
            logger.warning("normalize_failed", language=code, error=str(e)[:200])
            out = _WS.sub(" ", _STRIP.sub("", text)).strip()

        self._cache.put(key, out)
        return out

    def _normalize_uncached(self, text: str, code: str) -> str:
        c = self._s.text
        info = langmod.get(code)

        text = _STRIP.sub("", text)
        if self._lib:
            text = self._lib["sanitize"](text)
        if c.lowercase:
            text = text.lower()

        # Native digits -> ASCII must happen before any number handling.
        table = langmod.native_digit_table(code)
        if table:
            text = text.translate(table)

        if self._lib and info.normalizer_lang:
            nl = info.normalizer_lang
            if c.contractions and nl in ("en", "en-IN"):
                text = self._lib["contractions"](text)
            if c.datetime:
                text = self._lib["datetime"](text, nl)
            if c.urls_emails:
                text = self._lib["urls"](text)
            if c.otp_digit_splitting:
                text = self._lib["digits"](text, nl)
            if c.numbers:
                text = self._lib["numbers"](text, nl)
            if c.abbreviations:
                text = self._lib["abbreviations"](text, nl)
            if c.symbols:
                text = self._lib["symbols"](text, nl)
            return self._lib["collapse"](text)

        # Partial tier: no number-to-words backend exists for this language, so
        # digits are left as digits. DhVaani pronounces bare digits acceptably
        # in most Indic scripts; spelling them out via another language's number
        # words would be worse than leaving them alone.
        if code not in self._warned_partial:
            self._warned_partial.add(code)
            logger.info("normalize_partial_tier", language=code, script=info.script)
        if c.symbols:
            text = self._expand_symbols_local(text, code)
        return _WS.sub(" ", text).strip()

    @staticmethod
    def _expand_symbols_local(text: str, code: str) -> str:
        for sym, per_lang in _SYMBOL_WORDS.items():
            if sym in text:
                word = per_lang.get(code) or per_lang.get("en")
                if word:
                    text = text.replace(sym, " " + word + " ")
        return text

    # -- async ---------------------------------------------------------------
    async def normalize_async(self, text: str, language: str) -> str:
        # Serve cache hits inline: hopping to a thread costs more than the
        # dictionary lookup it would be dispatching.
        code = langmod.resolve(language, text, self._s.text.default_language)
        hit = self._cache.get((text, code, self._fingerprint))
        if hit is not None:
            return hit
        if self._pool is None:
            return self.normalize(text, language)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, self.normalize, text, language)

    # -- misc ----------------------------------------------------------------
    @staticmethod
    def oov_report(text: str, token2id: dict) -> int:
        return sum(1 for ch in text if ch not in token2id)

    def cache_stats(self) -> dict:
        d = self._cache.stats()
        d["library_available"] = self.available
        d["lowercase"] = self._s.text.lowercase
        return d

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
