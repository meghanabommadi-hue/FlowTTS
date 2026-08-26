"""Pipeline position: LANGUAGE REGISTRY — the 27 languages DhVaani speaks.

Role in pipeline:
  Resolves a request's language (explicit code, or detected from the script the
  text is written in) and tells the normaliser how much it can do for that
  language.

Two different language sets are in play and they do not match:
  * DhVaani speaks 27 Indian languages.
  * `indic_tts_normalizer` has full number/date/abbreviation support for 14.
The other 13 get a "partial" pass here -- native digits are mapped to ASCII and
symbols are expanded, but numbers are left as digits rather than spelled out.
That degrades gracefully instead of raising, and `/v1/languages` reports the
tier per language so callers know what they are getting.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

# Unicode "digit zero" codepoint per Brahmic script, used to map native digits
# to ASCII. Perso-Arabic and Latin scripts have no single mapping here (Urdu can
# use either ASCII or Eastern-Arabic digits), handled separately.
SCRIPT_ZERO = {
    "Deva": 0x0966,
    "Beng": 0x09E6,
    "Guru": 0x0A66,
    "Gujr": 0x0AE6,
    "Orya": 0x0B66,
    "Taml": 0x0BE6,
    "Telu": 0x0C66,
    "Knda": 0x0CE6,
    "Mlym": 0x0D66,
    "Olck": None,     # Ol Chiki has its own digits but they are rare in practice
    "Arab": 0x0660,   # Eastern Arabic-Indic
    "Mtei": None,
    "Latn": None,
}

# Unicode block ranges used for script detection, most specific first.
_BLOCKS = [
    (0x0900, 0x097F, "Deva"),
    (0x0980, 0x09FF, "Beng"),
    (0x0A00, 0x0A7F, "Guru"),
    (0x0A80, 0x0AFF, "Gujr"),
    (0x0B00, 0x0B7F, "Orya"),
    (0x0B80, 0x0BFF, "Taml"),
    (0x0C00, 0x0C7F, "Telu"),
    (0x0C80, 0x0CFF, "Knda"),
    (0x0D00, 0x0D7F, "Mlym"),
    (0x0600, 0x06FF, "Arab"),
    (0x0750, 0x077F, "Arab"),
    (0xFB50, 0xFDFF, "Arab"),
    (0xFE70, 0xFEFF, "Arab"),
    (0x1C50, 0x1C7F, "Olck"),
    (0xABC0, 0xABFF, "Mtei"),
    (0x0041, 0x024F, "Latn"),
]


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    native_name: str
    script: str
    # The code to hand indic_tts_normalizer, or None when it has no profile.
    normalizer_lang: str | None

    @property
    def tier(self) -> str:
        return "full" if self.normalizer_lang else "partial"

    @property
    def digit_zero(self) -> int | None:
        return SCRIPT_ZERO.get(self.script)


# All 27 languages from the DhVaani model card, in its own order.
DHVAANI_LANGUAGES: dict[str, Language] = {
    lang.code: lang
    for lang in [
        # --- well represented in training ---
        Language("en",  "English",       "English",       "Latn", "en-IN"),
        Language("hi",  "Hindi",         "हिन्दी",          "Deva", "hi"),
        Language("bn",  "Bengali",       "বাংলা",          "Beng", "bn"),
        Language("mr",  "Marathi",       "मराठी",          "Deva", "mr"),
        Language("kn",  "Kannada",       "ಕನ್ನಡ",          "Knda", "kn"),
        Language("te",  "Telugu",        "తెలుగు",         "Telu", "te"),
        Language("mai", "Maithili",      "मैथिली",          "Deva", "mai"),
        Language("mag", "Magahi",        "मगही",           "Deva", "mag"),
        Language("hne", "Chhattisgarhi", "छत्तीसगढ़ी",       "Deva", "hne"),
        Language("bho", "Bhojpuri",      "भोजपुरी",         "Deva", "bho"),
        Language("as",  "Assamese",      "অসমীয়া",        "Beng", None),
        Language("ta",  "Tamil",         "தமிழ்",          "Taml", "ta"),
        # --- mid resource ---
        Language("gu",  "Gujarati",      "ગુજરાતી",         "Gujr", "gu"),
        Language("ml",  "Malayalam",     "മലയാളം",        "Mlym", "ml"),
        Language("mni", "Manipuri",      "ꯃꯤꯇꯩꯂꯣꯟ",       "Mtei", None),
        Language("or",  "Odia",          "ଓଡ଼ିଆ",           "Orya", None),
        Language("pa",  "Punjabi",       "ਪੰਜਾਬੀ",          "Guru", "pa"),
        Language("ne",  "Nepali",        "नेपाली",          "Deva", None),
        Language("sd",  "Sindhi",        "سنڌي",          "Arab", None),
        Language("kok", "Konkani",       "कोंकणी",          "Deva", None),
        Language("sat", "Santali",       "ᱥᱟᱱᱛᱟᱲᱤ",        "Olck", None),
        Language("brx", "Bodo",          "बड़ो",            "Deva", None),
        Language("ur",  "Urdu",          "اردو",           "Arab", None),
        # --- low resource ---
        Language("sa",  "Sanskrit",      "संस्कृतम्",        "Deva", None),
        Language("doi", "Dogri",         "डोगरी",           "Deva", None),
        Language("ks",  "Kashmiri",      "کٲشُر",          "Arab", None),
        Language("raj", "Rajasthani",    "राजस्थानी",       "Deva", None),
    ]
}

# Alternate spellings seen in the wild.
_ALIASES = {
    "eng": "en", "en-us": "en", "en_us": "en", "en-gb": "en",
    "en-in": "en", "en_in": "en", "hin": "hi", "ben": "bn", "asm": "as",
    "mar": "mr", "kan": "kn", "tel": "te", "tam": "ta", "guj": "gu",
    "mal": "ml", "ori": "or", "ory": "or", "pan": "pa", "pun": "pa",
    "nep": "ne", "snd": "sd", "kon": "kok", "san": "sa", "urd": "ur",
    "mni-mtei": "mni", "bodo": "brx", "raj-in": "raj",
}

# When we can only tell the script, this is the most probable language for it.
# Devanagari is overwhelmingly Hindi in production traffic; Bengali script is
# ambiguous between Bengali and Assamese and we cannot distinguish them from
# characters alone, so an explicit `language` is the only way to get Assamese.
_SCRIPT_DEFAULT = {
    "Deva": "hi", "Beng": "bn", "Guru": "pa", "Gujr": "gu", "Orya": "or",
    "Taml": "ta", "Telu": "te", "Knda": "kn", "Mlym": "ml", "Arab": "ur",
    "Olck": "sat", "Mtei": "mni", "Latn": "en",
}

SUPPORTED_LANGUAGES = tuple(DHVAANI_LANGUAGES)


def _block_of(ch: str) -> str | None:
    cp = ord(ch)
    for lo, hi, name in _BLOCKS:
        if lo <= cp <= hi:
            return name
    return None


def detect_script(text: str) -> str:
    """Dominant Unicode script in `text`.

    Latin is only reported when nothing else is present: Indic text routinely
    embeds English words ("EMI", "OTP"), and letting those outvote the Indic
    characters would misroute normalisation for exactly the code-switched
    traffic this system sees most.
    """
    counts: dict[str, int] = {}
    for ch in text:
        if ch.isspace() or unicodedata.category(ch).startswith(("P", "N", "S")):
            continue
        b = _block_of(ch)
        if b:
            counts[b] = counts.get(b, 0) + 1
    if not counts:
        return "Latn"
    non_latin = {k: v for k, v in counts.items() if k != "Latn"}
    src = non_latin or counts
    return max(src.items(), key=lambda kv: kv[1])[0]


def detect_language(text: str, default: str = "hi") -> str:
    script = detect_script(text)
    return _SCRIPT_DEFAULT.get(script, default)


def resolve(code: str | None, text: str = "", default: str = "hi") -> str:
    """Explicit code wins; otherwise detect from the script; otherwise default."""
    if code:
        raw = code.strip()
        if raw in DHVAANI_LANGUAGES:
            return raw
        low = raw.lower()
        if low in DHVAANI_LANGUAGES:
            return low
        if low in _ALIASES:
            return _ALIASES[low]
        base = low.split("-")[0].split("_")[0]
        if base in DHVAANI_LANGUAGES:
            return base
        if base in _ALIASES:
            return _ALIASES[base]
    if text:
        return detect_language(text, default)
    return default if default in DHVAANI_LANGUAGES else "hi"


def get(code: str) -> Language:
    return DHVAANI_LANGUAGES[resolve(code)]


def script_of(code: str) -> str:
    return get(code).script


def is_supported(code: str | None) -> bool:
    if not code:
        return False
    low = code.strip().lower()
    return low in DHVAANI_LANGUAGES or low in _ALIASES


# Perso-Arabic script is written with two different digit sets: Arabic-Indic
# (U+0660..U+0669) in Arabic proper, and Extended Arabic-Indic (U+06F0..U+06F9)
# in Urdu, Sindhi and Kashmiri. Real text mixes them, so map both.
_ARAB_ZEROS = (0x0660, 0x06F0)


def native_digit_table(code: str) -> dict | None:
    """`str.translate` table mapping this language's native digits to ASCII."""
    info = get(code)
    if info.script == "Arab":
        table = {}
        for zero in _ARAB_ZEROS:
            table.update({zero + d: str(d) for d in range(10)})
        return table
    zero = info.digit_zero
    if zero is None:
        return None
    return {zero + d: str(d) for d in range(10)}


def describe_all() -> list[dict]:
    """Payload for `GET /v1/languages`."""
    return [
        {
            "code": lang.code,
            "name": lang.name,
            "native_name": lang.native_name,
            "script": lang.script,
            "normalization": lang.tier,
        }
        for lang in DHVAANI_LANGUAGES.values()
    ]
