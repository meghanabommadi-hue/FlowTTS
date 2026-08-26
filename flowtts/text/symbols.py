"""Pipeline position: TEXT PREPROCESSING — standalone symbol → word.

Ported from github.com/Ajaj-Ali/text_preprocessor_for_TTS
(`indic_tts_normalizer.symbols`), extended to every language that has a profile
so an Indic segment never falls back to English wording mid-sentence.

Currency symbols are handled in numbers.py (they attach to a number); this
module covers symbols standing on their own. "%" is listed here as well for the
case where it appears without a preceding number ("discount %").
"""

from __future__ import annotations

import re

from flowtts.text.languages import resolve_language

_EN = {"&": "and", "@": "at", "#": "number", "+": "plus", "=": "equals",
       "*": "star", "~": "approximately", "%": "percent"}
_HI = {"&": "और", "@": "एट", "#": "हैश", "+": "प्लस", "=": "बराबर",
       "*": "स्टार", "~": "लगभग", "%": "प्रतिशत"}
_MR = {**_HI, "&": "आणि", "%": "टक्के"}
_BN = {"&": "এবং", "@": "অ্যাট", "#": "হ্যাশ", "+": "প্লাস", "=": "সমান",
       "*": "স্টার", "~": "প্রায়", "%": "শতাংশ"}
_TA = {"&": "மற்றும்", "@": "அட்", "#": "ஹாஷ்", "+": "பிளஸ்", "=": "சமம்",
       "*": "நட்சத்திரம்", "~": "தோராயமாக", "%": "சதவீதம்"}
_TE = {"&": "మరియు", "@": "అట్", "#": "హాష్", "+": "ప్లస్", "=": "సమానం",
       "*": "స్టార్", "~": "సుమారు", "%": "శాతం"}
_KN = {"&": "ಮತ್ತು", "@": "ಅಟ್", "#": "ಹ್ಯಾಶ್", "+": "ಪ್ಲಸ್", "=": "ಸಮ",
       "*": "ಸ್ಟಾರ್", "~": "ಸುಮಾರು", "%": "ಶೇಕಡಾ"}
_ML = {"&": "ഒപ്പം", "@": "അറ്റ്", "#": "ഹാഷ്", "+": "പ്ലസ്", "=": "സമം",
       "*": "സ്റ്റാർ", "~": "ഏകദേശം", "%": "ശതമാനം"}
_GU = {"&": "અને", "@": "એટ", "#": "હેશ", "+": "પ્લસ", "=": "બરાબર",
       "*": "સ્ટાર", "~": "આશરે", "%": "ટકા"}
_PA = {"&": "ਅਤੇ", "@": "ਐਟ", "#": "ਹੈਸ਼", "+": "ਪਲੱਸ", "=": "ਬਰਾਬਰ",
       "*": "ਸਟਾਰ", "~": "ਲਗਭਗ", "%": "ਫ਼ੀਸਦੀ"}
_OR = {"&": "ଏବଂ", "@": "ଆଟ", "#": "ହ୍ୟାସ୍", "+": "ପ୍ଲସ୍", "=": "ସମାନ",
       "*": "ଷ୍ଟାର", "~": "ପ୍ରାୟ", "%": "ପ୍ରତିଶତ"}
_UR = {"&": "اور", "@": "ایٹ", "#": "ہیش", "+": "پلس", "=": "برابر",
       "*": "سٹار", "~": "تقریباً", "%": "فیصد"}
_AS = {**_BN, "&": "আৰু"}

SYMBOLS: dict[str, dict[str, str]] = {
    "en": dict(_EN), "en-IN": dict(_EN),
    "hi": dict(_HI), "mr": dict(_MR), "bn": dict(_BN), "as": dict(_AS),
    "ta": dict(_TA), "te": dict(_TE), "kn": dict(_KN), "ml": dict(_ML),
    "gu": dict(_GU), "pa": dict(_PA), "or": dict(_OR), "ur": dict(_UR),
    # Devanagari-family dialects and Nepali/Sanskrit/Konkani share Hindi wording.
    "ne": dict(_HI), "sa": dict(_HI), "kok": dict(_MR), "mai": dict(_HI),
    "doi": dict(_HI), "brx": dict(_HI), "bho": dict(_HI), "hne": dict(_HI),
    "mag": dict(_HI), "awa": dict(_HI), "raj": dict(_HI),
    # Perso-Arabic family shares Urdu wording; Tulu shares Kannada.
    "sd": dict(_UR), "ks": dict(_UR), "tcy": dict(_KN),
    "mni": dict(_BN), "sat": dict(_EN),
}

_pattern_cache: dict[str, re.Pattern] = {}


def _rebuild_pattern(lang: str) -> None:
    table = SYMBOLS.get(lang) or _EN
    _pattern_cache[lang] = re.compile("|".join(re.escape(s) for s in table))


def register_symbol(lang: str, symbol: str, word: str) -> None:
    """Add or override a symbol's spoken word for a language at runtime."""
    canonical = resolve_language(lang)
    SYMBOLS.setdefault(canonical, {})[symbol] = word
    _rebuild_pattern(canonical)


def expand_symbols(text: str, lang: str) -> str:
    canonical = resolve_language(lang)
    if canonical not in _pattern_cache:
        _rebuild_pattern(canonical)
    table = SYMBOLS.get(canonical) or _EN
    return _pattern_cache[canonical].sub(lambda m: f" {table[m.group(0)]} ", text)
