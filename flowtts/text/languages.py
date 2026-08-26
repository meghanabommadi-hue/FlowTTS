"""Pipeline position: TEXT PREPROCESSING — language registry (pure stdlib).

Role in pipeline:
  Single source of truth for "what language is this, what script does it use,
  how do we say a number in it, and what code does OmniVoice want?".

  api / ws  → resolve_language("hindi") → "hi"
            → get_profile("hi")         → LanguageProfile(script="deva", …)
            → omnivoice_lang("hi")      → "hi"   (ISO 639-3 where OmniVoice differs)

Ported and extended from github.com/Ajaj-Ali/text_preprocessor_for_TTS
(`indic_tts_normalizer.languages`). Extensions over upstream:
  • all 22 scheduled Indian languages + common dialects (upstream had 9 Indic),
  • Perso-Arabic / Ol Chiki / Meetei Mayek / Odia digit tables,
  • per-language spoken words (point / minus / percent / currency / digits) so a
    language can degrade to digit-by-digit reading when no cardinal backend
    supports it, instead of leaving bare numerals in the text,
  • ``omnivoice_code`` — OmniVoice keys some Indic languages by ISO 639-3
    ("ory", "npi", "dgo", "knn") and rejects the 639-1 code, so requests must be
    translated before they reach ``model.generate(language=…)``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Unicode codepoint of the "zero" digit for each script we support. The full
# 0-9 run is derived from this one offset rather than hand-listing 10 digits.
_SCRIPT_ZERO_CODEPOINT = {
    "deva": 0x0966,  # Devanagari — hi, mr, ne, sa, kok, mai, doi, brx, bho, hne, mag, raj, awa
    "beng": 0x09E6,  # Bengali    — bn, as, mni (Bengali orthography)
    "guru": 0x0A66,  # Gurmukhi   — pa
    "gujr": 0x0AE6,  # Gujarati   — gu
    "orya": 0x0B66,  # Odia       — or
    "taml": 0x0BE6,  # Tamil      — ta
    "telu": 0x0C66,  # Telugu     — te
    "knda": 0x0CE6,  # Kannada    — kn
    "mlym": 0x0D66,  # Malayalam  — ml
    "arab": 0x0660,  # Arabic-Indic — ur, sd, ks  (U+06F0 Eastern variant handled below)
    "olck": 0x1C50,  # Ol Chiki   — sat
    "mtei": 0xABF0,  # Meetei Mayek — mni
}

# Urdu/Sindhi/Kashmiri text mixes Arabic-Indic (U+0660) and Extended
# Arabic-Indic (U+06F0) digits; both must map to ASCII.
_ARABIC_EXTENDED_ZERO = 0x06F0


def _native_digit_table(script: str) -> dict:
    base = _SCRIPT_ZERO_CODEPOINT[script]
    table = {chr(base + d): str(d) for d in range(10)}
    if script == "arab":
        table.update({chr(_ARABIC_EXTENDED_ZERO + d): str(d) for d in range(10)})
    return str.maketrans(table)


@dataclass(frozen=True)
class LanguageProfile:
    """Everything the normalizer needs to speak numbers/symbols in one language."""

    code: str
    name: str
    script: Optional[str]        # key into _SCRIPT_ZERO_CODEPOINT; None for Latin
    number_backend: str          # "num2words" | "indic_num2words" | "digits"
    number_lang: str             # code handed to that backend
    omnivoice_code: str          # what OmniVoice's LANG_IDS actually accepts
    month_names: dict = field(default_factory=dict)   # 1..12 → localized name
    digit_words: tuple = ()      # ("zero","one",…) in this language, for fallbacks
    words: dict = field(default_factory=dict)         # point/minus/percent/currency

    @property
    def digit_translation_table(self) -> Optional[dict]:
        if self.script is None:
            return None
        return _native_digit_table(self.script)


_ENGLISH_MONTHS = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November",
    12: "December",
}

_EN_CURRENCY = {"₹": "rupees", "$": "dollars", "€": "euros", "£": "pounds", "¥": "yen"}
_EN_WORDS = {"point": "point", "minus": "minus", "percent": "percent",
             "currency": _EN_CURRENCY}
_EN_DIGITS = ("zero", "one", "two", "three", "four", "five", "six", "seven",
              "eight", "nine")

# Per-language spoken forms. These lean toward everyday spoken usage (loanwords
# included) rather than the most literary equivalent — that is what a call-centre
# or IVR listener expects to hear.
_HI_WORDS = {"point": "दशमलव", "minus": "माइनस", "percent": "प्रतिशत",
             "currency": {"₹": "रुपये", "$": "डॉलर", "€": "यूरो", "£": "पाउंड", "¥": "येन"}}
_HI_DIGITS = ("शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ")

_MR_WORDS = {"point": "दशांश", "minus": "उणे", "percent": "टक्के",
             "currency": {"₹": "रुपये", "$": "डॉलर", "€": "युरो", "£": "पाउंड", "¥": "येन"}}
_MR_DIGITS = ("शून्य", "एक", "दोन", "तीन", "चार", "पाच", "सहा", "सात", "आठ", "नऊ")

_BN_WORDS = {"point": "দশমিক", "minus": "মাইনাস", "percent": "শতাংশ",
             "currency": {"₹": "টাকা", "$": "ডলার", "€": "ইউরো", "£": "পাউন্ড", "¥": "ইয়েন"}}
_BN_DIGITS = ("শূন্য", "এক", "দুই", "তিন", "চার", "পাঁচ", "ছয়", "সাত", "আট", "নয়")

_AS_WORDS = {"point": "দশমিক", "minus": "মাইনাচ", "percent": "শতাংশ",
             "currency": {"₹": "টকা", "$": "ডলাৰ", "€": "ইউৰো", "£": "পাউণ্ড", "¥": "য়েন"}}
_AS_DIGITS = ("শূন্য", "এক", "দুই", "তিনি", "চাৰি", "পাঁচ", "ছয়", "সাত", "আঠ", "ন")

_PA_WORDS = {"point": "ਦਸ਼ਮਲਵ", "minus": "ਮਾਈਨਸ", "percent": "ਫ਼ੀਸਦੀ",
             "currency": {"₹": "ਰੁਪਏ", "$": "ਡਾਲਰ", "€": "ਯੂਰੋ", "£": "ਪਾਊਂਡ", "¥": "ਯੇਨ"}}
_PA_DIGITS = ("ਸਿਫ਼ਰ", "ਇੱਕ", "ਦੋ", "ਤਿੰਨ", "ਚਾਰ", "ਪੰਜ", "ਛੇ", "ਸੱਤ", "ਅੱਠ", "ਨੌਂ")

_GU_WORDS = {"point": "દશાંશ", "minus": "માઈનસ", "percent": "ટકા",
             "currency": {"₹": "રૂપિયા", "$": "ડોલર", "€": "યુરો", "£": "પાઉન્ડ", "¥": "યેન"}}
_GU_DIGITS = ("શૂન્ય", "એક", "બે", "ત્રણ", "ચાર", "પાંચ", "છ", "સાત", "આઠ", "નવ")

_OR_WORDS = {"point": "ଦଶମିକ", "minus": "ମାଇନସ୍", "percent": "ପ୍ରତିଶତ",
             "currency": {"₹": "ଟଙ୍କା", "$": "ଡଲାର", "€": "ୟୁରୋ", "£": "ପାଉଣ୍ଡ", "¥": "ୟେନ"}}
_OR_DIGITS = ("ଶୂନ୍ୟ", "ଏକ", "ଦୁଇ", "ତିନି", "ଚାରି", "ପାଞ୍ଚ", "ଛଅ", "ସାତ", "ଆଠ", "ନଅ")

_TA_WORDS = {"point": "புள்ளி", "minus": "மைனஸ்", "percent": "சதவீதம்",
             "currency": {"₹": "ரூபாய்", "$": "டாலர்", "€": "யூரோ", "£": "பவுண்ட்", "¥": "யென்"}}
_TA_DIGITS = ("பூஜ்ஜியம்", "ஒன்று", "இரண்டு", "மூன்று", "நான்கு", "ஐந்து", "ஆறு",
              "ஏழு", "எட்டு", "ஒன்பது")

_TE_WORDS = {"point": "పాయింట్", "minus": "మైనస్", "percent": "శాతం",
             "currency": {"₹": "రూపాయలు", "$": "డాలర్లు", "€": "యూరోలు", "£": "పౌండ్లు", "¥": "యెన్"}}
_TE_DIGITS = ("సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు",
              "ఎనిమిది", "తొమ్మిది")

_KN_WORDS = {"point": "ಪಾಯಿಂಟ್", "minus": "ಮೈನಸ್", "percent": "ಶೇಕಡಾ",
             "currency": {"₹": "ರೂಪಾಯಿ", "$": "ಡಾಲರ್", "€": "ಯುರೋ", "£": "ಪೌಂಡ್", "¥": "ಯೆನ್"}}
_KN_DIGITS = ("ಸೊನ್ನೆ", "ಒಂದು", "ಎರಡು", "ಮೂರು", "ನಾಲ್ಕು", "ಐದು", "ಆರು", "ಏಳು",
              "ಎಂಟು", "ಒಂಬತ್ತು")

_ML_WORDS = {"point": "പോയിന്റ്", "minus": "മൈനസ്", "percent": "ശതമാനം",
             "currency": {"₹": "രൂപ", "$": "ഡോളർ", "€": "യൂറോ", "£": "പൗണ്ട്", "¥": "യെൻ"}}
_ML_DIGITS = ("പൂജ്യം", "ഒന്ന്", "രണ്ട്", "മൂന്ന്", "നാല്", "അഞ്ച്", "ആറ്", "ഏഴ്",
              "എട്ട്", "ഒൻപത്")

_UR_WORDS = {"point": "اعشاریہ", "minus": "مائنس", "percent": "فیصد",
             "currency": {"₹": "روپے", "$": "ڈالر", "€": "یورو", "£": "پاؤنڈ", "¥": "ین"}}
_UR_DIGITS = ("صفر", "ایک", "دو", "تین", "چار", "پانچ", "چھ", "سات", "آٹھ", "نو")

_SD_WORDS = {"point": "اعشاريہ", "minus": "مائنس", "percent": "سيڪڙو",
             "currency": {"₹": "رپيا", "$": "ڊالر", "€": "يورو", "£": "پائونڊ", "¥": "ين"}}
_SD_DIGITS = ("ٻڙي", "هڪ", "ٻه", "ٽي", "چار", "پنج", "ڇهه", "ست", "اٺ", "نو")

_KS_WORDS = {"point": "اعشاریہ", "minus": "مائنس", "percent": "فیصد",
             "currency": {"₹": "رۄپیہ", "$": "ڈالر", "€": "یورو", "£": "پاونڈ", "¥": "ین"}}
_KS_DIGITS = ("صفر", "اَکھ", "زٕ", "ترٚے", "ژور", "پانٛژ", "شے", "ست", "ٲٹھ", "نو")

_NE_WORDS = {"point": "दशमलव", "minus": "माइनस", "percent": "प्रतिशत",
             "currency": {"₹": "रुपैयाँ", "$": "डलर", "€": "युरो", "£": "पाउन्ड", "¥": "येन"}}
_NE_DIGITS = ("शून्य", "एक", "दुई", "तीन", "चार", "पाँच", "छ", "सात", "आठ", "नौ")

_SA_WORDS = {"point": "दशमलवः", "minus": "ऋणम्", "percent": "प्रतिशतम्",
             "currency": {"₹": "रूप्यकाणि", "$": "डॉलर्", "€": "यूरो", "£": "पाउण्ड्", "¥": "येन्"}}
_SA_DIGITS = ("शून्यम्", "एकम्", "द्वे", "त्रीणि", "चत्वारि", "पञ्च", "षट्", "सप्त",
              "अष्ट", "नव")

_KOK_WORDS = {"point": "दशांश", "minus": "उणे", "percent": "टक्के",
              "currency": {"₹": "रुपया", "$": "डॉलर", "€": "युरो", "£": "पाउंड", "¥": "येन"}}
_KOK_DIGITS = ("शून्य", "एक", "दोन", "तीन", "चार", "पांच", "सव", "सात", "आठ", "नोव")

_MNI_WORDS = {"point": "দশমিক", "minus": "মাইনস", "percent": "শতাংশ",
              "currency": {"₹": "টকা", "$": "ডলার", "€": "ইউরো", "£": "পাউন্ড", "¥": "য়েন"}}
_MNI_DIGITS = ("ফুম", "অমা", "অনি", "অহুম", "মরি", "মঙা", "তরুক", "তরেৎ",
               "নিপান", "মাপল")

_SAT_WORDS = {"point": "ᱫᱚᱥᱢᱤᱠ", "minus": "ᱢᱟᱭᱱᱟᱥ", "percent": "ᱥᱚᱛᱟᱝᱥᱚ",
              "currency": {"₹": "ᱴᱟᱠᱟ", "$": "ᱰᱚᱞᱟᱨ", "€": "ᱭᱩᱨᱚ", "£": "ᱯᱟᱳᱱᱰ", "¥": "ᱭᱮᱱ"}}
_SAT_DIGITS = ("ᱮᱴᱟᱜ", "ᱢᱤᱫ", "ᱵᱟᱨ", "ᱯᱮ", "ᱯᱩᱱ", "ᱢᱚᱬᱮ", "ᱛᱩᱨᱩᱭ", "ᱮᱭᱟᱭ",
               "ᱤᱨᱟᱹᱞ", "ᱟᱨᱮ")

_BRX_WORDS = {"point": "दसमलव", "minus": "माइनास", "percent": "प्रतिसत",
              "currency": {"₹": "रुपि", "$": "डलार", "€": "युरो", "£": "पाउनद", "¥": "येन"}}
_BRX_DIGITS = ("लाथिख", "से", "नै", "थाम", "ब्रै", "बा", "ऱो", "सनि", "दाइन", "गु")

# Every profile the normalizer knows. `omnivoice_code` is the value that must
# reach model.generate(language=…) — several Indic languages are keyed there by
# ISO 639-3 and a 639-1 code would be rejected.
_PROFILES = {
    # ── Latin script ──────────────────────────────────────────────────────
    "en":    LanguageProfile("en", "English", None, "num2words", "en", "en",
                             _ENGLISH_MONTHS, _EN_DIGITS, _EN_WORDS),
    "en-IN": LanguageProfile("en-IN", "Indian English", None, "num2words", "en_IN",
                             "en", _ENGLISH_MONTHS, _EN_DIGITS, _EN_WORDS),

    # ── 22 scheduled languages of India ───────────────────────────────────
    "hi":  LanguageProfile("hi", "Hindi", "deva", "indic_num2words", "hi", "hi",
                           {}, _HI_DIGITS, _HI_WORDS),
    "bn":  LanguageProfile("bn", "Bengali", "beng", "indic_num2words", "bn", "bn",
                           {}, _BN_DIGITS, _BN_WORDS),
    "mr":  LanguageProfile("mr", "Marathi", "deva", "indic_num2words", "mr", "mr",
                           {}, _MR_DIGITS, _MR_WORDS),
    "te":  LanguageProfile("te", "Telugu", "telu", "indic_num2words", "te", "te",
                           {}, _TE_DIGITS, _TE_WORDS),
    "ta":  LanguageProfile("ta", "Tamil", "taml", "indic_num2words", "ta", "ta",
                           {}, _TA_DIGITS, _TA_WORDS),
    "gu":  LanguageProfile("gu", "Gujarati", "gujr", "indic_num2words", "gu", "gu",
                           {}, _GU_DIGITS, _GU_WORDS),
    "ur":  LanguageProfile("ur", "Urdu", "arab", "num2words", "ur", "ur",
                           {}, _UR_DIGITS, _UR_WORDS),
    "kn":  LanguageProfile("kn", "Kannada", "knda", "indic_num2words", "kn", "kn",
                           {}, _KN_DIGITS, _KN_WORDS),
    "or":  LanguageProfile("or", "Odia", "orya", "indic_num2words", "or", "ory",
                           {}, _OR_DIGITS, _OR_WORDS),
    "ml":  LanguageProfile("ml", "Malayalam", "mlym", "indic_num2words", "ml", "ml",
                           {}, _ML_DIGITS, _ML_WORDS),
    "pa":  LanguageProfile("pa", "Punjabi", "guru", "indic_num2words", "pa", "pa",
                           {}, _PA_DIGITS, _PA_WORDS),
    "as":  LanguageProfile("as", "Assamese", "beng", "indic_num2words", "bn", "as",
                           {}, _AS_DIGITS, _AS_WORDS),
    "mai": LanguageProfile("mai", "Maithili", "deva", "indic_num2words", "hi", "mai",
                           {}, _HI_DIGITS, _HI_WORDS),
    "sat": LanguageProfile("sat", "Santali", "olck", "digits", "sat", "sat",
                           {}, _SAT_DIGITS, _SAT_WORDS),
    "ks":  LanguageProfile("ks", "Kashmiri", "arab", "digits", "ks", "ks",
                           {}, _KS_DIGITS, _KS_WORDS),
    "ne":  LanguageProfile("ne", "Nepali", "deva", "indic_num2words", "ne", "npi",
                           {}, _NE_DIGITS, _NE_WORDS),
    "sd":  LanguageProfile("sd", "Sindhi", "arab", "digits", "sd", "sd",
                           {}, _SD_DIGITS, _SD_WORDS),
    "kok": LanguageProfile("kok", "Konkani", "deva", "indic_num2words", "mr", "knn",
                           {}, _KOK_DIGITS, _KOK_WORDS),
    "doi": LanguageProfile("doi", "Dogri", "deva", "indic_num2words", "hi", "dgo",
                           {}, _HI_DIGITS, _HI_WORDS),
    "mni": LanguageProfile("mni", "Manipuri", "beng", "digits", "mni", "mni",
                           {}, _MNI_DIGITS, _MNI_WORDS),
    "brx": LanguageProfile("brx", "Bodo", "deva", "digits", "brx", "brx",
                           {}, _BRX_DIGITS, _BRX_WORDS),
    "sa":  LanguageProfile("sa", "Sanskrit", "deva", "digits", "sa", "sa",
                           {}, _SA_DIGITS, _SA_WORDS),

    # ── Hindi-belt dialects: Hindi's script, backend and wording ──────────
    "bho": LanguageProfile("bho", "Bhojpuri", "deva", "indic_num2words", "hi", "bho",
                           {}, _HI_DIGITS, _HI_WORDS),
    "hne": LanguageProfile("hne", "Chhattisgarhi", "deva", "indic_num2words", "hi", "hne",
                           {}, _HI_DIGITS, _HI_WORDS),
    "mag": LanguageProfile("mag", "Magahi", "deva", "indic_num2words", "hi", "mag",
                           {}, _HI_DIGITS, _HI_WORDS),
    "awa": LanguageProfile("awa", "Awadhi", "deva", "indic_num2words", "hi", "awa",
                           {}, _HI_DIGITS, _HI_WORDS),
    "raj": LanguageProfile("raj", "Rajasthani", "deva", "indic_num2words", "hi", "raj",
                           {}, _HI_DIGITS, _HI_WORDS),

    # ── Other widely-served Indian languages ──────────────────────────────
    "tcy": LanguageProfile("tcy", "Tulu", "knda", "indic_num2words", "kn", "tcy",
                           {}, _KN_DIGITS, _KN_WORDS),
}

# Languages with no cardinal backend (neither num2words nor indic-num2words
# covers them): numbers are read digit-by-digit in their own script via
# LanguageProfile.digit_words. Correct speech, less natural prosody — see
# numbers.say_integer for the fallback chain.
DIGIT_FALLBACK_LANGUAGES = ("ur", "sd", "ks", "sa", "mni", "sat", "brx")

SUPPORTED_LANGUAGES = tuple(sorted(_PROFILES))

# Alternate spellings seen in the wild. Extend at runtime with
# register_language_alias() rather than forking.
_ALIASES = {
    "eng": "en", "en-us": "en", "en_us": "en", "en-gb": "en", "english": "en",
    "en-in": "en-IN", "en_in": "en-IN", "enin": "en-IN", "indian english": "en-IN",
    "hin": "hi", "hindi": "hi", "hinglish": "hi",
    "ben": "bn", "bengali": "bn", "bangla": "bn",
    "mar": "mr", "marathi": "mr",
    "tel": "te", "telugu": "te",
    "tam": "ta", "tamil": "ta",
    "guj": "gu", "gujarati": "gu",
    "urd": "ur", "urdu": "ur",
    "kan": "kn", "kannada": "kn",
    "ory": "or", "ori": "or", "odia": "or", "oriya": "or",
    "mal": "ml", "malayalam": "ml",
    "pan": "pa", "pun": "pa", "punjabi": "pa", "panjabi": "pa",
    "asm": "as", "assamese": "as",
    "maithili": "mai",
    "santali": "sat", "sant": "sat",
    "kashmiri": "ks", "kas": "ks",
    "npi": "ne", "nep": "ne", "nepali": "ne",
    "snd": "sd", "sindhi": "sd",
    "knn": "kok", "gom": "kok", "konkani": "kok",
    "dgo": "doi", "dogri": "doi",
    "manipuri": "mni", "meitei": "mni", "meiteilon": "mni",
    "bodo": "brx", "boro": "brx",
    "san": "sa", "sanskrit": "sa",
    "bhojpuri": "bho", "chhattisgarhi": "hne", "magahi": "mag",
    "awadhi": "awa", "rajasthani": "raj", "marwari": "raj",
    "tulu": "tcy",
}

# Script → default language, used when detecting the language of a text run
# whose language was not declared. Devanagari defaults to Hindi and Bengali
# script to Bengali: by speaker count those are the overwhelmingly likely
# reading, and an explicit `language` on the request always wins.
SCRIPT_DEFAULT_LANGUAGE = {
    "deva": "hi", "beng": "bn", "guru": "pa", "gujr": "gu", "orya": "or",
    "taml": "ta", "telu": "te", "knda": "kn", "mlym": "ml", "arab": "ur",
    "olck": "sat", "mtei": "mni", "latn": "en-IN",
}


# One table mapping EVERY supported script's digits to ASCII. Applied once, up
# front, so a Devanagari numeral inside a Latin run (or the reverse) in
# code-mixed text still normalizes — a per-segment table would miss those.
_ALL_DIGITS_TABLE = {}
for _script in _SCRIPT_ZERO_CODEPOINT:
    _ALL_DIGITS_TABLE.update(_native_digit_table(_script))


def to_ascii_digits(text: str) -> str:
    """Rewrite native digits of any supported script as ASCII 0-9."""
    return text.translate(_ALL_DIGITS_TABLE)


def register_language_alias(alias: str, canonical: str) -> None:
    """Map an additional language-code spelling onto a supported canonical code."""
    key = canonical if canonical in _PROFILES else canonical.lower()
    if key not in _PROFILES:
        raise ValueError(f"Unknown canonical language code: {canonical!r}")
    _ALIASES[alias.lower()] = key


def resolve_language(code: Optional[str]) -> str:
    """Normalize any language spelling to a key of SUPPORTED_LANGUAGES.

    Falls back to ``"en"`` for unknown input — normalization must never raise on
    a language it does not recognize, since OmniVoice itself supports 600+
    languages we do not carry number tables for.
    """
    if not code:
        return "en"
    raw = code.strip()
    if raw in _PROFILES:
        return raw
    lowered = raw.lower()
    if lowered in _ALIASES:
        return _ALIASES[lowered]
    for key in _PROFILES:
        if key.lower() == lowered:
            return key
    base = lowered.split("-")[0].split("_")[0]
    if base in _PROFILES:
        return base
    if base in _ALIASES:
        return _ALIASES[base]
    logger.debug("Unrecognized language code %r; normalizing as 'en'", code)
    return "en"


def get_profile(code: Optional[str]) -> LanguageProfile:
    return _PROFILES[resolve_language(code)]


def is_known_language(code: Optional[str]) -> bool:
    """True if *code* maps to a profile we actually have tables for."""
    if not code:
        return False
    raw = code.strip()
    return raw in _PROFILES or raw.lower() in _ALIASES or raw.lower() in _PROFILES


def omnivoice_lang(code: Optional[str]) -> Optional[str]:
    """Translate a request language into the code OmniVoice's LANG_IDS accepts.

    Unknown codes are passed through untouched: OmniVoice covers 600+ languages
    and resolves names as well as codes, so a code we have no profile for is far
    more likely to be one of those than a typo. ``None`` stays ``None``
    (language-agnostic mode).
    """
    if not code:
        return None
    raw = code.strip()
    if not is_known_language(raw):
        return raw
    return _PROFILES[resolve_language(raw)].omnivoice_code
