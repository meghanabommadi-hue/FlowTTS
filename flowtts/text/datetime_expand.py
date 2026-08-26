"""Pipeline position: TEXT PREPROCESSING — dates and times → spoken form.

Ported from github.com/Ajaj-Ali/text_preprocessor_for_TTS
(`indic_tts_normalizer.datetime_expand`), with localized month names and hour
markers added for the major Indic languages — upstream ships them for English
only and every other language reads a date as three bare numbers.

Dates are ambiguous: "03/04/2026" is 3 April in India and 4 March in the US.
This module reads DD/MM/YYYY, matching the locale the service targets; the
component order is a module constant so it is one edit, not a rewrite, if that
ever needs to change.
"""

from __future__ import annotations

import re

from flowtts.text.languages import LanguageProfile, get_profile
from flowtts.text.numbers import say_integer, say_ordinal

# Interpret bare numeric dates as day-first (Indian/European convention).
DAY_FIRST = True

_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_NUMERIC_DATE_RE = re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b")
_TIME_RE = re.compile(
    r"\b(\d{1,2}):(\d{2})(?::\d{2})?\s?(a\.?m\.?|p\.?m\.?)?", re.IGNORECASE
)

# "at N o'clock" markers. Indic languages put the marker after the number.
_HOUR_MARKER = {
    "hi": "बजे", "mr": "वाजता", "ne": "बजे", "bho": "बजे", "mag": "बजे",
    "hne": "बजे", "mai": "बजे", "doi": "बजे", "raj": "बजे", "awa": "बजे",
    "bn": "টা", "as": "বজাত", "gu": "વાગ્યે", "pa": "ਵਜੇ", "or": "ଟାରେ",
    "ta": "மணி", "te": "గంటలకు", "kn": "ಗಂಟೆಗೆ", "ml": "മണിക്ക്",
    "ur": "بجے", "sd": "وڳي", "ks": "بجے", "kok": "वराँ",
}

_MONTHS = {
    "hi": ["जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई",
           "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"],
    "mr": ["जानेवारी", "फेब्रुवारी", "मार्च", "एप्रिल", "मे", "जून", "जुलै",
           "ऑगस्ट", "सप्टेंबर", "ऑक्टोबर", "नोव्हेंबर", "डिसेंबर"],
    "bn": ["জানুয়ারি", "ফেব্রুয়ারি", "মার্চ", "এপ্রিল", "মে", "জুন", "জুলাই",
           "আগস্ট", "সেপ্টেম্বর", "অক্টোবর", "নভেম্বর", "ডিসেম্বর"],
    "gu": ["જાન્યુઆરી", "ફેબ્રુઆરી", "માર્ચ", "એપ્રિલ", "મે", "જૂન", "જુલાઈ",
           "ઓગસ્ટ", "સપ્ટેમ્બર", "ઓક્ટોબર", "નવેમ્બર", "ડિસેમ્બર"],
    "pa": ["ਜਨਵਰੀ", "ਫ਼ਰਵਰੀ", "ਮਾਰਚ", "ਅਪ੍ਰੈਲ", "ਮਈ", "ਜੂਨ", "ਜੁਲਾਈ",
           "ਅਗਸਤ", "ਸਤੰਬਰ", "ਅਕਤੂਬਰ", "ਨਵੰਬਰ", "ਦਸੰਬਰ"],
    "ta": ["ஜனவரி", "பிப்ரவரி", "மார்ச்", "ஏப்ரல்", "மே", "ஜூன்", "ஜூலை",
           "ஆகஸ்ட்", "செப்டம்பர்", "அக்டோபர்", "நவம்பர்", "டிசம்பர்"],
    "te": ["జనవరి", "ఫిబ్రవరి", "మార్చి", "ఏప్రిల్", "మే", "జూన్", "జూలై",
           "ఆగస్టు", "సెప్టెంబర్", "అక్టోబర్", "నవంబర్", "డిసెంబర్"],
    "kn": ["ಜನವರಿ", "ಫೆಬ್ರವರಿ", "ಮಾರ್ಚ್", "ಏಪ್ರಿಲ್", "ಮೇ", "ಜೂನ್", "ಜುಲೈ",
           "ಆಗಸ್ಟ್", "ಸೆಪ್ಟೆಂಬರ್", "ಅಕ್ಟೋಬರ್", "ನವೆಂಬರ್", "ಡಿಸೆಂಬರ್"],
    "ml": ["ജനുവരി", "ഫെബ്രുവരി", "മാർച്ച്", "ഏപ്രിൽ", "മേയ്", "ജൂൺ", "ജൂലൈ",
           "ഓഗസ്റ്റ്", "സെപ്റ്റംബർ", "ഒക്ടോബർ", "നവംബർ", "ഡിസംബർ"],
    "or": ["ଜାନୁଆରୀ", "ଫେବୃଆରୀ", "ମାର୍ଚ୍ଚ", "ଏପ୍ରିଲ", "ମଇ", "ଜୁନ", "ଜୁଲାଇ",
           "ଅଗଷ୍ଟ", "ସେପ୍ଟେମ୍ବର", "ଅକ୍ଟୋବର", "ନଭେମ୍ବର", "ଡିସେମ୍ବର"],
    "ur": ["جنوری", "فروری", "مارچ", "اپریل", "مئی", "جون", "جولائی",
           "اگست", "ستمبر", "اکتوبر", "نومبر", "دسمبر"],
    "as": ["জানুৱাৰী", "ফেব্ৰুৱাৰী", "মাৰ্চ", "এপ্ৰিল", "মে", "জুন", "জুলাই",
           "আগষ্ট", "ছেপ্তেম্বৰ", "অক্টোবৰ", "নৱেম্বৰ", "ডিচেম্বৰ"],
    "ne": ["जनवरी", "फेब्रुअरी", "मार्च", "अप्रिल", "मे", "जुन", "जुलाई",
           "अगस्ट", "सेप्टेम्बर", "अक्टोबर", "नोभेम्बर", "डिसेम्बर"],
}
# Devanagari dialects and Konkani read Hindi month names.
for _d in ("bho", "hne", "mag", "mai", "doi", "raj", "awa", "sa", "brx"):
    _MONTHS[_d] = _MONTHS["hi"]
_MONTHS["kok"] = _MONTHS["mr"]
_MONTHS["sd"] = _MONTHS["ur"]
_MONTHS["ks"] = _MONTHS["ur"]
_MONTHS["mni"] = _MONTHS["bn"]
_MONTHS["tcy"] = _MONTHS["kn"]


def _month_name(month: int, profile: LanguageProfile) -> str | None:
    if profile.month_names:
        return profile.month_names[month]
    table = _MONTHS.get(profile.code)
    return table[month - 1] if table else None


def _normalize_year(raw: str) -> int:
    year = int(raw)
    if len(raw) == 2:
        year += 2000 if year < 50 else 1900
    return year


def _speak_date(day: int, month: int, year: int, profile: LanguageProfile) -> str | None:
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return None
    month_word = _month_name(month, profile)
    if month_word is None:
        # No month table for this language: read the three components plainly.
        return " ".join(
            (say_integer(day, profile), say_integer(month, profile), say_integer(year, profile))
        )
    if profile.month_names:  # English-family: "the third of April two thousand and twenty six"
        return (
            f"the {say_ordinal(day, profile)} of {month_word.lower()} "
            f"{say_integer(year, profile)}"
        )
    # Indic order: <day> <month> <year>, e.g. "तीन अप्रैल दो हज़ार छब्बीस"
    return f"{say_integer(day, profile)} {month_word} {say_integer(year, profile)}"


def _speak_time(hour: int, minute: int, meridiem: str | None,
                profile: LanguageProfile) -> str | None:
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None

    if profile.month_names:  # English-family
        hour_word = say_integer(hour, profile)
        if minute == 0:
            spoken = f"{hour_word} o'clock"
        elif minute < 10:
            spoken = f"{hour_word} oh {say_integer(minute, profile)}"
        else:
            spoken = f"{hour_word} {say_integer(minute, profile)}"
        if meridiem:
            cleaned = meridiem.replace(".", "").lower()
            spoken += " a m" if cleaned == "am" else " p m"
        return spoken

    parts = [say_integer(hour, profile)]
    marker = _HOUR_MARKER.get(profile.code)
    if marker:
        parts.append(marker)
    if minute:
        parts.append(say_integer(minute, profile))
    return " ".join(parts)


def expand_datetime(text: str, lang: str) -> str:
    """Expand ISO/numeric dates and HH:MM[:SS] [am/pm] times into words."""
    profile = get_profile(lang)

    def _iso(m: re.Match) -> str:
        spoken = _speak_date(int(m.group(3)), int(m.group(2)), int(m.group(1)), profile)
        return spoken if spoken is not None else m.group(0)

    def _numeric(m: re.Match) -> str:
        a, b = int(m.group(1)), int(m.group(2))
        day, month = (a, b) if DAY_FIRST else (b, a)
        spoken = _speak_date(day, month, _normalize_year(m.group(3)), profile)
        return spoken if spoken is not None else m.group(0)

    def _time(m: re.Match) -> str:
        spoken = _speak_time(int(m.group(1)), int(m.group(2)), m.group(3), profile)
        if spoken is None:
            return m.group(0)
        # "…at 10:05 pm." — the meridiem group swallows that final period, which
        # would delete the sentence boundary the chunker splits on. Put it back
        # when the match ends the sentence (end of string, or followed by
        # whitespace) rather than sitting mid-clause.
        meridiem = m.group(3) or ""
        tail = m.string[m.end():]
        if meridiem.endswith(".") and (not tail or tail[0].isspace()):
            spoken += "."
        return spoken

    text = _ISO_DATE_RE.sub(_iso, text)
    text = _NUMERIC_DATE_RE.sub(_numeric, text)
    return _TIME_RE.sub(_time, text)
