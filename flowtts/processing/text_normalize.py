"""Text normalization before TTS synthesis.

Handles:
  - Rupee amounts → Hindi spoken words:
      ₹५,०००  / Rs. 5000 / rupees 5000  → "पाँच हज़ार रुपये"
  - Plain digit sequences → digit-by-digit Hindi words:
      खाता नंबर ९८७६५४३२१०  → खाता नंबर नौ आठ सात छह पाँच चार तीन दो एक शून्य
      ३० दिन  → तीन शून्य दिन
  - Abbreviation expansion (mr. → mister, amt → amount, etc.)
"""

import re

# ---------------------------------------------------------------------------
# Contraction expansion (English)
# ---------------------------------------------------------------------------

CONTRACTIONS: dict[str, str] = {
    # Negative contractions
    "aren't":    "are not",
    "can't":     "cannot",
    "couldn't":  "could not",
    "daren't":   "dare not",
    "didn't":    "did not",
    "doesn't":   "does not",
    "don't":     "do not",
    "hadn't":    "had not",
    "hasn't":    "has not",
    "haven't":   "have not",
    "isn't":     "is not",
    "mightn't":  "might not",
    "mustn't":   "must not",
    "needn't":   "need not",
    "shan't":    "shall not",
    "shouldn't": "should not",
    "wasn't":    "was not",
    "weren't":   "were not",
    "won't":     "will not",
    "wouldn't":  "would not",
    # Subject + verb
    "he'd":      "he would",
    "he'll":     "he will",
    "he's":      "he is",
    "how's":     "how is",
    "i'd":       "i would",
    "i'll":      "i will",
    "i'm":       "i am",
    "i've":      "i have",
    "it'd":      "it would",
    "it'll":     "it will",
    "it's":      "it is",
    "let's":     "let us",
    "she'd":     "she would",
    "she'll":    "she will",
    "she's":     "she is",
    "that'd":    "that would",
    "that'll":   "that will",
    "that's":    "that is",
    "there'd":   "there would",
    "there'll":  "there will",
    "there's":   "there is",
    "they'd":    "they would",
    "they'll":   "they will",
    "they're":   "they are",
    "they've":   "they have",
    "we'd":      "we would",
    "we'll":     "we will",
    "we're":     "we are",
    "we've":     "we have",
    "what'd":    "what did",
    "what'll":   "what will",
    "what's":    "what is",
    "what've":   "what have",
    "when's":    "when is",
    "where'd":   "where did",
    "where's":   "where is",
    "who'd":     "who would",
    "who'll":    "who will",
    "who's":     "who is",
    "who've":    "who have",
    "why's":     "why is",
    "you'd":     "you would",
    "you'll":    "you will",
    "you're":    "you are",
    "you've":    "you have",
}

# Pre-compiled patterns (word-boundary, case-insensitive).
# Longer keys first so e.g. "shouldn't" matches before any shorter suffix.
_CONTRACTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\b' + re.escape(k) + r'\b', re.IGNORECASE), v)
    for k, v in sorted(CONTRACTIONS.items(), key=lambda x: -len(x[0]))
]


ALWAYS_EXPAND = {
    "mr.": "mister",
    "mrs.": "missus",
    "ms.": "miss",
    "dr.": "doctor",
    "prof.": "professor",
    "sr.": "senior",
    "jr.": "junior",
    "hon.": "honorable",
    "rev.": "reverend",
    "fr.": "father",

    "ltd.": "limited",
    "pvt.": "private",
    "inc.": "incorporated",
    "corp.": "corporation",
    "co.": "company",
    "m/s": "messrs",

    "rd.": "road",
    "st.": "street",
    "ave.": "avenue",
    "blvd.": "boulevard",
    "ln.": "lane",
    "apt.": "apartment",
    "fl.": "floor",
    "bldg.": "building",
    "no.": "number",

    "amt": "amount",
    "a/c": "account",
    "bal.": "balance",
    "min.": "minimum",
    "max.": "maximum",
    "dept.": "department",

    "etc.": "etcetera",
    "vs.": "versus",
    "e.g.": "for example",
    "i.e.": "that is",
    "approx.": "approximately",
}

# Devanagari digit → ASCII digit translation table
_HINDI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

# Pre-compile abbreviation patterns once at import time
_ABBR_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\b' + re.escape(abbr) + r'\b', re.IGNORECASE), full)
    for abbr, full in ALWAYS_EXPAND.items()
]

# ---------------------------------------------------------------------------
# Hindi number-to-words (for rupee amounts only)
# ---------------------------------------------------------------------------

_ONES = [
    "", "एक", "दो", "तीन", "चार", "पाँच",
    "छह", "सात", "आठ", "नौ", "दस",
    "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह",
    "सोलह", "सत्रह", "अठारह", "उन्नीस", "बीस",
    "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस",
    "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस", "तीस",
    "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस",
    "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस", "चालीस",
    "इकतालीस", "बयालीस", "तैंतालीस", "चौंतालीस", "पैंतालीस",
    "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास", "पचास",
    "इक्यावन", "बावन", "तिरपन", "चौवन", "पचपन",
    "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ", "साठ",
    "इकसठ", "बासठ", "तिरसठ", "चौंसठ", "पैंसठ",
    "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर", "सत्तर",
    "इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर",
    "छिहत्तर", "सतहत्तर", "अठहत्तर", "उनासी", "अस्सी",
    "इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी",
    "छियासी", "सतासी", "अट्ठासी", "नवासी", "नब्बे",
    "इक्यानवे", "बानवे", "तिरानवे", "चौरानवे", "पचानवे",
    "छियानवे", "सतानवे", "अट्ठानवे", "निन्यानवे",
]


def _num_to_hindi(n: int) -> str:
    """Convert a non-negative integer to Hindi words (Indian numbering system)."""
    if n == 0:
        return "शून्य"
    parts = []
    if n >= 10_000_000:
        parts.append(_num_to_hindi(n // 10_000_000) + " करोड़")
        n %= 10_000_000
    if n >= 100_000:
        parts.append(_num_to_hindi(n // 100_000) + " लाख")
        n %= 100_000
    if n >= 1_000:
        parts.append(_num_to_hindi(n // 1_000) + " हज़ार")
        n %= 1_000
    if n >= 100:
        parts.append(_ONES[n // 100] + " सौ")
        n %= 100
    if n > 0:
        parts.append(_ONES[n])
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Rupee amounts: ₹५,०००  |  ₹5,000  |  Rs. 500  |  rs 500  |  rupee 500  |  rupees 500
_RUPEE_RE = re.compile(
    r'(?:₹\s*|[Rr][Ss]\.?\s*|rupees?\s+)'   # rupee prefix
    r'([०-९0-9][०-९0-9,]*)',                  # amount (digits + commas)
    re.IGNORECASE,
)

# Plain digit sequences (Devanagari or ASCII), no rupee prefix — read digit by digit
_PLAIN_NUM_RE = re.compile(r'[०-९0-9]+')

_DIGIT_WORDS = ["शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ"]


def _replace_rupee(m: re.Match) -> str:
    digits = m.group(1).replace(",", "").translate(_HINDI_DIGITS)
    return _num_to_hindi(int(digits)) + " रुपये"


def _replace_plain_digits(m: re.Match) -> str:
    # Translate Devanagari → ASCII first, then read each digit as a word
    ascii_digits = m.group(0).translate(_HINDI_DIGITS)
    return " ".join(_DIGIT_WORDS[int(d)] for d in ascii_digits)


_DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')

# ---------------------------------------------------------------------------
# English number-to-words
# ---------------------------------------------------------------------------

_EN_ONES = [
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty",
            "sixty", "seventy", "eighty", "ninety"]


def _num_to_english(n: int) -> str:
    """Convert a non-negative integer to English words."""
    if n == 0:
        return "zero"
    if n < 0:
        return "minus " + _num_to_english(-n)
    parts = []
    if n >= 1_000_000_000:
        parts.append(_num_to_english(n // 1_000_000_000) + " billion")
        n %= 1_000_000_000
    if n >= 1_000_000:
        parts.append(_num_to_english(n // 1_000_000) + " million")
        n %= 1_000_000
    if n >= 1_000:
        parts.append(_num_to_english(n // 1_000) + " thousand")
        n %= 1_000
    if n >= 100:
        parts.append(_EN_ONES[n // 100] + " hundred")
        n %= 100
    if n >= 20:
        t = _EN_TENS[n // 10]
        o = _EN_ONES[n % 10]
        parts.append(t + (" " + o if o else ""))
    elif n > 0:
        parts.append(_EN_ONES[n])
    return " ".join(parts)


# Matches currency prefix ($ £ € or "dollars"/"pounds") + number for English amounts
_EN_CURRENCY_RE = re.compile(
    r'(?:[$£€]\s*|(?:dollars?|pounds?|euros?)\s+)'
    r'([0-9][0-9,]*)',
    re.IGNORECASE,
)
_EN_PLAIN_NUM_RE = re.compile(r'[0-9]+')


def _replace_en_currency(m: re.Match) -> str:
    digits = m.group(1).replace(",", "")
    return _num_to_english(int(digits))


def _replace_en_plain(m: re.Match) -> str:
    return _num_to_english(int(m.group(0)))


def _is_hindi(text: str) -> bool:
    """Return True if the text contains any Devanagari characters."""
    return bool(_DEVANAGARI_RE.search(text))


def normalize_text(text: str) -> str:
    """Normalize text before passing to the TTS synthesizer.

    Pure English sentences: digits → English spoken words (thirty thousand, etc.).
    Hindi/mixed sentences:  rupee amounts → Hindi words; other digits → digit-by-digit Hindi words.
    """
    # Always: Hindi full stop → ASCII period
    text = text.replace("।", " .")

    # Normalise curly/smart apostrophes → straight, then expand contractions
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    for pattern, expansion in _CONTRACTION_PATTERNS:
        text = pattern.sub(expansion, text)
    # Add space before the last comma only
    # idx = text.rfind(",")
    # if idx != -1:
    #     text = text[:idx] + " ,.." + text[idx+1:]
    # Replace trailing "." with " ." (only the last one)
    # if text.endswith("."):
    #     text = text[:-1] + " ..."

    if not _is_hindi(text):
        # Pure English — convert numbers to English words
        text = _EN_CURRENCY_RE.sub(_replace_en_currency, text)
        text = _EN_PLAIN_NUM_RE.sub(_replace_en_plain, text)
        for pattern, full in _ABBR_PATTERNS:
            text = pattern.sub(full, text)
        return text

    # Hindi/mixed
    text = _RUPEE_RE.sub(_replace_rupee, text)
    text = _PLAIN_NUM_RE.sub(_replace_plain_digits, text)
    for pattern, full in _ABBR_PATTERNS:
        text = pattern.sub(full, text)
    return text
