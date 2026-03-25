"""Text normalization and sentence splitting before TTS synthesis.

Handles:
  - Rupee amounts → Hindi spoken words:
      ₹५,०००  / Rs. 5000 / rupees 5000  → "पाँच हज़ार रुपये"
  - Plain digit sequences → digit-by-digit Hindi words:
      खाता नंबर ९८७६५४३२१०  → खाता नंबर नौ आठ सात छह पाँच चार तीन दो एक शून्य
  - Abbreviation expansion (mr. → mister, amt → amount, etc.)
  - English contraction expansion (I'm → i am, don't → do not, etc.)
  - Sentence splitting with abbreviation/decimal protection
  - Loan/amount sentence splitting
"""

import re
from functools import lru_cache
from typing import List

# ---------------------------------------------------------------------------
# Abbreviation expansion
# ---------------------------------------------------------------------------

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
    "rs.": "rupees",          # also protects "Rs." from false sentence splits

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

# ---------------------------------------------------------------------------
# Contraction expansion (English)
# One combined regex + dict lookup — single pass instead of ~50 re.sub calls.
# ---------------------------------------------------------------------------

_CONTRACTIONS: dict[str, str] = {
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

# Longer keys first so "shouldn't" matches before "should".
# Curly apostrophes (\u2018/\u2019) normalised alongside straight ones.
_CONTRACTION_RE = re.compile(
    r'\b(?:' + '|'.join(
        re.escape(k).replace(r"\'", r"['\u2018\u2019]")
        for k in sorted(_CONTRACTIONS, key=len, reverse=True)
    ) + r')\b',
    re.IGNORECASE,
)
_CONTRACTION_MAP = {k.lower(): v for k, v in _CONTRACTIONS.items()}


def _expand_contraction(m: re.Match) -> str:
    key = m.group(0).lower().replace('\u2018', "'").replace('\u2019', "'")
    return _CONTRACTION_MAP.get(key, m.group(0))


# ---------------------------------------------------------------------------
# Devanagari digit → ASCII digit translation table
# ---------------------------------------------------------------------------
_HINDI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

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
# Digit / rupee regexes and helpers
# ---------------------------------------------------------------------------

# Rupee amounts: ₹५,०००  |  ₹5,000  |  Rs. 500  |  rs 500  |  rupee 500  |  rupees 500
_RUPEE_RE = re.compile(
    r'(?:₹\s*|[Rr][Ss]\.?\s*|rupees?\s+)'
    r'([०-९0-9][०-९0-9,]*)',
    re.IGNORECASE,
)

# Matches Devanagari and ASCII digit sequences.
# Used for both Hindi digit-by-digit replacement and loan/amount detection.
# Also used in the English branch (ASCII-only subset) — _EN_PLAIN_NUM_RE removed.
_PLAIN_NUM_RE = re.compile(r'[०-९0-9]+')

# Derived from _ONES to avoid duplicating the digit word list.
_DIGIT_WORDS = ["शून्य"] + _ONES[1:10]


def _replace_rupee(m: re.Match) -> str:
    digits = m.group(1).replace(",", "").translate(_HINDI_DIGITS)
    return _num_to_hindi(int(digits)) + " रुपये"


def _replace_plain_digits(m: re.Match) -> str:
    ascii_digits = m.group(0).translate(_HINDI_DIGITS)
    return " ".join(_DIGIT_WORDS[int(d)] for d in ascii_digits)


# ---------------------------------------------------------------------------
# Hindi detection — fast character scan, no regex overhead for short strings
# ---------------------------------------------------------------------------

def _is_hindi(text: str) -> bool:
    """Return True if the text contains any Devanagari characters."""
    return any('\u0900' <= c <= '\u097F' for c in text)


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


def _replace_en_currency(m: re.Match) -> str:
    digits = m.group(1).replace(",", "")
    return _num_to_english(int(digits))


def _replace_en_plain(m: re.Match) -> str:
    # translate covers Devanagari digits if they ever appear in this branch
    return _num_to_english(int(m.group(0).translate(_HINDI_DIGITS)))


# ---------------------------------------------------------------------------
# Abbreviation expansion
# One combined regex + dict lookup — single pass instead of ~25 re.sub calls.
# ---------------------------------------------------------------------------

_ABBR_RE = re.compile(
    r'(?:' + '|'.join(
        r'\b' + re.escape(k) + r'(?!\w)'
        for k in sorted(ALWAYS_EXPAND, key=len, reverse=True)
    ) + r')',
    re.IGNORECASE,
)
_ABBR_MAP = {k.lower(): v for k, v in ALWAYS_EXPAND.items()}


def _expand_abbr(m: re.Match) -> str:
    return _ABBR_MAP.get(m.group(0).lower(), m.group(0))


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

@lru_cache(maxsize=512)
def normalize_text(text: str) -> str:
    """Normalize text before passing to the TTS synthesizer.

    Pure English sentences: digits → English spoken words (thirty thousand, etc.).
    Hindi/mixed sentences:  rupee amounts → Hindi words; other digits → digit-by-digit.
    """
    # Hindi full stop → ASCII period
    text = text.replace("।", " .")

    # Expand English contractions — single regex pass
    text = _CONTRACTION_RE.sub(_expand_contraction, text)

    if not _is_hindi(text):
        # Pure English — convert numbers to words
        text = _EN_CURRENCY_RE.sub(_replace_en_currency, text)
        text = _PLAIN_NUM_RE.sub(_replace_en_plain, text)
    else:
        # Hindi / mixed — rupee amounts first, then digit-by-digit
        text = _RUPEE_RE.sub(_replace_rupee, text)
        text = _PLAIN_NUM_RE.sub(_replace_plain_digits, text)

    # Abbreviation expansion — single regex pass, hoisted out of both branches
    text = _ABBR_RE.sub(_expand_abbr, text)
    return text


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

# Derived from ALWAYS_EXPAND keys that end with '.' so the two lists stay
# in sync automatically when ALWAYS_EXPAND is updated.
_abbrev_stems = sorted(
    (k[:-1] for k in ALWAYS_EXPAND if k.endswith('.')),
    key=len, reverse=True,
)
_ABBREV_RE = re.compile(
    r'\b(' + '|'.join(re.escape(s) for s in _abbrev_stems) + r')\.',
    re.IGNORECASE,
)

# Decimal / float numbers like 15.5 — dot between digits must not split.
_DECIMAL_RE = re.compile(r'(\d)\.(\d)')

# Null byte as placeholder — safe because it never appears in normal text.
_DOT_PLACEHOLDER = '\x00'


def _split_sentences(text: str) -> List[str]:
    """
    Splits text into individual sentences on English full stop, Hindi danda (।),
    and question mark (?), while preserving abbreviations and decimal numbers.
    """
    # Step 1 — protect abbreviation dots
    protected = _ABBREV_RE.sub(lambda m: m.group(1) + _DOT_PLACEHOLDER, text)

    # Step 2 — protect decimal dots
    protected = _DECIMAL_RE.sub(r'\1' + _DOT_PLACEHOLDER + r'\2', protected)

    # Step 3 — split at sentence-ending punctuation followed by whitespace
    parts = re.split(r'(?<=[.?।])\s+', protected)

    # Step 4 — restore placeholder and strip
    return [p.replace(_DOT_PLACEHOLDER, '.').strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Loan / Amount sentence splitting
# ---------------------------------------------------------------------------

# Set to False to disable loan/amount split behaviour.
ENABLE_LOAN_AMOUNT_SPLIT = True

# Built from _EN_ONES, _EN_TENS, _DIGIT_WORDS — no hardcoded duplication.
# Terms sorted longest-first to help the regex engine match greedily.
_number_terms = sorted(filter(None, [
    *_EN_ONES, *_EN_TENS,
    'zero', 'hundred', 'thousand', 'lakh', 'crore', 'rupees', 'rupee',
    *_DIGIT_WORDS,
    'पांच', 'छः', 'दस',
    'सौ', 'हज़ार', 'हजार', 'लाख', 'करोड़', 'रुपये', 'रुपए',
]), key=len, reverse=True)

_NUMBER_WORDS_RE = re.compile(
    r'\b(' + '|'.join(_number_terms) + r')\b',
    re.IGNORECASE,
)

# Combined regex for finding the first number token (digit or word) in a string.
# Used by _split_before_numbers to avoid text.split() + linear word scan.
_FIRST_NUM_RE = re.compile(
    r'[०-९0-9]+|\b(?:' + '|'.join(_number_terms) + r')\b',
    re.IGNORECASE,
)

_LOAN_RE = re.compile(r'(loan|लोन)', re.IGNORECASE)
_AMOUNT_RE = re.compile(r'(amount|overdue|राशि|अमाउंट|रकम)', re.IGNORECASE)

_TRAILING_PUNCT = frozenset('.,!?;:')


def _has_numbers(text: str) -> bool:
    return bool(_PLAIN_NUM_RE.search(text)) or bool(_NUMBER_WORDS_RE.search(text))


def _is_loan_sentence(text: str) -> bool:
    return bool(_LOAN_RE.search(text))


def _is_amount_sentence(text: str) -> bool:
    return bool(_AMOUNT_RE.search(text))


def _split_before_numbers(text: str) -> List[str]:
    """
    Split text just before the first number token (digit or number word).
    Uses a single regex search instead of splitting into a word list.
    """
    m = _FIRST_NUM_RE.search(text)
    if not m or m.start() == 0:
        return [text]
    # Find the start of the word containing the match (last space before it)
    word_start = text.rfind(' ', 0, m.start())
    if word_start <= 0:
        return [text]
    first = text[:word_start].rstrip()
    second = text[word_start:].lstrip()
    if first and first[-1] not in _TRAILING_PUNCT:
        first += ' ,....'
    return [first, second]


def _expand_sentences(sentences: List[str]) -> List[str]:
    """
    For each sentence matching a loan-number or amount pattern, replace it
    with two parts split before the first number token.
    Returns the list unchanged when ENABLE_LOAN_AMOUNT_SPLIT is False.
    """
    if not ENABLE_LOAN_AMOUNT_SPLIT:
        return sentences
    expanded: List[str] = []
    for s in sentences:
        # Compute _has_numbers once and reuse for both loan and amount checks
        has_nums = _has_numbers(s)
        if has_nums and (_is_loan_sentence(s) or _is_amount_sentence(s)):
            expanded.extend(_split_before_numbers(s))
        else:
            expanded.append(s)
    return expanded


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def split_and_expand_sentences(text: str) -> tuple[str, ...]:
    """Split text into sentences and expand loan/amount phrases.

    Combines sentence splitting (with abbreviation/decimal protection) and
    loan/amount sub-splitting into a single call for use by the synthesizer.

    Returns a tuple (not a list) so the cached object is safe from mutation.
    """
    return tuple(_expand_sentences(_split_sentences(text)))
