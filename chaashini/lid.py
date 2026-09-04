"""Regex / Unicode-script language identification with code-mix composition.

Approach
--------
1. Tokenise the transcript into word tokens; classify each token's script with Unicode
   block regexes (Devanagari, Bengali, Gurmukhi, ... Latin, Perso-Arabic).
2. Script fractions give the composition.  Scripts used by several languages (Devanagari,
   Bengali-Assamese, Perso-Arabic, Kannada, Latin) are disambiguated with high-frequency
   function words, plus a small prior for the language the source was discovered under.
3. The primary language is the largest share; a secondary share above `code_mix_threshold`
   marks the utterance as code-mixed.  Confidence = share of primary x tie-break certainty.

Everything is deterministic and dependency-free, so it is cheap enough to run on every chunk.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .languages import LANGUAGES, SCRIPT_LANGS, SCRIPT_NAMES

_SCRIPT_RE: dict[str, re.Pattern] = {
    "devanagari": re.compile(r"[ऀ-ॿ꣠-ꣿ᳐-᳿]"),
    "bengali": re.compile(r"[ঀ-৿]"),
    "gurmukhi": re.compile(r"[਀-੿]"),
    "gujarati": re.compile(r"[઀-૿]"),
    "oriya": re.compile(r"[଀-୿]"),
    "tamil": re.compile(r"[஀-௿]"),
    "telugu": re.compile(r"[ఀ-౿]"),
    "kannada": re.compile(r"[ಀ-೿]"),
    "malayalam": re.compile(r"[ഀ-ൿ]"),
    "arabic": re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]"),
    "ol_chiki": re.compile(r"[᱐-᱿]"),
    "meetei_mayek": re.compile(r"[ꯀ-꯿ꫠ-꫿]"),
    "latin": re.compile(r"[A-Za-zÀ-ɏ]"),
}
_TOKEN_RE = re.compile(r"[^\s\d\.,;:!?\-।॥\"'()\[\]{}<>/\\|@#$%^&*_+=~`]+")
_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
_ASSAMESE_ONLY = re.compile(r"[ৰৱ]")   # ৰ ৱ never occur in Bengali


@dataclass
class LIDResult:
    lang: str = "und"
    confidence: float = 0.0        # how sure we are that `lang` is the right label for the dominant script
    dominance: float = 0.0         # share of tokens belonging to `lang` (1.0 = monolingual)
    composition: dict[str, float] = field(default_factory=dict)   # lang -> token share
    scripts: dict[str, float] = field(default_factory=dict)       # script key -> token share
    script: str = "unknown"        # display name of the dominant script
    script_key: str = "unknown"    # internal key of the dominant script
    code_mixed: bool = False
    n_tokens: int = 0
    matches_expected: bool | None = None
    consensus: bool = False        # label was aligned to the source-level majority

    def as_dict(self) -> dict:
        return {
            "lang": self.lang, "confidence": round(self.confidence, 4), "dominance": round(self.dominance, 4),
            "composition": {k: round(v, 4) for k, v in self.composition.items()},
            "scripts": {k: round(v, 4) for k, v in self.scripts.items()},
            "script": self.script, "code_mixed": self.code_mixed, "n_tokens": self.n_tokens,
            "matches_expected": self.matches_expected, "consensus": self.consensus,
        }


def token_script(tok: str) -> str | None:
    best, best_n = None, 0
    for name, rx in _SCRIPT_RE.items():
        n = len(rx.findall(tok))
        if n > best_n:
            best, best_n = name, n
    return best


def _disambiguate(script: str, tokens: list[str], expected: str | None, prior_weight: float) -> tuple[str, float]:
    """Pick one language among the candidates of a shared script. Returns (lang, certainty in [0,1])."""
    cands = SCRIPT_LANGS.get(script, ())
    if not cands:
        return "und", 0.0
    if len(cands) == 1:
        return cands[0], 1.0
    lowered = [t.lower() for t in tokens]
    scores: dict[str, float] = {}
    for c in cands:
        sw = LANGUAGES[c].stopwords
        hits = sum(1 for t in lowered if t in sw) if sw else 0
        scores[c] = hits
    # Script-specific character evidence
    if script == "bengali":
        joined = " ".join(tokens)
        n_as = len(_ASSAMESE_ONLY.findall(joined))
        n_bn = joined.count("র") + joined.count("ব")
        if n_as:
            scores["as"] = scores.get("as", 0) + (3.0 * n_as if n_as >= n_bn else 0.3 * n_as)
            if n_bn > n_as:
                scores["bn"] = scores.get("bn", 0) + 1.0
    if script == "devanagari":
        joined = " ".join(tokens)
        n_lla = joined.count("ळ")          # ळ : Marathi (and Konkani) marker
        if n_lla:
            scores["mr"] = scores.get("mr", 0) + 1.0 * n_lla
    total = sum(scores.values())
    # prior: expected language (from discovery) and, weakly, the head of the candidate list
    n = max(1, len(tokens))
    prior: dict[str, float] = {c: 0.0 for c in cands}
    if expected in prior:
        prior[expected] += prior_weight * n
    prior[cands[0]] += 0.02 * n
    final = {c: scores.get(c, 0.0) + prior[c] for c in cands}
    best = max(final, key=final.get)
    ranked = sorted(final.values(), reverse=True)
    top, second = ranked[0], (ranked[1] if len(ranked) > 1 else 0.0)
    if total == 0 and expected not in cands:
        certainty = 0.35                      # no evidence at all: default candidate, low certainty
    else:
        certainty = (top - second) / max(top, 1e-9)
        certainty = 0.5 + 0.5 * min(1.0, certainty)  # margin -> [0.5, 1]
        if total >= 3:
            certainty = min(1.0, certainty + 0.1)
    return best, float(certainty)


def identify(text: str, expected: str | None = None, prior_weight: float = 0.15,
             code_mix_threshold: float = 0.15) -> LIDResult:
    res = LIDResult()
    if not text:
        return res
    tokens = [t for t in _TOKEN_RE.findall(text) if _LETTER_RE.search(t)]
    res.n_tokens = len(tokens)
    if not tokens:
        return res
    by_script: dict[str, list[str]] = {}
    for t in tokens:
        s = token_script(t)
        if s:
            by_script.setdefault(s, []).append(t)
    n_scripted = sum(len(v) for v in by_script.values())
    if n_scripted == 0:
        return res
    res.scripts = {s: len(v) / n_scripted for s, v in by_script.items()}
    comp: dict[str, float] = {}
    certainty: dict[str, float] = {}
    for s, toks in by_script.items():
        lang, cert = _disambiguate(s, toks, expected, prior_weight)
        comp[lang] = comp.get(lang, 0.0) + len(toks) / n_scripted
        certainty[lang] = max(certainty.get(lang, 0.0), cert)
    res.composition = dict(sorted(comp.items(), key=lambda kv: -kv[1]))
    primary = next(iter(res.composition))
    res.lang = primary
    res.script_key = max(res.scripts, key=res.scripts.get)
    res.script = SCRIPT_NAMES.get(res.script_key, "unknown")
    share = res.composition[primary]
    res.dominance = float(share)
    res.confidence = float(certainty.get(primary, 1.0))
    others = [v for k, v in res.composition.items() if k != primary]
    res.code_mixed = bool(others and max(others) >= code_mix_threshold)
    if expected:
        res.matches_expected = (primary == expected)
    return res
