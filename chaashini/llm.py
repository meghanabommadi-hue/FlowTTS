"""OpenAI-compatible chat client (bolt-surge) used to invent discovery queries."""
from __future__ import annotations

import json
import logging
import random
import re
import time

import httpx

from .config import LLMCfg
from .languages import DISCOVERY_DOMAINS, DISCOVERY_GENRES, Language

log = logging.getLogger("chaashini.llm")


class LLM:
    def __init__(self, cfg: LLMCfg):
        self.cfg = cfg
        self.client = httpx.Client(base_url=cfg.base_url.rstrip("/"), timeout=cfg.timeout_s,
                                   headers={"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"})

    def chat(self, messages: list[dict], temperature: float | None = None, max_tokens: int | None = None,
             retries: int = 4) -> str:
        body = {"model": self.cfg.model, "messages": messages,
                "temperature": self.cfg.temperature if temperature is None else temperature,
                "max_tokens": max_tokens or self.cfg.max_tokens}
        delay = 2.0
        for attempt in range(retries):
            try:
                r = self.client.post("/chat/completions", json=body)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"] or ""
            except Exception as e:  # noqa: BLE001
                log.warning("llm call failed (%s/%s): %s", attempt + 1, retries, e)
                time.sleep(delay)
                delay = min(delay * 2, 30)
        raise RuntimeError("LLM unavailable")


_JSON_ARRAY = re.compile(r"\[.*\]", re.S)


def parse_string_list(text: str) -> list[str]:
    m = _JSON_ARRAY.search(text)
    if m:
        try:
            arr = json.loads(m.group(0))
            return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:  # noqa: BLE001
            pass
    # fallback: one per line, strip bullets/numbers
    out = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"').strip("'")
        if 3 <= len(line) <= 120 and not line.lower().startswith(("here", "sure", "json")):
            out.append(line)
    return out


def generate_queries(llm: LLM, lang: Language, n: int, known: list[str], good: list[str], bad: list[str]) -> list[dict]:
    """Ask the LLM for `n` diverse search queries for `lang`, written in that language.
    Each round rotates through content domains and spoken-word formats. Returns [{query, genre}]."""
    domains = random.sample(DISCOVERY_DOMAINS, k=min(10, len(DISCOVERY_DOMAINS)))
    formats = random.sample(DISCOVERY_GENRES, k=min(8, len(DISCOVERY_GENRES)))
    regions = ", ".join(lang.regions[:4]) if lang.regions else "India"
    hint_line = f"SPECIAL RULE FOR THIS LANGUAGE: {lang.query_hint}\n" if lang.query_hint else ""
    sys_prompt = (
        "You help build a clean, topically broad speech corpus. You write search queries that surface LONG-FORM SPOKEN-WORD "
        "videos in a given Indian language: podcasts, interviews, audiobooks, narrated stories, lectures, talks, explainers, "
        "radio shows, commentary, sermons, panel discussions. A single clear speaker on a good microphone is ideal; two-person "
        "interviews are fine. NEVER target songs, music, DJ mixes, remixes, movie scenes, trailers, comedy skits, kids rhymes, "
        "ASMR, vlogs with loud music, gaming, or anything with background music.\n"
        "RULES: (1) Write the queries IN THE TARGET LANGUAGE, in its native script, the way a native speaker would type them; "
        "make about one in four queries Latin-script transliterations or natural language+English mixes (e.g. 'share market podcast "
        "in <language>'). (2) Every query must name a concrete DOMAIN/TOPIC and a spoken FORMAT. (3) Spread the queries across ALL "
        "the domains given; do not cluster on one topic. (4) Vary speaker types (teacher, doctor, farmer, lawyer, journalist, "
        "author, entrepreneur, historian, coach, monk, engineer, homemaker, student) and regions. "
        "Output ONLY a JSON array of objects: [{\"query\": \"...\", \"genre\": \"<domain> / <format>\"}]."
    )
    user = (
        f"Target language: {lang.name} ({lang.native}); language name in searches: {', '.join(lang.query_names)}. "
        f"Regions/cities for flavour: {regions}.\n"
        f"Domains to cover this round: {', '.join(domains)}.\n"
        f"Spoken formats to combine with them: {', '.join(formats)}.\n"
        f"Give {n} NEW, diverse queries.\n"
        f"{hint_line}"
        f"Queries already used (do not repeat or trivially rephrase): {json.dumps(known[-60:], ensure_ascii=False)}\n"
        f"Queries that yielded excellent clean speech (make more like these, different topics): {json.dumps(good[:15], ensure_ascii=False)}\n"
        f"Queries that yielded music/noise (avoid these patterns): {json.dumps(bad[:15], ensure_ascii=False)}\n"
        "Return the JSON array only."
    )
    text = llm.chat([{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}])
    m = _JSON_ARRAY.search(text)
    items: list[dict] = []
    if m:
        try:
            for o in json.loads(m.group(0)):
                if isinstance(o, dict) and o.get("query"):
                    items.append({"query": str(o["query"]).strip()[:150], "genre": str(o.get("genre", "")).strip()[:80]})
                elif isinstance(o, str):
                    items.append({"query": o.strip()[:150], "genre": ""})
        except Exception:  # noqa: BLE001
            items = []
    if not items:
        items = [{"query": q, "genre": ""} for q in parse_string_list(text)]
    seen, out = set(), []
    for it in items:
        k = it["query"].lower()
        if k and k not in seen and len(k) >= 4:
            seen.add(k)
            out.append(it)
    return out[:n]


_INDIA_CUES = (
    "india", "indian", "bharat", "delhi", "mumbai", "bombay", "bangalore", "bengaluru", "chennai", "madras", "kolkata", "calcutta",
    "hyderabad", "pune", "ahmedabad", "jaipur", "lucknow", "kochi", "kerala", "tamil", "telugu", "kannada", "marathi", "gujarat",
    "punjab", "bengal", "bihar", "rajasthan", "odisha", "assam", "goa", "kashmir", "ladakh", "rupee", "rupees", "lakh", "lakhs",
    "crore", "crores", "iit", "iim", "iisc", "aiims", "upsc", "ias", "ips", "neet", "jee", "cat exam", "isro", "drdo", "rbi",
    "sebi", "nifty", "sensex", "lok sabha", "rajya sabha", "niti aayog", "modi", "gandhi", "nehru", "ambedkar", "bollywood",
    "cricket", "ipl", "bcci", "kohli", "dhoni", "sachin", "diwali", "holi", "hindu", "hindi", "sanskrit", "ayurveda", "yoga",
    "chai", "dal", "roti", "biryani", "auto rickshaw", "aadhaar", "upi", "paytm", "jio", "tata", "reliance", "infosys", "wipro",
    "hdfc", "icici", "sbi", "zomato", "swiggy", "flipkart", "ola", "startup india", "make in india", "panchayat", "gram", "sarkar",
    "ji", "yaar", "na", "achha", "bhai", "didi", "beta", "prepone", "timepass", "itself", "only", "do the needful", "kindly",
)


def india_cue_count(text: str) -> int:
    """Number of Indian-context cue words in a transcript (case-insensitive, whole words)."""
    words = set(re.findall(r"[a-z]+", (text or "").lower()))
    n = 0
    for cue in _INDIA_CUES:
        if " " in cue:
            if cue in (text or "").lower():
                n += 1
        elif cue in words:
            n += 1
    return n


def judge_indian_english(llm: "LLM", title: str, channel: str, description: str, tags: list[str] | None = None) -> tuple[bool, float, str]:
    """Ask the LLM whether this English-language video is spoken by Indian speakers (Indian English).
    Returns (indian, confidence 0-1, reason). Errors default to (False, 0, 'llm error') so nothing slips through."""
    sys_prompt = (
        "You classify videos for a speech corpus that must contain INDIAN ENGLISH only: English spoken by people from India "
        "(Indian accent), typically Indian creators, professionals, academics, journalists or public figures based in India. "
        "American, British, Australian, Canadian, other non-Indian speakers, and Indian-diaspora creators based abroad do NOT qualify. "
        "EVERY main speaker must be Indian: if a featured guest, co-host or interviewee is non-Indian (for example an American scientist "
        "or a British author interviewed by an Indian host), answer false, because most of the audio would be a non-Indian accent. "
        "Panel shows, reaction videos to foreign content, and dubbed or narrated foreign material also do not qualify. "
        "BEWARE RE-UPLOADS: an Indian-sounding channel name is NOT evidence of an Indian speaker. Indian channels routinely "
        "re-upload or mirror foreign courses, lectures and talks (for example a Coursera/Stanford/MIT course, a Western "
        "professor's lecture series, a foreign conference talk). If the title, description or transcript names a foreign "
        "course, university, professor or speaker, answer false even when the channel looks Indian. When the evidence does "
        "not clearly identify an Indian speaker, answer false with the confidence you have rather than guessing true. "
        "Judge from the title, channel name, description and tags. Output ONLY JSON: {\"indian\": true|false, \"confidence\": 0.0-1.0, \"reason\": \"...\"}."
    )
    user = f"Title: {title[:200]}\nChannel: {channel[:100]}\nTags: {', '.join((tags or [])[:20])[:300]}\nDescription: {(description or '')[:1200]}"
    try:
        text = llm.chat([{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}], temperature=0.0, max_tokens=200, retries=2)
        m = re.search(r"\{.*\}", text, re.S)
        o = json.loads(m.group(0)) if m else {}
        return bool(o.get("indian")), float(o.get("confidence", 0.0) or 0.0), str(o.get("reason", ""))[:200]
    except Exception as e:  # noqa: BLE001
        log.warning("indian-english judge failed: %s", e)
        return False, 0.0, "llm error"
