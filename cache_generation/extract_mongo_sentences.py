#!/usr/bin/env python3
"""
Extract bot raw_body texts from MongoDB conversation_history collections.

Each xPertVoiceXXXPQ entry is a DATABASE containing a conversation_history
collection. Documents have a "messages" array where each message has a "role"
and "raw_body". Collects raw_body for role == "bot", deduplicates, writes to
a .txt file (one sentence per line).

Usage:
    python extract_mongo_sentences.py
    python extract_mongo_sentences.py --out bot_sentences.txt
    python extract_mongo_sentences.py --dbs xPertVoice23May26PQ xPertVoice24May26PQ
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).parent

MONGO_URI = "mongodb+srv://readml:zBYU41BFBaUKSlJh@mlcluster0.e1q1y.mongodb.net/"
COLLECTION = "conversation_history"
DEFAULT_DBS = [
    "xPertVoice23May26PQ",
    "xPertVoice24May26PQ",
]
DEFAULT_OUT = str(HERE / "bot_sentences.txt")


def extract_bot_texts_from_db(client, db_name: str) -> list[str]:
    """Stream through conversation_history and collect bot raw_body values."""
    col = client[db_name][COLLECTION]
    total_docs = col.count_documents({})
    print(f"  [{db_name}] {total_docs:,} documents — streaming...")

    texts: list[str] = []
    for doc in col.find({}, {"messages": 1, "_id": 0}):
        messages = doc.get("messages", [])
        if not isinstance(messages, list):
            continue
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "bot":
                continue
            raw_body = msg.get("raw_body", "")
            if isinstance(raw_body, str):
                raw_body = raw_body.strip()
            elif isinstance(raw_body, list):
                raw_body = " ".join(str(x) for x in raw_body).strip()
            else:
                raw_body = str(raw_body).strip()
            if raw_body:
                texts.append(raw_body)

    print(f"  [{db_name}] found {len(texts):,} bot raw_body entries")
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract bot raw_body from MongoDB conversation_history")
    parser.add_argument("--uri", default=MONGO_URI, help="MongoDB connection URI")
    parser.add_argument("--dbs", nargs="+", default=DEFAULT_DBS,
                        help="Database names to extract from")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output .txt file path")
    args = parser.parse_args()

    try:
        from pymongo import MongoClient
    except ImportError:
        print("[ERROR] pymongo not installed. Run: pip install pymongo", file=sys.stderr)
        sys.exit(1)

    print("Connecting to MongoDB...")
    client = MongoClient(args.uri)

    all_texts: list[str] = []

    for db_name in args.dbs:
        print(f"\nProcessing: {db_name}")
        texts = extract_bot_texts_from_db(client, db_name)
        all_texts.extend(texts)

    client.close()

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        for line in all_texts:
            f.write(line + "\n")

    print(f"\nTotal bot raw_body texts : {len(all_texts):,}")
    print(f"Saved to                 : {out_path.resolve()}")


if __name__ == "__main__":
    main()
