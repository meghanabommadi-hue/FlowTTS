import re
from collections import defaultdict

from pathlib import Path

HERE = Path(__file__).parent
INPUT_FILE  = str(HERE / "bot_sentences.txt")
OUTPUT_FILE = str(HERE / "norm1_sentences.txt")
REPORT_FILE = str(HERE / "normalization_1_report.txt")


NORMALIZATIONS = [
    "code_block",
    "image",
    "link",
    "bold_italic",
    "header",
    "blockquote",
    "horizontal_rule",
    "list_marker",
    "char_removal",
    "exclamation_to_comma",
    "collapse_whitespace",
    "strip",
]


def normalize(text):
    applied = set()

    # 1. Strip code blocks ```...```
    t = re.sub(r'```[\s\S]*?```', '', text)
    if t != text:
        applied.add("code_block")
    text = t

    # 2. Remove images ![alt](url)
    t = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    if t != text:
        applied.add("image")
    text = t

    # 3. Links [text](url) → text
    t = re.sub(r'\[([^\]]*)\]\([^\)]*\)', r'\1', text)
    if t != text:
        applied.add("link")
    text = t

    # 4. Remove bold/italic markers (**text**, *text*, __text__, _text_)
    t = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    t = re.sub(r'\*(.*?)\*', r'\1', t)
    t = re.sub(r'__(.*?)__', r'\1', t)
    t = re.sub(r'_(.*?)_', r'\1', t)
    if t != text:
        applied.add("bold_italic")
    text = t

    # 5. Remove headers (## Heading → Heading)
    t = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    if t != text:
        applied.add("header")
    text = t

    # 6. Remove blockquotes (> text → text)
    t = re.sub(r'^>\s?', '', text, flags=re.MULTILINE)
    if t != text:
        applied.add("blockquote")
    text = t

    # 7. Remove horizontal rules (--- / *** / ___ on their own)
    t = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    if t != text:
        applied.add("horizontal_rule")
    text = t

    # 8. Remove list markers at line start: *, -, 1.
    t = re.sub(r'^\s*(?:\*|-|\d+\.)\s+', '', text, flags=re.MULTILINE)
    if t != text:
        applied.add("list_marker")
    text = t

    # 9. Character removal: * # ` "
    t = re.sub(r'[*#`"]', '', text)
    if t != text:
        applied.add("char_removal")
    text = t

    # 10. Character replacement: ! → ,
    t = text.replace('!', ',')
    if t != text:
        applied.add("exclamation_to_comma")
    text = t

    # 11. Collapse whitespace (multiple spaces/newlines → single space)
    t = re.sub(r'\s+', ' ', text)
    if t != text:
        applied.add("collapse_whitespace")
    text = t

    # 12. Strip
    t = text.strip()
    if t != text:
        applied.add("strip")
    text = t

    return text, applied


def main():
    total_lines = 0
    parse_errors = 0
    lines_changed = 0
    lines_dropped = 0
    norm_counts = defaultdict(int)

    output_rows = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue

            total_lines += 1

            try:
                if "\t" in line:
                    sentence, count = line.rsplit("\t", 1)
                    count = int(count)
                else:
                    sentence = line
                    count = 1
            except Exception:
                parse_errors += 1
                continue

            normalized, applied = normalize(sentence)

            for norm in applied:
                norm_counts[norm] += 1

            if applied:
                lines_changed += 1

            if not normalized:
                lines_dropped += 1
                continue

            output_rows.append((normalized, count))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sentence, count in output_rows:
            f.write(f"{sentence}\t{count}\n")

    lines_written = len(output_rows)

    report_lines = [
        "=" * 50,
        "NORMALIZATION REPORT — normalization_1.py",
        "=" * 50,
        f"Input file         : {INPUT_FILE}",
        f"Output file        : {OUTPUT_FILE}",
        "",
        "--- Summary ---",
        f"Total lines read   : {total_lines}",
        f"Parse errors       : {parse_errors}",
        f"Lines changed      : {lines_changed}",
        f"Lines dropped      : {lines_dropped}  (became empty after normalization)",
        f"Lines written      : {lines_written}",
        "",
        "--- Normalization hits (lines affected per rule) ---",
    ]

    for norm in NORMALIZATIONS:
        count = norm_counts.get(norm, 0)
        report_lines.append(f"  {norm:<25} : {count}")

    report_text = "\n".join(report_lines) + "\n"

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"Saved {lines_written} lines to {OUTPUT_FILE}")
    print(f"Report saved to {REPORT_FILE}")


if __name__ == "__main__":
    main()
