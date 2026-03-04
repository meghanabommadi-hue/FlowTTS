import subprocess
import time
import re
import os


def get_logs() -> str:
    result = subprocess.run(
        ["journalctl", "-u", "flowtts*", "--since", "today"],
        capture_output=True,
        text=True,
    )
    return result.stdout


def get_latency_histograms() -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    logs = get_logs()
    re_done = re.compile(r"done\s+llm=(\d+)ms\s+decode=(\d+)ms\s+\S+\s+total=(\d+)ms")
    llm_hist: dict[int, int] = {}
    dec_hist: dict[int, int] = {}
    tot_hist: dict[int, int] = {}
    for m in re_done.finditer(logs):
        for hist, val in (
            (llm_hist, int(m.group(1))),
            (dec_hist, int(m.group(2))),
            (tot_hist, int(m.group(3))),
        ):
            bucket = (val // 100) * 100
            hist[bucket] = hist.get(bucket, 0) + 1
    return llm_hist, dec_hist, tot_hist


def print_hist(title: str, hist: dict[int, int]) -> None:
    print(f"{title}:")
    print("-" * 44)
    if not hist:
        print("  (no data)")
        return
    max_count = max(hist.values())
    total = sum(hist.values())
    for bucket in range(0, max(hist) + 100, 100):
        n = hist.get(bucket, 0)
        bar = "█" * int(n / max_count * 28)
        print(f"  {bucket:>5}-{bucket+99:<5} | {bar:<28} {n:>5}  ({n/total*100:4.1f}%)")


def main(interval: float = 1.0):
    while True:
        llm_hist, dec_hist, tot_hist = get_latency_histograms()

        os.system("clear")

        total = sum(tot_hist.values())
        print(f"Latency monitor  —  {total} requests today")
        print()
        print_hist("LLM latency", llm_hist)
        print()
        print_hist("Decoder latency", dec_hist)
        print()
        print_hist("End-to-end latency", tot_hist)

        time.sleep(interval)


if __name__ == "__main__":
    main(interval=1.0)
