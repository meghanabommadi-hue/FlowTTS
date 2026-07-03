"""Model download helper for FlowTTS + Fish Audio S2 Pro.

The TTS model weights (`fishaudio/s2-pro`) live in the **sglang backend** container,
not in this gateway — the sglang-omni image runs `hf download fishaudio/s2-pro` at
build/first-run. This gateway is CPU-only and needs no model weights.

This script is kept as a convenience to pre-fetch the S2 Pro weights into a shared
HuggingFace cache (e.g. when priming a volume before starting the backend).

Usage:
    python -m flowtts.setup.download_models                 # fishaudio/s2-pro
    python -m flowtts.setup.download_models --repo fishaudio/s2-pro
"""

import argparse
import os
import sys

DEFAULT_REPO = "fishaudio/s2-pro"


def download(repo: str) -> None:
    print(f"[download] {repo} → HuggingFace cache")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface-hub")
        sys.exit(1)

    token = os.environ.get("HF_TOKEN") or None  # empty string → None (avoids illegal "Bearer " header)
    path = snapshot_download(
        repo_id=repo,
        token=token,
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )
    print(f"[done] {repo} → {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-fetch Fish S2 Pro weights into the HF cache")
    ap.add_argument("--repo", default=DEFAULT_REPO, help=f"HF repo id (default: {DEFAULT_REPO})")
    args = ap.parse_args()
    download(args.repo)
    print("\nDownload complete.")


if __name__ == "__main__":
    main()
