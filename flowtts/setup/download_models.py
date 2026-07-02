"""Model download for FlowTTS (OmniVoice).

Pre-fetches the k2-fsa/OmniVoice weights into the HuggingFace cache so the first
server start doesn't pay the download cost. `OmniVoice.from_pretrained(repo_id)`
then loads straight from the cache.

Usage:
    python -m flowtts.setup.download_models
    python -m flowtts.setup.download_models --repo k2-fsa/OmniVoice
"""

import argparse
import os
import sys

from flowtts.core.config import settings


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
    ap = argparse.ArgumentParser(description="Download OmniVoice weights from HuggingFace")
    ap.add_argument("--repo", default=settings.omnivoice.model_repo,
                    help=f"HF repo id (default: {settings.omnivoice.model_repo})")
    args = ap.parse_args()
    download(args.repo)
    print("\nDownload complete.")


if __name__ == "__main__":
    main()
