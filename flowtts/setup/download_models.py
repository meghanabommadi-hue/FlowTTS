"""
Model download script for FlowTTS.

Downloads models from HuggingFace into /root/CleanTTSData/inference/models/
matching the paths expected by flowtts/core/config.py.

Usage:
    python flowtts/setup/download_models.py
    python flowtts/setup/download_models.py --model MeghanaKap/MiraTTSTelugu
"""

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Model registry
# Each entry maps:
#   hf_repo  : HuggingFace repo id (owner/name)
#   local_dir: absolute path where the model lands (matches config.py defaults)
# ---------------------------------------------------------------------------
MODELS = {
    "MeghanaKap/MiraTTSTelugu": {
        "hf_repo": "Shubhangi7/mira_hindi_second_round",
        "local_dir": "/root/models/Shubhangi7-mira_hindi_second_round",
    },
}


def download(hf_repo: str, local_dir: str) -> None:
    dest = Path(local_dir)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"[download] {hf_repo}  →  {dest}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed.  Run:  pip install huggingface-hub")
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")  # optional – needed for gated repos

    snapshot_download(
        repo_id=hf_repo,
        local_dir=str(dest),
        token=token,
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )

    print(f"[done]     saved to {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FlowTTS models from HuggingFace")
    parser.add_argument(
        "--model",
        default=None,
        help="HuggingFace repo id to download (e.g. MeghanaKap/MiraTTSTelugu). "
             "Omit to download all registered models.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all registered models and their local paths, then exit.",
    )
    args = parser.parse_args()

    if args.list:
        print("Registered models:")
        for repo, cfg in MODELS.items():
            print(f"  {repo:45s}  →  {cfg['local_dir']}")
        return

    if args.model:
        if args.model not in MODELS:
            print(f"ERROR: '{args.model}' is not in the registry.")
            print("Registered models:", list(MODELS.keys()))
            sys.exit(1)
        targets = {args.model: MODELS[args.model]}
    else:
        targets = MODELS

    for repo, cfg in targets.items():
        download(cfg["hf_repo"], cfg["local_dir"])

    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()
