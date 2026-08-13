"""
Model upload script for FlowTTS.

Pushes a local model folder to a HuggingFace Hub repo.
Auth token is read from the HF_TOKEN environment variable; the source folder
and target repo are hardcoded below (edit to point at a different folder/repo).

Usage:
    export HF_TOKEN=hf_xxx
    python flowtts/setup/upload_model.py
    python flowtts/setup/upload_model.py /path/to/other/model/folder
"""

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Hardcoded source folder and target repo
# ---------------------------------------------------------------------------
MODEL_FOLDER = "/home/kapture/models/Shubhangi7-mira_hindi_second_round"
REPO_ID = "MeghanaKap/mira_hi_en"
REPO_TYPE = "model"  # "model", "dataset", or "space"


def upload(folder_path: str, repo_id: str = REPO_ID, repo_type: str = REPO_TYPE) -> None:
    src = Path(folder_path)
    if not src.is_dir():
        print(f"ERROR: '{src}' is not a directory.")
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN environment variable is not set.")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub is not installed.  Run:  pip install huggingface-hub")
        sys.exit(1)

    api = HfApi(token=token)

    print(f"[upload] {src}  →  {repo_id} ({repo_type})")

    api.create_repo(repo_id=repo_id, repo_type=repo_type, token=token, exist_ok=True)

    api.upload_folder(
        folder_path=str(src),
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
    )

    print(f"[done]   pushed to https://huggingface.co/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a local model folder to HuggingFace Hub")
    parser.add_argument(
        "folder",
        nargs="?",
        default=MODEL_FOLDER,
        help=f"Path to the local model folder to upload (default: {MODEL_FOLDER}).",
    )
    parser.add_argument(
        "--repo-id",
        default=REPO_ID,
        help=f"HuggingFace repo id to push to (default: {REPO_ID}).",
    )
    parser.add_argument(
        "--repo-type",
        default=REPO_TYPE,
        choices=["model", "dataset", "space"],
        help=f"HuggingFace repo type (default: {REPO_TYPE}).",
    )
    args = parser.parse_args()

    upload(args.folder, repo_id=args.repo_id, repo_type=args.repo_type)


if __name__ == "__main__":
    main()
