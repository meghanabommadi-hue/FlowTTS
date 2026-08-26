"""Pipeline position: SETUP — fetch the gated DhVaani snapshot and the vocoder.

Run this once before the first server start:

    export HF_TOKEN=hf_xxxxxxxxxxxx
    python -m flowtts.dhvaani.setup.fetch_model

The model repository is GATED: you must accept its terms while signed in at
https://huggingface.co/ARTPARK-IISc/DhVaani-0.5 before a token will work.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from flowtts.dhvaani.config import dhv_settings

_REQUIRED = ("model.safetensors", "tokens.txt", "model.json", "config.json", "_backend")


def _du(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def check(model_dir: Path) -> bool:
    print(f"Checking {model_dir}")
    ok = True
    for name in _REQUIRED:
        p = model_dir / name
        mark = "ok  " if p.exists() else "MISS"
        print(f"  [{mark}] {name}")
        ok = ok and p.exists()
    if ok:
        print(f"  total size: {_du(model_dir) / 2**20:.0f} MiB")
    return ok


def fetch_vocoder(repo: str) -> None:
    print(f"Fetching vocoder {repo} ...")
    try:
        from vocos import Vocos

        Vocos.from_pretrained(repo)
        print("  ok")
    except ImportError:
        print("  the `vocos` package is not installed: pip install vocos", file=sys.stderr)
    except Exception as e:
        print(f"  failed: {e}", file=sys.stderr)


def main() -> int:
    s = dhv_settings.model
    ap = argparse.ArgumentParser(description="Fetch the DhVaani model + vocoder")
    ap.add_argument("--token", default=None, help="HF token (else $HF_TOKEN)")
    ap.add_argument("--dir", default=s.local_dir, help="Destination snapshot dir")
    ap.add_argument("--repo", default=s.repo_id)
    ap.add_argument("--force", action="store_true", help="Re-download even if present")
    ap.add_argument("--check", action="store_true", help="Only verify what is on disk")
    ap.add_argument("--vocoder-only", action="store_true")
    args = ap.parse_args()

    if args.token:
        os.environ[s.hf_token_env] = args.token

    if args.vocoder_only:
        fetch_vocoder(s.vocoder_repo)
        return 0

    target = Path(args.dir).expanduser()
    if args.check:
        return 0 if check(target) else 1

    from flowtts.dhvaani.model.loader import ModelDownloadError, download_model

    dhv_settings.model.local_dir = str(target)
    dhv_settings.model.repo_id = args.repo
    try:
        path = download_model(dhv_settings, force=args.force)
    except ModelDownloadError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1

    print(f"\nSnapshot ready at {path}")
    check(Path(path))
    fetch_vocoder(s.vocoder_repo)
    print(
        "\nNext:\n"
        "  1. create a voice:  POST /v1/voices  (file + transcript + voice_id)\n"
        "  2. start the server: python -m flowtts.dhvaani.server --ports 1\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
