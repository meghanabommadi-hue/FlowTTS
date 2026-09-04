"""Upload built shards to the Hugging Face Hub with retries; refresh the dataset card."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

log = logging.getLogger("chaashini.pusher")


def _api(token: str):
    # huggingface_hub >= 1.0 moves large-file transfer to Xet; hf_transfer is deprecated.
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    from huggingface_hub import HfApi
    return HfApi(token=token)


def ensure_repo(token: str, repo_id: str, private: bool = False) -> None:
    api = _api(token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)


def upload_shards(token: str, repo_id: str, shards: list[dict], readme: str | None, max_retries: int = 8,
                  commit_message: str = "add shards") -> str:
    """Upload shard files in one commit (CommitOperationAdd). Returns commit URL. Retries with backoff."""
    from huggingface_hub import CommitOperationAdd
    api = _api(token)
    ops = [CommitOperationAdd(path_in_repo=s["hf_path"], path_or_fileobj=s["path"]) for s in shards]
    if readme is not None:
        ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme.encode("utf-8")))
    delay = 10.0
    last: Exception | None = None
    for attempt in range(max_retries):
        try:
            info = api.create_commit(repo_id=repo_id, repo_type="dataset", operations=ops, commit_message=commit_message)
            return getattr(info, "commit_url", "") or ""
        except Exception as e:  # noqa: BLE001
            last = e
            log.warning("upload attempt %d/%d failed: %s", attempt + 1, max_retries, e)
            time.sleep(delay)
            delay = min(delay * 2, 600)
    raise RuntimeError(f"upload failed after {max_retries} attempts: {last}")


def verify_present(token: str, repo_id: str, hf_paths: list[str]) -> bool:
    api = _api(token)
    try:
        files = set(api.list_repo_files(repo_id, repo_type="dataset"))
    except Exception as e:  # noqa: BLE001
        log.warning("verify failed: %s", e)
        return False
    return all(p in files for p in hf_paths)


def update_card(token: str, repo_id: str, readme: str) -> None:
    from huggingface_hub import CommitOperationAdd
    api = _api(token)
    api.create_commit(repo_id=repo_id, repo_type="dataset", operations=[CommitOperationAdd("README.md", readme.encode("utf-8"))],
                      commit_message="refresh dataset card")
