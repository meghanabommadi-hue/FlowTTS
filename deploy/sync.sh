#!/usr/bin/env bash
# Push the code to both boxes (rsync over ssh). Run from the repo root on your laptop.
#   deploy/sync.sh cpu|gpu|all
set -euo pipefail
CPU=root@101.53.139.186
GPU=root@101.53.138.193
HERE=$(cd "$(dirname "$0")/.." && pwd)
EXCL=(--exclude .git --exclude __pycache__ --exclude '*.pyc' --exclude data --exclude logs --exclude .venv --exclude 'configs/local.yaml' --exclude 'configs/cookies.txt')
sync_cpu() {
  rsync -az --delete "${EXCL[@]}" "$HERE/" $CPU:/opt/chaashini/app/
  ssh $CPU 'chmod +x /opt/chaashini/app/deploy/chaashinictl; ln -sf /opt/chaashini/app/deploy/chaashinictl /usr/local/bin/chaashinictl; cd /opt/chaashini && ./venv/bin/pip install -q -e app >/dev/null 2>&1 || true; echo "cpu synced"'
}
sync_gpu() {
  rsync -az --delete "${EXCL[@]}" "$HERE/" $GPU:/opt/chaashini/app/
  ssh $GPU 'chmod +x /opt/chaashini/app/deploy/gpu/gpuctl; ln -sf /opt/chaashini/app/deploy/gpu/gpuctl /usr/local/bin/gpuctl; echo "gpu synced"'
}
case "${1:-all}" in
  cpu) sync_cpu ;; gpu) sync_gpu ;; all) sync_cpu; sync_gpu ;;
esac
