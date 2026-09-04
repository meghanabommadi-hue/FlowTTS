#!/usr/bin/env bash
# Push the code to both boxes over ssh (tar stream; no rsync dependency on the remote).
#   deploy/sync.sh cpu|gpu|all
set -euo pipefail
CPU=root@101.53.139.186
GPU=root@101.53.138.193
HERE=$(cd "$(dirname "$0")/.." && pwd)
EXCL=(--exclude .git --exclude __pycache__ --exclude '*.pyc' --exclude ./data --exclude ./logs --exclude .venv --exclude 'configs/local.yaml' --exclude 'configs/cookies.txt' --exclude .DS_Store)
push() {  # host
  COPYFILE_DISABLE=1 tar czf - --no-xattrs -C "$HERE" "${EXCL[@]}" . | ssh "$1" 'set -e; mkdir -p /opt/chaashini/app; cd /opt/chaashini/app; rm -rf chaashini gpu ui deploy docs tests; tar xzf -; echo "synced -> $(hostname)"'
}
sync_cpu() {
  push $CPU
  ssh $CPU 'chmod +x /opt/chaashini/app/deploy/chaashinictl /opt/chaashini/app/deploy/*.sh; ln -sf /opt/chaashini/app/deploy/chaashinictl /usr/local/bin/chaashinictl; cd /opt/chaashini && ./venv/bin/pip install -q -e app 2>&1 | tail -1; echo "cpu ready"'
}
sync_gpu() {
  push $GPU
  ssh $GPU 'chmod +x /opt/chaashini/app/deploy/gpu/gpuctl /opt/chaashini/app/deploy/gpu/*.sh; ln -sf /opt/chaashini/app/deploy/gpu/gpuctl /usr/local/bin/gpuctl; echo "gpu ready"'
}
case "${1:-all}" in
  cpu) sync_cpu ;; gpu) sync_gpu ;; all) sync_cpu; sync_gpu ;;
esac
