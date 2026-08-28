#!/usr/bin/env bash
# Keep the progressive stack alive over a long unattended run.
#
# Restarts anything that dies, and reclaims the GPU if a crashed CUDA process
# leaks its VRAM (SIGKILL on a CUDA process holds memory until every holder
# exits, which stalled the whole box once already).
set -uo pipefail
BASE=/home/jovyan/omnivoice-train
LOG=$BASE/run_prog/logs/watchdog.log
mkdir -p "$(dirname "$LOG")"
wlog(){ echo "[$(date -u +%m-%d\ %H:%M:%S)] $*" >> "$LOG"; }

alive(){ ps -eo args --no-headers | tr -d '\0' | grep -a "venv/bin/python.*$1" | grep -qav grep; }
alive_sh(){ ps -eo args --no-headers | tr -d '\0' | grep -a "bash.*$1" | grep -qav grep; }

ALIGN_FAILS=0
while :; do
  if curl -s -m 60 http://127.0.0.1:8899/health >/dev/null 2>&1; then
    ALIGN_FAILS=0
  else
    ALIGN_FAILS=$((${ALIGN_FAILS:-0}+1))
    wlog "aligner probe failed (${ALIGN_FAILS}/3)"
    if [ "$ALIGN_FAILS" -ge 3 ]; then
      wlog "aligner down after 3 probes -> restarting"
      ASR_CONCURRENCY=1 ALIGN_BATCH=1 ALIGN_VRAM_FRACTION=0.24 \
        $BASE/start_align.sh >> "$LOG" 2>&1
      ALIGN_FAILS=0
    fi
  fi
  if ! alive chunk_producer.py; then
    wlog "producer down -> restarting"
    setsid nohup $BASE/.venv/bin/python $BASE/omnivoice_training/chunk_producer.py \
      --langs hau,ibo,yor,pcm --clips-per-batch 24 --shards-per-pass 2 \
      --vram-floor-mb 8000 >> $BASE/run_prog/logs/producer.log 2>&1 < /dev/null &
    disown || true
  fi
  if ! alive progress_status.py; then
    wlog "status collector down -> restarting"
    setsid nohup $BASE/.venv/bin/python $BASE/omnivoice_training/progress_status.py \
      >> $BASE/run_prog/logs/status.log 2>&1 < /dev/null &
    disown || true
  fi
  if ! alive_sh progressive_train.sh; then
    wlog "trainer down -> restarting"
    rm -f $BASE/run_prog/progressive.pid
    setsid nohup $BASE/omnivoice_training/progressive_train.sh \
      >> $BASE/run_prog/logs/progressive_stdout.log 2>&1 < /dev/null &
    disown || true
  fi

  # leaked-VRAM detector: memory in use but nothing computing and nothing of
  # ours holding an nvidia fd
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  holders=0
  for p in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
    ls -l /proc/$p/fd 2>/dev/null | grep -q nvidia && holders=$((holders+1))
  done
  if [ "${used:-0}" -gt 20000 ] && [ "$holders" -eq 0 ]; then
    wlog "WARNING ${used}MiB held with no live nvidia fd holders (leaked)"
  fi
  sleep 120
done
