#!/usr/bin/env bash
# Stop the pipeline and any training it spawned. Leaves nginx/tensorboard up.
RUN=${RUN:-/opt/omnivoice-train/run}
for pat in "pipeline.sh" "supervise_train.sh" "train_ohun.py" "extract_audio_tokens" "ohun_prepare.py"; do
  pids=$(pgrep -f "omnivoice_training/$pat|omnivoice.scripts.$pat" 2>/dev/null)
  [ -n "$pids" ] && { echo "killing $pat: $pids"; kill $pids 2>/dev/null; }
done
sleep 5
pkill -9 -f "omnivoice_training/" 2>/dev/null
rm -f "$RUN/pipeline.pid"
echo "stopped (tensorboard + nginx left running)"
