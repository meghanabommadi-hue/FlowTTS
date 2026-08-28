#!/usr/bin/env bash
# Full-corpus pipeline: prepare -> tokenize -> discard WAVs, one shard-window
# at a time, so disk never holds more than one window of audio.
#
# The whole corpus is ~1 TB of WAV if materialised at once, which does not fit.
# Audio TOKENS are tiny (~9 GB for everything), so only the intermediate WAVs
# need bounding. Each window is independent and leaves a marker, so a restart
# (or another container wipe) resumes instead of redoing the work.
set -uo pipefail

BASE=${BASE:-/home/jovyan/omnivoice-train}
RUN=${RUN:-$BASE/run}
SRC=${SRC:-$BASE/OmniVoice-src}
PKG=${PKG:-$BASE/omnivoice_training}
PY=${PY:-$BASE/.venv/bin/python}

DATA_REPO=${DATA_REPO:-kapturecx/ohun}
LANGS=${LANGS:-"ibo yor hau pcm"}
MAX_SEC=${MAX_SEC:-30}
MIN_SEC=${MIN_SEC:-1.5}
READ_WORKERS=${READ_WORKERS:-12}
NJ_PER_GPU=${NJ_PER_GPU:-4}
LOADER_WORKERS=${LOADER_WORKERS:-32}
# window sizes tuned to ~10-15 GB of parquet per chunk per language
declare -A WIN=( [ibo]=600 [yor]=28 [hau]=28 [pcm]=28 )
# Reader parallelism is per-language: ibo shards are ~28 MB, the others ~500 MB,
# and each in-flight shard is held in RAM as Python objects.
declare -A RW=( [ibo]=32 [yor]=14 [hau]=14 [pcm]=14 )
MIN_FREE_GB=${MIN_FREE_GB:-150}
# BALANCE: the corpus is heavily skewed (pcm 335k utts vs hau 40k). Cap every
# language at the same number of hours so the model does not just learn pidgin.
TARGET_HOURS=${TARGET_HOURS:-90}
DEV_HOURS=${DEV_HOURS:-0.5}

mkdir -p "$RUN"/{stage,logs,prep,tokens,exp,eval_wav,ui}
export PYTHONPATH="$SRC:${PYTHONPATH:-}"
# Reads use the Pro token (24.9 MB/s vs 7.2), pushes use the write token.
export HF_TOKEN=${HF_TOKEN:-$(cat "$BASE/token.read" 2>/dev/null || cat "$BASE/token.write" 2>/dev/null)}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$RUN/logs/full_pipeline.log"; }

shard_count(){ # lang split -> number of shards in the newest set
  $PY - "$1" "$2" <<'PY'
import sys, os
sys.path.insert(0, os.environ["PKG"])
from hf_parquet import list_shards
lang, split = sys.argv[1], sys.argv[2]
print(len(list_shards(os.environ["DATA_REPO"], split, token=os.environ.get("HF_TOKEN"))))
PY
}

free_gb(){ df -BG --output=avail "$RUN" | tail -1 | tr -dc '0-9'; }
free_ram_gb(){ awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }
free_vram_mb(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'; }

wait_for_resources(){
  local waited=0
  while :; do
    local d=$(free_gb) r=$(free_ram_gb) v=$(free_vram_mb)
    local ok=1
    [ "${d:-0}" -lt "$MIN_FREE_GB" ] && ok=0
    [ "${r:-999}" -lt "${MIN_FREE_RAM_GB:-40}" ] && ok=0
    [ "${v:-99999}" -lt "${MIN_FREE_VRAM_MB:-8000}" ] && ok=0
    [ "$ok" = "1" ] && return 0
    [ $((waited % 300)) -eq 0 ] && \
      log "  waiting on resources: disk ${d}G (need $MIN_FREE_GB) ram ${r}G (need ${MIN_FREE_RAM_GB:-40}) vram ${v}M (need ${MIN_FREE_VRAM_MB:-8000})"
    sleep 30; waited=$((waited+30))
  done
}

lang_hours(){ # hours accumulated so far for a language/split
  local f="$RUN/tokens/$2/$1/data.lst"
  [ -s "$f" ] || { echo 0; return; }
  awk '{t+=$4} END{printf "%.2f", t/3600}' "$f"
}

process_window(){ # lang split start count
  local lang=$1 split=$2 start=$3 count=$4
  local tag="${lang}_${split}_${start}"
  [ -f "$RUN/stage/win_$tag.done" ] && return 0

  wait_for_resources

  local wdir="$RUN/prep/$tag"
  rm -rf "$wdir"; mkdir -p "$wdir"
  local rw=${RW[$lang]:-8}
  $PY "$PKG/ohun_prepare.py" --repo "$DATA_REPO" --out "$wdir" --langs "$lang" \
      --split "$split" --hours-per-lang 100000 --dev-minutes-per-lang 0 \
      --min-sec "$MIN_SEC" --max-sec "$MAX_SEC" --read-workers "$rw" \
      --skip-shards "$start" --max-shards "$count" \
      >> "$RUN/logs/prep_$tag.log" 2>&1
  local n=$(cat "$wdir"/*_train.jsonl "$wdir"/*_dev.jsonl 2>/dev/null | wc -l)
  if [ "${n:-0}" -eq 0 ]; then
    log "  [$tag] no utterances survived filtering - skipping"
    rm -rf "$wdir"; touch "$RUN/stage/win_$tag.done"; return 0
  fi

  cat "$wdir"/*_train.jsonl "$wdir"/*_dev.jsonl 2>/dev/null > "$wdir/all.jsonl"
  local tdir="$RUN/tokens/$split/$tag"
  mkdir -p "$tdir"/{audios,txts}
  $PY -m omnivoice.scripts.extract_audio_tokens \
      --input_jsonl "$wdir/all.jsonl" \
      --tar_output_pattern "$tdir/audios/shard-%06d.tar" \
      --jsonl_output_pattern "$tdir/txts/shard-%06d.jsonl" \
      --tokenizer_path eustlb/higgs-audio-v2-tokenizer \
      --min_length "$MIN_SEC" --max_length "$MAX_SEC" \
      --nj_per_gpu "$NJ_PER_GPU" --loader_workers "$LOADER_WORKERS" \
      --skip_errors --shuffle True --min_num_shards 4 \
      >> "$RUN/logs/tok_$tag.log" 2>&1
  if [ -s "$tdir/data.lst" ]; then
    mkdir -p "$RUN/tokens/$split/$lang"
    cat "$tdir/data.lst" >> "$RUN/tokens/$split/$lang/data.lst"
    log "  [$tag] +$(wc -l < "$tdir/data.lst") shards, $n utts -> $(lang_hours "$lang" "$split")h for $lang/$split"
    touch "$RUN/stage/win_$tag.done"
  else
    log "  [$tag] TOKENIZE PRODUCED NOTHING - see logs/tok_$tag.log"
  fi
  rm -rf "$wdir"          # the WAVs have served their purpose
}

export PKG DATA_REPO
log "=== full-corpus pipeline: langs=$LANGS max_sec=$MAX_SEC ==="
for split in dev train; do
  for lang in $LANGS; do
    total=$(shard_count "$lang" "$split" 2>/dev/null || echo 0)
    [ "${total:-0}" -eq 0 ] && { log "[$lang/$split] no shards found - skipped"; continue; }
    w=${WIN[$lang]:-24}
    log "[$lang/$split] $total shards, windows of $w"
    local_target=$TARGET_HOURS
    [ "$split" = "dev" ] && local_target=$DEV_HOURS
    start=0
    while [ "$start" -lt "$total" ]; do
      have=$(lang_hours "$lang" "$split")
      if awk "BEGIN{exit !($have >= $local_target)}"; then
        log "[$lang/$split] reached ${have}h (target ${local_target}h) - stopping early"
        break
      fi
      process_window "$lang" "$split" "$start" "$w"
      start=$((start + w))
    done
  done
done

log "=== balancing languages to equal hours ==="
$PY "$PKG/balance_tokens.py" --run "$RUN" --langs "$LANGS" 2>&1 | tee -a "$RUN/logs/full_pipeline.log"

log "=== tokenization complete - per-language totals ==="
for s in train dev; do
  : > "$RUN/tokens/$s/data.lst"
  for lang in $LANGS; do
    f="$RUN/tokens/$s/$lang/data.lst"
    [ -s "$f" ] || continue
    log "  $s/$lang: $(wc -l < "$f") shards, $(awk '{n+=$3; t+=$4} END{printf "%d utts %.1fh", n, t/3600}' "$f")"
    cat "$f" >> "$RUN/tokens/$s/data.lst"
  done
  [ -s "$RUN/tokens/$s/data.lst" ] && \
    log "  $s TOTAL: $(wc -l < "$RUN/tokens/$s/data.lst") shards, $(awk '{n+=$3; t+=$4} END{printf "%d utts %.1fh", n, t/3600}' "$RUN/tokens/$s/data.lst")"
done
touch "$RUN/stage/tokenize_all.done"
