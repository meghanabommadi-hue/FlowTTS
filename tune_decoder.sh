#!/usr/bin/env bash
# FlowTTS benchmark + decoder parameter tuner.
#
# Usage:
#   ./tune_decoder.sh [--max-batch N] [--batch-timeout-ms N]
#                     [--gpu-chunk-size N] [--onnx-workers N]
#
# Runs a full sweep if no decoder args given.
# Pass specific values to benchmark a single configuration.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VIRTUAL_ENV:-${HOME}/FlowTTS/.venv}"
PYTHON="${VENV}/bin/python3"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${SCRIPT_DIR}/tune_logs/${RUN_TS}"
mkdir -p "${LOG_DIR}"

SERVER_LOG="${LOG_DIR}/server.log"
RESULTS_FILE="${LOG_DIR}/results.tsv"
BEST_FILE="${LOG_DIR}/best_config.txt"

REQUESTS=75
CONCURRENCY=9
ROUNDS=5
TTFF_TARGET=1.0
MAX_REQUESTS=200
CTRL_PORT=8764
WS_PORT=8765
SERVER_PID=""

# ── Decoder override args (empty = use config.py default) ────────────────────
ARG_MAX_BATCH=""
ARG_BATCH_TIMEOUT_MS=""
ARG_GPU_CHUNK_SIZE=""
ARG_ONNX_WORKERS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-batch)         ARG_MAX_BATCH="$2";        shift 2 ;;
        --batch-timeout-ms)  ARG_BATCH_TIMEOUT_MS="$2"; shift 2 ;;
        --gpu-chunk-size)    ARG_GPU_CHUNK_SIZE="$2";   shift 2 ;;
        --onnx-workers)      ARG_ONNX_WORKERS="$2";     shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

BASELINE_MAX_BATCH="${ARG_MAX_BATCH:-256}"
BASELINE_TIMEOUT="${ARG_BATCH_TIMEOUT_MS:-0.5}"
BASELINE_CHUNK="${ARG_GPU_CHUNK_SIZE:-160}"
BASELINE_WORKERS="${ARG_ONNX_WORKERS:-2}"

printf "param\tvalue\tttff_min\tttff_avg\tttff_max\trtf_bad_pct\n" > "${RESULTS_FILE}"

# ── env setup ─────────────────────────────────────────────────────────────────
setup_env() {
    export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
    export PATH="${VENV}/bin:${PATH}"
    export LD_LIBRARY_PATH="\
${VENV}/lib/python3.12/site-packages/torch/lib:\
${VENV}/lib/python3.12/site-packages/nvidia/cudnn/lib:\
${VENV}/lib/python3.12/site-packages/nvidia/cu13/lib:\
${VENV}/lib/python3.12/site-packages/tensorrt_libs:\
${LD_LIBRARY_PATH:-}"
}

# ── decoder env vars ──────────────────────────────────────────────────────────
apply_decoder_env() {
    local param="$1" value="$2"
    case "${param}" in
        max_batch)        export FLOWTTS_DECODER__MAX_BATCH="${value}" ;;
        batch_timeout_ms) export FLOWTTS_DECODER__BATCH_TIMEOUT_MS="${value}" ;;
        gpu_chunk_size)   export FLOWTTS_DECODER__GPU_CHUNK_SIZE="${value}" ;;
        onnx_workers)     export FLOWTTS_DECODER__ONNX_WORKERS="${value}" ;;
    esac
}

clear_decoder_env() {
    unset FLOWTTS_DECODER__MAX_BATCH          2>/dev/null || true
    unset FLOWTTS_DECODER__BATCH_TIMEOUT_MS   2>/dev/null || true
    unset FLOWTTS_DECODER__GPU_CHUNK_SIZE     2>/dev/null || true
    unset FLOWTTS_DECODER__ONNX_WORKERS       2>/dev/null || true
}

# Apply any args passed on the command line as the starting baseline
[[ -n "${ARG_MAX_BATCH}"        ]] && export FLOWTTS_DECODER__MAX_BATCH="${ARG_MAX_BATCH}"
[[ -n "${ARG_BATCH_TIMEOUT_MS}" ]] && export FLOWTTS_DECODER__BATCH_TIMEOUT_MS="${ARG_BATCH_TIMEOUT_MS}"
[[ -n "${ARG_GPU_CHUNK_SIZE}"   ]] && export FLOWTTS_DECODER__GPU_CHUNK_SIZE="${ARG_GPU_CHUNK_SIZE}"
[[ -n "${ARG_ONNX_WORKERS}"     ]] && export FLOWTTS_DECODER__ONNX_WORKERS="${ARG_ONNX_WORKERS}"

# ── kill any existing server ──────────────────────────────────────────────────
stop_server() {
    echo "Stopping server..."
    pkill -TERM -f "flowtts\.server" 2>/dev/null || true
    pkill -TERM -f "flowtts\.main"   2>/dev/null || true
    pkill -TERM -f "run\.sh"         2>/dev/null || true
    if [[ -n "${SERVER_PID}" ]]; then
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
    fi
    sleep 2
    pkill -KILL -f "flowtts\.server" 2>/dev/null || true
    pkill -KILL -f "flowtts\.main"   2>/dev/null || true
    local waited=0
    while (echo >/dev/tcp/localhost/"${WS_PORT}") 2>/dev/null; do
        sleep 1; waited=$((waited + 1))
        [[ "${waited}" -ge 30 ]] && { echo "WARNING: port ${WS_PORT} still open after 30s"; break; }
    done
    SERVER_PID=""
}

# ── start server ──────────────────────────────────────────────────────────────
start_server() {
    setup_env
    echo "Starting server..."
    # Truncate log so wait_for_ready only scans the current boot
    > "${SERVER_LOG}"
    VIRTUAL_ENV="${VENV}" bash "${SCRIPT_DIR}/run.sh" --ctrl-port "${CTRL_PORT}" \
        >> "${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!
    echo "Server PID=${SERVER_PID}  log=${SERVER_LOG}"
}

# ── wait for ready via log (no fixed sleep) ───────────────────────────────────
wait_for_ready() {
    local timeout=600 waited=0
    echo "Waiting for warmup..."
    while true; do
        if grep -qE "(FlowTTS.*ready|all ports warmed up)" "${SERVER_LOG}" 2>/dev/null; then
            echo "Server ready (${waited}s)"
            return 0
        fi
        # Bail if the process died
        if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "ERROR: server process ${SERVER_PID} died — check ${SERVER_LOG}"
            exit 1
        fi
        if [[ "${waited}" -ge "${timeout}" ]]; then
            echo "ERROR: server not ready after ${timeout}s — check ${SERVER_LOG}"
            exit 1
        fi
        sleep 2
        waited=$((waited + 2))
    done
}

# ── trap: always kill server on exit ─────────────────────────────────────────
cleanup() {
    echo ""
    stop_server
    clear_decoder_env
}
trap cleanup EXIT

# ── run N benchmark rounds ────────────────────────────────────────────────────
run_rounds() {
    local tag="$1" requests="${2:-${REQUESTS}}"
    local ttff_sum=0 ttff_min=99 ttff_max=0
    local rtf_bad_total=0 rtf_total=0 valid_rounds=0

    setup_env
    echo "  [${tag}] ${ROUNDS} rounds × ${requests} requests"

    for i in $(seq 1 "${ROUNDS}"); do
        local run_log="${LOG_DIR}/run_${tag//[^a-zA-Z0-9]/_}_${i}.log"
        echo "  Running test ${i}..."

        "${PYTHON}" -m flowtts.test.test_pipeline \
            --ctrl-port "${CTRL_PORT}" \
            --no-launch \
            --requests "${requests}" \
            --concurrency "${CONCURRENCY}" \
            --streaming \
            2>&1 | tee "${run_log}"

        local line
        line=$(grep "time-to-first" "${run_log}" || true)
        if [[ -z "${line}" ]]; then
            echo "  run ${i}: FAILED — no time-to-first in output"
            continue
        fi

        local avg min max
        avg=$(echo "${line}" | grep -oP "avg=\K[0-9.]+")
        min=$(echo "${line}" | grep -oP "min=\K[0-9.]+")
        max=$(echo "${line}" | grep -oP "max=\K[0-9.]+")

        local rtf_line bad=0 total_r=0
        rtf_line=$(grep "rtf > 1.0" "${run_log}" || true)
        if [[ -n "${rtf_line}" ]]; then
            bad=$(echo "${rtf_line}"   | grep -oP "\K[0-9]+(?=/)")
            total_r=$(echo "${rtf_line}" | grep -oP "/\K[0-9]+")
        fi

        echo "  run ${i}: ttff avg=${avg}s  min=${min}s  max=${max}s  rtf>1=${bad}/${total_r}"

        ttff_sum=$(awk "BEGIN{print ${ttff_sum}+${avg}}")
        ttff_min=$(awk "BEGIN{print (${min}<${ttff_min})?${min}:${ttff_min}}")
        ttff_max=$(awk "BEGIN{print (${max}>${ttff_max})?${max}:${ttff_max}}")
        rtf_bad_total=$((rtf_bad_total + bad))
        rtf_total=$((rtf_total + total_r))
        valid_rounds=$((valid_rounds + 1))
    done

    if [[ "${valid_rounds}" -eq 0 ]]; then
        echo "  → ALL rounds failed"
        ROUND_AVG="99.000"; ROUND_MIN="99"; ROUND_MAX="0"; RTF_PCT="N/A"
        return
    fi

    ROUND_AVG=$(awk "BEGIN{printf \"%.3f\", ${ttff_sum}/${valid_rounds}}")
    ROUND_MIN="${ttff_min}"
    ROUND_MAX="${ttff_max}"
    RTF_PCT=$(awk "BEGIN{if(${rtf_total}>0) printf \"%.1f\", 100*${rtf_bad_total}/${rtf_total}; else print \"N/A\"}")
    echo "  → [${tag}] TTFF: avg=${ROUND_AVG}s  min=${ROUND_MIN}s  max=${ROUND_MAX}s  rtf>1=${RTF_PCT}%"
}

# ── evaluate one candidate: stop → set env → start → wait → test ─────────────
BEST_TTFF=99
BEST_PARAMS="baseline"
BEST_REQUESTS="${REQUESTS}"

evaluate() {
    local param="$1" value="$2"
    local tag="${param}=${value}"

    local desc=""
    case "${param}" in
        max_batch)        desc="max tokens per decode batch → decoder TTFF / decode lag" ;;
        batch_timeout_ms) desc="ms before batch dispatch → time-to-first (biggest lever)" ;;
        gpu_chunk_size)   desc="tokens per GPU iteration → RTF / total latency" ;;
        onnx_workers)     desc="ONNX threads feeding GPU → rtf > 1.0% under load" ;;
    esac

    echo ""
    echo "══ ${tag}  [${desc}] ══"

    stop_server
    clear_decoder_env
    apply_decoder_env "${param}" "${value}"
    start_server
    wait_for_ready
    run_rounds "${tag}"

    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${param}" "${value}" \
        "${ROUND_MIN}" "${ROUND_AVG}" "${ROUND_MAX}" "${RTF_PCT}" \
        >> "${RESULTS_FILE}"

    if awk "BEGIN{exit !(${ROUND_AVG} < ${BEST_TTFF})}"; then
        BEST_TTFF="${ROUND_AVG}"
        BEST_PARAMS="${tag}"
        echo "  ★ New best: ${tag}  avg=${BEST_TTFF}s"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  FlowTTS decoder tuner  [${RUN_TS}]"
echo "  Baseline: max_batch=${BASELINE_MAX_BATCH}  batch_timeout_ms=${BASELINE_TIMEOUT}"
echo "            gpu_chunk_size=${BASELINE_CHUNK}  onnx_workers=${BASELINE_WORKERS}"
echo "  Requests: ${REQUESTS}  Rounds: ${ROUNDS}  Target TTFF: <${TTFF_TARGET}s"
echo "  Logs: ${LOG_DIR}"
echo "════════════════════════════════════════════════════════════"

# ── Kill any stale server first ───────────────────────────────────────────────
stop_server

# ── Baseline run ──────────────────────────────────────────────────────────────
echo ""
echo "══ BASELINE ══"
start_server
wait_for_ready
run_rounds "baseline"
printf "baseline\t-\t%s\t%s\t%s\t%s\n" \
    "${ROUND_MIN}" "${ROUND_AVG}" "${ROUND_MAX}" "${RTF_PCT}" >> "${RESULTS_FILE}"
BEST_TTFF="${ROUND_AVG}"
echo "  Baseline TTFF avg=${BEST_TTFF}s"

# If specific args were passed, only do the baseline run (single benchmark mode)
if [[ -n "${ARG_MAX_BATCH}${ARG_BATCH_TIMEOUT_MS}${ARG_GPU_CHUNK_SIZE}${ARG_ONNX_WORKERS}" ]]; then
    echo ""
    echo "Single-config benchmark complete. Results: ${RESULTS_FILE}"
    exit 0
fi

# ── Phase 1: sweep ────────────────────────────────────────────────────────────
echo ""
echo "═══ Phase 1: parameter sweep ═══"

echo "  [sweep] batch_timeout_ms — targeting: time-to-first"
for v in 0.1 0.2 1.0 2.0; do
    [[ "$v" == "${BASELINE_TIMEOUT}" ]] && continue
    evaluate "batch_timeout_ms" "$v"
done

echo "  [sweep] max_batch — targeting: decoder TTFF / decode lag"
for v in 64 128 512; do
    [[ "$v" == "${BASELINE_MAX_BATCH}" ]] && continue
    evaluate "max_batch" "$v"
done

echo "  [sweep] gpu_chunk_size — targeting: RTF / total latency"
for v in 64 128 256 320; do
    [[ "$v" == "${BASELINE_CHUNK}" ]] && continue
    evaluate "gpu_chunk_size" "$v"
done

echo "  [sweep] onnx_workers — targeting: rtf > 1.0%"
for v in 1 4 6 8; do
    [[ "$v" == "${BASELINE_WORKERS}" ]] && continue
    evaluate "onnx_workers" "$v"
done

echo ""
echo "═══ Phase 1 complete — Best: ${BEST_PARAMS}  avg TTFF=${BEST_TTFF}s ═══"

# ── Phase 2: verify best ──────────────────────────────────────────────────────
echo ""
echo "═══ Phase 2: verify best param ═══"

BEST_PARAM_NAME="${BEST_PARAMS%%=*}"
BEST_PARAM_VAL="${BEST_PARAMS##*=}"

if [[ "${BEST_PARAMS}" != "baseline" ]]; then
    stop_server
    clear_decoder_env
    apply_decoder_env "${BEST_PARAM_NAME}" "${BEST_PARAM_VAL}"
    start_server
    wait_for_ready
fi

run_rounds "verify:${BEST_PARAMS}"

TARGET_HIT=0
if awk "BEGIN{exit !(${ROUND_AVG} < ${TTFF_TARGET})}"; then
    echo "  ✓ Target <${TTFF_TARGET}s ACHIEVED (avg=${ROUND_AVG}s)"
    TARGET_HIT=1
    BEST_TTFF="${ROUND_AVG}"
else
    echo "  ✗ Target not achieved (avg=${ROUND_AVG}s)"
fi

# ── Phase 3: scale requests if target hit ────────────────────────────────────
if [[ "${TARGET_HIT}" -eq 1 ]]; then
    echo ""
    echo "═══ Phase 3: scaling requests ═══"
    CUR_REQ="${REQUESTS}"

    while [[ "${CUR_REQ}" -lt "${MAX_REQUESTS}" ]]; do
        NEXT_REQ=$(( CUR_REQ + 25 ))
        [[ "${NEXT_REQ}" -gt "${MAX_REQUESTS}" ]] && NEXT_REQ="${MAX_REQUESTS}"

        echo ""
        echo "  Trying ${NEXT_REQ} requests..."
        run_rounds "scale@${NEXT_REQ}" "${NEXT_REQ}"

        printf "scale\t%s@%s\t%s\t%s\t%s\t%s\n" \
            "${BEST_PARAMS}" "${NEXT_REQ}" \
            "${ROUND_MIN}" "${ROUND_AVG}" "${ROUND_MAX}" "${RTF_PCT}" \
            >> "${RESULTS_FILE}"

        if awk "BEGIN{exit !(${ROUND_AVG} < ${TTFF_TARGET})}"; then
            echo "  ✓ Still under target at ${NEXT_REQ} requests (avg=${ROUND_AVG}s)"
            BEST_REQUESTS="${NEXT_REQ}"
            CUR_REQ="${NEXT_REQ}"
        else
            echo "  ✗ Degraded at ${NEXT_REQ} requests (avg=${ROUND_AVG}s) — max stable: ${CUR_REQ}"
            break
        fi
    done
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  TUNING COMPLETE"
echo "  Best param    : ${BEST_PARAMS}"
echo "  Best avg TTFF : ${BEST_TTFF}s"
echo "  Max requests  : ${BEST_REQUESTS}"
echo "  Target hit    : $([[ ${TARGET_HIT} -eq 1 ]] && echo YES || echo NO)"
echo "  Logs dir      : ${LOG_DIR}"
echo "════════════════════════════════════════════════════════════"

{
    echo "best_param=${BEST_PARAMS}"
    echo "best_ttff=${BEST_TTFF}"
    echo "requests=${BEST_REQUESTS}"
    echo "target_hit=${TARGET_HIT}"
} > "${BEST_FILE}"
cat "${BEST_FILE}"
