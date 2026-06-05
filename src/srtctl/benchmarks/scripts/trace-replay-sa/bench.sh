#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Trace Replay (SA) Benchmark using aiperf
# Replays a SemiAnalysis-style agentic-coding trace dataset pulled from
# HuggingFace via aiperf's --public-dataset, at configurable concurrency levels.
#
# Unlike trace-replay (which takes a local JSONL trace file via --input-file +
# --custom-dataset-type mooncake_trace), this variant downloads a pre-registered
# public dataset (e.g. semianalysis_cc_traces_weka_no_subagents) and runs it as a
# chat workload. The public-dataset loaders live in cquil11/aiperf@cjq/agentx-v0.3,
# so the recipe must pin aiperf_package to that branch.
#
# Usage: bench.sh ENDPOINT MODEL_NAME PUBLIC_DATASET NUM_DATASET_ENTRIES CONCURRENCIES \
#                 [TTFT_THRESHOLD] [ITL_THRESHOLD] [TOKENIZER_PATH] [EXTRA_ARGS...]
#
# EXTRA_ARGS: additional aiperf flags (passed through from the recipe's aiperf_args,
#             e.g. --benchmark-duration 1200 --workers-max 200 --export-http-trace)
#
# Profiling support (optional):
#   PROFILING_BACKEND: set to "trtllm" to use the no-op TRTLLM profiling lib
#                      (profiling is managed by worker env vars at launch time)
#   PROFILE_TYPE: "nsys" or "nsys-time" -- logged for diagnostics

set -e

SCRIPT_DIR="$(dirname "$0")"
LIB_DIR="${SCRIPT_DIR}/../lib"

# Source the appropriate profiling library
if [[ "${PROFILING_BACKEND:-}" == "trtllm" ]]; then
    # shellcheck source=../lib/profiling_trtllm.sh
    source "${LIB_DIR}/profiling_trtllm.sh"
else
    # shellcheck source=../lib/profiling.sh
    source "${LIB_DIR}/profiling.sh"
fi
profiling_init_from_env

cleanup() { stop_all_profiling; }
trap cleanup EXIT

# Ensure Python output is unbuffered for real-time logging
export PYTHONUNBUFFERED=1

ENDPOINT=$1
MODEL_NAME=${2:-"test-model"}
PUBLIC_DATASET=$3
NUM_DATASET_ENTRIES=${4:-}
CONCURRENCIES=${5:-"1"}
TTFT_THRESHOLD=${6:-2000}
ITL_THRESHOLD=${7:-25}
TOKENIZER_PATH=${8:-"/model"}
# Remaining args are extra aiperf flags
shift 8 2>/dev/null || true
EXTRA_ARGS=("$@")

# --public-dataset value is required (the dataset is pulled from HuggingFace, no local file)
if [ -z "${PUBLIC_DATASET}" ]; then
    echo "ERROR: public dataset name is required (arg 3)"
    exit 1
fi

# Build dataset args: --num-dataset-entries is optional (omit to let aiperf use the full corpus)
DATASET_ARGS=(--public-dataset "${PUBLIC_DATASET}")
if [ -n "${NUM_DATASET_ENTRIES}" ]; then
    DATASET_ARGS+=(--num-dataset-entries "${NUM_DATASET_ENTRIES}")
fi

# Optional: extra Prometheus endpoints for AIPerf server metrics
SERVER_METRICS_ARGS=()
if [ -n "${AIPERF_SERVER_METRICS_URLS:-}" ]; then
    IFS=',' read -r -a server_metrics_urls <<< "${AIPERF_SERVER_METRICS_URLS}"
    if [ ${#server_metrics_urls[@]} -gt 0 ]; then
        SERVER_METRICS_ARGS+=(--server-metrics "${server_metrics_urls[@]}")
        SERVER_METRICS_ARGS+=(--server-metrics-formats json jsonl)
    fi
fi

# Setup directories (BASE_DIR defaults to /logs inside container, overridable for testing)
BASE_DIR="${BASE_DIR:-/logs}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${BASE_DIR}/artifacts}"
mkdir -p "${ARTIFACT_DIR}"

# Increase file descriptor limit for high concurrency
ulimit -n 600000 2>/dev/null || ulimit -n 65536 2>/dev/null || true

# Increase aiperf HTTP timeout
export AIPERF_HTTP_SO_RCVTIMEO=120

echo "=============================================="
echo "Trace Replay (SA) Benchmark (aiperf)"
echo "=============================================="
echo "Endpoint: ${ENDPOINT}"
echo "Model: ${MODEL_NAME}"
echo "Public Dataset: ${PUBLIC_DATASET}"
echo "Num Dataset Entries: ${NUM_DATASET_ENTRIES:-<all>}"
echo "Concurrencies: ${CONCURRENCIES}"
echo "TTFT Threshold: ${TTFT_THRESHOLD}ms"
echo "ITL Threshold: ${ITL_THRESHOLD}ms"
echo "Tokenizer Path: ${TOKENIZER_PATH}"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "Extra Args: ${EXTRA_ARGS[*]}"
fi
if [[ "${PROFILE_TYPE:-none}" != "none" ]]; then
    echo "Profiling: ${PROFILE_TYPE} (backend=${PROFILING_BACKEND:-sglang})"
fi
echo "=============================================="

# Create isolated aiperf environment (avoids polluting container packages)
# AIPERF_PACKAGE env var controls the version. For the SA public datasets this
# must point at the fork that registers them, e.g.
#   git+https://github.com/cquil11/aiperf.git@cjq/agentx-v0.3
AIPERF_SPEC="${AIPERF_PACKAGE:-aiperf}"
AIPERF_VENV="/tmp/aiperf-${SLURM_JOB_ID:-$$}"

echo "Setting up aiperf environment: ${AIPERF_SPEC}"

# Install uv if not in container
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv venv "${AIPERF_VENV}"
uv pip install -p "${AIPERF_VENV}" "${AIPERF_SPEC}" tiktoken
export PATH="${AIPERF_VENV}/bin:${PATH}"
echo "aiperf $(aiperf --version 2>/dev/null || echo 'installed') in ${AIPERF_VENV}"

# Run small benchmark for warmup (synthetic prompts, no dataset download needed).
# CAP the warmup output with max_tokens — the warmup is SYNTHETIC (no trace to set
# per-request output_length), so ignore_eos:true alone makes each request generate to
# max-model-len (~256k tokens on the 235B/262144 config => ~40 min/request, i.e. the
# whole walltime burned in warmup). 128 tokens primes the decode path in seconds.
# (The MAIN run below is correctly bounded: the WekaTrace loader sets max_tokens =
# req.output_length per request, so do NOT add a global cap there — it would override
# the recorded trace output lengths and break faithful replay.)
echo "Running warmup..."
WARMUP_DIR="${ARTIFACT_DIR}/warmup"
mkdir -p "${WARMUP_DIR}"
aiperf profile \
    -m "${MODEL_NAME}" \
    --tokenizer "${TOKENIZER_PATH}" \
    --tokenizer-trust-remote-code \
    --url "${ENDPOINT}" \
    --endpoint-type chat \
    --streaming \
    --ui simple \
    --extra-inputs ignore_eos:true max_tokens:128 \
    --concurrency 1 \
    --request-count 5 \
    --artifact-dir "${WARMUP_DIR}"
echo "Warmup complete"

# Setup artifact directory
MODEL_BASE_NAME="${MODEL_NAME##*/}"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# Parse concurrencies (comma-separated)
IFS=',' read -r -a CONCURRENCY_LIST <<< "${CONCURRENCIES}"

# a no-op if profiling is not enabled
start_all_profiling

for C in "${CONCURRENCY_LIST[@]}"; do
    echo ""
    echo "=============================================="
    echo "Running concurrency=${C}"
    echo "=============================================="
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting benchmark at concurrency ${C}"

    RUN_ARTIFACT_DIR="${ARTIFACT_DIR}/${MODEL_BASE_NAME}_sa_trace_c${C}_${TIMESTAMP}"
    mkdir -p "${RUN_ARTIFACT_DIR}"

    aiperf profile \
        -m "${MODEL_NAME}" \
        --tokenizer "${TOKENIZER_PATH}" \
        --tokenizer-trust-remote-code \
        "${DATASET_ARGS[@]}" \
        --url "${ENDPOINT}" \
        --endpoint-type chat \
        --streaming \
        --extra-inputs ignore_eos:true \
        --concurrency "${C}" \
        --random-seed 42 \
        --ui simple \
        --artifact-dir "${RUN_ARTIFACT_DIR}" \
        "${SERVER_METRICS_ARGS[@]}" \
        --goodput "time_to_first_token:${TTFT_THRESHOLD} inter_token_latency:${ITL_THRESHOLD}" \
        "${EXTRA_ARGS[@]}"

    echo "$(date '+%Y-%m-%d %H:%M:%S') - Concurrency ${C} complete"

    # List artifacts
    ls -la "${RUN_ARTIFACT_DIR}" 2>/dev/null || true
done

# a no-op if profiling is not enabled
stop_all_profiling

echo ""
echo "=============================================="
echo "Trace Replay (SA) Benchmark Complete"
echo "Results saved to: ${ARTIFACT_DIR}"
echo "=============================================="
