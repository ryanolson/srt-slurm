#!/bin/bash
# kvbm-dev-install.sh — srt-slurm setup_script for the kvbm-hub plane on the DEV
# image. Installs the locally-built kvbm + ai_dynamo + ai_dynamo_runtime wheels
# (from the mounted dynamo repo target/wheels) into the image venv BEFORE the
# frontend / decode / prefill commands run. Mirrors harness/install-dynamo.sh
# (same wheels, same order, --no-deps so vllm stays pinned).
#
# Runs IN-container, once per node, via the worker/frontend preamble (the prefill
# aside passes it too). The kvbm_hub binary needs no wheel (runs from
# /workspace/target/release or, in prod, /opt/kvbm/bin).
#
# Pair with `dynamo.install: false` in the recipe so srt-slurm does NOT also
# pip-install ai-dynamo from PyPI (wrong version; would clobber these wheels).
#
# PRODUCTION image (baked; /opt/kvbm/IMAGE_VARIANT=production): verify-only no-op.
set -euo pipefail
VENV="${KVBM_VENV:-/opt/dynamo/venv}"
WHEELS="${KVBM_WHEELS_DIR:-/workspace/target/wheels}"
PIP="$VENV/bin/pip"
PY="$VENV/bin/python3"

verify() {
  "$PY" - <<'PYEOF'
import vllm, kvbm
import dynamo._core, dynamo.frontend, dynamo.vllm
print(f"[kvbm-setup] OK: vllm {vllm.__version__} | kvbm {kvbm.__version__} | dynamo._core+frontend+vllm import")
PYEOF
}

# Production image: wheels are baked — verify, don't reinstall.
if [ -f /opt/kvbm/IMAGE_VARIANT ] && grep -qi '^production' /opt/kvbm/IMAGE_VARIANT 2>/dev/null; then
  echo "[kvbm-setup] production image (marker present) — wheels baked; verifying instead of reinstalling"
  verify
  exit 0
fi

KVBM_WHL=$(ls -t "$WHEELS"/kvbm-*-cp310-abi3-linux_aarch64.whl 2>/dev/null | head -1)
RT_WHL=$(ls -t "$WHEELS"/ai_dynamo_runtime-*.whl 2>/dev/null | head -1)
PY_WHL=$(ls -t "$WHEELS"/ai_dynamo-*-py3-none-any.whl 2>/dev/null | head -1)
[ -n "$KVBM_WHL" ] || { echo "[kvbm-setup] FATAL: no kvbm wheel in $WHEELS (is the dynamo repo mounted at /workspace?)" >&2; exit 1; }
[ -n "$RT_WHL" ]   || { echo "[kvbm-setup] FATAL: no ai_dynamo_runtime wheel in $WHEELS" >&2; exit 1; }
[ -n "$PY_WHL" ]   || { echo "[kvbm-setup] FATAL: no ai_dynamo py3-none-any wheel in $WHEELS" >&2; exit 1; }

echo "[kvbm-setup] host=$(hostname) venv=$VENV"
echo "[kvbm-setup] installing (--no-deps): $(basename "$KVBM_WHL") $(basename "$RT_WHL") $(basename "$PY_WHL")"
"$PIP" install --force-reinstall --no-deps "$KVBM_WHL" "$RT_WHL" "$PY_WHL"
verify
