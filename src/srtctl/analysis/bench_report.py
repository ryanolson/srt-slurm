# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline agg-vs-CD benchmark-report generator.

Turns a SET of srt-slurm run-output dirs (or job ids) into a high-impact-first
comparison report (markdown + CSV). Pure-python + offline: reads only the
artifacts already on disk (aiperf aggregates, per-request jsonl, server-metrics
jsonl) plus the recipe ``config.yaml`` (loaded through the srtctl schema so we
reuse :class:`ResourceConfig`). No compute node, no network, no new deps.

Usage::

    uv run python -m srtctl.analysis.bench_report --runs 2187812,2187813,2187814

See ``docs/design/benchmark_report_design.md`` for the report layout, column
schema, and the verified formulas. The CD-internal-leverage section (4) is N/A
for any run lacking ``kvbm_metrics_snapshot.json`` (written by the post-benchmark
snapshot hook in ``benchmark_stage.py``); everything else works on existing runs.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import html
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (the user-SLA knobs; native SLA is recovered from the cli_command)
# ---------------------------------------------------------------------------

DEFAULT_TTFT_SLA_MS = 5000.0
DEFAULT_ITL_SLA_MS = 7.0

# Default output-root search: ./outputs/<jobid> relative to the srt-slurm root.
# We resolve relative to the cwd first, then to this module's repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]  # .../srt-slurm
_DEFAULT_OUTPUTS = _REPO_ROOT / "outputs"

# The metrics snapshot the post-benchmark hook writes (section-4 source of truth).
SNAPSHOT_FILENAME = "kvbm_metrics_snapshot.json"


# ---------------------------------------------------------------------------
# Artifact discovery
# ---------------------------------------------------------------------------


def resolve_run_dir(token: str) -> Path | None:
    """Resolve a ``--runs`` token (job id or path) to a run-output dir.

    A run-output dir is the top-level ``outputs/<jobid>/`` directory that holds
    ``config.yaml`` and ``logs/``. Accepts an absolute/relative path or a bare
    job id (looked up under ./outputs and the srt-slurm repo outputs).
    """
    token = token.strip()
    if not token:
        return None
    candidates: list[Path] = []
    p = Path(token)
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(Path.cwd() / token)
        candidates.append(Path.cwd() / "outputs" / token)
        candidates.append(_DEFAULT_OUTPUTS / token)
    for c in candidates:
        if c.is_dir():
            return c.resolve()
    return None


def find_real_artifact_dir(run_dir: Path) -> Path | None:
    """Return the timestamped real-run artifacts dir (NOT warmup).

    aiperf writes two artifact subtrees: ``artifacts/warmup/`` (the throwaway
    warmup phase) and ``artifacts/<model>_sa_trace_c<N>_<ts>/`` (the real run).
    Exclude warmup BY DIRECTORY and pick the newest timestamped run dir.

    Searches ``<run>/logs/artifacts`` (the on-disk layout) then ``<run>/artifacts``.
    """
    for base in (run_dir / "logs" / "artifacts", run_dir / "artifacts"):
        if not base.is_dir():
            continue
        runs = [
            d
            for d in base.iterdir()
            if d.is_dir() and d.name != "warmup" and (d / "profile_export_aiperf.json").is_file()
        ]
        if runs:
            # Newest by mtime (timestamps in the name also sort, but mtime is robust).
            return sorted(runs, key=lambda d: d.stat().st_mtime)[-1]
    return None


def find_snapshot(run_dir: Path, artifact_dir: Path | None) -> Path | None:
    """Locate ``kvbm_metrics_snapshot.json`` (the CD-leverage source).

    SHARED CONTRACT with the snapshot hook in ``benchmark_stage.py``: the hook
    writes the snapshot into the real artifacts dir if it can resolve it, else
    the run ``logs/`` dir. This reader searches BOTH (artifacts dir first), so
    the two sides cannot drift. See benchmark_report_design.md "Metrics-timing".
    """
    search: list[Path] = []
    if artifact_dir is not None:
        search.append(artifact_dir / SNAPSHOT_FILENAME)
    search.append(run_dir / "logs" / SNAPSHOT_FILENAME)
    search.append(run_dir / SNAPSHOT_FILENAME)
    for s in search:
        if s.is_file():
            return s
    return None


# ---------------------------------------------------------------------------
# aiperf aggregate parsing
# ---------------------------------------------------------------------------


def _avg(metric: dict[str, Any] | None) -> float | None:
    if not isinstance(metric, dict):
        return None
    return metric.get("avg")


def _pct(metric: dict[str, Any] | None, pct: str) -> float | None:
    if not isinstance(metric, dict):
        return None
    return metric.get(pct)


def _compute_active_gpus(r: Any) -> int:
    """Active GPUs across the serving plane (NOT provisioned nodes).

    Three serving modes:
    - aggregated: ``num_agg * gpus_per_agg`` (+ CD prefill-aside if any).
    - CD (decode-AGG + hub-owned prefill aside): ``num_agg * gpus_per_agg +
      kvbm_prefill_nodes * kvbm_prefill_tp``.
    - native (trad) P/D disagg: ``num_decode * gpus_per_decode + num_prefill *
      gpus_per_prefill`` — BOTH planes count. (Earlier this omitted the native
      prefill plane, halving the per-GPU denominator for trad-P/D runs.)
    """
    prefill_nodes = r.kvbm_prefill_nodes or 0  # CD prefill-aside (0 for native disagg)
    prefill_tp = r.kvbm_prefill_tp or 0
    gpus = r.num_agg * r.gpus_per_agg + prefill_nodes * prefill_tp
    if not gpus and r.num_decode:
        # native P/D disagg: count the native prefill plane too (num_prefill is 0 for
        # agg/CD, so this term only fires for trad-P/D and never perturbs agg/CD).
        gpus = r.num_decode * r.gpus_per_decode + r.num_prefill * r.gpus_per_prefill + prefill_nodes * prefill_tp
    return gpus


def parse_cli_goodput_sla(cli_command: str | None) -> tuple[float | None, float | None]:
    """Recover the NATIVE goodput SLA (ttft_ms, itl_ms) from the cli_command.

    aiperf records the run's own ``--goodput 'time_to_first_token:5000 inter_token_latency:10'``
    inside ``input_config.cli_command``. We parse it so the report's "native"
    goodput line matches exactly what aiperf scored as good_request_count.
    """
    if not cli_command:
        return None, None
    m = re.search(r"--goodput\s+'([^']*)'", cli_command)
    if not m:
        m = re.search(r"--goodput\s+\"([^\"]*)\"", cli_command)
    if not m:
        return None, None
    spec = m.group(1)
    ttft = itl = None
    t = re.search(r"time_to_first_token:([0-9.]+)", spec)
    i = re.search(r"inter_token_latency:([0-9.]+)", spec)
    if t:
        ttft = float(t.group(1))
    if i:
        itl = float(i.group(1))
    return ttft, itl


def parse_cli_field(cli_command: str | None, flag: str) -> str | None:
    """Extract a single-quoted/space value for ``--flag`` from a cli_command."""
    if not cli_command:
        return None
    m = re.search(rf"--{re.escape(flag)}\s+'([^']*)'", cli_command)
    if m:
        return m.group(1)
    m = re.search(rf"--{re.escape(flag)}\s+(\S+)", cli_command)
    if m:
        return m.group(1).strip("'\"")
    return None


# ---------------------------------------------------------------------------
# Per-request goodput recompute (user SLA + native SLA, same denominator)
# ---------------------------------------------------------------------------


@dataclass
class GoodputResult:
    sla_ttft_ms: float | None
    sla_itl_ms: float | None
    good: int
    denom: int
    median_itl_ms: float | None = None

    @property
    def fraction(self) -> float | None:
        if self.denom <= 0:
            return None
        return self.good / self.denom

    @property
    def discriminatory(self) -> bool:
        """False when the ITL SLA is at/under the median ITL (goodput is noise)."""
        if self.sla_itl_ms is None or self.median_itl_ms is None:
            return True
        return self.sla_itl_ms > self.median_itl_ms


def recompute_goodput(jsonl_path: Path, ttft_sla_ms: float, itl_sla_ms: float) -> GoodputResult:
    """Recompute goodput as a FRACTION over request_count from per-request jsonl.

    Denominator = total records (== request_count). A record is GOOD iff it has
    both a TTFT and an ITL value AND ttft < sla AND itl < sla. Records missing an
    ITL (errors) and records that fail a threshold count as NOT-good but STAY in
    the denominator (we never drop error/overflow records). Also returns the
    median ITL so callers can flag a non-discriminatory SLA.
    """
    good = 0
    denom = 0
    itls: list[float] = []
    if not jsonl_path.is_file():
        return GoodputResult(ttft_sla_ms, itl_sla_ms, 0, 0, None)
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            denom += 1
            m = rec.get("metrics", {}) or {}
            ttft = (m.get("time_to_first_token") or {}).get("value")
            itl = (m.get("inter_token_latency") or {}).get("value")
            if itl is not None:
                itls.append(itl)
            if ttft is not None and itl is not None and ttft < ttft_sla_ms and itl < itl_sla_ms:
                good += 1
    median_itl = None
    if itls:
        itls.sort()
        n = len(itls)
        median_itl = itls[n // 2] if n % 2 == 1 else (itls[n // 2 - 1] + itls[n // 2]) / 2.0
    return GoodputResult(ttft_sla_ms, itl_sla_ms, good, denom, median_itl)


# ---------------------------------------------------------------------------
# server-metrics jsonl parsing (KV blocks, prefix cache, util peak)
# ---------------------------------------------------------------------------


def _metric_value(entries: Any) -> float | None:
    """Pull the scalar ``value`` from a server-metrics list-of-{labels,value}."""
    if isinstance(entries, list) and entries:
        return entries[0].get("value")
    return None


@dataclass
class KvMetrics:
    total_kv_blocks: float | None = None
    block_size: float | None = None
    util_max_perc: float | None = None
    prefix_hits: float = 0.0
    prefix_queries: float = 0.0
    external_prefix_hits: float = 0.0
    decode_worker_count: int = 0

    @property
    def kv_reused_blocks(self) -> float | None:
        if not self.block_size:
            return None
        return self.prefix_hits / self.block_size

    @property
    def kv_missed_blocks(self) -> float | None:
        if not self.block_size:
            return None
        return max(0.0, self.prefix_queries - self.prefix_hits) / self.block_size

    @property
    def kv_token_capacity(self) -> float | None:
        """G1 token capacity = total_blocks * block_size.

        Block-count alone does not give GiB without dtype/heads geometry, so we
        report the token capacity (the honest, model-agnostic proxy) instead.
        """
        if self.total_kv_blocks is None or self.block_size is None:
            return None
        return self.total_kv_blocks * self.block_size

    @property
    def local_prefix_hit_rate(self) -> float | None:
        if self.prefix_queries <= 0:
            return None
        return self.prefix_hits / self.prefix_queries


def parse_server_metrics(jsonl_path: Path) -> KvMetrics:
    """Parse KV-cache + prefix-cache metrics from server_metrics_export.jsonl.

    The file is one JSON object per (endpoint, timestamp) snapshot. We:
      * take the LAST snapshot per endpoint for the counters (they are monotonic);
      * sum prefix hits/queries/external over the per-worker endpoints
        (``:75xx/metrics``, the vLLM decode workers);
      * read total_kv_blocks + block_size from the frontend endpoint
        (``localhost:8000/metrics``) last snapshot;
      * track the PEAK ``vllm:kv_cache_usage_perc`` across ALL snapshots (the last
        snapshot is typically 0.0 because the run has drained).
    """
    km = KvMetrics()
    if not jsonl_path.is_file():
        return km

    last_by_ep: dict[str, dict[str, Any]] = {}
    util_max = 0.0
    saw_util = False
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ep = rec.get("endpoint_url", "")
            last_by_ep[ep] = rec
            # Peak util scan across every snapshot of every worker endpoint.
            mt = rec.get("metrics", {}) or {}
            uv = _metric_value(mt.get("vllm:kv_cache_usage_perc"))
            if uv is not None:
                saw_util = True
                util_max = max(util_max, uv)

    worker_eps = [ep for ep in last_by_ep if re.search(r":75\d\d/metrics$", ep)]
    km.decode_worker_count = len(worker_eps)
    for ep in worker_eps:
        mt = last_by_ep[ep].get("metrics", {}) or {}
        h = _metric_value(mt.get("vllm:prefix_cache_hits"))
        q = _metric_value(mt.get("vllm:prefix_cache_queries"))
        e = _metric_value(mt.get("vllm:external_prefix_cache_hits"))
        if h:
            km.prefix_hits += h
        if q:
            km.prefix_queries += q
        if e:
            km.external_prefix_hits += e

    # Frontend endpoint carries the model-level total kv blocks + block size.
    fe = None
    for ep in last_by_ep:
        if "localhost:8000" in ep or ep.endswith(":8000/metrics"):
            fe = last_by_ep[ep]
            break
    if fe is not None:
        mt = fe.get("metrics", {}) or {}
        km.total_kv_blocks = _metric_value(mt.get("dynamo_frontend_model_total_kv_blocks"))
        km.block_size = _metric_value(mt.get("dynamo_frontend_model_kv_cache_block_size"))
    # block_size fallback: any worker that reports a block size, else None.
    if km.block_size is None:
        for ep in worker_eps:
            mt = last_by_ep[ep].get("metrics", {}) or {}
            bs = _metric_value(mt.get("dynamo_frontend_model_kv_cache_block_size"))
            if bs:
                km.block_size = bs
                break

    km.util_max_perc = util_max if saw_util else None
    return km


# ---------------------------------------------------------------------------
# CD-internal-leverage snapshot (section 4)
# ---------------------------------------------------------------------------


@dataclass
class CdLeverage:
    # snapshot_found = a kvbm_metrics_snapshot.json existed; present = it actually
    # carried CD content (>=1 instance with a non-null cd block). An aggregated run
    # can produce a snapshot (decode_workers only, no hub) — that is NOT CD content,
    # so section 4 stays N/A for it (design: "Agg: N/A").
    snapshot_found: bool = False
    local_decisions: int = 0  # Q1: policy-Local DECISIONS (request count, NOT tokens)
    remote_decisions: int = 0  # Q2: TRUE-remote DECISIONS (label=="remote"); excludes downgrades
    local_prefill_tokens: int = 0  # Q3: tokens the decode computes locally, net of G1+G2 prefix match
    remote_prefill_tokens: int = 0  # Q4: tokens the remote prefill computes (remote_blocks*bs)
    prefill_computed_tokens: int = 0  # Q6: prefill-worker forward-pass tokens; ≡ Q4 by construction
    prefill_pulled_tokens: int = 0  # prefix window the prefill worker pulled from decode (over-pull)
    prefill_local_hit_tokens: int = 0  # Q7: of the pulled prefix, what prefill already had cached (PNCT opp.)
    declined_by_reason: dict[str, int] = field(default_factory=dict)
    # Policy-Remote decisions that did NOT disaggregate (computed LOCALLY instead): breaker-HOT
    # coarse downgrade, B-GNMT overload downgrade, zero-block, budget-rejected. Their local
    # compute is already folded into local_prefill_tokens (Q3).
    downgrades: dict[str, int] = field(default_factory=dict)
    instances: int = 0
    raw_error: str | None = None

    @property
    def present(self) -> bool:
        """True only when the snapshot carried actual CD content (>=1 CD instance)."""
        return self.instances > 0

    @property
    def remote_fraction(self) -> float | None:
        """DECISION-based remote share by FINAL placement: true-remote requests / all CD-policy
        requests. local_decisions already folds in the downgrade-to-local labels, so the
        denominator covers every decision exactly once (downgrades are a sub-count of local)."""
        total = self.local_decisions + self.remote_decisions
        if total <= 0:
            return None
        return self.remote_decisions / total

    @property
    def remote_compute_fraction(self) -> float | None:
        """TOKEN-LOAD remote share = remote tokens / (local + remote tokens). The load-bearing
        leverage number: where the prefill FLOPs actually went, net of the G1+G2 prefix match.
        Differs from remote_fraction because requests vary wildly in prompt length and because
        breaker/overload downgrades move would-be-remote requests into the local token bucket."""
        total = self.local_prefill_tokens + self.remote_prefill_tokens
        if total <= 0:
            return None
        return self.remote_prefill_tokens / total

    @property
    def prefill_pull_supplement(self) -> int:
        """Genuine tokens the prefill worker must pull from decode after crediting its own
        cache = pulled − local-hit. Today the worker over-pulls the full window, so this is
        the PNCT savings opportunity, not yet realized."""
        return max(0, self.prefill_pulled_tokens - self.prefill_local_hit_tokens)


def parse_cd_snapshot(snapshot_path: Path | None) -> CdLeverage:
    """Aggregate the kvbm_cd_* surface from a hub /v1/metrics fanout snapshot.

    The hook writes ``{"hub_fanout": <MetricsFanoutResponse>, "decode_workers":
    {...}, "_meta": {...}}`` where the fanout is ``{"instances": {id: {...
    "cd": {...}|null}}}``. We unwrap ``hub_fanout`` (also tolerating a raw
    fanout) and aggregate prefill decisions (local/remote), token totals,
    declines, and downgrades across all instances that carry a non-null ``cd``
    block. ``present`` (a property) is True only when >=1 such CD instance is
    found — so an aggregated run's decode-workers-only snapshot stays N/A.
    Best-effort: any structural surprise leaves the section empty, not raising.
    """
    if snapshot_path is None:
        return CdLeverage(snapshot_found=False)
    try:
        data = json.loads(snapshot_path.read_text())
    except Exception as e:  # noqa: BLE001 - report stays best-effort
        return CdLeverage(snapshot_found=False, raw_error=str(e))

    lev = CdLeverage(snapshot_found=True)
    # Unwrap the hook's wrapper: prefer hub_fanout, else treat data as the fanout.
    fanout = data.get("hub_fanout") if isinstance(data, dict) else None
    if not isinstance(fanout, dict):
        fanout = data if isinstance(data, dict) else {}
    instances = fanout.get("instances")
    if not isinstance(instances, dict):
        # Tolerate a flat single-instance snapshot too.
        instances = {"_": fanout} if isinstance(fanout, dict) and "cd" in fanout else {}
    for _id, inst in instances.items():
        if not isinstance(inst, dict):
            continue
        snap = inst.get("snapshot", inst)
        cd = snap.get("cd") if isinstance(snap, dict) else None
        if not isinstance(cd, dict):
            continue
        lev.instances += 1
        decisions = cd.get("prefill_decisions", {}) or {}
        if isinstance(decisions, dict):
            for k, v in decisions.items():
                try:
                    iv = int(v)
                except (TypeError, ValueError):
                    continue
                # Classify each decision by where prefill ACTUALLY ran, so local_decisions
                # pairs with local_prefill_tokens (and remote with remote_prefill_tokens) — the
                # decode connector records local tokens on EVERY final-local path. Mapping mirrors
                # the record sites in decode_leader.rs:
                #   final-LOCAL : "local", "*downgraded*" (breaker-hot/warm, overload, zero-block),
                #                 "*rejected*" (budget) — all call record_local_prefill_tokens
                #   final-REMOTE: "remote", "*admitted*" (warm admit→remote, P3)
                # Non-{local,remote} labels are ALSO kept in `downgrades` as a diagnostic breakdown
                # (a sub-count of local_decisions), so the table can show why local got the load.
                if k == "local" or "downgraded" in k or "rejected" in k:
                    lev.local_decisions += iv
                elif k == "remote" or "admitted" in k:
                    lev.remote_decisions += iv
                if k not in ("local", "remote"):
                    lev.downgrades[k] = lev.downgrades.get(k, 0) + iv
        for k, dst in (
            ("local_prefill_tokens_total", "local_prefill_tokens"),
            ("remote_prefill_tokens_total", "remote_prefill_tokens"),
            ("prefill_computed_tokens_total", "prefill_computed_tokens"),
            ("prefill_pulled_tokens_total", "prefill_pulled_tokens"),
            ("prefill_local_hit_tokens_total", "prefill_local_hit_tokens"),
        ):
            v = cd.get(k)
            if v is not None:
                with contextlib.suppress(TypeError, ValueError):
                    setattr(lev, dst, getattr(lev, dst) + int(v))
        # Declines: the live schema is "remote_prefill_declined" (older/test snapshots: "declined").
        declined = cd.get("remote_prefill_declined", {}) or cd.get("declined", {}) or {}
        if isinstance(declined, dict):
            for k, v in declined.items():
                with contextlib.suppress(TypeError, ValueError):
                    lev.declined_by_reason[k] = lev.declined_by_reason.get(k, 0) + int(v)
        # A dedicated downgrades/breaker_downgrades map, if a future snapshot emits one, folds in too.
        downgrades = cd.get("downgrades", {}) or cd.get("breaker_downgrades", {}) or {}
        if isinstance(downgrades, dict):
            for k, v in downgrades.items():
                with contextlib.suppress(TypeError, ValueError):
                    lev.downgrades[k] = lev.downgrades.get(k, 0) + int(v)
    return lev


# ---------------------------------------------------------------------------
# Per-run aggregate
# ---------------------------------------------------------------------------


@dataclass
class RunReport:
    job_id: str
    run_dir: Path
    job_name: str
    is_baseline_candidate: bool  # no kvbm_hub block => agg baseline
    dataset: str | None
    concurrency: int | None
    # topology
    topology_label: str
    tp: int | None
    agg_workers: int
    gpus_per_agg: int
    prefill_nodes: int
    prefill_tp: int
    active_gpus: int
    provisioned_nodes: int | None
    provisioned_gpus: int | None
    image: str | None
    condp_policy: int | None  # CD ThresholdRemote = backend.kvbm_hub.min_remote_prefill_tokens (N/A for agg)
    max_num_tokens: int | None  # max-model-len (closest vLLM analog to the user-schema column)
    # aiperf headline
    request_throughput: float | None
    request_count: int | None
    ttft_p50: float | None
    ttft_p90: float | None
    ttft_p99: float | None
    itl_p50: float | None
    itl_p90: float | None
    itl_p99: float | None
    output_tput: float | None
    output_tput_per_user: float | None
    total_tput: float | None
    input_tput: float | None
    isl_avg: float | None
    osl_avg: float | None
    effective_concurrency: float | None
    theoretical_prefix_cache_hit: float | None
    was_cancelled: bool
    error_count: int
    runtime_error: str  # compact "{type}×{count}" summary of aiperf error_summary buckets ("" = clean)
    # native goodput (recovered from cli_command + aiperf good_request_count)
    native_sla_ttft: float | None
    native_sla_itl: float | None
    native_good_request_count: int | None
    native_goodput_fraction: float | None
    # recomputed goodput
    native_goodput_recompute: GoodputResult
    user_goodput: GoodputResult
    # KV
    kv: KvMetrics
    # CD leverage
    cd: CdLeverage

    @property
    def output_tput_per_gpu(self) -> float | None:
        if self.output_tput is None or self.active_gpus <= 0:
            return None
        return self.output_tput / self.active_gpus

    @property
    def total_tput_per_gpu(self) -> float | None:
        if self.total_tput is None or self.active_gpus <= 0:
            return None
        return self.total_tput / self.active_gpus

    @property
    def error_pct(self) -> float | None:
        if not self.request_count:
            return None
        return 100.0 * self.error_count / self.request_count

    @property
    def group_key(self) -> tuple[str, int]:
        return (self.dataset or "unknown", self.concurrency or -1)


def summarize_errors(error_summary: Any) -> tuple[int, str]:
    """Failed-request COUNT (sum of per-bucket counts, NOT len) + a compact
    ``{type}×{count}`` string, from aiperf's ``error_summary``.

    aiperf groups failures into BUCKETS: ``[{error_details:{type,code,message}, count}]``.
    Counting ``len()`` counts distinct error CATEGORIES and ~halves the true error
    rate when several requests share a type — so we sum the per-bucket ``count``.
    """
    if not isinstance(error_summary, list):
        return 0, ""
    count = sum(int(e.get("count", 0)) for e in error_summary if isinstance(e, dict))
    parts: list[str] = []
    for e in error_summary:
        if not isinstance(e, dict):
            continue
        ed = e.get("error_details", {}) if isinstance(e.get("error_details"), dict) else {}
        etype = ed.get("type") or (f"HTTP {ed.get('code')}" if ed.get("code") else "error")
        parts.append(f"{etype}×{int(e.get('count', 1))}")
    return count, "; ".join(parts)


def _topology_label(cfg_resources: Any, is_baseline: bool) -> tuple[str, int | None]:
    """Synthesize a compact topology label + TP from ResourceConfig.

    ``RunMetadata.topology_label`` named in the design does not exist in the
    tree; we build the label here from the live ResourceConfig.
    """
    agg = cfg_resources.num_agg
    tp = cfg_resources.gpus_per_agg if agg else cfg_resources.gpus_per_decode
    pn = cfg_resources.kvbm_prefill_nodes or 0
    # Native (trad) P/D disagg: no agg plane, distinct prefill+decode planes (num_agg==0,
    # num_prefill>0, num_decode>0). MUST be checked FIRST — these runs are is_baseline (no
    # kvbm_hub) with pn==0, so the agg branch below would otherwise mislabel them "agg-0xTEP2".
    if agg == 0 and cfg_resources.num_prefill > 0 and cfg_resources.num_decode > 0:
        return f"pd-{cfg_resources.num_prefill}p{cfg_resources.num_decode}d TEP{tp}", tp
    if is_baseline or pn == 0:
        return f"agg-{agg}xTEP{tp}", tp
    return f"cd-{agg}d{pn}p TEP{tp}", tp


def build_run_report(
    run_dir: Path,
    ttft_sla_ms: float,
    itl_sla_ms: float,
) -> RunReport:
    """Build the full per-run aggregate from one run-output dir (offline)."""
    from srtctl.core.config import load_config

    job_id = run_dir.name
    cfg = load_config(run_dir / "config.yaml")
    r = cfg.resources
    hub = getattr(cfg.backend, "kvbm_hub", None)
    is_baseline = not bool(hub)

    job_name = getattr(cfg, "name", None) or job_id
    image = getattr(cfg.model, "container", None) if getattr(cfg, "model", None) else None

    topo_label, tp = _topology_label(r, is_baseline)
    prefill_nodes = r.kvbm_prefill_nodes or 0
    prefill_tp = r.kvbm_prefill_tp or 0
    condp_policy = getattr(hub, "min_remote_prefill_tokens", None) if hub else None
    _vc = getattr(cfg.backend, "vllm_config", None)
    _agg = getattr(_vc, "aggregated", None) if _vc else None
    max_num_tokens = _agg.get("max-model-len") if isinstance(_agg, dict) else None
    # ACTIVE GPUs across the serving plane (NOT provisioned nodes).
    active_gpus = _compute_active_gpus(r)
    try:
        provisioned_nodes = r.total_nodes
    except Exception:  # noqa: BLE001
        provisioned_nodes = None
    provisioned_gpus = provisioned_nodes * r.gpus_per_node if provisioned_nodes else None

    art = find_real_artifact_dir(run_dir)
    aiperf: dict[str, Any] = {}
    cli_command = None
    dataset = None
    concurrency = None
    if art is not None:
        aiperf_path = art / "profile_export_aiperf.json"
        if aiperf_path.is_file():
            try:
                aiperf = json.loads(aiperf_path.read_text())
            except json.JSONDecodeError:
                aiperf = {}
        ic = aiperf.get("input_config", {}) or {}
        cli_command = ic.get("cli_command")
        dataset = parse_cli_field(cli_command, "public-dataset")
        c = parse_cli_field(cli_command, "concurrency")
        if c and c.isdigit():
            concurrency = int(c)
    # Fall back to recipe benchmark config for dataset/concurrency.
    if dataset is None:
        dataset = getattr(cfg.benchmark, "public_dataset", None)
    if concurrency is None:
        conc = getattr(cfg.benchmark, "concurrencies", None)
        if conc:
            first = str(conc).split(",")[0].strip()
            if first.isdigit():
                concurrency = int(first)

    native_ttft, native_itl = parse_cli_goodput_sla(cli_command)

    def m(name: str) -> dict[str, Any] | None:
        v = aiperf.get(name)
        return v if isinstance(v, dict) else None

    request_count = None
    rc = _avg(m("request_count"))
    if rc is not None:
        request_count = int(rc)
    native_good = None
    grc = _avg(m("good_request_count"))
    if grc is not None:
        native_good = int(grc)
    native_goodput_fraction = None
    if native_good is not None and request_count:
        native_goodput_fraction = native_good / request_count

    jsonl = art / "profile_export.jsonl" if art is not None else Path("/nonexistent")
    # Native-SLA recompute (over the SAME denominator) — should match good_request_count.
    if native_ttft is not None and native_itl is not None:
        native_recompute = recompute_goodput(jsonl, native_ttft, native_itl)
    else:
        native_recompute = recompute_goodput(jsonl, ttft_sla_ms, itl_sla_ms)
    user_goodput = recompute_goodput(jsonl, ttft_sla_ms, itl_sla_ms)

    server_metrics = art / "server_metrics_export.jsonl" if art is not None else Path("/nonexistent")
    kv = parse_server_metrics(server_metrics)

    snapshot = find_snapshot(run_dir, art)
    cd = parse_cd_snapshot(snapshot)

    error_count, runtime_error = summarize_errors(aiperf.get("error_summary"))

    return RunReport(
        job_id=job_id,
        run_dir=run_dir,
        job_name=job_name,
        is_baseline_candidate=is_baseline,
        dataset=dataset,
        concurrency=concurrency,
        topology_label=topo_label,
        tp=tp,
        agg_workers=r.num_agg,
        gpus_per_agg=r.gpus_per_agg,
        prefill_nodes=prefill_nodes,
        prefill_tp=prefill_tp,
        active_gpus=active_gpus,
        provisioned_nodes=provisioned_nodes,
        provisioned_gpus=provisioned_gpus,
        image=image,
        condp_policy=condp_policy,
        max_num_tokens=max_num_tokens,
        request_throughput=_avg(m("request_throughput")),
        request_count=request_count,
        ttft_p50=_pct(m("time_to_first_token"), "p50"),
        ttft_p90=_pct(m("time_to_first_token"), "p90"),
        ttft_p99=_pct(m("time_to_first_token"), "p99"),
        itl_p50=_pct(m("inter_token_latency"), "p50"),
        itl_p90=_pct(m("inter_token_latency"), "p90"),
        itl_p99=_pct(m("inter_token_latency"), "p99"),
        output_tput=_avg(m("output_token_throughput")),
        # MEDIAN (p50), not avg: aiperf's per-user avg is skewed high by a heavy tail of
        # fast short requests (e.g. agg-c64 avg=187 vs p50=43), which would misrepresent the
        # typical user's output rate. The p50 matches 1/ITL_p50 and is the honest comparator.
        output_tput_per_user=_pct(m("output_token_throughput_per_user"), "p50"),
        total_tput=_avg(m("total_token_throughput")),
        input_tput=_avg(m("input_token_throughput")),
        isl_avg=_avg(m("input_sequence_length")),
        osl_avg=_avg(m("output_sequence_length")),
        effective_concurrency=_avg(m("effective_concurrency")),
        theoretical_prefix_cache_hit=_avg(m("theoretical_prefix_cache_hit")),
        was_cancelled=bool(aiperf.get("was_cancelled")),
        error_count=error_count,
        runtime_error=runtime_error,
        native_sla_ttft=native_ttft,
        native_sla_itl=native_itl,
        native_good_request_count=native_good,
        native_goodput_fraction=native_goodput_fraction,
        native_goodput_recompute=native_recompute,
        user_goodput=user_goodput,
        kv=kv,
        cd=cd,
    )


# ---------------------------------------------------------------------------
# Curated manifest (the durable "latest results" source of truth)
# ---------------------------------------------------------------------------

# The three run kinds a manifest entry may declare.
KIND_BASELINE = "baseline"  # the delta baseline (exactly one expected)
KIND_BASELINE_REFERENCE = "baseline_reference"  # shown but marked NOT the baseline
KIND_CD = "cd"  # a conditional-disaggregation run (flagged as CD)


@dataclass
class ManifestRun:
    """One curated run entry from the manifest (label/kind/tpcb override config)."""

    job_id: str
    label: str
    kind: str
    tpcb: bool
    note: str

    @property
    def is_baseline(self) -> bool:
        return self.kind == KIND_BASELINE

    @property
    def is_reference(self) -> bool:
        return self.kind == KIND_BASELINE_REFERENCE

    @property
    def is_cd(self) -> bool:
        return self.kind == KIND_CD


@dataclass
class ManifestChangelogEntry:
    date: str
    note: str


@dataclass
class Manifest:
    """The parsed curated manifest. ``runs`` is ORDERED (report column order)."""

    title: str
    dataset: str | None
    concurrency: int | None
    generated_note: str
    changelog: list[ManifestChangelogEntry]
    runs: list[ManifestRun]
    # Optional parallel axis: a conditional-threshold sweep over ONE topology (5d1p),
    # rendered in its own section (NOT mixed into the topology comparison). Ordered job ids
    # (the threshold is read from each run's condp_policy); may overlap `runs` (the anchor).
    sweep_job_ids: list[str] = field(default_factory=list)
    sweep_note: str = ""
    # Optional parallel axis: traditional (native dynamo) P/D disaggregation runs, a distinct
    # serving mode rendered in its own section. Carry explicit labels (the auto topology_label
    # mislabels native P/D as "agg-0xTEP2"). Ordered ManifestRuns (job_id + label).
    pd_disagg_runs: list[ManifestRun] = field(default_factory=list)
    pd_disagg_note: str = ""
    # Optional parallel axis: the Pareto frontier (output tok/s/user p50 vs output tok/s/GPU),
    # one line per serving mode across a concurrency sweep, rendered as a PNG. Ordered job ids
    # (may overlap `runs`); the line key is each run's topology_label.
    pareto_job_ids: list[str] = field(default_factory=list)
    pareto_note: str = ""
    source_path: Path | None = None

    def baseline_run(self) -> ManifestRun | None:
        """The kind=baseline entry (the delta baseline). First match wins."""
        for run in self.runs:
            if run.is_baseline:
                return run
        return None


def parse_manifest(path: Path) -> Manifest:
    """Parse a curated ``latest_results.yaml`` manifest.

    PyYAML is already a srtctl dependency; we import it lazily so the
    ``--runs`` path keeps working without it. Job ids parse as ints from YAML —
    we ``str()`` them so they feed straight into :func:`resolve_run_dir`.
    """
    import yaml  # lazy: keep the --runs path dependency-light

    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"manifest {path} is not a mapping")

    conc = data.get("concurrency")
    with contextlib.suppress(TypeError, ValueError):
        conc = int(conc) if conc is not None else None

    changelog: list[ManifestChangelogEntry] = []
    for entry in data.get("changelog") or []:
        if isinstance(entry, dict):
            changelog.append(
                ManifestChangelogEntry(
                    date=str(entry.get("date", "")),
                    note=str(entry.get("note", "")),
                )
            )

    runs: list[ManifestRun] = []
    for entry in data.get("runs") or []:
        if not isinstance(entry, dict) or entry.get("job_id") is None:
            continue
        runs.append(
            ManifestRun(
                job_id=str(entry.get("job_id")),
                label=str(entry.get("label", entry.get("job_id"))),
                kind=str(entry.get("kind", KIND_CD)),
                tpcb=bool(entry.get("tpcb", False)),
                note=str(entry.get("note", "")),
            )
        )

    # Optional threshold sweep: `threshold_sweep: {note?, runs: [{job_id}, ...]}`
    # (also tolerates a bare list of {job_id}). Ordered; threshold comes from condp_policy.
    sweep_job_ids: list[str] = []
    sweep_note = ""
    sweep_block = data.get("threshold_sweep")
    if isinstance(sweep_block, dict):
        sweep_note = str(sweep_block.get("note", ""))
        sweep_entries = sweep_block.get("runs") or []
    elif isinstance(sweep_block, list):
        sweep_entries = sweep_block
    else:
        sweep_entries = []
    for entry in sweep_entries:
        if isinstance(entry, dict) and entry.get("job_id") is not None:
            sweep_job_ids.append(str(entry.get("job_id")))
        elif isinstance(entry, (str, int)):
            sweep_job_ids.append(str(entry))

    # Optional trad P/D disagg: `pd_disagg: {note?, runs: [{job_id, label, note?}, ...]}`.
    pd_disagg_runs: list[ManifestRun] = []
    pd_disagg_note = ""
    pd_block = data.get("pd_disagg")
    if isinstance(pd_block, dict):
        pd_disagg_note = str(pd_block.get("note", ""))
        for entry in pd_block.get("runs") or []:
            if isinstance(entry, dict) and entry.get("job_id") is not None:
                pd_disagg_runs.append(
                    ManifestRun(
                        job_id=str(entry.get("job_id")),
                        label=str(entry.get("label", entry.get("job_id"))),
                        kind=str(entry.get("kind", KIND_CD)),
                        tpcb=bool(entry.get("tpcb", False)),
                        note=str(entry.get("note", "")),
                    )
                )

    # Optional Pareto axis: `pareto: {note?, runs: [{job_id}, ...]}` (also tolerates a bare list
    # of {job_id}). Ordered; the per-mode line key is each run's topology_label. Mirrors threshold_sweep.
    pareto_job_ids: list[str] = []
    pareto_note = ""
    pareto_block = data.get("pareto")
    if isinstance(pareto_block, dict):
        pareto_note = str(pareto_block.get("note", ""))
        pareto_entries = pareto_block.get("runs") or []
    elif isinstance(pareto_block, list):
        pareto_entries = pareto_block
    else:
        pareto_entries = []
    for entry in pareto_entries:
        if isinstance(entry, dict) and entry.get("job_id") is not None:
            pareto_job_ids.append(str(entry.get("job_id")))
        elif isinstance(entry, (str, int)):
            pareto_job_ids.append(str(entry))

    return Manifest(
        title=str(data.get("title", "KVBM benchmark report")),
        dataset=(str(data["dataset"]) if data.get("dataset") is not None else None),
        concurrency=conc,
        generated_note=str(data.get("generated_note", "")),
        changelog=changelog,
        runs=runs,
        sweep_job_ids=sweep_job_ids,
        sweep_note=sweep_note,
        pd_disagg_runs=pd_disagg_runs,
        pd_disagg_note=pd_disagg_note,
        pareto_job_ids=pareto_job_ids,
        pareto_note=pareto_note,
        source_path=path,
    )


# ---------------------------------------------------------------------------
# Grouping + baseline selection
# ---------------------------------------------------------------------------


def group_runs(runs: list[RunReport]) -> dict[tuple[str, int], list[RunReport]]:
    groups: dict[tuple[str, int], list[RunReport]] = {}
    for run in runs:
        groups.setdefault(run.group_key, []).append(run)
    return groups


def pick_baseline(group: list[RunReport], explicit_baseline: str | None) -> RunReport | None:
    """Baseline = explicit job id if given+present, else the no-kvbm_hub run."""
    if explicit_baseline:
        for run in group:
            if run.job_id == explicit_baseline:
                return run
    for run in group:
        if run.is_baseline_candidate:
            return run
    return None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def fmt(v: Any, prec: int = 2, suffix: str = "") -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{prec}f}{suffix}"
    return f"{v}{suffix}"


def delta_pct(cur: float | None, base: float | None) -> str:
    """Signed (cur-base)/base*100, formatted; N/A when not computable."""
    if cur is None or base is None or base == 0:
        return "N/A"
    d = (cur - base) / base * 100.0
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.1f}%"


# ---------------------------------------------------------------------------
# Threshold sweep (parallel axis — its own section, not the topology comparison)
# ---------------------------------------------------------------------------

# The interpretive thesis of the sweep, kept in one place so the md + HTML agree.
_SWEEP_THESIS = (
    "**Conclusion: in this workload the conditional threshold is a weak throughput lever but a "
    "useful TTFT / prefill-pressure knob — and it tells us the small prefills don't matter, the "
    "large ones do.** Raising `min_remote_prefill_tokens` from 1024 to 8192 cuts remote *decisions* "
    "~3x (fewer requests clear the bar to disaggregate), yet the token-compute remote share stays "
    "flat (~56%): the requests we stop disaggregating carry almost no prefill FLOPs, so the prefill "
    "aside does ~the same total work either way. In other words the small-remainder prefills (uncached "
    "remainder in [1024, 8192)) are numerous but FLOP-negligible; the prefill load is dominated by the "
    "few large prefills (remainder > 8192), which disaggregate at every threshold. Net effect of a "
    "higher threshold: throughput ~flat (within run-noise — the server is not saturated at these "
    "concurrencies), TTFT p50 modestly better (~8-15%, less prefill-aside queueing), and the circuit "
    "breaker fires far less (it stops policing the cheap small disaggregations). The over-pull stays "
    "~50-64% across the whole range, so the PNCT opportunity is threshold-independent."
)


def _sweep_by_concurrency(sweep_runs: list[RunReport]) -> dict[int, list[RunReport]]:
    """Group sweep runs by declared concurrency, each list sorted by threshold (condp_policy)."""
    by_conc: dict[int, list[RunReport]] = {}
    for r in sweep_runs:
        by_conc.setdefault(r.concurrency or -1, []).append(r)
    for runs in by_conc.values():
        runs.sort(key=lambda r: (r.condp_policy if r.condp_policy is not None else 0))
    return by_conc


def _sweep_overpull(cd: CdLeverage) -> float | None:
    return cd.prefill_local_hit_tokens / cd.prefill_pulled_tokens if cd.prefill_pulled_tokens else None


def _sweep_breaker_hot(cd: CdLeverage) -> int:
    return sum(v for k, v in cd.downgrades.items() if "breaker_hot" in k)


def _pct0(frac: float | None) -> str:
    """Format a [0,1] fraction as a whole-percent string (N/A when None)."""
    return f"{frac * 100:.0f}%" if frac is not None else "N/A"


def render_threshold_sweep(sweep_runs: list[RunReport], note: str = "") -> str:
    """Section 7: the conditional-threshold sweep, one table per concurrency.

    Columns are the per-threshold rows (req/s with delta vs the lowest threshold, TTFT/ITL
    p50+p90, and the CD-leverage that actually moves: remote decisions, token-compute remote%,
    prefill over-pull%, breaker-HOT count)."""
    lines: list[str] = []
    lines.append("## 7. Conditional-threshold sweep")
    lines.append("")
    lines.append(_SWEEP_THESIS)
    lines.append("")
    if note:
        lines.append(f"> {note}")
        lines.append("")
    by_conc = _sweep_by_concurrency(sweep_runs)
    for conc in sorted(by_conc):
        runs = by_conc[conc]
        topo = runs[0].topology_label if runs else "?"
        lines.append(f"### {topo} @ concurrency {conc}")
        lines.append("")
        lines.append(
            "| threshold | job | req/s (Δ vs min) | TTFT p50 | TTFT p90 | ITL p50 | ITL p90 | "
            "remote dec | token rem% | over-pull% | breaker HOT |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        base_req = next((r.request_throughput for r in runs if r.request_throughput), None)
        for r in runs:
            cd = r.cd
            req = r.request_throughput
            dpart = ""
            if req is not None and base_req:
                dpart = f" ({(req - base_req) / base_req * 100:+.1f}%)"
            lines.append(
                f"| {r.condp_policy} | {r.job_id} | {fmt(req, 3)}{dpart} | {fmt(r.ttft_p50, 0)} | "
                f"{fmt(r.ttft_p90, 0)} | {fmt(r.itl_p50, 1)} | {fmt(r.itl_p90, 1)} | {cd.remote_decisions} | "
                f"{_pct0(cd.remote_compute_fraction)} | {_pct0(_sweep_overpull(cd))} | {_sweep_breaker_hot(cd)} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Traditional (native dynamo) P/D disaggregation (distinct serving mode — its own section)
# ---------------------------------------------------------------------------

_PD_DISAGG_THESIS = (
    "**Traditional (native dynamo) prefill/decode disaggregation — a 3rd serving mode alongside "
    "aggregated and CD.** Configured for KV-aware P-first routing (`router-mode: kv`, requests sized "
    "on a prefill worker first); the prefill workers run KVBM + the in-process consolidator (G2 "
    "160GB/rank, same as agg/CD) and hand KV to plain-vLLM decode workers over NIXL. Sweeping the "
    "prefill:decode split at fixed total GPUs (6 nodes × 2-GPU TEP2). `kv_reused_blk` > 0 proves the "
    "KVBM-on-prefill consolidator engaged (this prefill-disagg+KVBM shape had no prior live precedent "
    "here); the prefill-side local prefix-hit rates are comparable to the KV-routed CD runs, "
    "consistent with KV-routing working — though the router's runtime overlap-scoring was not directly "
    "confirmed. READING THE NUMBERS: at this (unsaturated) c48 load the throughput differences between "
    "the best split and agg / CD-3d3p in §1-2 are within run-noise — do NOT over-read a few-percent "
    "edge. The above-noise signals are (1) the whole native-P/D family sits ~9% BELOW CD-5d1p (1.94 "
    "req/s) — the CD prefill-aside wins in this huge-prefix (~85k ISL) workload — and (2) the TTFT-p50 "
    "spread across the split (4p2d ~886ms prefill-rich but decode-bound → 2p4d ~6556ms prefill-starved) "
    "is the real prefill:decode tradeoff. Among the splits the balanced 3p3d is the throughput sweet spot."
)


def _pd_disagg_valid(rr: RunReport) -> bool:
    """A trad-P/D run is valid iff it actually served traffic (req/s>0, requests>0, no all-error)."""
    return bool(rr.request_throughput) and bool(rr.request_count) and (rr.error_count or 0) < (rr.request_count or 0)


def render_pd_disagg(paired: list[tuple[ManifestRun, RunReport]], note: str = "") -> str:
    """Section 8: traditional P/D disaggregation runs (one row per prefill:decode split).

    Uses the manifest LABEL (the auto topology_label mislabels native P/D as 'agg-0xTEP2').
    Invalid runs (all requests errored) are shown but explicitly flagged, never silently dropped."""
    lines: list[str] = []
    lines.append("## 8. Traditional P/D disaggregation")
    lines.append("")
    lines.append(_PD_DISAGG_THESIS)
    lines.append("")
    if note:
        lines.append(f"> {note}")
        lines.append("")
    lines.append(
        "| config | conc | job | req/s | out tok/s | TTFT p50 | TTFT p90 | ITL p50 | ITL p90 | reqs | kv_reused_blk | status |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    # order by (concurrency, req/s desc) so each concurrency block is grouped, best on top
    ordered = sorted(paired, key=lambda pr: (pr[1].concurrency or 0, -(pr[1].request_throughput or 0.0)))
    for mr, rr in ordered:
        if _pd_disagg_valid(rr):
            status = "valid"
        else:
            why = f"all requests errored ({rr.runtime_error})" if rr.runtime_error else "no served traffic"
            status = f"**INVALID — {why}**"
        lines.append(
            f"| {mr.label} | {rr.concurrency if rr.concurrency is not None else 'N/A'} | {rr.job_id} | "
            f"{fmt(rr.request_throughput, 3)} | {fmt(rr.output_tput, 0)} | "
            f"{fmt(rr.ttft_p50, 0)} | {fmt(rr.ttft_p90, 0)} | {fmt(rr.itl_p50, 1)} | {fmt(rr.itl_p90, 1)} | "
            f"{rr.request_count if rr.request_count is not None else 'N/A'} | {fmt(rr.kv.kv_reused_blocks, 0)} | {status} |"
        )
    lines.append("")
    # Best valid split PER concurrency (req/s is only comparable within a concurrency).
    valid_pairs = [(mr, rr) for mr, rr in paired if _pd_disagg_valid(rr)]
    by_conc: dict[int, list[tuple[ManifestRun, RunReport]]] = {}
    for mr, rr in valid_pairs:
        by_conc.setdefault(rr.concurrency or -1, []).append((mr, rr))
    for conc in sorted(by_conc):
        mr, rr = max(by_conc[conc], key=lambda pr: pr[1].request_throughput or 0.0)
        lines.append(
            f"- **Best valid trad-P/D @ c{conc}: {mr.label}** ({fmt(rr.request_throughput, 3)} req/s, "
            f"{fmt(rr.output_tput, 0)} out tok/s, TTFT p50 {fmt(rr.ttft_p50, 0)}ms)."
        )
    if valid_pairs:
        lines.append(f"- {len(valid_pairs)}/{len(paired)} configs valid.")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Pareto frontier (output tok/s/user p50 vs output tok/s/GPU, one line per mode)
# ---------------------------------------------------------------------------


def _pareto_series(runs: list[RunReport]) -> dict[str, list[tuple[float, float, int | None, float | None]]]:
    """Group runs by topology_label (the serving-mode line key) into plottable point series.

    Each point is ``(x, y, concurrency, ttft_p50)`` where x = output tok/s/user (p50) and
    y = output tok/s/GPU. Runs missing x or y are dropped (can't be plotted). Each series is
    sorted by concurrency so the line traces the concurrency sweep. Matplotlib-free so it is
    unit-testable without rendering."""
    series: dict[str, list[tuple[float, float, int | None, float | None]]] = {}
    for r in runs:
        x = r.output_tput_per_user
        y = r.output_tput_per_gpu
        if x is None or y is None:
            continue
        series.setdefault(r.topology_label, []).append((x, y, r.concurrency, r.ttft_p50))
    for pts in series.values():
        pts.sort(key=lambda p: p[2] if p[2] is not None else -1)
    return series


def render_pareto_png(pareto_runs: list[RunReport], out_path: Path, title: str) -> bool:
    """Render a Pareto-frontier PNG: x = output tok/s/user (p50), y = output tok/s/GPU.

    One line+marker series per serving mode (``topology_label`` — correct after the native-disagg
    label fix), in sorted order for stable colors from the default cycle, with each series sorted by
    concurrency. Every point is annotated ``c{concurrency}, {ttft_p50:.0f}ms``. Returns True iff at
    least one point was plotted (so the caller can gate the markdown/HTML embeds)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = _pareto_series(pareto_runs)
    if not series:
        return False  # nothing plottable — skip writing an empty PNG the caller would not embed
    fig, ax = plt.subplots(figsize=(9, 6))
    plotted = 0
    for label in sorted(series):
        pts = series[label]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", label=label)
        for x, y, conc, ttft in pts:
            ann = f"c{conc}"
            if ttft is not None:
                ann += f", {ttft:.0f}ms"
            ax.annotate(ann, (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
        plotted += len(pts)

    ax.set_xlabel("output tok/s/user (p50)")
    ax.set_ylabel("output tok/s/GPU")
    ax.set_title(title)
    ax.grid(True)
    if plotted:
        ax.legend()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return plotted >= 1


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _winner(runs: list[RunReport], key, higher_is_better: bool) -> str:
    vals = [(run, key(run)) for run in runs if key(run) is not None]
    if not vals:
        return "N/A"
    best = max(vals, key=lambda x: x[1]) if higher_is_better else min(vals, key=lambda x: x[1])
    return best[0].topology_label


def render_markdown(
    runs: list[RunReport],
    groups: dict[tuple[str, int], list[RunReport]],
    explicit_baseline: str | None,
    ttft_sla_ms: float,
    itl_sla_ms: float,
) -> str:
    lines: list[str] = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append("# KVBM agg-vs-CD benchmark report")
    lines.append("")
    lines.append(f"_Generated {ts} · {len(runs)} run(s) · user SLA TTFT<{ttft_sla_ms:.0f}ms ITL<{itl_sla_ms:g}ms_")
    lines.append("")

    # ---- 0. Run-set header + validity guards ----
    lines.append("## 0. Run set + validity guards")
    lines.append("")
    multi_dataset = len({k[0] for k in groups}) > 1
    lines.append(
        "> GUARD: deltas are computed ONLY within a (dataset, concurrency) group. "
        "Cross-dataset pairings are shown side-by-side and labelled NOT COMPARABLE."
    )
    if multi_dataset:
        lines.append(">")
        lines.append("> WARNING: this run set spans MULTIPLE datasets — comparisons across groups are NOT COMPARABLE.")
    lines.append("")
    # No status / no node-count columns: a run with BLOCKING errors is invalid and must not
    # be in the set at all; non-blocking errors don't affect the outcome, so we don't report
    # them. Topology counts (decode/prefill) are in the per-run detail; the header shows only
    # the active-GPU count (the comparison denominator), not provisioned nodes.
    lines.append("| job | name | topology | dataset | conc | active GPUs | image |")
    lines.append("|---|---|---|---|---|---|---|")
    for run in runs:
        img = Path(run.image).name if run.image else "N/A"
        lines.append(
            f"| {run.job_id} | {run.job_name} | {run.topology_label} | {run.dataset} | "
            f"{run.concurrency} | {run.active_gpus} | {img} |"
        )
    lines.append("")
    canonical = max(groups.items(), key=lambda kv: len(kv[1]))[0] if groups else None
    if canonical:
        lines.append(
            f"**Canonical comparison group:** dataset=`{canonical[0]}` @ concurrency={canonical[1]} "
            f"({len(groups[canonical])} runs)."
        )
    lines.append("")

    # ---- 1. Headline punchline per group ----
    lines.append("## 1. Headline (per (dataset, concurrency) group)")
    lines.append("")
    for gkey, group in groups.items():
        baseline = pick_baseline(group, explicit_baseline)
        lines.append(f"### {gkey[0]} @ c{gkey[1]}")
        lines.append("")
        if baseline is None:
            lines.append("_No baseline (no no-kvbm_hub run in this group) — deltas omitted._")
        else:
            lines.append(f"_Baseline = `{baseline.job_id}` ({baseline.topology_label})._")
        lines.append("")
        lines.append(
            "| metric | " + " | ".join(f"{run.topology_label}<br>`{run.job_id}`" for run in group) + " | winner |"
        )
        lines.append("|---" * (len(group) + 2) + "|")

        def row(label: str, key, prec, suffix, higher_better, _group=group, _baseline=baseline, winner_override=None):
            cells = []
            base_v = key(_baseline) if _baseline else None
            for run in _group:
                v = key(run)
                cell = fmt(v, prec, suffix)
                if _baseline is not None and run is not _baseline:
                    cell += f" ({delta_pct(v, base_v)})"
                cells.append(cell)
            win = winner_override if winner_override is not None else _winner(_group, key, higher_better)
            lines.append(f"| {label} | " + " | ".join(cells) + f" | {win} |")

        # Goodput is only a valid winner axis when discriminatory (ITL SLA above
        # the median ITL); else the winner cell is "—" so we never crown a topology
        # on a near-zero, load-skewed goodput artifact. Checked per SLA flavour.
        native_disc = all(r.native_goodput_recompute.discriminatory for r in group)
        user_disc = all(r.user_goodput.discriminatory for r in group)

        row("req/s", lambda r: r.request_throughput, 3, "", True)
        row("total tok/s", lambda r: r.total_tput, 0, "", True)
        row("total tok/s/GPU", lambda r: r.total_tput_per_gpu, 0, "", True)
        row("output tok/s", lambda r: r.output_tput, 1, "", True)
        row("output tok/s/GPU", lambda r: r.output_tput_per_gpu, 1, "", True)
        row("output tok/s/user (p50)", lambda r: r.output_tput_per_user, 1, "", True)
        row("TTFT p50 (ms)", lambda r: r.ttft_p50, 0, "", False)
        row("TTFT p90 (ms)", lambda r: r.ttft_p90, 0, "", False)
        row("TTFT p99 (ms)", lambda r: r.ttft_p99, 0, "", False)
        row("ITL p50 (ms)", lambda r: r.itl_p50, 1, "", False)
        row("ITL p90 (ms)", lambda r: r.itl_p90, 1, "", False)
        row("ITL p99 (ms)", lambda r: r.itl_p99, 1, "", False)
        # BOTH goodput rows over the SAME denominator (request count incl. errors):
        # native = recompute at the native SLA; user = recompute at --ttft/--itl.
        row(
            "native goodput",
            lambda r: r.native_goodput_recompute.fraction,
            3,
            "",
            True,
            winner_override=("non-discriminatory" if not native_disc else None),
        )
        row(
            "user goodput",
            lambda r: r.user_goodput.fraction,
            3,
            "",
            True,
            winner_override=("non-discriminatory" if not user_disc else None),
        )
        # No error row: runs in the set are valid (completed); non-blocking errors (e.g.
        # context-length 400s, aiperf client ValueErrors) don't affect the outcome.
        lines.append("")

        # Findings + the load-bearing goodput caveat (covers BOTH SLA flavours).
        if not user_disc or not native_disc:
            med = ", ".join(f"{run.topology_label}={fmt(run.user_goodput.median_itl_ms, 1)}ms" for run in group)
            flavours = []
            if not user_disc:
                flavours.append(f"user ITL SLA {itl_sla_ms:g}ms")
            if not native_disc:
                nat = next((r.native_sla_itl for r in group if r.native_sla_itl is not None), None)
                flavours.append(f"native ITL SLA {nat:g}ms" if nat is not None else "native ITL SLA")
            lines.append(
                f"- **Goodput is NON-DISCRIMINATORY** here ({' AND '.join(flavours)} at/under the median ITL "
                f"[{med}]) — goodput collapses toward zero, so the goodput rows are marked non-discriminatory and "
                f"name NO winner. A higher goodput fraction here mostly reflects LOWER offered load (fewer "
                f"completed requests in the denominator), NOT a CD win/regression — do not read it as one. "
                f"Relax the ITL SLA above the median to make goodput discriminatory."
            )
        out_win = _winner(group, lambda r: r.output_tput_per_gpu, True)
        lines.append(
            f"- Output-tput/GPU is the load-bearing throughput axis (total-tput is input-dominated at "
            f"~{fmt(group[0].isl_avg, 0)} ISL): leader = **{out_win}**."
        )
        lines.append("")

    # ---- 2. Full comparison table ----
    lines.append("## 2. Full comparison table")
    lines.append("")
    lines.append(
        "This markdown table is a CURATED subset; the sibling `.csv` carries the FULL column "
        "schema (one row per run, every metric/derived column)."
    )
    lines.append("")
    cols = [
        ("job", lambda r: r.job_id),
        ("topology", lambda r: r.topology_label),
        ("active_gpus", lambda r: r.active_gpus),
        ("req/s", lambda r: fmt(r.request_throughput, 3)),
        ("req_count", lambda r: r.request_count),
        ("total_tok/s", lambda r: fmt(r.total_tput, 0)),
        ("out_tok/s", lambda r: fmt(r.output_tput, 0)),
        ("out_tok/s/GPU", lambda r: fmt(r.output_tput_per_gpu, 1)),
        ("TTFT_p50", lambda r: fmt(r.ttft_p50, 0)),
        ("TTFT_p90", lambda r: fmt(r.ttft_p90, 0)),
        ("TTFT_p99", lambda r: fmt(r.ttft_p99, 0)),
        ("ITL_p50", lambda r: fmt(r.itl_p50, 1)),
        ("ITL_p90", lambda r: fmt(r.itl_p90, 1)),
        ("ITL_p99", lambda r: fmt(r.itl_p99, 1)),
        ("native_gp", lambda r: fmt(r.native_goodput_fraction, 3)),
        ("user_gp", lambda r: fmt(r.user_goodput.fraction, 3)),
        ("err%", lambda r: fmt(r.error_pct, 2)),
        ("util_max%", lambda r: fmt(r.kv.util_max_perc, 1)),
        ("kv_reused_blk", lambda r: fmt(r.kv.kv_reused_blocks, 0)),
        ("kv_missed_blk", lambda r: fmt(r.kv.kv_missed_blocks, 0)),
    ]
    lines.append("| " + " | ".join(c[0] for c in cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for run in runs:
        lines.append("| " + " | ".join(str(c[1](run)) for c in cols) + " |")
    lines.append("")

    # ---- 3. SLA / goodput detail ----
    lines.append("## 3. SLA / goodput detail")
    lines.append("")
    lines.append(
        "Both goodput fractions use the SAME denominator = `request_count` (per-request "
        "`profile_export.jsonl`). Records with no ITL (errors) and threshold failures count as "
        "NOT-good but STAY in the denominator (we never drop error/overflow records)."
    )
    lines.append("")
    lines.append(
        "| job | topology | native SLA | native gp (aiperf) | native gp (recompute) | user SLA | user gp | median ITL | discriminatory? |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for run in runs:
        ng = run.native_goodput_recompute
        ug = run.user_goodput
        native_sla = (
            f"{run.native_sla_ttft:.0f}/{run.native_sla_itl:g}"
            if run.native_sla_ttft is not None and run.native_sla_itl is not None
            else "N/A"
        )
        native_aiperf = (
            f"{run.native_goodput_fraction:.3f} ({run.native_good_request_count}/{run.request_count})"
            if run.native_goodput_fraction is not None
            else "N/A"
        )
        native_recompute = f"{ng.fraction:.3f} ({ng.good}/{ng.denom})" if ng.fraction is not None else "N/A"
        user_sla = f"{ttft_sla_ms:.0f}/{itl_sla_ms:g}"
        user_gp = f"{ug.fraction:.3f} ({ug.good}/{ug.denom})" if ug.fraction is not None else "N/A"
        disc = "yes" if ug.discriminatory else "**NO (relax ITL SLA)**"
        lines.append(
            f"| {run.job_id} | {run.topology_label} | {native_sla} | {native_aiperf} | {native_recompute} | "
            f"{user_sla} | {user_gp} | {fmt(ug.median_itl_ms, 1)}ms | {disc} |"
        )
    lines.append("")

    # ---- 4. CD-internal leverage ----
    lines.append("## 4. CD-internal leverage")
    lines.append("")
    lines.append(
        "> **Decisions are request COUNTS; token columns are the prefill COMPUTE load — counted by "
        "FINAL placement.** `local dec`/`remote dec` count *requests* by where prefill actually ran: "
        "`local dec` includes policy-Local AND every policy-Remote request the breaker/overload/"
        "zero-block/budget path downgraded back to local (those are itemized in `downgrades` as a "
        "sub-count of `local dec`); `remote dec` = TRUE disaggregation only. So `local tok`/`local "
        "dec` and `remote tok`/`remote dec` are each correct per-request averages. `local tok` (Q3) "
        "and `remote tok` (Q4) are the tokens actually prefilled on the decode workers vs the "
        "prefill aside, each **net of the G1+G2 prefix match** (`remote tok` ≡ the prefill-worker "
        "forward-pass tokens by construction). `remote tok%` = remote / (local+remote) tokens — the "
        "load-bearing leverage number (where the prefill FLOPs went), which differs from the "
        "decision share because prompts vary wildly in length and downgrades shift would-be-remote "
        "load into `local tok`. NOTE: the decode-side match credits G1+G2 only — a remote/G3-tier "
        "prefix match is not separately credited."
    )
    lines.append("")
    any_cd = any(run.cd.present for run in runs)
    found_but_empty = [run for run in runs if run.cd.snapshot_found and not run.cd.present]
    if not any_cd:
        reason = "no `kvbm_metrics_snapshot.json` with CD content for any run"
        if found_but_empty:
            reason = (
                "snapshot(s) found but carrying NO CD content (e.g. an aggregated run with no hub "
                "plane — only per-worker metrics)"
            )
        lines.append(
            f"**N/A — {reason}.** The real CD leverage "
            "(local-vs-remote prefill decisions, remote-prefill token volume, "
            "decode⇄prefill reconciliation, declined-by-reason, breaker downgrades) comes from "
            "the hub `/v1/metrics` fanout snapshot, written by the post-benchmark snapshot hook."
        )
        lines.append("")
        lines.append(
            "> To populate this section, re-run a CD recipe (one with a `kvbm_hub` plane) with the "
            "snapshot hook enabled. The hook scrapes Prometheus/velo metrics, which are "
            "**LOG-LEVEL-INDEPENDENT** — `KVBM_CONTROL_METRICS=true` (already set in the CD recipes) "
            "plus the hook is sufficient. **`RUST_LOG` need NOT be raised.**"
        )
    else:
        lines.append(
            "| job | topology | local dec | remote dec | local tok | remote tok | remote tok% | downgrades |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for run in runs:
            cd = run.cd
            if not cd.present:
                why = "no CD content" if cd.snapshot_found else "no snapshot"
                lines.append(f"| {run.job_id} | {run.topology_label} | _N/A ({why})_ | | | | | |")
                continue
            downgr = ", ".join(f"{k}={v}" for k, v in sorted(cd.downgrades.items())) or "-"
            lines.append(
                f"| {run.job_id} | {run.topology_label} | {cd.local_decisions} | {cd.remote_decisions} | "
                f"{cd.local_prefill_tokens} | {cd.remote_prefill_tokens} | "
                f"{fmt(cd.remote_compute_fraction, 3)} | {downgr} |"
            )
        lines.append("")

        # ---- 4b. Prefill-side prefix acquisition ----
        lines.append("### 4b. Prefill-side prefix acquisition (the prefill aside)")
        lines.append("")
        lines.append(
            "> For each disaggregated request the prefill worker acquires the prompt prefix as "
            "`pulled` (RDMA-pulled from decode) and forward-passes only the net-new remainder "
            "(`computed` = the prefill PRESSURE gauge, ≡ decode `remote tok`). Today the worker "
            "OVER-PULLS the full `[0, DNPT)` window; `local hit` is the slice it already had cached "
            "(its own vLLM prefix-cache hit), so the genuine `supplement` it would still need after "
            "the PNCT optimization = `pulled − local hit`. `local hit`/`supplement` need the "
            "Q7-instrumented binary (kvbm-prod-260607+); older runs report `local hit`=0 ⇒ "
            "`supplement`=`pulled` (unmeasured, not truly zero). CAVEAT: `local hit` is the prefill "
            "worker's vLLM-G1 prefix-cache hit only — a LOWER BOUND on the true opportunity, which "
            "also includes the prefill worker's own G2/G3 tiers (not yet surfaced)."
        )
        lines.append("")
        lines.append(
            "| job | topology | computed (forward) | pulled (from decode) | local hit (cached) | supplement (pull−hit) | pull:compute |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for run in runs:
            cd = run.cd
            if not cd.present:
                why = "no CD content" if cd.snapshot_found else "no snapshot"
                lines.append(f"| {run.job_id} | {run.topology_label} | _N/A ({why})_ | | | | |")
                continue
            ratio = cd.prefill_pulled_tokens / cd.prefill_computed_tokens if cd.prefill_computed_tokens else None
            lines.append(
                f"| {run.job_id} | {run.topology_label} | {cd.prefill_computed_tokens} | "
                f"{cd.prefill_pulled_tokens} | {cd.prefill_local_hit_tokens} | "
                f"{cd.prefill_pull_supplement} | {fmt(ratio, 2)} |"
            )
    lines.append("")

    # ---- 5. KV / cache ----
    lines.append("## 5. KV / cache behavior")
    lines.append("")
    lines.append(
        "> The `vllm:external_prefix_cache_hits` column is a **remote-prefill-leverage proxy "
        "(NOT cache efficiency)** — it counts tokens served from a remote/G2 tier, which is a "
        "side effect of CD/tiering, not a hit-rate. The real CD leverage is section 4."
    )
    lines.append("")
    lines.append(
        "| job | topology | total_blocks | block_size | G1 tok cap | util_max% | local prefix-hit | kv_reused_blk | kv_missed_blk | ext-cache-hit (proxy) | theoretical prefix-hit |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for run in runs:
        kv = run.kv
        lines.append(
            f"| {run.job_id} | {run.topology_label} | {fmt(kv.total_kv_blocks, 0)} | {fmt(kv.block_size, 0)} | "
            f"{fmt(kv.kv_token_capacity, 0)} | {fmt(kv.util_max_perc, 1)} | {fmt(kv.local_prefix_hit_rate, 4)} | "
            f"{fmt(kv.kv_reused_blocks, 0)} | {fmt(kv.kv_missed_blocks, 0)} | "
            f"{fmt(kv.external_prefix_hits, 0)} | {fmt(run.theoretical_prefix_cache_hit, 1, '%')} |"
        )
    lines.append("")

    # ---- 6. Per-run detail ----
    lines.append("## 6. Per-run detail")
    lines.append("")
    for run in runs:
        lines.append(f"### {run.job_id} — {run.job_name} ({run.topology_label})")
        lines.append("")
        lines.append(f"- run dir: `{run.run_dir}`")
        lines.append(f"- image: `{run.image}`")
        lines.append(
            f"- topology: agg_workers={run.agg_workers} × TEP{run.gpus_per_agg}; "
            f"prefill aside: {run.prefill_nodes} × TP{run.prefill_tp}; "
            f"active GPUs={run.active_gpus}"
        )
        lines.append(
            f"- dataset: `{run.dataset}` · concurrency declared={run.concurrency} · "
            f"effective={fmt(run.effective_concurrency, 1)}"
        )
        if run.concurrency and run.effective_concurrency and run.effective_concurrency < 0.85 * run.concurrency:
            lines.append(
                f"  - NOTE: effective concurrency ({fmt(run.effective_concurrency, 1)}) is well below declared "
                f"({run.concurrency}) — the server was not saturated at the requested concurrency."
            )
        lines.append(
            f"- ISL avg={fmt(run.isl_avg, 0)} · OSL avg={fmt(run.osl_avg, 0)} · request_count={run.request_count}"
        )
        lines.append(
            f"- condp_policy (threshold)={run.condp_policy if run.condp_policy is not None else 'N/A (agg)'}"
            f" · max_num_tokens={run.max_num_tokens}"
        )
        if run.ttft_p99 and run.ttft_p50 and run.ttft_p99 > 5 * run.ttft_p50:
            lines.append(
                f"  - p99-noise flag: TTFT p99 ({fmt(run.ttft_p99, 0)}ms) >> p50 ({fmt(run.ttft_p50, 0)}ms) — "
                f"heavy tail; treat p99 cautiously."
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-contained HTML report (manifest mode)
# ---------------------------------------------------------------------------

# The headline metric rows: (label, accessor, precision, suffix, higher_is_better).
_HTML_METRIC_ROWS: list[tuple[str, Any, int, str, bool]] = [
    ("req/s", lambda r: r.request_throughput, 3, "", True),
    ("total tok/s", lambda r: r.total_tput, 0, "", True),
    ("total tok/s/GPU", lambda r: r.total_tput_per_gpu, 0, "", True),
    ("output tok/s", lambda r: r.output_tput, 1, "", True),
    ("output tok/s/GPU", lambda r: r.output_tput_per_gpu, 1, "", True),
    ("output tok/s/user (p50)", lambda r: r.output_tput_per_user, 1, "", True),
    ("TTFT p50 (ms)", lambda r: r.ttft_p50, 0, "", False),
    ("TTFT p90 (ms)", lambda r: r.ttft_p90, 0, "", False),
    ("TTFT p99 (ms)", lambda r: r.ttft_p99, 0, "", False),
    ("ITL p50 (ms)", lambda r: r.itl_p50, 1, "", False),
    ("ITL p90 (ms)", lambda r: r.itl_p90, 1, "", False),
    ("ITL p99 (ms)", lambda r: r.itl_p99, 1, "", False),
    ("native goodput", lambda r: r.native_goodput_recompute.fraction, 3, "", True),
    ("user goodput", lambda r: r.user_goodput.fraction, 3, "", True),
]

_HTML_CSS = """
:root {
  --fg: #1a1d23; --muted: #5b6470; --line: #d8dde4; --bg: #ffffff;
  --baseline: #1f5fb0; --baseline-bg: #eaf2fb;
  --reference: #8a6d00; --reference-bg: #fbf4dd;
  --cd: #0a7d4e; --cd-bg: #e9f7ef;
  --win: #0a7d4e; --pos: #0a7d4e; --neg: #b0202a;
}
* { box-sizing: border-box; }
body {
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  color: var(--fg); background: var(--bg); margin: 0; padding: 32px;
}
.wrap { max-width: 1180px; margin: 0 auto; }
h1 { font-size: 21px; line-height: 1.3; margin: 0 0 6px; }
h2 { font-size: 16px; margin: 30px 0 10px; border-bottom: 2px solid var(--line); padding-bottom: 5px; }
.sub { color: var(--muted); font-size: 13px; margin: 2px 0; }
.note { color: var(--muted); font-size: 13px; margin: 10px 0; max-width: 900px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 4px; font-size: 13px; }
th, td { border: 1px solid var(--line); padding: 6px 9px; text-align: right; white-space: nowrap; }
th.metric, td.metric { text-align: left; font-weight: 600; background: #f7f8fa; }
thead th { vertical-align: bottom; }
.coltag { display: block; font-size: 10.5px; font-weight: 600; letter-spacing: .03em; margin-top: 3px; }
.badge {
  display: inline-block; font-size: 10px; font-weight: 700; letter-spacing: .04em;
  padding: 1px 6px; border-radius: 9px; margin-left: 4px; vertical-align: middle;
}
.badge.tpcb { background: #ffe0e0; color: #b0202a; border: 1px solid #e6a4a4; }
.badge.cd { background: var(--cd-bg); color: var(--cd); border: 1px solid #a7d8bf; }
.badge.base { background: var(--baseline-bg); color: var(--baseline); border: 1px solid #aecbe8; }
.badge.ref { background: var(--reference-bg); color: var(--reference); border: 1px solid #ddcb8a; }
th.col-baseline { background: var(--baseline-bg); }
th.col-reference { background: var(--reference-bg); }
th.col-cd { background: var(--cd-bg); }
td.col-reference { background: #fdfaee; }
td.col-reference .delta { font-style: italic; }
.delta { font-size: 11px; display: block; }
.delta.pos { color: var(--pos); }
.delta.neg { color: var(--neg); }
.win { font-weight: 700; color: var(--win); }
td.winner { text-align: left; color: var(--win); font-weight: 600; }
td.winner.none { color: var(--muted); font-weight: 400; font-style: italic; }
ul.changelog { list-style: none; padding-left: 0; margin: 8px 0; }
ul.changelog li { padding: 6px 0; border-bottom: 1px solid var(--line); max-width: 900px; }
ul.changelog .date { font-weight: 700; margin-right: 8px; color: var(--baseline); }
.caveat {
  background: #fff7ec; border-left: 4px solid #e0a93a; padding: 10px 14px; margin: 10px 0;
  font-size: 13px; max-width: 900px; border-radius: 0 4px 4px 0;
}
.legend { font-size: 12px; color: var(--muted); margin: 6px 0 0; }
.prov { font-size: 12px; }
.prov th, .prov td { white-space: normal; }
.prov td.errors-yes { color: var(--neg); font-weight: 600; }
footer { margin-top: 28px; color: var(--muted); font-size: 11.5px; }
code { background: #f1f3f6; padding: 1px 4px; border-radius: 3px; font-size: 12px; }
"""


def _esc(s: Any) -> str:
    return html.escape(str(s), quote=True)


def _md_inline_to_html(s: str) -> str:
    """Minimal inline-markdown → HTML for report prose: **bold** + `code`, HTML-escaped first."""
    out = _esc(s)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    out = re.sub(r"`(.+?)`", r"<code>\1</code>", out)
    return out


def _col_class(mr: ManifestRun) -> str:
    if mr.is_baseline:
        return "col-baseline"
    if mr.is_reference:
        return "col-reference"
    return "col-cd"


def render_html(
    manifest: Manifest,
    paired: list[tuple[ManifestRun, RunReport]],
    ttft_sla_ms: float,
    itl_sla_ms: float,
    sweep_runs: list[RunReport] | None = None,
    pd_paired: list[tuple[ManifestRun, RunReport]] | None = None,
    pareto_png_name: str | None = None,
) -> str:
    """Render a SELF-CONTAINED HTML report from a curated manifest.

    ``paired`` is ordered (column order). The kind=baseline run is the delta
    baseline; the kind=baseline_reference run is shown but visually marked as a
    reference (italic deltas, a "REFERENCE" badge) and is NOT eligible to win an
    axis. CD columns carry a CD badge; the TPCB column carries a TPCB badge.

    Inline CSS only — no external/CDN dependency. All free text is HTML-escaped.
    """
    runs = [rr for _, rr in paired]
    mruns = [mr for mr, _ in paired]
    # Delta baseline = the kind=baseline run (explicit, never auto-picked by order).
    base_idx = next((i for i, mr in enumerate(mruns) if mr.is_baseline), None)
    base_run = runs[base_idx] if base_idx is not None else None
    # NOTE: winner eligibility EXCLUDES the round-robin reference column (done inline below).

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out: list[str] = []
    out.append("<!DOCTYPE html>")
    out.append('<html lang="en"><head><meta charset="utf-8">')
    out.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    out.append(f"<title>{_esc(manifest.title)}</title>")
    out.append(f"<style>{_HTML_CSS}</style>")
    out.append("</head><body><div class='wrap'>")

    # ---- Header ----
    out.append(f"<h1>{_esc(manifest.title)}</h1>")
    if manifest.dataset is not None:
        out.append(f"<p class='sub'>Dataset: <code>{_esc(manifest.dataset)}</code></p>")
    out.append(
        f"<p class='sub'>Concurrency: <b>{_esc(manifest.concurrency)}</b> &middot; "
        f"{len(paired)} run(s) &middot; user SLA TTFT&lt;{ttft_sla_ms:.0f}ms ITL&lt;{itl_sla_ms:g}ms "
        f"&middot; generated {ts}</p>"
    )
    if manifest.generated_note:
        out.append(f"<p class='note'>{_esc(manifest.generated_note)}</p>")
    if base_run is not None:
        base_mr = mruns[base_idx]
        out.append(
            f"<p class='sub'>Delta baseline = <b>{_esc(base_mr.label)}</b> "
            f"(job <code>{_esc(base_run.job_id)}</code>). "
            "Round-robin column is a <span class='badge ref'>REFERENCE</span> only — not the baseline.</p>"
        )

    # ---- Changelog ----
    out.append("<h2>Changelog</h2>")
    if manifest.changelog:
        out.append("<ul class='changelog'>")
        for entry in manifest.changelog:
            out.append(f"<li><span class='date'>{_esc(entry.date)}</span>{_esc(entry.note)}</li>")
        out.append("</ul>")
    else:
        out.append("<p class='note'>(no changelog entries)</p>")

    # ---- Headline comparison table ----
    out.append("<h2>Headline comparison</h2>")
    out.append("<table><thead><tr><th class='metric'>metric</th>")
    for mr in mruns:
        cls = _col_class(mr)
        badges = ""
        if mr.is_baseline:
            badges = "<span class='badge base'>BASELINE</span>"
            tag = "delta baseline"
        elif mr.is_reference:
            badges = "<span class='badge ref'>REFERENCE</span>"
            tag = "not the baseline"
        else:
            badges = "<span class='badge cd'>CD</span>"
            tag = "conditional disagg"
        if mr.tpcb:
            badges += "<span class='badge tpcb'>TPCB</span>"
        out.append(
            f"<th class='{cls}'>{_esc(mr.label)}{badges}"
            f"<span class='coltag'>{tag} &middot; job {_esc(mr.job_id)}</span></th>"
        )
    out.append("<th class='metric'>winner</th></tr></thead><tbody>")

    # Goodput rows name no winner when non-discriminatory (ITL SLA <= median ITL).
    native_disc = all(rr.native_goodput_recompute.discriminatory for rr in runs)
    user_disc = all(rr.user_goodput.discriminatory for rr in runs)

    for label, accessor, prec, suffix, higher in _HTML_METRIC_ROWS:
        out.append(f"<tr><td class='metric'>{_esc(label)}</td>")
        base_v = accessor(base_run) if base_run is not None else None
        for mr, rr in paired:
            cls = _col_class(mr)
            v = accessor(rr)
            cell = _esc(fmt(v, prec, suffix))
            delta_html = ""
            if base_run is not None and rr is not base_run and v is not None and base_v not in (None, 0):
                d = (v - base_v) / base_v * 100.0
                sign = "+" if d >= 0 else ""
                # "good" direction depends on the metric (higher_is_better).
                good = (d >= 0) if higher else (d <= 0)
                dcls = "pos" if good else "neg"
                delta_html = f"<span class='delta {dcls}'>{sign}{d:.1f}%</span>"
            out.append(f"<td class='{cls}'>{cell}{delta_html}</td>")
        # winner
        non_disc = (label == "native goodput" and not native_disc) or (label == "user goodput" and not user_disc)
        if non_disc:
            out.append("<td class='winner none'>non-discriminatory</td>")
        else:
            vals = [(mr, accessor(rr)) for mr, rr in paired if not mr.is_reference and accessor(rr) is not None]
            if not vals:
                out.append("<td class='winner none'>N/A</td>")
            else:
                best = max(vals, key=lambda x: x[1]) if higher else min(vals, key=lambda x: x[1])
                out.append(f"<td class='winner'>{_esc(best[0].label)}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    out.append(
        "<p class='legend'>Green delta = better than baseline on that axis, red = worse. "
        "The reference (round-robin) column is shown italic and is excluded from winner selection.</p>"
    )

    # ---- Caveats (load-bearing) ----
    out.append("<h2>Caveats</h2>")
    if not native_disc or not user_disc:
        med = ", ".join(f"{_esc(mr.label)}={fmt(rr.user_goodput.median_itl_ms, 1)}ms" for mr, rr in paired)
        out.append(
            "<div class='caveat'><b>Goodput is NON-DISCRIMINATORY here.</b> The ITL SLA is at/under the median "
            f"ITL [{med}], so the goodput fractions collapse toward zero and name NO winner. A higher goodput "
            "fraction here mostly reflects LOWER offered load (fewer completed requests in the denominator), "
            "NOT a CD win/regression — do not read it as one. Relax the ITL SLA above the median to make goodput "
            "discriminatory.</div>"
        )
    out.append(
        "<div class='caveat'><b>External-prefix-cache-hits is a remote-prefill-leverage proxy, NOT cache "
        "efficiency.</b> The <code>vllm:external_prefix_cache_hits</code> counter measures tokens served from a "
        "remote/G2 tier — a side effect of CD/tiering, not a hit-rate. Never read it as a headline efficiency "
        "metric.</div>"
    )

    # ---- Threshold sweep (parallel axis) ----
    if sweep_runs:
        out.append("<h2>Conditional-threshold sweep</h2>")
        out.append(f"<p class='note'>{_md_inline_to_html(_SWEEP_THESIS)}</p>")
        if manifest.sweep_note:
            out.append(f"<p class='caveat'>{_esc(manifest.sweep_note)}</p>")
        for conc in sorted(_sweep_by_concurrency(sweep_runs)):
            sruns = _sweep_by_concurrency(sweep_runs)[conc]
            topo = sruns[0].topology_label if sruns else "?"
            out.append(f"<h3>{_esc(topo)} @ concurrency {conc}</h3>")
            out.append("<table><thead><tr>")
            for h in (
                "threshold", "job", "req/s", "TTFT p50", "TTFT p90", "ITL p50",
                "ITL p90", "remote dec", "token rem%", "over-pull%", "breaker HOT",
            ):
                out.append(f"<th>{_esc(h)}</th>")
            out.append("</tr></thead><tbody>")
            base_req = next((r.request_throughput for r in sruns if r.request_throughput), None)
            for r in sruns:
                cd = r.cd
                req = r.request_throughput
                reqcell = fmt(req, 3)
                if req is not None and base_req:
                    reqcell += f" ({(req - base_req) / base_req * 100:+.1f}%)"
                cells = [
                    str(r.condp_policy), r.job_id, reqcell, fmt(r.ttft_p50, 0), fmt(r.ttft_p90, 0),
                    fmt(r.itl_p50, 1), fmt(r.itl_p90, 1), str(cd.remote_decisions),
                    _pct0(cd.remote_compute_fraction), _pct0(_sweep_overpull(cd)), str(_sweep_breaker_hot(cd)),
                ]
                out.append("<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in cells) + "</tr>")
            out.append("</tbody></table>")

    # ---- Traditional P/D disaggregation (distinct serving mode) ----
    if pd_paired:
        out.append("<h2>Traditional P/D disaggregation</h2>")
        out.append(f"<p class='note'>{_md_inline_to_html(_PD_DISAGG_THESIS)}</p>")
        if manifest.pd_disagg_note:
            out.append(f"<p class='caveat'>{_esc(manifest.pd_disagg_note)}</p>")
        out.append("<table><thead><tr>")
        for h in ("config", "conc", "job", "req/s", "out tok/s", "TTFT p50", "TTFT p90", "ITL p50", "ITL p90", "reqs", "kv_reused_blk", "status"):
            out.append(f"<th>{_esc(h)}</th>")
        out.append("</tr></thead><tbody>")
        for mr, rr in sorted(pd_paired, key=lambda pr: (pr[1].concurrency or 0, -(pr[1].request_throughput or 0.0))):
            valid = _pd_disagg_valid(rr)
            status = "valid" if valid else f"INVALID — {rr.runtime_error or 'no served traffic'}"
            cells = [
                mr.label, (rr.concurrency if rr.concurrency is not None else "N/A"), rr.job_id,
                fmt(rr.request_throughput, 3), fmt(rr.output_tput, 0),
                fmt(rr.ttft_p50, 0), fmt(rr.ttft_p90, 0), fmt(rr.itl_p50, 1), fmt(rr.itl_p90, 1),
                str(rr.request_count if rr.request_count is not None else "N/A"), fmt(rr.kv.kv_reused_blocks, 0), status,
            ]
            tdcls = "" if valid else " class='errors-yes'"
            out.append("<tr>" + "".join(f"<td{tdcls}>{_esc(c)}</td>" for c in cells) + "</tr>")
        out.append("</tbody></table>")

    # ---- Pareto frontier (relative PNG embed; emitted only when the PNG was written) ----
    # NOTE: this is the one place the report references an external asset (the sibling PNG, same
    # dir). The "self-contained" guard (no ` src=`) is intentionally relaxed for this embed, same
    # as the markdown relative image — so render_html only emits the <img> when a name is passed.
    if pareto_png_name:
        out.append("<h2>Pareto frontier</h2>")
        if manifest.pareto_note:
            out.append(f"<p class='note'>{_esc(manifest.pareto_note)}</p>")
        out.append(
            f'<img src="{_esc(pareto_png_name)}" '
            'alt="Pareto: output tok/s/user (p50) vs output tok/s/GPU" '
            'style="max-width:100%;height:auto">'
        )
        out.append(
            "<p class='legend'>Each line is a serving mode (agg / CD / trad-P/D) traced across the "
            "concurrency sweep; each point is labelled (concurrency, TTFT p50).</p>"
        )

    # ---- Per-run provenance footer ----
    out.append("<h2>Per-run provenance</h2>")
    out.append("<table class='prov'><thead><tr>")
    # No 'errors' column: runs in the set are valid; non-blocking errors are not reported.
    for h in ("job", "label", "image", "condp policy", "topology", "active GPUs"):
        cls = "metric" if h in ("job", "label") else ""
        out.append(f"<th class='{cls}'>{_esc(h)}</th>")
    out.append("</tr></thead><tbody>")
    for mr, rr in paired:
        img = Path(rr.image).name if rr.image else "N/A"
        condp = rr.condp_policy if rr.condp_policy is not None else "N/A (agg)"
        topo = (
            f"{rr.topology_label} (agg {rr.agg_workers}×TEP{rr.gpus_per_agg}"
            + (f", prefill {rr.prefill_nodes}×TP{rr.prefill_tp}" if rr.prefill_nodes else "")
            + ")"
        )
        out.append(
            f"<tr><td class='metric'>{_esc(rr.job_id)}</td>"
            f"<td class='metric'>{_esc(mr.label)}</td>"
            f"<td>{_esc(img)}</td>"
            f"<td>{_esc(condp)}</td>"
            f"<td>{_esc(topo)}</td>"
            f"<td style='text-align:right'>{_esc(rr.active_gpus)}</td></tr>"
        )
    out.append("</tbody></table>")

    out.append(
        f"<footer>Generated by <code>srtctl.analysis.bench_report</code> from "
        f"<code>{_esc(manifest.source_path.name if manifest.source_path else 'manifest')}</code> &middot; "
        f"self-contained (inline CSS, no external assets) &middot; {ts}</footer>"
    )
    out.append("</div></body></html>")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def csv_rows(runs: list[RunReport]) -> tuple[list[str], list[dict[str, Any]]]:
    fields = [
        "job_id",
        "job_name",
        "topology_label",
        "is_baseline_candidate",
        "dataset",
        "concurrency",
        "tp",
        "agg_workers",
        "gpus_per_agg",
        "prefill_nodes",
        "prefill_tp",
        "active_gpus",
        "provisioned_nodes",
        "provisioned_gpus",
        "image",
        "condp_policy",
        "max_num_tokens",
        "request_throughput",
        "request_count",
        "ttft_p50_ms",
        "ttft_p99_ms",
        "itl_p50_ms",
        "itl_p99_ms",
        "output_tput",
        "output_tput_per_gpu",
        "output_tput_per_user",
        "total_tput",
        "total_tput_per_gpu",
        "input_tput",
        "isl_avg",
        "osl_avg",
        "effective_concurrency",
        "theoretical_prefix_cache_hit",
        "was_cancelled",
        "error_count",
        "error_pct",
        "runtime_error",
        "native_sla_ttft_ms",
        "native_sla_itl_ms",
        "native_good_request_count",
        "native_goodput_fraction",
        "native_goodput_recompute_good",
        "native_goodput_recompute_denom",
        "native_goodput_recompute_fraction",
        "user_sla_ttft_ms",
        "user_sla_itl_ms",
        "user_goodput_good",
        "user_goodput_denom",
        "user_goodput_fraction",
        "user_goodput_median_itl_ms",
        "user_goodput_discriminatory",
        "kv_total_blocks",
        "kv_block_size",
        "kv_token_capacity",
        "kv_util_max_perc",
        "kv_local_prefix_hit_rate",
        "kv_reused_blocks",
        "kv_missed_blocks",
        "kv_external_prefix_hits",
        "kv_decode_worker_count",
        "cd_snapshot_found",
        "cd_snapshot_present",
        "cd_local_decisions",
        "cd_remote_decisions",
        "cd_remote_fraction",
        "cd_local_prefill_tokens",
        "cd_remote_prefill_tokens",
        "cd_remote_compute_fraction",
        "cd_prefill_computed_tokens",
        "cd_prefill_pulled_tokens",
        "cd_prefill_local_hit_tokens",
        "cd_prefill_pull_supplement",
        "cd_instances",
    ]
    rows: list[dict[str, Any]] = []
    for run in runs:
        ng = run.native_goodput_recompute
        ug = run.user_goodput
        kv = run.kv
        cd = run.cd
        rows.append(
            {
                "job_id": run.job_id,
                "job_name": run.job_name,
                "topology_label": run.topology_label,
                "is_baseline_candidate": run.is_baseline_candidate,
                "dataset": run.dataset,
                "concurrency": run.concurrency,
                "tp": run.tp,
                "agg_workers": run.agg_workers,
                "gpus_per_agg": run.gpus_per_agg,
                "prefill_nodes": run.prefill_nodes,
                "prefill_tp": run.prefill_tp,
                "active_gpus": run.active_gpus,
                "provisioned_nodes": run.provisioned_nodes,
                "provisioned_gpus": run.provisioned_gpus,
                "image": run.image,
                "condp_policy": run.condp_policy,
                "max_num_tokens": run.max_num_tokens,
                "request_throughput": run.request_throughput,
                "request_count": run.request_count,
                "ttft_p50_ms": run.ttft_p50,
                "ttft_p99_ms": run.ttft_p99,
                "itl_p50_ms": run.itl_p50,
                "itl_p99_ms": run.itl_p99,
                "output_tput": run.output_tput,
                "output_tput_per_gpu": run.output_tput_per_gpu,
                "output_tput_per_user": run.output_tput_per_user,
                "total_tput": run.total_tput,
                "total_tput_per_gpu": run.total_tput_per_gpu,
                "input_tput": run.input_tput,
                "isl_avg": run.isl_avg,
                "osl_avg": run.osl_avg,
                "effective_concurrency": run.effective_concurrency,
                "theoretical_prefix_cache_hit": run.theoretical_prefix_cache_hit,
                "was_cancelled": run.was_cancelled,
                "error_count": run.error_count,
                "error_pct": run.error_pct,
                "runtime_error": run.runtime_error,
                "native_sla_ttft_ms": run.native_sla_ttft,
                "native_sla_itl_ms": run.native_sla_itl,
                "native_good_request_count": run.native_good_request_count,
                "native_goodput_fraction": run.native_goodput_fraction,
                "native_goodput_recompute_good": ng.good,
                "native_goodput_recompute_denom": ng.denom,
                "native_goodput_recompute_fraction": ng.fraction,
                "user_sla_ttft_ms": ug.sla_ttft_ms,
                "user_sla_itl_ms": ug.sla_itl_ms,
                "user_goodput_good": ug.good,
                "user_goodput_denom": ug.denom,
                "user_goodput_fraction": ug.fraction,
                "user_goodput_median_itl_ms": ug.median_itl_ms,
                "user_goodput_discriminatory": ug.discriminatory,
                "kv_total_blocks": kv.total_kv_blocks,
                "kv_block_size": kv.block_size,
                "kv_token_capacity": kv.kv_token_capacity,
                "kv_util_max_perc": kv.util_max_perc,
                "kv_local_prefix_hit_rate": kv.local_prefix_hit_rate,
                "kv_reused_blocks": kv.kv_reused_blocks,
                "kv_missed_blocks": kv.kv_missed_blocks,
                "kv_external_prefix_hits": kv.external_prefix_hits,
                "kv_decode_worker_count": kv.decode_worker_count,
                "cd_snapshot_found": cd.snapshot_found,
                "cd_snapshot_present": cd.present,
                "cd_local_decisions": cd.local_decisions,
                "cd_remote_decisions": cd.remote_decisions,
                "cd_remote_fraction": cd.remote_fraction,
                "cd_local_prefill_tokens": cd.local_prefill_tokens,
                "cd_remote_prefill_tokens": cd.remote_prefill_tokens,
                "cd_remote_compute_fraction": cd.remote_compute_fraction,
                "cd_prefill_computed_tokens": cd.prefill_computed_tokens,
                "cd_prefill_pulled_tokens": cd.prefill_pulled_tokens,
                "cd_prefill_local_hit_tokens": cd.prefill_local_hit_tokens,
                "cd_prefill_pull_supplement": cd.prefill_pull_supplement,
                "cd_instances": cd.instances,
            }
        )
    return fields, rows


def write_csv(path: Path, runs: list[RunReport]) -> None:
    fields, rows = csv_rows(runs)
    with path.open("w", newline="") as f:
        # lineterminator="\n" (not the csv default "\r\n"): with newline="" the default
        # would write CRLF verbatim, which git (core.safecrlf) refuses to add. Keep LF.
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_run(rd_token: str, ttft_sla_ms: float, itl_sla_ms: float) -> RunReport | None:
    """Resolve a job-id/path token to a built RunReport (or None, logging why)."""
    rd = resolve_run_dir(rd_token)
    if rd is None:
        logger.warning("Could not resolve run token %r to a run-output dir — skipping", rd_token)
        return None
    if not (rd / "config.yaml").is_file():
        logger.warning("Run dir %s has no config.yaml — skipping", rd)
        return None
    try:
        return build_run_report(rd, ttft_sla_ms, itl_sla_ms)
    except Exception as e:  # noqa: BLE001 - one bad run must not kill the report
        logger.warning("Failed to build report for %s: %s", rd, e)
        return None


def run_manifest_mode(args: argparse.Namespace) -> int:
    """Manifest mode: resolve job ids from a curated manifest, write HTML (+md+csv).

    The default output is the manifest's own basename with the format suffix —
    so ``docs/runs/latest_results.yaml`` regenerates ``latest_results.{html,md,csv}``
    in place (the durable "latest results" semantics). ``--out-dir`` overrides the
    directory; the stem stays the manifest stem.
    """
    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.is_file():
        logger.error("Manifest %s not found.", manifest_path)
        return 1
    manifest = parse_manifest(manifest_path)
    if not manifest.runs:
        logger.error("Manifest %s declares no runs.", manifest_path)
        return 1

    paired: list[tuple[ManifestRun, RunReport]] = []
    for mr in manifest.runs:  # preserve manifest order (column order)
        rr = _build_run(mr.job_id, args.ttft_sla_ms, args.itl_sla_ms)
        if rr is not None:
            paired.append((mr, rr))
    if not paired:
        logger.error("No manifest runs could be parsed. Nothing to report.")
        return 1
    if manifest.baseline_run() is None:
        logger.warning("Manifest declares no kind=baseline run — deltas will be omitted in the HTML.")

    runs = [rr for _, rr in paired]

    # Threshold sweep (parallel axis): build RunReports for the sweep job ids, reusing the
    # already-built main runs by job id (the anchor overlaps) so each dir is parsed once.
    sweep_runs: list[RunReport] = []
    if manifest.sweep_job_ids:
        by_id = {rr.job_id: rr for rr in runs}
        for jid in manifest.sweep_job_ids:
            rr = by_id.get(jid) or _build_run(jid, args.ttft_sla_ms, args.itl_sla_ms)
            if rr is not None:
                sweep_runs.append(rr)

    # Trad P/D disagg (distinct serving mode): build paired (ManifestRun, RunReport) for the label.
    pd_paired: list[tuple[ManifestRun, RunReport]] = []
    if manifest.pd_disagg_runs:
        by_id = {rr.job_id: rr for rr in runs}
        for mr in manifest.pd_disagg_runs:
            rr = by_id.get(mr.job_id) or _build_run(mr.job_id, args.ttft_sla_ms, args.itl_sla_ms)
            if rr is not None:
                pd_paired.append((mr, rr))

    # Pareto frontier (parallel axis): build RunReports for the pareto job ids (reusing the
    # already-built main runs by job id, like sweep/pd) so each dir is parsed once.
    pareto_runs: list[RunReport] = []
    if manifest.pareto_job_ids:
        by_id = {rr.job_id: rr for rr in runs}
        for jid in manifest.pareto_job_ids:
            rr = by_id.get(jid) or _build_run(jid, args.ttft_sla_ms, args.itl_sla_ms)
            if rr is not None:
                pareto_runs.append(rr)

    # Output dir/stem computed up front: the PNG path + the written-bool gate the HTML img and md §9.
    out_dir = Path(args.out_dir).resolve() if args.out_dir else manifest_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = manifest_path.stem

    # Write the Pareto PNG (if any pareto runs) and only then reference it from the HTML/md.
    pareto_png_name: str | None = None
    if pareto_runs:
        png_name = f"{stem}_pareto.png"
        title = f"{manifest.title} — Pareto frontier"
        if render_pareto_png(pareto_runs, out_dir / png_name, title):
            pareto_png_name = png_name
            logger.info("Wrote %s", out_dir / png_name)

    html_doc = render_html(manifest, paired, args.ttft_sla_ms, args.itl_sla_ms, sweep_runs, pd_paired, pareto_png_name)

    # md + csv reuse the existing renderers (grouping + explicit baseline by kind).
    base_mr = manifest.baseline_run()
    explicit_baseline = base_mr.job_id if base_mr else None
    groups = group_runs(runs)
    md = render_markdown(runs, groups, explicit_baseline, args.ttft_sla_ms, args.itl_sla_ms)
    if sweep_runs:
        md = md.rstrip() + "\n\n" + render_threshold_sweep(sweep_runs, manifest.sweep_note)
    if pd_paired:
        md = md.rstrip() + "\n\n" + render_pd_disagg(pd_paired, manifest.pd_disagg_note)
    # §9 Pareto frontier: a GitHub/GitLab-renderable RELATIVE image embed (filename only, same dir).
    if pareto_png_name:
        pareto_md = ["## 9. Pareto frontier", ""]
        if manifest.pareto_note:
            pareto_md.append(f"> {manifest.pareto_note}")
            pareto_md.append("")
        pareto_md.append(f"![Pareto: output tok/s/user (p50) vs output tok/s/GPU]({pareto_png_name})")
        pareto_md.append("")
        pareto_md.append(
            "Each line is a serving mode (agg / CD / trad-P/D) traced across the concurrency sweep; "
            "each point is labelled (concurrency, TTFT p50)."
        )
        md = md.rstrip() + "\n\n" + "\n".join(pareto_md) + "\n"

    html_path = out_dir / f"{stem}.html"
    html_path.write_text(html_doc)
    logger.info("Wrote %s", html_path)
    if args.format in ("all", "md"):
        md_path = out_dir / f"{stem}.md"
        md_path.write_text(md)
        logger.info("Wrote %s", md_path)
    if args.format in ("all", "csv"):
        csv_path = out_dir / f"{stem}.csv"
        write_csv(csv_path, runs)
        logger.info("Wrote %s", csv_path)
        # Sibling CSV for the sweep axis (full schema, every threshold point).
        if sweep_runs:
            sweep_csv = out_dir / f"{stem}_threshold_sweep.csv"
            write_csv(sweep_csv, sweep_runs)
            logger.info("Wrote %s", sweep_csv)
        # Sibling CSV for the trad-P/D serving mode (full schema, every split).
        if pd_paired:
            pd_csv = out_dir / f"{stem}_pd_disagg.csv"
            write_csv(pd_csv, [rr for _, rr in pd_paired])
            logger.info("Wrote %s", pd_csv)
    return 0


def run_runs_mode(args: argparse.Namespace) -> int:
    """Legacy --runs mode: timestamped markdown + CSV (byte-for-byte as before)."""
    tokens = [t for t in args.runs.split(",") if t.strip()]
    runs: list[RunReport] = []
    for token in tokens:
        rr = _build_run(token, args.ttft_sla_ms, args.itl_sla_ms)
        if rr is not None:
            runs.append(rr)

    if not runs:
        logger.error("No runs could be parsed. Nothing to report.")
        return 1

    groups = group_runs(runs)
    md = render_markdown(runs, groups, args.baseline, args.ttft_sla_ms, args.itl_sla_ms)

    out_dir = Path(args.out_dir) if args.out_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    md_path = out_dir / f"BENCHMARK_REPORT_{stamp}.md"
    csv_path = out_dir / f"BENCHMARK_REPORT_{stamp}.csv"
    md_path.write_text(md)
    write_csv(csv_path, runs)

    logger.info("Wrote %s", md_path)
    logger.info("Wrote %s", csv_path)
    print(md)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="srtctl.analysis.bench_report",
        description="Offline agg-vs-CD benchmark report from srt-slurm run dirs. "
        "Use --runs for an ad-hoc set (markdown + CSV) or --manifest for the "
        "curated, durable latest-results report (HTML + markdown + CSV).",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--runs",
        help="Comma-separated job ids or run-output dirs (e.g. 2187812,2187813,2187814).",
    )
    src.add_argument(
        "--manifest",
        help="Path to a curated latest_results.yaml manifest (job ids + labels + changelog). "
        "Emits a self-contained HTML report (default out = the manifest's own dir/stem).",
    )
    parser.add_argument("--ttft-sla-ms", type=float, default=DEFAULT_TTFT_SLA_MS, help="User TTFT SLA (ms).")
    parser.add_argument("--itl-sla-ms", type=float, default=DEFAULT_ITL_SLA_MS, help="User ITL SLA (ms).")
    parser.add_argument(
        "--baseline", default=None, help="Explicit baseline job id (--runs mode; default: no-kvbm_hub run)."
    )
    parser.add_argument(
        "--all-columns",
        action="store_true",
        help="Reserved/no-op: the markdown table is a curated subset; the full column schema is "
        "always written to the sibling .csv.",
    )
    parser.add_argument(
        "--format",
        choices=("html", "all", "md", "csv"),
        default="all",
        help="Manifest mode: which sibling outputs to also write next to the HTML (default: all).",
    )
    parser.add_argument("--out-dir", default=None, help="Output dir for the report (default: cwd, or manifest dir).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.manifest:
        return run_manifest_mode(args)
    return run_runs_mode(args)


if __name__ == "__main__":
    sys.exit(main())
