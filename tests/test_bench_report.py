# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the offline agg-vs-CD benchmark report generator.

Exercises the pure parsing + aggregation helpers in
:mod:`srtctl.analysis.bench_report` against synthetic artifacts, plus the
discriminatory-power flag and goodput-denominator contract.
"""

import json
from pathlib import Path

from srtctl.analysis.bench_report import (
    CdLeverage,
    GoodputResult,
    KvMetrics,
    ManifestRun,
    RunReport,
    find_real_artifact_dir,
    parse_cd_snapshot,
    parse_cli_field,
    parse_cli_goodput_sla,
    parse_manifest,
    parse_server_metrics,
    recompute_goodput,
    render_html,
    summarize_errors,
    write_csv,
)


def test_write_csv_is_lf_not_crlf(tmp_path):
    """The CSV must use LF line endings: csv.DictWriter defaults to CRLF, which git
    (core.safecrlf) refuses to add (`CRLF would be replaced by LF`)."""
    run = _make_run("2187965", "agg-6xTEP2", is_baseline=True, req_s=1.74, itl_p50=17.0)
    out = tmp_path / "latest_results.csv"
    write_csv(out, [run])
    data = out.read_bytes()
    assert b"\r" not in data, "CSV must be LF-only (no CR)"
    assert data.count(b"\n") >= 2  # header + at least one data row


def test_summarize_errors_sums_buckets_not_categories():
    # aiperf error_summary = BUCKETS; the count is the SUM of per-bucket counts,
    # NOT len() (which counts distinct categories and ~halves the real error rate).
    es = [
        {"error_details": {"type": "ValueError"}, "count": 3},
        {"error_details": {"code": 500, "type": "Internal Server Error"}, "count": 1},
        {"error_details": {"code": 500, "type": "Internal Server Error"}, "count": 2},
    ]
    count, summary = summarize_errors(es)
    assert count == 6  # NOT len(es) == 3
    assert "ValueError×3" in summary
    assert "Internal Server Error×1" in summary and "Internal Server Error×2" in summary
    # empty / non-list are clean
    assert summarize_errors([]) == (0, "")
    assert summarize_errors(None) == (0, "")
    # code-only bucket (no type) falls back to HTTP <code>
    c2, s2 = summarize_errors([{"error_details": {"code": 503}, "count": 4}])
    assert c2 == 4 and "HTTP 503×4" in s2


def test_parse_cli_goodput_sla():
    cli = (
        "aiperf profile -m foo --concurrency 48 "
        "--goodput 'time_to_first_token:5000 inter_token_latency:10' --benchmark-duration 1200"
    )
    ttft, itl = parse_cli_goodput_sla(cli)
    assert ttft == 5000.0
    assert itl == 10.0


def test_parse_cli_field():
    cli = "aiperf profile --public-dataset 'semianalysis_cc_traces' --concurrency 48 --streaming"
    assert parse_cli_field(cli, "public-dataset") == "semianalysis_cc_traces"
    assert parse_cli_field(cli, "concurrency") == "48"
    assert parse_cli_field(cli, "missing-flag") is None
    assert parse_cli_field(None, "concurrency") is None


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_recompute_goodput_keeps_error_records_in_denominator(tmp_path):
    # 4 records: 2 good (under SLA), 1 over-ITL, 1 error (no ITL value).
    jsonl = tmp_path / "profile_export.jsonl"
    _write_jsonl(
        jsonl,
        [
            {"metrics": {"time_to_first_token": {"value": 100}, "inter_token_latency": {"value": 5}}},
            {"metrics": {"time_to_first_token": {"value": 200}, "inter_token_latency": {"value": 6}}},
            {"metrics": {"time_to_first_token": {"value": 300}, "inter_token_latency": {"value": 50}}},
            {"metrics": {"time_to_first_token": {"value": None}, "inter_token_latency": {"value": None}}},
        ],
    )
    res = recompute_goodput(jsonl, ttft_sla_ms=5000, itl_sla_ms=7)
    # Error/overflow records stay in the denominator (4), only the 2 good count.
    assert res.denom == 4
    assert res.good == 2
    assert abs(res.fraction - 0.5) < 1e-9


def test_goodput_discriminatory_flag(tmp_path):
    jsonl = tmp_path / "profile_export.jsonl"
    # ITLs around 15-20ms: a 7ms SLA is below the median => non-discriminatory.
    _write_jsonl(
        jsonl,
        [
            {"metrics": {"time_to_first_token": {"value": 100}, "inter_token_latency": {"value": 15}}},
            {"metrics": {"time_to_first_token": {"value": 100}, "inter_token_latency": {"value": 20}}},
            {"metrics": {"time_to_first_token": {"value": 100}, "inter_token_latency": {"value": 18}}},
        ],
    )
    tight = recompute_goodput(jsonl, ttft_sla_ms=5000, itl_sla_ms=7)
    assert tight.median_itl_ms == 18
    assert tight.discriminatory is False  # SLA 7 <= median 18
    loose = recompute_goodput(jsonl, ttft_sla_ms=5000, itl_sla_ms=30)
    assert loose.discriminatory is True  # SLA 30 > median 18


def test_parse_server_metrics_sums_workers_and_peaks_util(tmp_path):
    jsonl = tmp_path / "server_metrics_export.jsonl"
    block = 64

    def worker(url, hits, queries, ext, util):
        return {
            "endpoint_url": url,
            "metrics": {
                "vllm:prefix_cache_hits": [{"labels": {}, "value": hits}],
                "vllm:prefix_cache_queries": [{"labels": {}, "value": queries}],
                "vllm:external_prefix_cache_hits": [{"labels": {}, "value": ext}],
                "vllm:kv_cache_usage_perc": [{"labels": {}, "value": util}],
            },
        }

    frontend = {
        "endpoint_url": "http://localhost:8000/metrics",
        "metrics": {
            "dynamo_frontend_model_total_kv_blocks": [{"labels": {}, "value": 29683}],
            "dynamo_frontend_model_kv_cache_block_size": [{"labels": {}, "value": block}],
        },
    }
    _write_jsonl(
        jsonl,
        [
            # earlier snapshot with high util (peak should be picked up here)
            worker("http://10.0.0.1:7500/metrics", 100, 400, 10, 0.9),
            worker("http://10.0.0.2:7501/metrics", 200, 600, 20, 0.5),
            # later snapshots with the final monotonic counters + drained util
            worker("http://10.0.0.1:7500/metrics", 640, 1280, 64, 0.0),
            worker("http://10.0.0.2:7501/metrics", 1280, 2560, 128, 0.0),
            frontend,
        ],
    )
    km = parse_server_metrics(jsonl)
    assert km.total_kv_blocks == 29683
    assert km.block_size == block
    assert km.decode_worker_count == 2
    # last-snapshot counters, summed across the two workers
    assert km.prefix_hits == 640 + 1280
    assert km.prefix_queries == 1280 + 2560
    assert km.external_prefix_hits == 64 + 128
    # peak util across ALL snapshots (not the 0.0 final)
    assert km.util_max_perc == 0.9
    # reused/missed blocks
    assert km.kv_reused_blocks == (640 + 1280) / block
    assert km.kv_missed_blocks == ((1280 + 2560) - (640 + 1280)) / block
    assert km.kv_token_capacity == 29683 * block


def test_kv_metrics_empty_is_safe():
    km = KvMetrics()
    assert km.kv_reused_blocks is None
    assert km.kv_missed_blocks is None
    assert km.local_prefix_hit_rate is None
    assert km.kv_token_capacity is None


def test_compute_active_gpus_all_serving_modes():
    """active_gpus must count the full serving plane in all 3 modes — including the NATIVE
    prefill plane for trad P/D disagg (which the old formula omitted, halving the per-GPU
    denominator and inflating tok/s/GPU for trad-P/D runs)."""
    from types import SimpleNamespace

    from srtctl.analysis.bench_report import _compute_active_gpus

    def res(**kw):
        base = dict(
            num_agg=0, gpus_per_agg=0, num_decode=0, gpus_per_decode=0,
            num_prefill=0, gpus_per_prefill=0, kvbm_prefill_nodes=0, kvbm_prefill_tp=0,
        )
        base.update(kw)
        return SimpleNamespace(**base)

    # aggregated 6×TEP2 = 12
    assert _compute_active_gpus(res(num_agg=6, gpus_per_agg=2)) == 12
    # CD: 5 decode-AGG ×TEP2 + 1 hub prefill-aside ×TP2 = 12
    assert _compute_active_gpus(res(num_agg=5, gpus_per_agg=2, kvbm_prefill_nodes=1, kvbm_prefill_tp=2)) == 12
    # native P/D disagg (the fix): prefill plane + decode plane both count = 12
    assert _compute_active_gpus(res(num_decode=3, gpus_per_decode=2, num_prefill=3, gpus_per_prefill=2)) == 12  # 3p3d
    assert _compute_active_gpus(res(num_decode=2, gpus_per_decode=2, num_prefill=4, gpus_per_prefill=2)) == 12  # 4p2d
    assert _compute_active_gpus(res(num_decode=4, gpus_per_decode=2, num_prefill=2, gpus_per_prefill=2)) == 12  # 2p4d


def test_topology_label_agg_cd_and_native_disagg():
    """_topology_label must label native (trad) P/D disagg as 'pd-<p>p<d>d TEP<tp>' (num_agg==0,
    num_prefill>0, num_decode>0) — WITHOUT changing the agg label or the CD label. The native-disagg
    case is is_baseline=True with pn==0, so it must be checked before the agg branch."""
    from types import SimpleNamespace

    from srtctl.analysis.bench_report import _topology_label

    def res(**kw):
        base = dict(
            num_agg=0, gpus_per_agg=0, num_prefill=0, num_decode=0,
            gpus_per_decode=0, kvbm_prefill_nodes=0,
        )
        base.update(kw)
        return SimpleNamespace(**base)

    # aggregated: num_agg>0, no prefill aside -> agg-<n>xTEP<gpus_per_agg> (UNCHANGED)
    assert _topology_label(res(num_agg=6, gpus_per_agg=2), is_baseline=True) == ("agg-6xTEP2", 2)
    # CD: num_agg>0 + a hub prefill aside -> cd-<agg>d<pn>p TEP<gpus_per_agg> (UNCHANGED)
    assert _topology_label(res(num_agg=5, gpus_per_agg=2, kvbm_prefill_nodes=1), is_baseline=False) == (
        "cd-5d1p TEP2",
        2,
    )
    # native P/D disagg (the fix): no agg plane, distinct prefill+decode -> pd-3p3d TEP2.
    # It is is_baseline=True with pn==0, so the agg branch would have mislabelled it 'agg-0xTEP2'.
    assert _topology_label(
        res(num_prefill=3, num_decode=3, gpus_per_decode=2), is_baseline=True
    ) == ("pd-3p3d TEP2", 2)
    assert _topology_label(
        res(num_prefill=4, num_decode=2, gpus_per_decode=2), is_baseline=True
    ) == ("pd-4p2d TEP2", 2)


def test_pareto_series_groups_by_type_sorts_by_concurrency():
    """_pareto_series must group runs by topology_label, sort each series by concurrency, and yield
    (x=output_tput_per_user, y=output_tput_per_gpu, concurrency, ttft_p50) tuples. Runs missing x or y
    are dropped."""
    from srtctl.analysis.bench_report import _pareto_series

    def pt(job, topo, conc, per_user, out_tput, gpus, ttft):
        r = _make_run(job, topo, is_baseline=False, req_s=1.0, itl_p50=15.0)
        r.concurrency = conc
        r.output_tput_per_user = per_user
        r.output_tput = out_tput  # output_tput_per_gpu = out_tput / active_gpus
        r.active_gpus = gpus
        r.ttft_p50 = ttft
        return r

    runs = [
        # agg: out of concurrency order on purpose (c64 before c48) -> must sort to (c48, c64)
        pt("a64", "agg-6xTEP2", 64, 40.0, 1200.0, 12, 2000.0),  # y = 100.0
        pt("a48", "agg-6xTEP2", 48, 50.0, 600.0, 12, 1000.0),  # y = 50.0
        pt("c48", "cd-5d1p TEP2", 48, 55.0, 720.0, 12, 900.0),  # y = 60.0
        pt("c64", "cd-5d1p TEP2", 64, 45.0, 1080.0, 12, 1500.0),  # y = 90.0
    ]
    series = _pareto_series(runs)
    assert set(series) == {"agg-6xTEP2", "cd-5d1p TEP2"}
    # sorted by concurrency; tuple = (x=per_user, y=out/gpu, conc, ttft)
    assert series["agg-6xTEP2"] == [(50.0, 50.0, 48, 1000.0), (40.0, 100.0, 64, 2000.0)]
    assert series["cd-5d1p TEP2"] == [(55.0, 60.0, 48, 900.0), (45.0, 90.0, 64, 1500.0)]

    # a run missing x (per_user None) or y (active_gpus 0 -> per-gpu None) is dropped
    drop_x = pt("dx", "pd-3p3d TEP2", 48, None, 600.0, 12, 900.0)
    drop_x.output_tput_per_user = None
    drop_y = pt("dy", "pd-3p3d TEP2", 64, 50.0, 600.0, 0, 900.0)  # active_gpus=0 -> per-gpu None
    series2 = _pareto_series([drop_x, drop_y])
    assert "pd-3p3d TEP2" not in series2


def test_render_pareto_png_writes_nonempty_png(tmp_path):
    """render_pareto_png writes a non-empty PNG for >=2 serving modes x >=2 concurrencies."""
    from srtctl.analysis.bench_report import render_pareto_png

    def pt(job, topo, conc, per_user, out_tput, ttft):
        r = _make_run(job, topo, is_baseline=False, req_s=1.0, itl_p50=15.0)
        r.concurrency = conc
        r.output_tput_per_user = per_user
        r.output_tput = out_tput  # active_gpus=12 from the factory
        r.ttft_p50 = ttft
        return r

    runs = [
        pt("a48", "agg-6xTEP2", 48, 50.0, 600.0, 1000.0),
        pt("a64", "agg-6xTEP2", 64, 40.0, 1200.0, 2000.0),
        pt("c48", "cd-5d1p TEP2", 48, 55.0, 720.0, 900.0),
        pt("c64", "cd-5d1p TEP2", 64, 45.0, 1080.0, 1500.0),
    ]
    out = tmp_path / "report_pareto.png"
    ok = render_pareto_png(runs, out, "Pareto smoke")
    assert ok is True
    assert out.is_file()
    data = out.read_bytes()
    assert len(data) > 0
    assert data[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic


def test_parse_cd_snapshot_absent_is_na():
    lev = parse_cd_snapshot(None)
    assert isinstance(lev, CdLeverage)
    assert lev.snapshot_found is False
    assert lev.present is False


def test_parse_cd_snapshot_aggregates(tmp_path):
    snap = tmp_path / "kvbm_metrics_snapshot.json"
    snap.write_text(
        json.dumps(
            {
                # The hook wraps the fanout under "hub_fanout"; parser must unwrap it.
                "hub_fanout": {
                    "instances": {
                        "leaderA": {
                            "snapshot": {
                                "cd": {
                                    "prefill_decisions": {"local": 3, "remote": 8},
                                    "remote_prefill_tokens_total": 13824,
                                    "declined": {"budget": 2},
                                }
                            }
                        },
                        "leaderB": {
                            "snapshot": {
                                "cd": {
                                    "prefill_decisions": {"remote": 7},
                                    "remote_prefill_tokens_total": 11968,
                                }
                            }
                        },
                        "router": {"snapshot": {"cd": None}},  # non-CD instance
                    }
                },
            }
        )
    )
    lev = parse_cd_snapshot(snap)
    assert lev.snapshot_found is True
    assert lev.present is True
    assert lev.instances == 2  # router (cd: null) excluded
    assert lev.local_decisions == 3
    assert lev.remote_decisions == 15
    assert lev.remote_prefill_tokens == 13824 + 11968
    assert lev.declined_by_reason == {"budget": 2}
    assert abs(lev.remote_fraction - 15 / 18) < 1e-9


def test_parse_cd_snapshot_token_load_and_breaker_downgrades(tmp_path):
    """Token-load split (Q3 local / Q4 remote) + the breaker-downgrade wrinkle: a
    `remote_downgraded_breaker_hot` decision is a policy-Remote request that computed LOCALLY,
    so it must land in `downgrades` (NOT `remote_decisions`) and its tokens in `local_prefill_tokens`.
    The live snapshot schema for declines is `remote_prefill_declined`."""
    snap = tmp_path / "kvbm_metrics_snapshot.json"
    snap.write_text(
        json.dumps(
            {
                "hub_fanout": {
                    "instances": {
                        "decode": {
                            "snapshot": {
                                "cd": {
                                    "prefill_decisions": {
                                        "local": 100,
                                        "remote": 60,
                                        "remote_downgraded_breaker_hot": 40,
                                    },
                                    "remote_prefill_declined": {"breaker_hot": 40},
                                    "local_prefill_tokens_total": 800_000,
                                    "remote_prefill_tokens_total": 1_200_000,
                                }
                            }
                        },
                        "prefill": {
                            "snapshot": {
                                "cd": {
                                    "prefill_decisions": {},
                                    "prefill_computed_tokens_total": 1_200_000,
                                    "prefill_pulled_tokens_total": 9_000_000,
                                    "prefill_local_hit_tokens_total": 2_000_000,
                                }
                            }
                        },
                    }
                },
            }
        )
    )
    lev = parse_cd_snapshot(snap)
    # local_decisions counts by FINAL placement: policy-Local (100) + breaker-HOT downgrade (40)
    # = 140, so it pairs with local_prefill_tokens for a correct per-request average.
    assert lev.local_decisions == 140
    assert lev.remote_decisions == 60  # TRUE remote only — breaker-HOT computed locally
    assert lev.downgrades == {"remote_downgraded_breaker_hot": 40}  # diagnostic sub-count of local
    assert lev.declined_by_reason == {"breaker_hot": 40}
    assert lev.local_prefill_tokens == 800_000
    assert lev.remote_prefill_tokens == 1_200_000
    assert lev.prefill_computed_tokens == 1_200_000  # Q6 ≡ Q4
    # token-load remote share = 1.2M / 2.0M = 0.6, vs decision share 60/200 = 0.3
    assert abs(lev.remote_compute_fraction - 0.6) < 1e-9
    assert abs(lev.remote_fraction - 60 / 200) < 1e-9
    # prefill-side: pulled window, the cached slice (Q7), and the derived supplement
    assert lev.prefill_pulled_tokens == 9_000_000
    assert lev.prefill_local_hit_tokens == 2_000_000
    assert lev.prefill_pull_supplement == 7_000_000  # pulled − local-hit


def test_parse_cd_snapshot_decode_workers_only_is_not_present(tmp_path):
    # An aggregated run produces a snapshot with ONLY per-worker metrics (no hub,
    # no CD instances). It must NOT count as CD content => section 4 stays N/A.
    snap = tmp_path / "kvbm_metrics_snapshot.json"
    snap.write_text(
        json.dumps(
            {
                "decode_workers": {"http://10.0.0.1:7500/metrics": "vllm:... 1.0\n"},
                "_meta": {"hub_present": False},
            }
        )
    )
    lev = parse_cd_snapshot(snap)
    assert lev.snapshot_found is True  # the file existed
    assert lev.present is False  # but carried no CD content
    assert lev.instances == 0


def test_find_real_artifact_dir_excludes_warmup(tmp_path):
    base = tmp_path / "logs" / "artifacts"
    warmup = base / "warmup"
    real = base / "Model_sa_trace_c48_20260604_111159"
    for d in (warmup, real):
        d.mkdir(parents=True)
        (d / "profile_export_aiperf.json").write_text("{}")
    found = find_real_artifact_dir(tmp_path)
    assert found == real
    assert found.name != "warmup"


# ---------------------------------------------------------------------------
# Curated manifest + self-contained HTML
# ---------------------------------------------------------------------------

_MANIFEST_YAML = """
title: "KVBM Agg vs CD — test"
dataset: "semianalysis_cc_traces_weka_with_subagents_256k (470 traces)"
concurrency: 48
generated_note: "free-text note here"
changelog:
  - { date: "2026-06-04", note: "KV-routed agg baseline replaces round-robin; ~449 HOT downgrades." }
runs:
  - { job_id: 2187965, label: "agg 6xTEP2 (KV-routed)",   kind: baseline,           tpcb: false, note: "primary baseline" }
  - { job_id: 2187812, label: "agg 6xTEP2 (round-robin)", kind: baseline_reference, tpcb: false, note: "reference only" }
  - { job_id: 2187937, label: "CD 5d+1p (+TPCB)",         kind: cd,                 tpcb: true,  note: "TPCB ON" }
threshold_sweep:
  note: "sweep note text"
  runs:
    - { job_id: 2187937 }
    - { job_id: 2189847 }
    - { job_id: 2189850 }
pd_disagg:
  note: "pd note text"
  runs:
    - { job_id: 2190163, label: "3p3d" }
    - { job_id: 2190164, label: "2p4d" }
pareto:
  note: "pareto note text"
  runs:
    - { job_id: 2187965 }
    - { job_id: 2188128 }
    - { job_id: 2190163 }
"""


def test_parse_manifest(tmp_path):
    path = tmp_path / "latest_results.yaml"
    path.write_text(_MANIFEST_YAML)
    m = parse_manifest(path)
    assert m.title == "KVBM Agg vs CD — test"
    assert m.dataset.startswith("semianalysis_cc_traces_weka_with_subagents_256k")
    assert m.concurrency == 48
    assert m.generated_note == "free-text note here"
    # changelog newest-first; date + note preserved
    assert len(m.changelog) == 1
    assert m.changelog[0].date == "2026-06-04"
    assert "449 HOT" in m.changelog[0].note
    # runs ordered; job ids stringified; kind/tpcb parsed
    assert [r.job_id for r in m.runs] == ["2187965", "2187812", "2187937"]
    assert m.runs[0].kind == "baseline" and m.runs[0].is_baseline
    assert m.runs[1].kind == "baseline_reference" and m.runs[1].is_reference
    assert m.runs[2].kind == "cd" and m.runs[2].is_cd and m.runs[2].tpcb is True
    # baseline selection is by kind, not by order
    assert m.baseline_run().job_id == "2187965"
    # threshold-sweep axis parsed separately (ordered job ids + note); may overlap runs
    assert m.sweep_job_ids == ["2187937", "2189847", "2189850"]
    assert m.sweep_note == "sweep note text"
    # trad-P/D serving mode parsed separately (ordered ManifestRuns w/ explicit labels + note)
    assert [(r.job_id, r.label) for r in m.pd_disagg_runs] == [("2190163", "3p3d"), ("2190164", "2p4d")]
    assert m.pd_disagg_note == "pd note text"
    # pareto axis parsed separately (ordered job ids + note); may overlap runs
    assert m.pareto_job_ids == ["2187965", "2188128", "2190163"]
    assert m.pareto_note == "pareto note text"


def test_render_pd_disagg_valid_and_invalid_runs():
    from srtctl.analysis.bench_report import ManifestRun, render_pd_disagg

    good = _make_run("2190163", "agg-0xTEP2", is_baseline=False, req_s=1.783, itl_p50=16.4)
    good.kv = KvMetrics(block_size=1, prefix_hits=3561000)  # kv_reused_blocks=3561000 => consolidator engaged
    bad = _make_run("2190162", "agg-0xTEP2", is_baseline=False, req_s=0.0, itl_p50=0.0)
    bad.request_count = 4188
    bad.error_count = 4188  # all requests errored => INVALID
    bad.runtime_error = "Not Found×4188"
    paired = [
        (ManifestRun("2190162", "4p2d", "cd", False, ""), bad),
        (ManifestRun("2190163", "3p3d", "cd", False, ""), good),
    ]
    md = render_pd_disagg(paired, note="pd note")
    assert "## 8. Traditional P/D disaggregation" in md
    assert "small prefixes" not in md  # (sanity: not the sweep thesis)
    assert "3rd serving mode" in md  # the pd thesis
    assert "pd note" in md
    # the 4p2d invalid run is shown but flagged, not silently dropped
    assert "4p2d" in md and "INVALID" in md and "Not Found" in md
    # the valid 3p3d is named best (only valid run); kv_reused proves consolidator engaged
    assert "Best valid trad-P/D @ c48: 3p3d" in md
    assert "3561000" in md
    # invalid run sinks below the valid one (ordered by req/s desc)
    assert md.index("| 3p3d |") < md.index("| 4p2d |")


def test_render_threshold_sweep_table_and_thesis():
    from srtctl.analysis.bench_report import render_threshold_sweep

    def sweep_run(job, thr, req, brk_hot):
        r = _make_run(job, "cd-5d1p TEP2", is_baseline=False, req_s=req, itl_p50=15.0)
        r.condp_policy = thr
        r.cd = CdLeverage(
            instances=2,
            remote_decisions=1000 - thr // 4,  # falls as threshold rises
            local_prefill_tokens=8_000_000,
            remote_prefill_tokens=10_000_000,
            prefill_pulled_tokens=60_000_000,
            prefill_local_hit_tokens=36_000_000,  # 60% over-pull
            downgrades={"remote_downgraded_breaker_hot": brk_hot},
        )
        return r

    runs = [sweep_run("J1024", 1024, 1.942, 438), sweep_run("J8192", 8192, 1.986, 110)]
    md = render_threshold_sweep(runs, note="my sweep note")
    assert "## 7. Conditional-threshold sweep" in md
    assert "small prefills don't matter, the large ones do" in md  # the thesis
    assert "my sweep note" in md
    assert "@ concurrency 48" in md
    # rows ordered by threshold; req/s delta vs the lowest threshold (1024 = +0.0%)
    assert "| 1024 | J1024 | 1.942 (+0.0%)" in md
    assert "| 8192 | J8192 | 1.986 (+2.3%)" in md
    # over-pull% derived = local_hit / pulled = 36M/60M = 0.6
    assert "0.600" in md or "60%" in md
    # breaker-HOT count surfaced
    assert "438" in md and "110" in md


def _make_run(job_id: str, label_topo: str, *, is_baseline: bool, req_s: float, itl_p50: float) -> RunReport:
    """A minimal-but-complete RunReport for hermetic HTML tests (no on-disk artifacts)."""
    gp = GoodputResult(sla_ttft_ms=5000.0, sla_itl_ms=7.0, good=0, denom=100, median_itl_ms=itl_p50)
    return RunReport(
        job_id=job_id,
        run_dir=Path(f"/nonexistent/{job_id}"),
        job_name=f"job-{job_id}",
        is_baseline_candidate=is_baseline,
        dataset="semianalysis_cc_traces_weka_with_subagents_256k",
        concurrency=48,
        topology_label=label_topo,
        tp=2,
        agg_workers=6,
        gpus_per_agg=2,
        prefill_nodes=0,
        prefill_tp=0,
        active_gpus=12,
        provisioned_nodes=6,
        provisioned_gpus=12,
        image="/img/kvbm-prod.sqsh",
        condp_policy=None if is_baseline else 1024,
        max_num_tokens=262144,
        request_throughput=req_s,
        request_count=100,
        ttft_p50=1195.0,
        ttft_p90=9287.0,
        ttft_p99=12638.0,
        itl_p50=itl_p50,
        itl_p90=42.0,
        itl_p99=126.9,
        output_tput=994.6,
        output_tput_per_user=55.5,
        total_tput=145039.0,
        input_tput=140000.0,
        isl_avg=82818.0,
        osl_avg=572.0,
        effective_concurrency=27.0,
        theoretical_prefix_cache_hit=92.3,
        was_cancelled=False,
        error_count=0,
        runtime_error="",
        native_sla_ttft=5000.0,
        native_sla_itl=10.0,
        native_good_request_count=2,
        native_goodput_fraction=0.02,
        native_goodput_recompute=gp,
        user_goodput=gp,
        kv=KvMetrics(),
        cd=CdLeverage(),
    )


def test_render_html_contains_runs_changelog_badges(tmp_path):
    from srtctl.analysis.bench_report import Manifest, ManifestChangelogEntry

    base_mr = ManifestRun("2187965", "agg 6xTEP2 (KV-routed)", "baseline", False, "primary baseline")
    ref_mr = ManifestRun("2187812", "agg 6xTEP2 (round-robin)", "baseline_reference", False, "reference only")
    cd_mr = ManifestRun("2187937", "CD 5d+1p (+TPCB)", "cd", True, "TPCB ON")
    manifest = Manifest(
        title="KVBM Agg vs CD — test",
        dataset="semianalysis_cc_traces_weka_with_subagents_256k (470 traces)",
        concurrency=48,
        generated_note="free-text note here",
        changelog=[ManifestChangelogEntry("2026-06-04", "KV-routed agg baseline; ~449 HOT downgrades.")],
        runs=[base_mr, ref_mr, cd_mr],
        source_path=tmp_path / "latest_results.yaml",
    )
    paired = [
        (base_mr, _make_run("2187965", "agg-6xTEP2", is_baseline=True, req_s=1.739, itl_p50=17.2)),
        (ref_mr, _make_run("2187812", "agg-6xTEP2", is_baseline=True, req_s=1.439, itl_p50=19.5)),
        (cd_mr, _make_run("2187937", "cd-5d1p TEP2", is_baseline=False, req_s=1.970, itl_p50=15.0)),
    ]
    doc = render_html(manifest, paired, ttft_sla_ms=5000.0, itl_sla_ms=7.0)

    # Self-contained: no external/CDN assets.
    for forbidden in ("<script src", "<link ", "cdn.", "@import", " src="):
        assert forbidden not in doc, f"HTML must be self-contained; found {forbidden!r}"

    # All three manifest LABELS appear (the two agg runs are otherwise indistinguishable).
    assert "agg 6xTEP2 (KV-routed)" in doc
    assert "agg 6xTEP2 (round-robin)" in doc
    assert "CD 5d+1p (+TPCB)" in doc

    # Changelog note substring + date.
    assert "449 HOT downgrades" in doc
    assert "2026-06-04" in doc

    # Badges: TPCB on the 5d+1p column, REFERENCE marker on round-robin, BASELINE on the baseline.
    assert "badge tpcb" in doc and "TPCB</span>" in doc
    assert "badge ref" in doc and "REFERENCE</span>" in doc
    assert "BASELINE</span>" in doc

    # Baseline column carries NO delta; the CD column carries a signed % delta.
    # The baseline req/s cell is 1.739 with no <span class='delta'> attached.
    assert "1.739</td>" in doc  # baseline cell, no delta span
    assert "+13.3%" in doc  # CD 5d+1p req/s delta vs baseline 1.739
    # Round-robin is the reference and is excluded from winners → never crowned.
    assert "<td class='winner'>agg 6xTEP2 (round-robin)</td>" not in doc
