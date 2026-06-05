# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Never-fail contract tests for the post-benchmark KVBM metrics snapshot hook.

The hook in :class:`BenchmarkStageMixin` must NEVER raise or block teardown,
even when the hub is absent (aggregated runs) or the scrape times out. These
tests poke the hook's helpers on a minimal stand-in object.
"""

from pathlib import Path
from types import SimpleNamespace

from srtctl.cli.mixins.benchmark_stage import (
    KVBM_METRICS_SNAPSHOT_FILENAME,
    BenchmarkStageMixin,
)


class _Harness(BenchmarkStageMixin):
    """Minimal object exposing just what the snapshot hook touches."""

    def __init__(self, log_dir: Path, hub_present: bool):
        backend = SimpleNamespace(kvbm_hub=object() if hub_present else None)
        self.config = SimpleNamespace(backend=backend)
        self.runtime = SimpleNamespace(
            log_dir=log_dir,
            infra_node_ip="10.255.255.1",  # unroutable -> GET fails fast
            network_interface="eth0",
            job_id="99999",
        )
        self._backend_processes: list = []

    @property
    def endpoints(self):  # pragma: no cover - unused here
        return []

    @property
    def backend_processes(self):
        return self._backend_processes


def test_snapshot_never_fails_with_no_hub(tmp_path, monkeypatch):
    # Aggregated run: no hub, no workers. Hook must be a clean no-op (no file).
    h = _Harness(tmp_path, hub_present=False)
    # Force any accidental HTTP to fail instantly rather than hang the test.
    monkeypatch.setattr(
        "srtctl.cli.mixins.benchmark_stage.urllib.request.urlopen",
        lambda *a, **k: (_ for _ in ()).throw(OSError("no network")),
    )
    h._snapshot_kvbm_metrics()  # must not raise
    assert not (tmp_path / KVBM_METRICS_SNAPSHOT_FILENAME).exists()


def test_snapshot_swallows_scrape_error_with_hub(tmp_path, monkeypatch):
    # Hub present but every GET fails: hook must swallow and write nothing.
    h = _Harness(tmp_path, hub_present=True)
    monkeypatch.setattr(
        "srtctl.cli.mixins.benchmark_stage.urllib.request.urlopen",
        lambda *a, **k: (_ for _ in ()).throw(OSError("connection refused")),
    )
    h._snapshot_kvbm_metrics()  # must not raise
    assert not (tmp_path / KVBM_METRICS_SNAPSHOT_FILENAME).exists()


def test_snapshot_writes_when_hub_responds(tmp_path, monkeypatch):
    h = _Harness(tmp_path, hub_present=True)

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"instances": {"leaderA": {"snapshot": {"cd": {"prefill_decisions": {"remote": 5}}}}}}'

    monkeypatch.setattr(
        "srtctl.cli.mixins.benchmark_stage.urllib.request.urlopen",
        lambda *a, **k: _Resp(),
    )
    h._snapshot_kvbm_metrics()
    out = tmp_path / KVBM_METRICS_SNAPSHOT_FILENAME
    assert out.exists()
    import json

    data = json.loads(out.read_text())
    assert "hub_fanout" in data
    assert data["_meta"]["hub_present"] is True


def test_resolve_artifacts_dir_prefers_real_run(tmp_path):
    h = _Harness(tmp_path, hub_present=True)
    artifacts = tmp_path / "artifacts"
    (artifacts / "warmup").mkdir(parents=True)
    real = artifacts / "Model_sa_trace_c48_20260604_111159"
    real.mkdir(parents=True)
    assert h._resolve_artifacts_dir() == real


def test_resolve_artifacts_dir_falls_back_to_log_dir(tmp_path):
    h = _Harness(tmp_path, hub_present=True)
    # No artifacts dir at all -> fall back to log_dir.
    assert h._resolve_artifacts_dir() == tmp_path
