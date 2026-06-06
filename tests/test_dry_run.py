# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for dry-run config details display (mounts, env vars)."""

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from srtctl.backends.vllm import KvbmHubConfig, VLLMProtocol, VLLMServerConfig
from srtctl.cli.submit import show_config_details
from srtctl.core.schema import SrtConfig
from srtctl.ports import KV_EVENTS_PORT_BASE

# Minimal valid config that all tests build on
BASE_CONFIG = {
    "name": "test-job",
    "model": {
        "path": "/models/test-model",
        "container": "test-container.sqsh",
        "precision": "fp8",
    },
    "resources": {
        "gpu_type": "h100",
        "gpus_per_node": 8,
        "prefill_nodes": 1,
        "decode_nodes": 1,
        "prefill_workers": 1,
        "decode_workers": 1,
    },
    "benchmark": {"type": "manual"},
}


def _make_config(overrides: dict | None = None) -> SrtConfig:
    """Build an SrtConfig from BASE_CONFIG with optional overrides merged in."""
    data = {**BASE_CONFIG}
    if overrides:
        for key, value in overrides.items():
            if isinstance(value, dict) and key in data and isinstance(data[key], dict):
                data[key] = {**data[key], **value}
            else:
                data[key] = value
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        tmp_path = Path(f.name)
    return SrtConfig.from_yaml(tmp_path)


class TestDryRunMounts:
    """Test that container mounts from all sources appear in dry-run output."""

    def test_builtin_mounts_always_shown(self, capsys):
        config = _make_config()
        show_config_details(config)
        output = capsys.readouterr().out
        assert "/model" in output
        assert "/logs" in output

    def test_extra_mount_from_recipe(self, capsys):
        config = _make_config({"extra_mount": ["/data/custom:/custom", "/shared/cache:/cache"]})
        show_config_details(config)
        output = capsys.readouterr().out
        assert "/data/custom" in output
        assert "/custom" in output
        assert "/shared/cache" in output
        assert "/cache" in output
        assert "recipe" in output

    def test_extra_mount_expands_env_and_user_in_dry_run(self, capsys):
        with patch.dict(os.environ, {"SRT_EXTRA_ROOT": "/expanded/extra", "HOME": "/home/tester"}):
            config = _make_config({"extra_mount": ["$SRT_EXTRA_ROOT:/extra", "~/cache:/cache"]})
            show_config_details(config)
        output = capsys.readouterr().out
        assert "/expanded/extra" in output
        assert "/home/tester/cache" in output
        assert "$SRT_EXTRA_ROOT" not in output
        assert "~/cache" not in output

    def test_cluster_mounts_from_srtslurm_yaml(self, capsys):
        cluster_mounts = {"/shared/datasets": "/datasets", "/shared/models": "/models"}
        with patch("srtctl.cli.submit.get_srtslurm_setting", return_value=cluster_mounts):
            config = _make_config()
            show_config_details(config)
        output = capsys.readouterr().out
        assert "/shared/datasets" in output
        assert "/datasets" in output
        assert "srtslurm.yaml" in output

    def test_mounts_from_both_cluster_and_recipe(self, capsys):
        """Mounts from srtslurm.yaml AND recipe extra_mount should both appear."""
        cluster_mounts = {"/cluster/data": "/data"}

        def mock_setting(key, default=None):
            if key == "default_mounts":
                return cluster_mounts
            return default

        with patch("srtctl.cli.submit.get_srtslurm_setting", side_effect=mock_setting):
            config = _make_config({"extra_mount": ["/recipe/models:/models"]})
            show_config_details(config)
        output = capsys.readouterr().out
        assert "/cluster/data" in output
        assert "srtslurm.yaml" in output
        assert "/recipe/models" in output
        assert "recipe" in output

    def test_no_extra_mounts_only_builtins(self, capsys):
        config = _make_config()
        show_config_details(config)
        output = capsys.readouterr().out
        assert "/model" in output
        assert "recipe" not in output


class TestDryRunEnvironment:
    """Test that environment variables from all levels appear in dry-run output."""

    def test_global_environment(self, capsys):
        config = _make_config({"environment": {"NCCL_SOCKET_IFNAME": "eth0", "MY_VAR": "hello"}})
        show_config_details(config)
        output = capsys.readouterr().out
        assert "NCCL_SOCKET_IFNAME" in output
        assert "eth0" in output
        assert "MY_VAR" in output
        assert "global" in output

    def test_backend_prefill_decode_environment(self, capsys):
        config = _make_config(
            {
                "backend": {
                    "type": "sglang",
                    "prefill_environment": {
                        "TORCH_DISTRIBUTED_DEFAULT_TIMEOUT": "1800",
                        "PYTHONUNBUFFERED": "1",
                    },
                    "decode_environment": {
                        "SGLANG_ENABLE_FLASHINFER_GEMM": "1",
                    },
                },
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "TORCH_DISTRIBUTED_DEFAULT_TIMEOUT" in output
        assert "1800" in output
        assert "prefill" in output
        assert "SGLANG_ENABLE_FLASHINFER_GEMM" in output
        assert "decode" in output

    def test_global_and_backend_env_together(self, capsys):
        """Global environment AND backend per-mode env should both appear."""
        config = _make_config(
            {
                "environment": {"GLOBAL_VAR": "global_val"},
                "backend": {
                    "type": "sglang",
                    "prefill_environment": {"PREFILL_VAR": "prefill_val"},
                },
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "GLOBAL_VAR" in output
        assert "global" in output
        assert "PREFILL_VAR" in output
        assert "prefill" in output

    def test_no_environment_shows_message(self, capsys):
        config = _make_config()
        show_config_details(config)
        output = capsys.readouterr().out
        assert "No custom environment variables configured" in output

    def test_trtllm_backend_environment(self, capsys):
        config = _make_config(
            {
                "backend": {
                    "type": "trtllm",
                    "prefill_environment": {
                        "TRTLLM_ENABLE_PDL": "1",
                        "NCCL_GRAPH_MIXING_SUPPORT": "0",
                    },
                    "decode_environment": {
                        "TRTLLM_SERVER_DISABLE_GC": "1",
                    },
                },
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "TRTLLM_ENABLE_PDL" in output
        assert "prefill" in output
        assert "TRTLLM_SERVER_DISABLE_GC" in output
        assert "decode" in output

    def test_custom_benchmark_environment(self, capsys):
        config = _make_config(
            {
                "benchmark": {
                    "type": "custom",
                    "command": "python /bench/run.py",
                    "env": {"BENCH_FOO": "bar"},
                }
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "BENCH_FOO" in output
        assert "benchmark" in output


class TestDryRunSrunOptions:
    """Test that srun options appear in dry-run output."""

    def test_srun_options_shown(self, capsys):
        config = _make_config({"srun_options": {"export": "ALL", "cpu-bind": "none"}})
        show_config_details(config)
        output = capsys.readouterr().out
        assert "--export=ALL" in output
        assert "--cpu-bind=none" in output

    def test_no_srun_options_no_output(self, capsys):
        config = _make_config()
        show_config_details(config)
        output = capsys.readouterr().out
        assert "srun options" not in output


class TestDryRunExecutionExtensions:
    """Test custom benchmark and telemetry details display."""

    def test_custom_benchmark_details_shown(self, capsys):
        config = _make_config(
            {
                "benchmark": {
                    "type": "custom",
                    "command": "python /bench/run.py",
                    "container_image": "nvcr.io/nvidia/python:3.11",
                }
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "Execution Extensions" in output
        assert "container_image" in output
        assert "nvcr.io/nvidia/python:3.11" in output

    def test_telemetry_details_shown(self, capsys):
        config = _make_config(
            {
                "telemetry": {
                    "enabled": True,
                    "container_image": "telemetry:latest",
                    "dcgm_exporter": {"container_image": "dcgm:latest", "port": 9401},
                    "node_exporter": {"container_image": "node:latest", "port": 9101},
                }
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "telemetry" in output
        assert "scraper" in output
        assert "storage_subdir" in output

    def test_mooncake_kv_store_details_shown(self, capsys):
        """mooncake_kv_store should appear in env vars and execution extensions."""
        config = _make_config(
            {
                "backend": {
                    "type": "sglang",
                    "mooncake_kv_store": {
                        "container": "nvcr.io/nvidia/mooncake:latest",
                        "env": {
                            "MOONCAKE_PROTOCOL": "rdma",
                            "MOONCAKE_GLOBAL_SEGMENT_SIZE": "4gb",
                        },
                    },
                    "sglang_config": {
                        "prefill": {"disaggregation-transfer-backend": "mooncake"},
                        "decode": {"disaggregation-transfer-backend": "mooncake"},
                    },
                }
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        # Env table shows mooncake-scoped env vars
        assert "mooncake" in output
        assert "MOONCAKE_PROTOCOL" in output
        assert "rdma" in output
        assert "MOONCAKE_GLOBAL_SEGMENT_SIZE" in output
        # Execution extensions shows master + container
        assert "nvcr.io/nvidia/mooncake:latest" in output
        assert "master_port" in output

    def test_mooncake_kv_store_no_container_shows_default(self, capsys):
        """mooncake_kv_store without explicit container falls back to job container label."""
        config = _make_config(
            {
                "backend": {
                    "type": "sglang",
                    "mooncake_kv_store": {"env": {"MOONCAKE_PROTOCOL": "tcp"}},
                    "sglang_config": {
                        "prefill": {"disaggregation-transfer-backend": "mooncake"},
                        "decode": {"disaggregation-transfer-backend": "mooncake"},
                    },
                }
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "<job container>" in output
        assert "MOONCAKE_PROTOCOL" in output


class TestDryRunKvbmHub:
    """KVBM hub config surfaces in dry-run and renders the decode connector."""

    def test_kvbm_hub_details_shown(self, capsys):
        """kvbm_hub should appear in the execution-extensions panel."""
        config = _make_config(
            {
                "backend": {
                    "type": "vllm",
                    "kvbm_hub": {
                        "block_layout": "universal",
                        "min_remote_prefill_tokens": 256,
                        "host_cache_gb": 400.0,
                    },
                    "vllm_config": {"aggregated": {"block-size": 64, "max-model-len": 40960}},
                }
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "kvbm_hub" in output
        assert "discovery_port" in output
        assert "min_remote_prefill_tokens" in output
        assert "universal" in output
        assert "indexer,p2p,disagg" in output

    def test_kvbm_hub_connector_built(self):
        """The static decode connector matches the validated kvbm v2 config; a
        prefill is a pure CD target (remote_search explicitly OFF, no threshold)."""
        backend = VLLMProtocol(
            kvbm_hub=KvbmHubConfig(block_layout="universal", min_remote_prefill_tokens=256, host_cache_gb=400.0),
            vllm_config=VLLMServerConfig(aggregated={"block-size": 64, "max-model-len": 40960}),
        )
        assert backend.kvbm_block_size() == 64
        assert backend.kvbm_max_seq_len() == 40960

        cfg = json.loads(backend.build_kvbm_hub_connector("decode", "http://infra0:1337"))
        assert cfg["kv_connector"] == "DynamoConnector"
        assert cfg["kv_connector_module_path"] == "kvbm.v2.vllm.connector"
        leader = cfg["kv_connector_extra_config"]["leader"]
        assert leader["hub"]["url"] == "http://infra0:1337"
        assert leader["hub"]["features"] == ["indexer", "p2p", "disagg"]
        assert leader["disagg"] == {"role": "decode", "min_remote_prefill_tokens": 256}
        assert leader["remote_search"] == {"enabled": True}
        assert cfg["kv_connector_extra_config"]["default"]["block_layout"] == "universal"
        assert leader["max_seq_len"] == 40960

        prefill = json.loads(backend.build_kvbm_hub_connector("prefill", "http://infra0:1337"))
        pleader = prefill["kv_connector_extra_config"]["leader"]
        assert pleader["disagg"] == {"role": "prefill"}
        # prefill carries NO threshold (decode-only) and remote_search EXPLICITLY off
        # (provable in the static config; default decode=on / prefill=off).
        assert "min_remote_prefill_tokens" not in pleader["disagg"]
        assert pleader["remote_search"] == {"enabled": False}

    def test_kvbm_hub_connector_overflow_budget(self):
        """max_inflight_remote_prefill_tokens arms decode-side B-GNMT downgrade-to-local;
        absent by default (connector usize::MAX => inert), present iff set in the recipe."""
        # default: key omitted => connector default usize::MAX => feature inert
        default_backend = VLLMProtocol(
            kvbm_hub=KvbmHubConfig(min_remote_prefill_tokens=1024),
            vllm_config=VLLMServerConfig(aggregated={"block-size": 64, "max-model-len": 40960}),
        )
        dleader = json.loads(default_backend.build_kvbm_hub_connector("decode", "http://infra0:1337"))[
            "kv_connector_extra_config"
        ]["leader"]
        assert "max_inflight_remote_prefill_tokens" not in dleader["disagg"]

        # finite budget => key surfaces on the DECODE connector only
        backend = VLLMProtocol(
            kvbm_hub=KvbmHubConfig(min_remote_prefill_tokens=1024, max_inflight_remote_prefill_tokens=131072),
            vllm_config=VLLMServerConfig(aggregated={"block-size": 64, "max-model-len": 40960}),
        )
        leader = json.loads(backend.build_kvbm_hub_connector("decode", "http://infra0:1337"))[
            "kv_connector_extra_config"
        ]["leader"]
        assert leader["disagg"]["max_inflight_remote_prefill_tokens"] == 131072
        assert leader["disagg"]["min_remote_prefill_tokens"] == 1024
        # the prefill connector never carries the decode-only overflow budget
        pleader = json.loads(backend.build_kvbm_hub_connector("prefill", "http://infra0:1337"))[
            "kv_connector_extra_config"
        ]["leader"]
        assert "max_inflight_remote_prefill_tokens" not in pleader["disagg"]

    def test_kvbm_hub_connector_omits_breaker_knobs(self):
        """The CD circuit breaker is configured on the HUB CLI, NOT on the
        connector. The cd_breaker_* knobs must NEVER appear in the connector's
        --kv-transfer-config JSON, even when opted in (the connector reads only
        the hub-pushed tier at runtime). Asserted on decode and prefill."""
        backend = VLLMProtocol(
            kvbm_hub=KvbmHubConfig(
                min_remote_prefill_tokens=1024,
                cd_breaker_enabled=True,
                cd_breaker_warm_high=0.4,
                cd_breaker_hot_high=0.1,
                cd_breaker_clear_low=0.8,
                cd_breaker_clear_debounce_ticks=5,
            ),
            vllm_config=VLLMServerConfig(aggregated={"block-size": 64, "max-model-len": 40960}),
        )
        for role in ("decode", "prefill"):
            leader = json.loads(backend.build_kvbm_hub_connector(role, "http://infra0:1337"))[
                "kv_connector_extra_config"
            ]["leader"]
            for key in (
                "cd_breaker_enabled",
                "cd_breaker_warm_high",
                "cd_breaker_hot_high",
                "cd_breaker_clear_low",
                "cd_breaker_queue_depth_warm",
                "cd_breaker_queue_depth_hot",
                "cd_breaker_clear_debounce_ticks",
            ):
                assert key not in leader["disagg"], f"{key} must never appear in the {role} connector"

    def test_kvbm_hub_launch_command_circuit_breaker(self):
        """The hub LAUNCH command carries --cd-breaker (+ watermark/debounce
        flags) ONLY when cd_breaker_enabled is True; omitting it (or False) =>
        no --cd-breaker => the hub never constructs the breaker (byte-identical
        to today). The queue-depth knobs have NO hub flag, so they are never
        emitted. The breaker requires --prefill-router, which is always passed."""
        from srtctl.cli.do_sweep import build_kvbm_hub_command

        common = dict(
            block_size=64,
            max_seq_len=40960,
            infra_node="infra0",
            discovery_port=1337,
            control_port=8337,
            velo_port=4317,
        )

        # OFF (default): no --cd-breaker anywhere in the argv.
        off_cmd = build_kvbm_hub_command(KvbmHubConfig(min_remote_prefill_tokens=1024), **common)
        assert "--prefill-router" in off_cmd, "prefill-router is always passed (breaker gate)"
        assert "--cd-breaker" not in off_cmd
        assert not any(a.startswith("--cd-breaker") for a in off_cmd), "no breaker flags when OFF"

        # Explicit False is also OFF.
        false_cmd = build_kvbm_hub_command(
            KvbmHubConfig(min_remote_prefill_tokens=1024, cd_breaker_enabled=False), **common
        )
        assert "--cd-breaker" not in false_cmd

        # ON: --cd-breaker plus each set watermark/debounce flag, in order.
        on_cmd = build_kvbm_hub_command(
            KvbmHubConfig(
                min_remote_prefill_tokens=1024,
                cd_breaker_enabled=True,
                cd_breaker_warm_high=0.4,
                cd_breaker_hot_high=0.1,
                cd_breaker_clear_low=0.8,
                cd_breaker_clear_debounce_ticks=5,
                # queue-depth set but NOT routable (no hub flag) — must be ignored.
                cd_breaker_queue_depth_warm=64,
                cd_breaker_queue_depth_hot=256,
            ),
            **common,
        )
        assert "--cd-breaker" in on_cmd
        assert "--prefill-router" in on_cmd
        # Each flag immediately precedes its value (argv pair).
        for flag, val in (
            ("--cd-breaker-warm-high", "0.4"),
            ("--cd-breaker-hot-high", "0.1"),
            ("--cd-breaker-clear-low", "0.8"),
            ("--cd-breaker-clear-debounce-ticks", "5"),
        ):
            assert flag in on_cmd, f"{flag} missing"
            assert on_cmd[on_cmd.index(flag) + 1] == val, f"{flag} value"
        # The queue-depth axis has no hub CLI flag — never emitted.
        assert not any("queue-depth" in a for a in on_cmd), "queue-depth has no hub flag"

    def test_kvbm_consolidator_agg_baseline(self):
        """The hub-less agg KV-routing baseline reuses the EXACT CD decode consolidator
        treatment (force prefix-caching ON + emit per-worker KV events — the consolidator's
        source), so agg routes iso with CD. Shared helper == identical methodology."""
        backend = VLLMProtocol(
            kvbm_consolidator=True,
            connector=(
                '{"kv_connector":"DynamoConnector","kv_role":"kv_both",'
                '"kv_connector_module_path":"kvbm.v2.vllm.connector","kv_connector_extra_config":{}}'
            ),
        )
        assert backend.kvbm_consolidator is True
        assert backend.kvbm_hub is None  # hub-less by construction
        assert VLLMProtocol().kvbm_consolidator is False  # default OFF (round-robin agg unchanged)

        # The shared helper applies the CD methodology to an agg worker.
        class _Proc:
            kv_events_port = 5200

        cmd: list[str] = []
        cfg: dict = {"no-enable-prefix-caching": True}
        backend._enable_kvbm_consolidator(cmd, cfg, _Proc())
        assert cfg.get("enable-prefix-caching") is True  # consolidator hard-requires it
        assert "no-enable-prefix-caching" not in cfg  # a recipe opt-out is dropped
        assert "--kv-events-config" in cmd
        kv_cfg = cmd[cmd.index("--kv-events-config") + 1]
        assert '"enable_kv_cache_events": true' in kv_cfg and "5200" in kv_cfg

        # No kv_events_port => no kv-events flag, but prefix-caching is still forced.
        class _ProcNoPort:
            kv_events_port = None

        cmd2: list[str] = []
        cfg2: dict = {}
        backend._enable_kvbm_consolidator(cmd2, cfg2, _ProcNoPort())
        assert cfg2.get("enable-prefix-caching") is True
        assert "--kv-events-config" not in cmd2

    def test_kvbm_consolidator_prefill_disagg_pd_connector(self):
        """TRADITIONAL (native dynamo) P/D disagg with KVBM on the PREFILL workers:
        kvbm_consolidator=True (hub-less) now also applies the consolidator treatment to a
        mode=='prefill' worker (force prefix-caching ON + emit per-worker KV events), while a
        plain-vLLM decode worker stays untouched (no kvbm, no kv-events, no forced prefix
        caching). The prefill connector is a PdConnector wrapping DynamoConnector(+leader) +
        NixlConnector so KV still transfers prefill->decode over NIXL."""
        import json
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from srtctl.core.topology import Process

        pd_connector = json.dumps(
            {
                "kv_connector": "PdConnector",
                "kv_role": "kv_both",
                "kv_connector_module_path": "kvbm.v2.vllm.connector",
                "kv_connector_extra_config": {
                    "connectors": [
                        {
                            "kv_connector": "DynamoConnector",
                            "kv_role": "kv_both",
                            "kv_connector_module_path": "kvbm.v2.vllm.connector",
                            "kv_connector_extra_config": {
                                "default": {"block_layout": "operational"},
                                "leader": {"cache": {"host": {"cache_size_gb": 160.0}}, "max_seq_len": 262144},
                                "worker": {"nixl": {"backends": {"UCX": {}, "POSIX": {}}}},
                            },
                        },
                        {"kv_connector": "NixlConnector", "kv_role": "kv_both"},
                    ]
                },
            }
        )
        backend = VLLMProtocol(
            kvbm_consolidator=True,
            vllm_config=VLLMServerConfig(
                prefill={"connector": pd_connector, "block-size": 64, "max-model-len": 262144},
                decode={"connector": "nixl"},
            ),
        )

        def _cmd(mode: str) -> list[str]:
            proc = Process(
                node="node0",
                gpu_indices=frozenset([0, 1]),
                sys_port=8081,
                http_port=30000,
                endpoint_mode=mode,
                endpoint_index=0,
                node_rank=0,
                kv_events_port=5200,
            )
            rt = MagicMock()
            rt.model_path = Path("/model")
            rt.is_hf_model = False
            with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
                return backend.build_worker_command(process=proc, endpoint_processes=[proc], runtime=rt)

        # PREFILL: gets the PdConnector + consolidator wiring.
        p = _cmd("prefill")
        assert "--disaggregation-mode" in p and p[p.index("--disaggregation-mode") + 1] == "prefill"
        assert "--kv-transfer-config" in p
        ktc = json.loads(p[p.index("--kv-transfer-config") + 1])
        assert ktc["kv_connector"] == "PdConnector"
        child_kinds = {c["kv_connector"] for c in ktc["kv_connector_extra_config"]["connectors"]}
        assert child_kinds == {"DynamoConnector", "NixlConnector"}
        assert "--enable-prefix-caching" in p  # consolidator hard-requires it on prefill
        assert "--kv-events-config" in p  # the consolidator's KV-events source

        # DECODE: plain vLLM NIXL — NO kvbm consolidator wiring.
        d = _cmd("decode")
        assert "--disaggregation-mode" in d and d[d.index("--disaggregation-mode") + 1] == "decode"
        dktc = json.loads(d[d.index("--kv-transfer-config") + 1])
        assert dktc["kv_connector"] == "NixlConnector"
        assert "--kv-events-config" not in d  # decode does NOT publish kvbm KV events
        assert "--enable-prefix-caching" not in d  # not forced on the plain decode worker

    def test_kvbm_consolidator_gets_per_worker_zmq_ports(self):
        """The hub-less consolidator path MUST get the per-worker DYN_KVBM_LEADER_ZMQ_PUB_PORT
        offset (like the hub path), else 2 bin-packed agg workers/node collide on the default
        56001 consolidator egress port and EngineCore init fails (job 2187950)."""

        class _Proc:
            def __init__(self, kvp):
                self.kv_events_port = kvp
                self.nixl_port = None

        backend = VLLMProtocol(
            kvbm_consolidator=True,
            connector='{"kv_connector_module_path":"kvbm.v2.vllm.connector"}',
        )
        e0 = backend.get_process_environment(_Proc(KV_EVENTS_PORT_BASE))
        e1 = backend.get_process_environment(_Proc(KV_EVENTS_PORT_BASE + 1))
        assert "DYN_KVBM_LEADER_ZMQ_PUB_PORT" in e0 and "DYN_KVBM_LEADER_ZMQ_PUB_PORT" in e1
        # distinct per co-located worker => no collision
        assert e0["DYN_KVBM_LEADER_ZMQ_PUB_PORT"] != e1["DYN_KVBM_LEADER_ZMQ_PUB_PORT"]
        assert e0["DYN_KVBM_LEADER_ZMQ_ACK_PORT"] != e1["DYN_KVBM_LEADER_ZMQ_ACK_PORT"]
        # a plain agg (no consolidator, no hub) is unchanged — no kvbm ZMQ ports.
        plain = VLLMProtocol().get_process_environment(_Proc(KV_EVENTS_PORT_BASE))
        assert "DYN_KVBM_LEADER_ZMQ_PUB_PORT" not in plain

    def test_kvbm_prefill_aside_reserves_extra_nodes(self):
        """kvbm_prefill_nodes folds into total_nodes (on top of the decode plane)
        without flipping the deployment into dynamo-disaggregated mode."""
        config = _make_config(
            {
                "resources": {
                    "gpu_type": "h100",
                    "gpus_per_node": 8,
                    "agg_nodes": 1,
                    "agg_workers": 2,
                    "kvbm_prefill_nodes": 1,
                    "kvbm_prefill_tp": 2,
                    # Clear the disagg defaults from BASE_CONFIG so this is an agg deployment.
                    "prefill_nodes": None,
                    "decode_nodes": None,
                    "prefill_workers": None,
                    "decode_workers": None,
                },
                "backend": {"type": "vllm", "kvbm_hub": {}},
            }
        )
        assert config.resources.is_disaggregated is False
        assert config.resources.total_nodes == 1 + 1  # agg_nodes + kvbm_prefill_nodes

    def test_kvbm_prefill_command_built(self):
        """The prefill aside launches kvbm.vllm.prefill (NOT dynamo.vllm) with the
        role=prefill connector and the requested TP/EP."""
        backend = VLLMProtocol(
            kvbm_hub=KvbmHubConfig(),
            vllm_config=VLLMServerConfig(
                aggregated={"block-size": 64, "max-model-len": 40960, "enable-expert-parallel": True}
            ),
        )
        runtime = SimpleNamespace(model_path=Path("Qwen/Qwen3-235B-A22B-NVFP4"), is_hf_model=True)
        cmd = backend.build_kvbm_prefill_command(runtime, tp=4, hub_url="http://infra0:1337")
        assert cmd[:3] == ["python3", "-m", "kvbm.vllm.prefill"]
        assert "--tensor-parallel-size" in cmd and cmd[cmd.index("--tensor-parallel-size") + 1] == "4"
        assert "--enable-expert-parallel" in cmd
        kv = json.loads(cmd[cmd.index("--kv-transfer-config") + 1])
        assert kv["kv_connector_extra_config"]["leader"]["disagg"]["role"] == "prefill"

    def test_kvbm_recipe_file_wires_local_wheel_install(self):
        """The shipped recipe installs dynamo+kvbm from LOCAL wheels (setup_script)
        and disables srt-slurm's PyPI installer (dynamo.install=false), so the dev
        kvbm image (no `dynamo` pkg) is provisioned without pulling ai-dynamo 0.8.0."""
        recipe = Path(__file__).resolve().parents[1] / "recipes/vllm/kvbm/agg-kv-router-hub.yaml"
        config = SrtConfig.from_yaml(recipe)
        assert config.setup_script == "kvbm-dev-install.sh"
        assert config.dynamo.install is False
        # the setup_script must actually exist where it'll be staged (/configs)
        assert (Path(__file__).resolve().parents[1] / "configs" / config.setup_script).is_file()


class TestDryRunHetJobs:
    """Het structure panel appears only when het is enabled."""

    def test_het_panel_rendered_when_enabled(self, capsys):
        config = _make_config(
            {
                "resources": {
                    "gpu_type": "gb200",
                    "gpus_per_node": 4,
                    "prefill_nodes": 12,
                    "decode_nodes": 10,
                    "prefill_workers": 12,
                    "decode_workers": 10,
                    "het_jobs": True,
                },
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "Heterogeneous Job" in output
        assert "prefill" in output
        assert "decode" in output

    def test_het_panel_hidden_when_disabled(self, capsys):
        """No het panel when het_jobs is unset (recipe default)."""
        config = _make_config()
        show_config_details(config)
        output = capsys.readouterr().out
        assert "Heterogeneous Job" not in output

    def test_het_panel_shows_infra_folded_into_prefill(self, capsys):
        config = _make_config(
            {
                "resources": {
                    "gpu_type": "gb200",
                    "gpus_per_node": 4,
                    "prefill_nodes": 12,
                    "decode_nodes": 10,
                    "prefill_workers": 12,
                    "decode_workers": 10,
                    "het_jobs": True,
                },
                "infra": {"etcd_nats_dedicated_node": True},
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "Heterogeneous Job" in output
        assert "first node" in output  # infra note on the prefill row


def _binpack_placement(config):
    """Mirror do_sweep's endpoint + aside placement for a non-het vLLM config.

    Returns (total_nodes, list[(label, node, gpu_indices, ports_dict)]) where
    ports_dict holds the TCP ports each process binds (nixl/kv/zmq). Used to
    assert per-node GPU + port disjointness at the object level (dry-run does
    not print per-worker GPU/port assignments, and the aside is carved at
    runtime in do_sweep, so this is the only place to verify them).
    """
    from srtctl.core.topology import (
        NodePortAllocator,
        compute_aside_placement,
        endpoints_to_processes,
    )
    from srtctl.ports import KVBM_ZMQ_PORT_BASE, VLLM_NIXL_PORT_BASE

    r = config.resources
    backend = config.backend
    # backend.allocate_endpoints internally applies should_colocate_prefill_decode.
    nodes = tuple(f"node{i}" for i in range(config.total_nodes))
    eps = backend.allocate_endpoints(
        num_prefill=r.num_prefill,
        num_decode=r.num_decode,
        num_agg=r.num_agg,
        gpus_per_prefill=r.gpus_per_prefill,
        gpus_per_decode=r.gpus_per_decode,
        gpus_per_agg=r.gpus_per_agg,
        gpus_per_node=r.gpus_per_node,
        available_nodes=nodes,
        spread_workers=r.spread_workers,
    )
    procs = endpoints_to_processes(eps, port_allocator=NodePortAllocator())
    placement = []
    for p in procs:
        env = backend.get_process_environment(p)
        ports = {"nixl": p.nixl_port, "kv": p.kv_events_port}
        if "DYN_KVBM_LEADER_ZMQ_PUB_PORT" in env:
            ports["zmq_pub"] = int(env["DYN_KVBM_LEADER_ZMQ_PUB_PORT"])
            ports["zmq_ack"] = int(env["DYN_KVBM_LEADER_ZMQ_ACK_PORT"])
        placement.append((f"{p.endpoint_mode}{p.endpoint_index}", p.node, frozenset(p.gpu_indices), ports))

    n_aside = r.kvbm_prefill_nodes or 0
    if n_aside and backend.kvbm_hub is not None:
        tp = r.kvbm_prefill_tp or r.gpus_per_node
        infra = None if config.infra.etcd_nats_dedicated_node else nodes[0]
        asides = compute_aside_placement(
            endpoints=eps,
            num_workers=n_aside,
            gpus_per_worker=tp,
            gpus_per_node=r.gpus_per_node,
            available_nodes=nodes,
            infra_node=infra,
            colocate=r.kvbm_prefill_colocate,
        )
        n_ep = len(procs)
        for i, pl in enumerate(asides):
            # Mirror do_sweep: only a CO-LOCATED aside gets explicit ports.
            if pl.colocated:
                off = n_ep + i
                ports = {
                    "nixl": VLLM_NIXL_PORT_BASE + off,
                    "zmq_pub": KVBM_ZMQ_PORT_BASE + off * 2,
                    "zmq_ack": KVBM_ZMQ_PORT_BASE + off * 2 + 1,
                }
            else:
                ports = {}
            placement.append((f"aside{i}", pl.node, pl.gpu_indices, ports))

    return config.total_nodes, placement


def _assert_2_workers_per_node_distinct(total_nodes, placement, gpus_per_node):
    """All workers pack 2/node; per-node GPUs and ports are disjoint."""
    by_node: dict[str, list] = {}
    for label, node, gpus, ports in placement:
        by_node.setdefault(node, []).append((label, gpus, ports))

    assert len(by_node) == total_nodes
    for node, items in by_node.items():
        # GPU disjointness
        seen_gpus: set[int] = set()
        for _label, gpus, _ports in items:
            assert seen_gpus.isdisjoint(gpus), f"GPU overlap on {node}"
            seen_gpus |= set(gpus)
        # 2 workers/node => 4 GPUs used on a 4-GPU node
        assert len(items) == 2, f"{node} has {len(items)} workers, expected 2"
        assert len(seen_gpus) == gpus_per_node
        # Port disjointness
        seen_ports: set[int] = set()
        for _label, _gpus, ports in items:
            for port in ports.values():
                if port is None:
                    continue
                assert port not in seen_ports, f"port {port} collision on {node}"
                seen_ports.add(port)


# Recipe-shaped resource blocks (mirror the SOP binpack120 recipes; gb200 4-GPU nodes).
_CD_RESOURCES = {
    "gpu_type": "gb200",
    "gpus_per_node": 4,
    "agg_nodes": 3,
    "agg_workers": 5,
    "gpus_per_agg": 2,
    "kvbm_prefill_nodes": 1,
    "kvbm_prefill_tp": 2,
    "prefill_nodes": None,
    "decode_nodes": None,
    "prefill_workers": None,
    "decode_workers": None,
}

_TRAD_RESOURCES = {
    "gpu_type": "gb200",
    "gpus_per_node": 4,
    "prefill_nodes": 2,
    "prefill_workers": 3,
    "gpus_per_prefill": 2,
    "decode_nodes": 2,
    "decode_workers": 3,
    "gpus_per_decode": 2,
}


class TestBinpack3Node:
    """CD (prefill-aside co-location) and TRAD (P/D co-location) bin-pack to a
    TRUE 3 nodes (12 GPUs, 2 workers/node) with distinct GPUs + no port
    collisions; AGG and the default-off paths are unchanged."""

    def test_cd_aside_colocate_packs_to_three_nodes(self):
        config = _make_config(
            {
                "resources": {**_CD_RESOURCES, "kvbm_prefill_colocate": True},
                "backend": {"type": "vllm", "kvbm_hub": {}},
                "infra": {"etcd_nats_dedicated_node": False},
            }
        )
        assert config.total_nodes == 3
        total_nodes, placement = _binpack_placement(config)
        assert total_nodes == 3
        _assert_2_workers_per_node_distinct(total_nodes, placement, gpus_per_node=4)

        # The aside co-locates on the partial decode node (the 5th agg leaves 2
        # idle GPUs there) and is carved its FREE indices, not the agg's.
        aside = next(p for p in placement if p[0] == "aside0")
        agg_on_aside_node = [p for p in placement if p[1] == aside[1] and p[0] != "aside0"]
        assert len(agg_on_aside_node) == 1  # exactly one agg shares the aside node
        assert aside[2].isdisjoint(agg_on_aside_node[0][2])  # distinct GPUs

    def test_cd_aside_default_off_reserves_extra_node(self):
        """Without kvbm_prefill_colocate the CD aside still takes its own node
        (4 nodes) — byte-identical to the historical carve-out."""
        config = _make_config(
            {
                "resources": {**_CD_RESOURCES},  # colocate defaults False
                "backend": {"type": "vllm", "kvbm_hub": {}},
                "infra": {"etcd_nats_dedicated_node": False},
            }
        )
        assert config.resources.kvbm_prefill_colocate is False
        assert config.total_nodes == 4  # agg_nodes(3) + kvbm_prefill_nodes(1)

    def test_trad_pd_colocate_packs_to_three_nodes(self):
        config = _make_config(
            {
                "resources": {**_TRAD_RESOURCES},
                "backend": {"type": "vllm", "allow_prefill_decode_colocation": True, "connector": "nixl"},
                "infra": {"etcd_nats_dedicated_node": False},
            }
        )
        assert config.total_nodes == 3
        total_nodes, placement = _binpack_placement(config)
        assert total_nodes == 3
        _assert_2_workers_per_node_distinct(total_nodes, placement, gpus_per_node=4)

        # Exactly one node mixes a prefill and a decode worker (n1).
        by_node: dict[str, set] = {}
        for label, node, _gpus, _ports in placement:
            by_node.setdefault(node, set()).add(label[:-1])  # strip index
        mixed = [n for n, modes in by_node.items() if modes == {"prefill", "decode"}]
        assert len(mixed) == 1

    def test_trad_default_off_keeps_four_nodes(self):
        """Without allow_prefill_decode_colocation, 3P+3D stays on 4 nodes
        (2 prefill + 2 decode) — unrelated trad recipes unchanged."""
        config = _make_config(
            {
                "resources": {**_TRAD_RESOURCES},
                "backend": {"type": "vllm", "connector": "nixl"},
            }
        )
        assert config.total_nodes == 4

    def test_agg_unchanged_three_nodes(self):
        """The aggregated baseline (6 TEP=2 agg workers, no aside, no P/D
        colocation) packs 2/node onto 3 nodes — unaffected by either change."""
        config = _make_config(
            {
                "resources": {
                    "gpu_type": "gb200",
                    "gpus_per_node": 4,
                    "agg_nodes": 3,
                    "agg_workers": 6,
                    "gpus_per_agg": 2,
                    "prefill_nodes": None,
                    "decode_nodes": None,
                    "prefill_workers": None,
                    "decode_workers": None,
                },
                "backend": {"type": "vllm", "connector": "nixl"},
            }
        )
        assert config.total_nodes == 3
        total_nodes, placement = _binpack_placement(config)
        _assert_2_workers_per_node_distinct(total_nodes, placement, gpus_per_node=4)
