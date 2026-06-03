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
