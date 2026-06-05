# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
vLLM backend configuration.

Implements BackendProtocol for vLLM inference serving with prefill/decode disaggregation.
Uses dynamo.vllm integration module.
"""

from __future__ import annotations

import builtins
import json
from collections.abc import Sequence
from dataclasses import field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
)

from marshmallow import Schema
from marshmallow_dataclass import dataclass

from srtctl.ports import (
    DYN_SYSTEM_PORT_BASE,
    KV_EVENTS_PORT_BASE,
    KVBM_HUB_DISCOVERY_PORT,
    KVBM_ZMQ_PORT_BASE,
    VLLM_DATA_PARALLEL_RPC_PORT,
)

if TYPE_CHECKING:
    from srtctl.backends.base import SrunConfig
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Endpoint, NodePortAllocator, Process

# Type alias for worker modes
WorkerMode = Literal["prefill", "decode", "agg"]


@dataclass(frozen=True)
class VLLMServerConfig:
    """vLLM server CLI configuration per mode (prefill/decode/aggregated).

    Each mode can have its own configuration dict that gets converted
    to CLI flags when starting the worker. These are passed directly to
    vLLM's AsyncEngineArgs.
    """

    prefill: dict[str, Any] | None = None
    decode: dict[str, Any] | None = None
    aggregated: dict[str, Any] | None = None

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class KvbmHubConfig:
    """KVBM hub (conditional-disagg + remote-search) configuration.

    When present on a vLLM backend, srtslurm:
    1. Launches ``kvbm_hub`` on the infra node (do_sweep.start_kvbm_hub) with the
       indexer+p2p+disagg feature set and the prefill-router enabled.
    2. Injects ``KVBM_HUB_URL=http://<infra>:1337`` on every worker.
    3. Builds the decode (agg) workers' ``--kv-transfer-config`` to register the
       kvbm v2 connector (role=decode) against that hub, with prefix-caching + KV
       events ON so the in-process consolidator relays to the dynamo KV-router.

    Decode MUST run in AGGREGATED mode (``--disaggregation-mode decode`` disables
    the KV-event publisher → the router goes blind). The hub's block_size /
    max_seq_len / block_layout must match the workers'; block_size and max_seq_len
    are read from the decode/aggregated vllm_config (``--block-size`` /
    ``--max-model-len``) to avoid drift, block_layout is set here (use ``universal``
    for asymmetric TP between decode and the prefill aside).

    Example YAML:
        backend:
          type: vllm
          kvbm_hub:
            block_layout: operational        # ``universal`` for asymmetric TP
            min_remote_prefill_tokens: 256    # ThresholdRemote: <N local, >=N disagg
            host_cache_gb: 100.0
    """

    container: str | None = None
    hub_binary: str = "/workspace/target/release/kvbm_hub"
    features: str = "indexer,p2p,disagg"
    block_layout: str = "operational"
    min_remote_prefill_tokens: int = 256
    # Decode-side B-GNMT overflow budget: max in-flight remote-prefill TOKENS across all
    # outstanding disaggregations. When exhausted, a request that would disaggregate is
    # DOWNGRADED to local prefill on the decode worker (the "use decode to help prefill in
    # a heavy wave" adaptation). None => connector default usize::MAX => feature inert.
    max_inflight_remote_prefill_tokens: int | None = None
    # CD prefill-overload circuit breaker (router-sourced 3-tier). The breaker is
    # configured ENTIRELY on the hub: when cd_breaker_enabled is True these route
    # to the kvbm_hub --cd-breaker* CLI flags (do_sweep.build_kvbm_hub_command).
    # cd_breaker_enabled None/False => no --cd-breaker => the hub never constructs
    # the breaker => decode stays CALM (byte-identical to today). Set
    # cd_breaker_enabled=True to opt in; the watermarks are expressed against the
    # router's free-capacity fraction [0.0, 1.0] (lower = more pressure). See the
    # kvbm-hub CircuitBreaker for the exact semantics.
    cd_breaker_enabled: bool | None = None
    cd_breaker_warm_high: float | None = None
    cd_breaker_hot_high: float | None = None
    cd_breaker_clear_low: float | None = None
    # NOTE: the queue-depth trip axis has NO kvbm_hub CLI flag yet — the hub
    # hardcodes queue_depth_warm/hot = 0 (DISABLED) pending the P2 queue-depth
    # accessor. These two fields are retained for forward-compat but are NOT
    # routed to the hub today (setting them is a no-op until the flag lands).
    cd_breaker_queue_depth_warm: int | None = None
    cd_breaker_queue_depth_hot: int | None = None
    cd_breaker_clear_debounce_ticks: int | None = None
    host_cache_gb: float = 100.0
    prefill_max_num_seqs: int | None = None  # cap the prefill aside's in-flight requests (vLLM --max-num-seqs)
    remote_search: bool = True          # decode: remote-search ON by default
    remote_search_prefill: bool = False  # prefill: OFF by default (pure CD target; no indexer)
    onboard_mode: str = "inter"
    connector_module_path: str = "kvbm.v2.vllm.connector"
    env: dict[str, str] = field(default_factory=dict)

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class VLLMProtocol:
    """vLLM protocol - implements BackendProtocol.

    This frozen dataclass both holds configuration AND implements the
    BackendProtocol methods for process allocation and launching.

    dynamo 1.0.0+: ``--connector`` was removed; the ``connector`` field is now
    translated to ``--kv-transfer-config`` with the appropriate JSON payload.

    Example YAML:
        backend:
          type: vllm
          connector: nixl  # translated to --kv-transfer-config JSON
          allow_prefill_decode_colocation: true  # pack P/D on one node when all workers fit
          prefill_environment:
            PYTHONUNBUFFERED: "1"
          vllm_config:
            prefill:
              tensor-parallel-size: 2
              gpu-memory-utilization: 0.9
              connector: lmcache  # override connector for prefill
            decode:
              tensor-parallel-size: 2
              gpu-memory-utilization: 0.85
              # uses default connector (nixl)
    """

    type: Literal["vllm"] = "vllm"

    # Environment variables per mode
    prefill_environment: dict[str, str] = field(default_factory=dict)
    decode_environment: dict[str, str] = field(default_factory=dict)
    aggregated_environment: dict[str, str] = field(default_factory=dict)

    # vLLM server CLI config per mode
    vllm_config: VLLMServerConfig | None = None

    # Default KV connector: "nixl", "lmcache", or a raw JSON string for --kv-transfer-config.
    # Can be overridden per mode by setting "connector" in vllm_config.prefill/decode/aggregated.
    # dynamo 1.0.0+: translated to --kv-transfer-config (--connector was removed).
    connector: str | None = "nixl"

    # KVBM hub (conditional-disagg + remote-search). When set, the decode workers
    # register the kvbm v2 connector against the hub launched on the infra node,
    # and their --kv-transfer-config is built from this config (overriding
    # ``connector`` for agg/decode workers). See KvbmHubConfig.
    kvbm_hub: KvbmHubConfig | None = None

    # Allow prefill and decode workers to share one node when the combined GPU
    # request fits within gpus_per_node. Defaults off to preserve existing P/D
    # node separation.
    allow_prefill_decode_colocation: bool = False

    # Aggregated KV-aware-routing baseline: enable the in-process KV consolidator +
    # per-worker KV events on AGG workers using the recipe's raw kvbm v2 ``connector``,
    # WITHOUT a hub or prefill aside. This is the CD decode plane's routing methodology
    # (consolidator relays G1+G2 events to the dynamo KV-router; prefix-caching forced ON)
    # minus the disagg sidecar — so an agg baseline routes iso with CD and the agg-vs-CD
    # delta isolates conditional disaggregation. Requires a kvbm v2 ``connector`` + a
    # KV-router frontend (router-mode: kv, router-kv-events: true). Ignored if kvbm_hub is set.
    kvbm_consolidator: bool = False

    Schema: ClassVar[builtins.type[Schema]] = Schema

    # =========================================================================
    # BackendProtocol Implementation
    # =========================================================================

    def get_srun_config(self) -> SrunConfig:
        """vLLM uses per-process launching (one srun per node)."""
        from srtctl.backends.base import SrunConfig

        return SrunConfig(mpi=None, oversubscribe=False, launch_per_endpoint=False)

    def get_config_for_mode(self, mode: WorkerMode) -> dict[str, Any]:
        """Get merged config dict for a worker mode."""
        if not self.vllm_config:
            return {}

        if mode == "prefill":
            return dict(self.vllm_config.prefill or {})
        elif mode == "decode":
            return dict(self.vllm_config.decode or {})
        elif mode == "agg":
            return dict(self.vllm_config.aggregated or {})
        return {}

    def get_environment_for_mode(self, mode: WorkerMode) -> dict[str, str]:
        """Get environment variables for a worker mode."""
        if mode == "prefill":
            return dict(self.prefill_environment)
        elif mode == "decode":
            return dict(self.decode_environment)
        elif mode == "agg":
            return dict(self.aggregated_environment)
        return {}

    def get_process_environment(self, process: Process) -> dict[str, str]:
        """Get process-specific environment variables for vLLM workers.

        vLLM with dynamo requires unique ports for each worker:
        - DYN_VLLM_KV_EVENT_PORT: ZMQ port for KV events publishing
        - VLLM_NIXL_SIDE_CHANNEL_PORT: Port for NIXL side channel transfers
        """
        env: dict[str, str] = {}
        if process.kv_events_port is not None:
            env["DYN_VLLM_KV_EVENT_PORT"] = str(process.kv_events_port)
        if process.nixl_port is not None:
            env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(process.nixl_port)
        # Deterministic per-worker KVBM leader ZMQ ports so co-located workers don't
        # collide on the default 56001 (the in-process consolidator derives its egress
        # port from DYN_KVBM_LEADER_ZMQ_PUB_PORT). Required for BOTH consolidator paths:
        # the CD decode plane (kvbm_hub) AND the hub-less agg consolidator baseline
        # (kvbm_consolidator) — else 2 bin-packed agg workers/node both bind the default
        # port and EngineCore init fails. Mirrors worker_stage._apply_kvbm_endpoint_env
        # (DYN_CONNECTOR=kvbm path); our connector arrives via --kv-transfer-config.
        if (self.kvbm_hub is not None or self.kvbm_consolidator) and process.kv_events_port is not None:
            port_offset = max(0, process.kv_events_port - KV_EVENTS_PORT_BASE)
            pub_port = KVBM_ZMQ_PORT_BASE + (port_offset * 2)
            if pub_port + 1 <= 65535:
                env["DYN_KVBM_LEADER_ZMQ_PUB_PORT"] = str(pub_port)
                env["DYN_KVBM_LEADER_ZMQ_ACK_PORT"] = str(pub_port + 1)
        return env

    def get_served_model_name(self, default: str) -> str:
        """Get served model name from vLLM config, or return default."""
        if self.vllm_config:
            for cfg in [self.vllm_config.prefill, self.vllm_config.aggregated, self.vllm_config.decode]:
                if cfg:
                    name = cfg.get("served-model-name") or cfg.get("served_model_name")
                    if name:
                        return name
        return default

    # =========================================================================
    # KVBM hub (conditional-disagg + remote-search) helpers
    # =========================================================================

    def _vllm_cfg_value(self, *keys: str, default: Any = None) -> Any:
        """First matching value across aggregated/decode/prefill vllm_config dicts."""
        if not self.vllm_config:
            return default
        for cfg in (self.vllm_config.aggregated, self.vllm_config.decode, self.vllm_config.prefill):
            if not cfg:
                continue
            for key in keys:
                if cfg.get(key) is not None:
                    return cfg[key]
        return default

    def kvbm_block_size(self) -> int:
        return int(self._vllm_cfg_value("block-size", "block_size", default=64))

    def kvbm_max_seq_len(self) -> int:
        return int(self._vllm_cfg_value("max-model-len", "max_model_len", default=40960))

    def _enable_kvbm_consolidator(self, cmd: list[str], config: dict[str, Any], process: Process) -> None:
        """Apply the in-process KV-consolidator prerequisites to an AGG worker — the
        single source of truth shared by the CD decode plane and the hub-less agg
        consolidator baseline, so both route with the identical methodology.

        The consolidator hard-requires prefix-caching (force it ON; drop any recipe
        no-enable-prefix-caching that would silently blind the router) and consumes the
        worker's vLLM KV events, so emit them on the per-worker ZMQ port (its source).
        The consolidator itself is default-on in the kvbm v2 connector once a
        kv_transfer_config + kv-events are present (DYN_KVBM_KV_EVENTS_CONSOLIDATOR_MODE
        defaults to "dedup"); it relays G1+G2 events to the dynamo KV-router in-process,
        independent of the hub.
        """
        config.pop("no-enable-prefix-caching", None)
        config.pop("no_enable_prefix_caching", None)
        config["enable-prefix-caching"] = True
        if process.kv_events_port is not None:
            kv_events = {"endpoint": f"tcp://*:{process.kv_events_port}", "enable_kv_cache_events": True}
            cmd.extend(["--kv-events-config", json.dumps(kv_events)])

    def build_kvbm_hub_connector(self, role: str, hub_url: str) -> str:
        """Build the STATIC kvbm v2 connector --kv-transfer-config JSON for a
        hub-registered worker (role ``decode`` or ``prefill``).

        Matches the hub's must-match register check by construction (same
        block_layout / max_seq_len as the hub flags), so no runtime ``kvbmctl``
        render is needed. Remote-search is role-keyed: ON for decode (override
        ``kvbm_hub.remote_search``), explicitly OFF for prefill (override
        ``kvbm_hub.remote_search_prefill``). A prefill declares only ``disagg``
        (no ``indexer``), so enabling remote-search there trips the connector's
        startup validator — keep it off unless the prefill also carries indexer.
        """
        h = self.kvbm_hub
        assert h is not None
        leader: dict[str, Any] = {
            "hub": {"url": hub_url, "features": [f.strip() for f in h.features.split(",") if f.strip()]},
            "max_seq_len": self.kvbm_max_seq_len(),
            "cache": {"host": {"cache_size_gb": float(h.host_cache_gb)}},
            "disagg": {"role": role},
            "onboard": {"mode": h.onboard_mode},
        }
        if role == "decode":
            leader["disagg"]["min_remote_prefill_tokens"] = int(h.min_remote_prefill_tokens)
            if h.max_inflight_remote_prefill_tokens is not None:
                # Finite budget => B-GNMT downgrade-to-local arms on overload (heavy wave).
                leader["disagg"]["max_inflight_remote_prefill_tokens"] = int(h.max_inflight_remote_prefill_tokens)
            # NOTE: the CD circuit breaker is configured ENTIRELY on the hub via
            # the kvbm_hub --cd-breaker* CLI flags (see do_sweep.build_kvbm_hub_command),
            # NOT on the connector. The breaker lives in the hub's prefill-router
            # and PUSHES the resulting tier to decodes over velo; the connector
            # only consumes the pushed tier at runtime. The cd_breaker_* fields
            # on KvbmHubConfig are the recipe's representation and are routed to
            # the hub CLI, never into this connector JSON.
            if h.remote_search:
                leader["remote_search"] = {"enabled": True}
        elif role == "prefill":
            # Pure CD target: remote-search OFF explicitly (provable in the static
            # config, not reliant on a connector default). Overridable, but only
            # valid if the prefill also carries the indexer feature.
            leader["remote_search"] = {"enabled": bool(h.remote_search_prefill)}
        cfg = {
            "kv_connector": "DynamoConnector",
            "kv_role": "kv_both",
            "kv_connector_module_path": h.connector_module_path,
            "kv_connector_extra_config": {
                "default": {"block_layout": h.block_layout},
                "leader": leader,
                "worker": {"nixl": {"backends": {"UCX": {}, "POSIX": {}}}},
            },
        }
        return json.dumps(cfg)

    def build_kvbm_prefill_command(self, runtime: RuntimeContext, tp: int, hub_url: str) -> list[str]:
        """Build the `python -m kvbm.vllm.prefill` command for a hub-owned prefill
        ASIDE worker (NOT a dynamo endpoint).

        Joins the hub prefill-router velo fleet via the kvbm v2 connector
        (role=prefill, remote_search off). kvbm.vllm.prefill reuses vLLM's arg
        parser, so the flags pass through identically. model / TP / EP / max-len
        mirror the decode plane (TP may differ via kvbm_prefill_tp).
        """
        assert self.kvbm_hub is not None
        model_arg = str(runtime.model_path) if runtime.is_hf_model else "/model"
        served = self.get_served_model_name(runtime.model_path.name)
        agg = (self.vllm_config.aggregated if self.vllm_config else None) or {}
        gmu = agg.get("gpu-memory-utilization") or agg.get("gpu_memory_utilization") or 0.85
        cmd = [
            "python3",
            "-m",
            "kvbm.vllm.prefill",
            "--model",
            model_arg,
            "--served-model-name",
            served,
            "--tensor-parallel-size",
            str(tp),
            "--gpu-memory-utilization",
            str(gmu),
            "--block-size",
            str(self.kvbm_block_size()),
            "--max-model-len",
            str(self.kvbm_max_seq_len()),
        ]
        if agg.get("enable-expert-parallel") or agg.get("enable_expert_parallel"):
            cmd.append("--enable-expert-parallel")
        # Cap the prefill aside's in-flight requests (vLLM running batch). The prefill
        # aside is compute-bound and lightly loaded under CD; a small cap bounds its
        # activation memory (headroom for weights + KV on tight TP=1 prefill) without
        # hurting throughput. None => vLLM default.
        if self.kvbm_hub.prefill_max_num_seqs is not None:
            cmd.extend(["--max-num-seqs", str(self.kvbm_hub.prefill_max_num_seqs)])
        cmd.extend(["--kv-transfer-config", self.build_kvbm_hub_connector("prefill", hub_url)])
        return cmd

    def should_colocate_prefill_decode(
        self,
        *,
        num_prefill: int,
        num_decode: int,
        num_agg: int,
        gpus_per_prefill: int,
        gpus_per_decode: int,
        gpus_per_agg: int,
        gpus_per_node: int,
    ) -> bool:
        """Whether all vLLM workers should be packed onto one node."""
        if not self.allow_prefill_decode_colocation:
            return False
        if num_prefill <= 0 or num_decode <= 0 or gpus_per_node <= 0:
            return False

        total_worker_gpus = num_prefill * gpus_per_prefill + num_decode * gpus_per_decode + num_agg * gpus_per_agg
        return total_worker_gpus <= gpus_per_node

    def allocate_endpoints(
        self,
        num_prefill: int,
        num_decode: int,
        num_agg: int,
        gpus_per_prefill: int,
        gpus_per_decode: int,
        gpus_per_agg: int,
        gpus_per_node: int,
        available_nodes: Sequence[str],
        spread_workers: bool = False,
    ) -> list[Endpoint]:
        """Allocate endpoints to nodes."""
        from srtctl.core.topology import allocate_endpoints

        return allocate_endpoints(
            num_prefill=num_prefill,
            num_decode=num_decode,
            num_agg=num_agg,
            gpus_per_prefill=gpus_per_prefill,
            gpus_per_decode=gpus_per_decode,
            gpus_per_agg=gpus_per_agg,
            gpus_per_node=gpus_per_node,
            available_nodes=available_nodes,
            spread_workers=spread_workers,
            allow_prefill_decode_colocation=self.should_colocate_prefill_decode(
                num_prefill=num_prefill,
                num_decode=num_decode,
                num_agg=num_agg,
                gpus_per_prefill=gpus_per_prefill,
                gpus_per_decode=gpus_per_decode,
                gpus_per_agg=gpus_per_agg,
                gpus_per_node=gpus_per_node,
            ),
        )

    def _is_dp_mode(self, mode: WorkerMode) -> bool:
        """Check if this mode uses Data Parallel + Expert Parallel pattern.

        DP+EP mode is detected when data-parallel-size is set in the mode's config.
        In this mode, each GPU runs its own process (rather than TP across GPUs).
        """
        config = self.get_config_for_mode(mode)
        return config.get("data-parallel-size") is not None or config.get("data_parallel_size") is not None

    def _get_dp_size(self, mode: WorkerMode) -> int | None:
        """Get the data-parallel-size for a mode, or None if not in DP mode."""
        config = self.get_config_for_mode(mode)
        return config.get("data-parallel-size") or config.get("data_parallel_size")

    def endpoints_to_processes(
        self,
        endpoints: list[Endpoint],
        base_sys_port: int = DYN_SYSTEM_PORT_BASE,
        port_allocator: NodePortAllocator | None = None,
    ) -> list[Process]:
        """Convert endpoints to processes.

        For DP+EP mode (data-parallel-size set), creates one process per GPU.
        For standard TP mode, creates one process per node.
        """
        from srtctl.core.topology import NodePortAllocator, Process, endpoints_to_processes

        # Check if any endpoint uses DP mode
        has_dp_mode = any(self._is_dp_mode(ep.mode) for ep in endpoints)

        if not has_dp_mode:
            # Standard TP mode: one process per node
            return endpoints_to_processes(endpoints, base_sys_port=base_sys_port, port_allocator=port_allocator)

        # DP+EP mode: one process per GPU
        processes: list[Process] = []
        current_sys_port = base_sys_port
        if port_allocator is None:
            port_allocator = NodePortAllocator()

        for endpoint in endpoints:
            if not self._is_dp_mode(endpoint.mode):
                # Non-DP endpoints get standard processing
                # (This shouldn't happen in practice since all modes should be consistent)
                for node_rank, node in enumerate(endpoint.nodes):
                    is_leader = node_rank == 0
                    http_port = port_allocator.next_http_port(node) if is_leader else 0
                    bootstrap_port = (
                        port_allocator.next_bootstrap_port(node) if endpoint.mode == "prefill" and is_leader else None
                    )
                    kv_events_port = port_allocator.next_kv_events_port()
                    nixl_port = port_allocator.next_nixl_port()

                    processes.append(
                        Process(
                            node=node,
                            gpu_indices=endpoint.gpu_indices,
                            sys_port=current_sys_port,
                            http_port=http_port,
                            endpoint_mode=endpoint.mode,
                            endpoint_index=endpoint.index,
                            node_rank=node_rank,
                            bootstrap_port=bootstrap_port,
                            kv_events_port=kv_events_port,
                            nixl_port=nixl_port,
                        )
                    )
                    current_sys_port += 1
            else:
                # DP+EP mode: one process per GPU
                # Each process gets a single GPU and a unique dp_rank
                dp_rank = 0
                # Allocate a unique DP RPC port for this endpoint's leader node
                dp_rpc_port = port_allocator.next_dp_rpc_port(endpoint.leader_node)
                # Allocate a single NIXL base port for this endpoint.
                # vLLM internally computes: actual_port = base + data_parallel_rank
                # so all DP ranks in the endpoint share the same base port.
                dp_size = self._get_dp_size(endpoint.mode) or len(endpoint.gpu_indices)
                nixl_base_port = port_allocator.next_nixl_port_block(dp_size)
                for _node_rank, node in enumerate(endpoint.nodes):
                    for gpu_idx in sorted(endpoint.gpu_indices):
                        is_leader = dp_rank == 0
                        http_port = port_allocator.next_http_port(node) if is_leader else 0
                        bootstrap_port = (
                            port_allocator.next_bootstrap_port(node)
                            if endpoint.mode == "prefill" and is_leader
                            else None
                        )
                        kv_events_port = port_allocator.next_kv_events_port()
                        nixl_port = nixl_base_port

                        processes.append(
                            Process(
                                node=node,
                                gpu_indices=frozenset([gpu_idx]),  # Single GPU per process
                                sys_port=current_sys_port,
                                http_port=http_port,
                                endpoint_mode=endpoint.mode,
                                endpoint_index=endpoint.index,
                                node_rank=dp_rank,  # dp_rank stored in node_rank for now
                                bootstrap_port=bootstrap_port,
                                kv_events_port=kv_events_port,
                                nixl_port=nixl_port,
                                dp_rpc_port=dp_rpc_port,
                            )
                        )
                        current_sys_port += 1
                        dp_rank += 1

        return processes

    def build_worker_command(
        self,
        process: Process,
        endpoint_processes: list[Process],
        runtime: RuntimeContext,
        frontend_type: str = "dynamo",
        nsys_prefix: list[str] | None = None,
        dump_config_path: Path | None = None,
    ) -> list[str]:
        """Build the command to start a vLLM worker process.

        Args:
            process: The process to start
            endpoint_processes: All processes for this endpoint (for multi-node)
            runtime: Runtime context with paths and settings
            frontend_type: Frontend type (currently only "dynamo" supported for vLLM)
            nsys_prefix: Optional nsys profiling command prefix
            dump_config_path: Path to dump config JSON
        """
        from srtctl.core.slurm import get_hostname_ip

        mode = process.endpoint_mode
        config = self.get_config_for_mode(mode)

        # Determine if multi-node
        endpoint_nodes = list(dict.fromkeys(p.node for p in endpoint_processes))
        is_multi_node = len(endpoint_nodes) > 1

        # Get leader IP for distributed init
        leader_ip = get_hostname_ip(endpoint_nodes[0])

        # Determine model path: HF model ID or container mount path
        # For HF models (hf:prefix), model_path contains the HF model ID (e.g., "facebook/opt-125m")
        # For local models, model is mounted to /model in the container
        model_arg = str(runtime.model_path) if runtime.is_hf_model else "/model"

        # Get served model name from config or use model path name
        served_model_name = self.get_served_model_name(runtime.model_path.name)

        # Start with nsys prefix if provided
        cmd: list[str] = list(nsys_prefix) if nsys_prefix else []

        # Base command - use dynamo.vllm module
        cmd.extend(
            [
                "python3",
                "-m",
                "dynamo.vllm",
                "--model",
                model_arg,
                "--served-model-name",
                served_model_name,
            ]
        )

        # Disaggregation mode (dynamo 1.0.0+: --is-prefill-worker/--is-decode-worker are deprecated)
        if mode in ("prefill", "decode"):
            cmd.extend(["--disaggregation-mode", mode])

        # KV connector → --kv-transfer-config (dynamo 1.0.0+: --connector was removed)
        if self.kvbm_hub is not None and mode == "agg":
            # CD decode plane: register the kvbm v2 connector against the hub on the
            # infra node (role=decode). Decode runs AGGREGATED (note: no
            # --disaggregation-mode above for "agg") so the KV-event publisher
            # stays alive and the in-process consolidator relays to the dynamo
            # KV-router. Prefix-caching + kv-events are REQUIRED for that relay.
            config.pop("connector", None)
            hub_url = f"http://{runtime.nodes.infra}:{KVBM_HUB_DISCOVERY_PORT}"
            cmd.extend(["--kv-transfer-config", self.build_kvbm_hub_connector("decode", hub_url)])
            self._enable_kvbm_consolidator(cmd, config, process)
        elif self.kvbm_consolidator and self.kvbm_hub is None and mode == "agg":
            # Aggregated KV-aware-routing baseline: the SAME in-process consolidator +
            # KV-router methodology as the CD decode plane, but WITHOUT the hub/prefill
            # sidecar — use the recipe's RAW kvbm v2 connector (no leader.hub, no disagg).
            # So agg routes iso with CD (G1+G2 dedup) and agg-vs-CD isolates disaggregation.
            mode_connector = config.pop("connector", None)
            connector = mode_connector if mode_connector is not None else self.connector
            if connector and connector not in ("null", "none", None):
                cmd.extend(["--kv-transfer-config", _connector_to_kv_transfer_config(connector)])
            self._enable_kvbm_consolidator(cmd, config, process)
        else:
            # Check for mode-specific override first, then fall back to default.
            # Pop from config so it doesn't get added again by _config_to_cli_args.
            mode_connector = config.pop("connector", None)
            connector = mode_connector if mode_connector is not None else self.connector

            if connector and connector not in ("null", "none", None):
                kv_transfer_cfg = _connector_to_kv_transfer_config(connector)
                cmd.extend(["--kv-transfer-config", kv_transfer_cfg])

        # Check if this is DP+EP mode (data-parallel-size set)
        is_dp_mode = self._is_dp_mode(mode)

        if is_dp_mode:
            # DP+EP mode: each GPU runs its own process
            # process.node_rank is the dp_rank (set in endpoints_to_processes)
            dp_rank = process.node_rank
            # Use the per-endpoint dp_rpc_port allocated by NodePortAllocator
            # (avoids port collisions when multiple endpoints share a node)
            dp_rpc_port = (
                process.dp_rpc_port
                or config.pop("data-parallel-rpc-port", None)
                or config.pop("data_parallel_rpc_port", VLLM_DATA_PARALLEL_RPC_PORT)
            )
            # Pop from config so it doesn't get added again by _config_to_cli_args
            config.pop("data-parallel-rpc-port", None)
            config.pop("data_parallel_rpc_port", None)

            cmd.extend(
                [
                    "--data-parallel-rank",
                    str(dp_rank),
                    "--data-parallel-address",
                    leader_ip,
                    "--data-parallel-rpc-port",
                    str(dp_rpc_port),
                ]
            )
            # Note: --data-parallel-size is added via _config_to_cli_args from vllm_config
        elif is_multi_node:
            # Standard TP+PP multi-node coordination flags
            node_rank = endpoint_nodes.index(process.node)
            cmd.extend(
                [
                    "--master-addr",
                    leader_ip,
                    "--nnodes",
                    str(len(endpoint_nodes)),
                    "--node-rank",
                    str(node_rank),
                ]
            )

            # Non-leader nodes run headless
            if node_rank > 0:
                cmd.append("--headless")

        # Add config dump path
        if dump_config_path:
            cmd.extend(["--dump-config-to", str(dump_config_path)])

        # Add all config flags from vllm_config
        cmd.extend(_config_to_cli_args(config))

        return cmd


_CONNECTOR_MAP: dict[str, dict[str, str]] = {
    "nixl": {"kv_connector": "NixlConnector", "kv_role": "kv_both"},
    "lmcache": {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"},
    "kvbm": {
        "kv_connector": "DynamoConnector",
        "kv_connector_module_path": "kvbm.vllm_integration.connector",
        "kv_role": "kv_both",
    },
}


def _connector_to_kv_transfer_config(connector: str) -> str:
    """Translate a connector shorthand to a --kv-transfer-config JSON string.

    Known shorthands (e.g. "nixl", "lmcache") are expanded to the full JSON
    config expected by vLLM.  Anything else is passed through as-is (assumed
    to already be a valid JSON string).
    """
    preset = _CONNECTOR_MAP.get(connector.lower())
    if preset is not None:
        return json.dumps(preset)
    return connector


def _config_to_cli_args(config: dict[str, Any]) -> list[str]:
    """Convert config dict to CLI arguments."""
    args: list[str] = []
    for key, value in sorted(config.items()):
        flag_name = key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(f"--{flag_name}")
        elif isinstance(value, list):
            args.append(f"--{flag_name}")
            args.extend(str(v) for v in value)
        elif value is not None:
            args.extend([f"--{flag_name}", str(value)])
    return args
