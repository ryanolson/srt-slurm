# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trace replay (SA) benchmark runner using aiperf with a public dataset.

Like trace-replay, but instead of a user-provided JSONL trace file this variant
pulls a pre-registered public dataset from HuggingFace via aiperf's
``--public-dataset`` (e.g. ``semianalysis_cc_traces_weka_no_subagents``) and
replays it as a chat workload across concurrency levels.

The SemiAnalysis public-dataset loaders are not in stock aiperf; they live in
cquil11/aiperf@cjq/agentx-v0.3, so a recipe using this benchmark must pin
``benchmark.aiperf_package`` to that branch, e.g.::

    aiperf_package: "git+https://github.com/cquil11/aiperf.git@cjq/agentx-v0.3"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.benchmarks.base import SCRIPTS_DIR, AIPerfBenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


@register_benchmark("trace-replay-sa")
class TraceReplaySARunner(AIPerfBenchmarkRunner):
    """Trace replay benchmark using aiperf with a public (HuggingFace) dataset.

    Downloads a pre-registered public dataset and replays it at various
    concurrency levels to measure serving performance on realistic agentic
    workloads.

    Required config fields:
        - benchmark.public_dataset: aiperf public-dataset name
          (e.g. "semianalysis_cc_traces_weka_no_subagents")
        - benchmark.concurrencies: Concurrency levels to sweep

    Optional config fields:
        - benchmark.num_dataset_entries: Number of entries to load (default: all)
        - benchmark.ttft_threshold_ms: Goodput TTFT threshold (default: 2000)
        - benchmark.itl_threshold_ms: Goodput ITL threshold (default: 25)
        - benchmark.aiperf_package: aiperf install spec (pin to the fork branch
          that registers the SemiAnalysis datasets)
        - benchmark.aiperf_args: extra aiperf flags (e.g. benchmark-duration,
          workers-max, export-http-trace)
    """

    @property
    def name(self) -> str:
        return "Trace-Replay-SA-Bench"

    @property
    def script_path(self) -> str:
        return "/srtctl-benchmarks/trace-replay-sa/bench.sh"

    @property
    def local_script_dir(self) -> str:
        return str(SCRIPTS_DIR / "trace-replay-sa")

    def validate_config(self, config: SrtConfig) -> list[str]:
        errors = []
        b = config.benchmark

        if not b.public_dataset:
            errors.append("benchmark.public_dataset is required for trace-replay-sa")

        if b.concurrencies is None:
            errors.append("benchmark.concurrencies is required for trace-replay-sa")

        return errors

    def build_command(
        self,
        config: SrtConfig,
        runtime: RuntimeContext,
    ) -> list[str]:
        b = config.benchmark
        endpoint = f"http://localhost:{runtime.frontend_port}"

        model_name = config.served_model_name or config.model.path

        # Format concurrencies as comma-separated string
        concurrencies = b.concurrencies
        if isinstance(concurrencies, list):
            concurrencies = ",".join(str(c) for c in concurrencies)

        # num_dataset_entries is optional; pass "" to let aiperf use the full corpus
        num_dataset_entries = "" if b.num_dataset_entries is None else str(b.num_dataset_entries)

        ttft_threshold = getattr(b, "ttft_threshold_ms", None) or 2000
        itl_threshold = getattr(b, "itl_threshold_ms", None) or 25

        tokenizer_path = str(runtime.model_path) if runtime.is_hf_model else "/model"

        cmd = [
            "bash",
            self.script_path,
            endpoint,
            model_name,
            b.public_dataset or "",
            num_dataset_entries,
            str(concurrencies) if concurrencies else "",
            str(ttft_threshold),
            str(itl_threshold),
            tokenizer_path,
        ]

        self.append_aiperf_args(cmd, config)

        return cmd
