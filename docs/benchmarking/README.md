# Benchmark Suites

vLLM provides comprehensive benchmarking tools for performance testing and evaluation:

- **[Benchmark CLI](./cli.md)**: `vllm bench` CLI tools and specialized benchmark scripts for interactive performance testing.
- **[MoE DBO](./moe_dbo.md)**: Compare online serving throughput with MoE Dual Batch Overlap disabled and enabled.
- **[Parameter Sweeps](./sweeps.md)**: Automate `vllm bench` runs for multiple configurations, useful for [optimization and tuning](../configuration/optimization.md).
- **[Performance Dashboard](./dashboard.md)**: Automated CI that publishes benchmarks on each commit.
