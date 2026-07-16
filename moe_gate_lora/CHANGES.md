# MoE Gate LoRA 迁移与修改说明

本文档记录当前工作区相对上游 vLLM 的 MoE trace、流式 gate-LoRA
训练、serving 侧下一层 gate 预测、kernel 与 benchmark 修改。它描述的是
当前代码状态，不代表所有实验路径都已在 GPU 上验证。

## 1. 范围与目标

本次迁移只保留两类功能：MoE/gate-LoRA，以及为该功能采集 trace、训练、
评估和画图的工具。TurboQuant 和 prefix-cache benchmark 明确没有迁移。

整体训练流程改为：

```text
推理一个小 batch
  -> 只产生当前 batch 的 MoE trace
  -> 立即执行一次 gate-LoRA 训练或评估
  -> 更新在线统计、checkpoint 和图
  -> 删除本 batch trace
  -> 继续下一个推理 batch
```

这样不再先保存完整数据集的 activations、router logits 和 top-k，再进行
第二遍离线训练。

## 2. 环境变量

`vllm/envs.py` 新增并集中保留五个环境变量：

| 环境变量 | 作用 | 使用方 |
| --- | --- | --- |
| `VLLM_MOE_TRACE_DIR` | 启用 trace 并指定输出目录 | 训练/评估采集 |
| `VLLM_MOE_TRACE_MAX_STEPS` | 限制每个 worker 记录的 forward 数 | 训练/评估采集 |
| `VLLM_MOE_TRACE_MODE` | 选择完整训练 trace 或 expert 计数 trace | 训练/分布采集 |
| `VLLM_SC_EPLB` | 启用 serving 侧下一层 gate 预测实验 | vLLM worker |
| `VLLM_SC_EPLB_LORA_DIR` | serving 侧 gate-LoRA checkpoint 目录 | vLLM worker |

trace/训练开关与 serving/EPLB 实验开关被分开。训练 worker 会显式设置
`VLLM_SC_EPLB=0`，并删除 `VLLM_SC_EPLB_LORA_DIR`，因此训练不会误用
vLLM 内部 serving LoRA 路径。

三个 trace 变量被登记为已知运行时变量，并从 compile cache key 中忽略；
trace 强制使用 eager mode，不会生成依赖这些选项的编译图。

## 3. 根目录训练工具

所有训练、trace 消费、统计和画图代码集中到 `moe_gate_lora/`，避免把研究
脚本继续散落在 vLLM 内部模块。

### `moe_gate_lora/cli.py`

提供 `pipeline` 和 `plot` 两个子命令。`pipeline` 依次执行流式训练和使用
最终 checkpoint 的流式评估；`plot` 根据已有 `metrics.json` 重画图。CLI
统一管理模型、EP size、batch size、训练 epoch、LoRA rank/alpha、优化器参数
和工作目录。`--epochs` 只作用于训练，最终评估固定执行一遍。

### `moe_gate_lora/collect.py`

负责启动 vLLM worker、切分 prompts、同步 trace 与训练进程。每个 EP rank
完成一个 `llm.generate(batch)` 后写 ready 标记并暂停；parent 等待所有活跃
rank，消费当前 batch 的 `.pt` 文件，执行训练/评估，再写 ack 释放 worker。
每个 epoch 结束还有一次全 rank barrier，避免 shard 较短的 rank 提前进入下一
epoch。

因此内存和磁盘上只需保留当前 batch trace。所有 epoch 共用同一组 vLLM
workers、adapter 和 optimizer；每个 epoch 重新推理 prompts。prefix cache 被
显式禁用，确保后续 epoch 仍重新计算完整 prefill trace。异常时会终止并 join
所有子进程；输出目录必须为空，以免旧 trace 与新实验混合。

### `moe_gate_lora/trainer.py`

每对相邻 MoE 层维护一组 gate LoRA：

```text
predicted_logits = base_next_gate_logits
                 + scale * (X @ A.T) @ B.T
```

当前训练目标是 predicted logits 与下一层真实 router logits 的 MSE。每消费
一个推理 batch，每个相邻层对执行一次 optimizer step。

`StreamingProcessor` 在线维护原始 gate overlap、LoRA gate overlap、训练
loss、batch/step/token 数。训练模式每个 batch 覆盖保存 LoRA checkpoint；
评估模式加载最终 checkpoint，不创建 optimizer，也不修改参数。

### `moe_gate_lora/stats.py`

`RunningMoments` 在线累计 `count/sum/sum_sq`。在样本集合相同的情况下，
最终 mean 和 population std 与加载全部样本后计算严格一致。

统计等价不代表任意训练过程等价：多 epoch 流式训练只与 epoch 数、batch
边界、样本顺序和重新生成数据均相同的离线训练对应。当前每个 epoch 使用
相同 prompt 顺序，尚未实现 shuffle。

### `moe_gate_lora/plot.py`

每处理一个 batch，根据当前在线聚合结果覆盖 `overlap.png`。画图不保存所有
token 的预测，只使用 running statistics。

### 其他文件

- `__main__.py`：支持 `.venv/bin/python -m moe_gate_lora`。
- `__init__.py`：包定义。
- `README.md`：使用示例和环境变量说明。
- `tests/test_stats.py`：running moments 与 top-k overlap 测试。
- `tests/test_trainer.py`：adapter 训练、保存/加载和流式处理测试。

多 epoch 测试会连续处理两轮数据，断言 batch/optimizer step 从 2 累加到 4、
AdamW state 没有重置，并检查 epoch 只能按顺序完成一次。

## 4. vLLM 内部 MoE Trace

### `vllm/model_executor/layers/fused_moe/moe_trace.py`

新增 opt-in `MoETraceCollector`。只有设置 `VLLM_MOE_TRACE_DIR` 时启用，且
当前要求 `--enforce-eager`。`VLLM_MOE_TRACE_MODE` 区分两种记录格式：

- `expert_distribution` 在 router 内部对全部 logical top-k IDs 执行
  `torch.bincount`，只保存每个 expert 的计数、scheduled token 数和轻量元数据。
- `lora_training` 保存训练需要的完整 tensor；gate-LoRA pipeline 固定选择此模式。

完整训练记录包括：

- rank、forward step、layer id/name、request id。
- 当前层 router 输入 activations 和 router logits。
- 当前层 top-k weights 和 logical top-k IDs。
- 可选的下一层 base gate logits。

top-k IDs 在 EPLB logical-to-physical 映射之前记录，因此两种模式都统计 logical
expert。只有 `lora_training` 的大 tensor 最多选择 4096 token；
`expert_distribution` 的内部计数覆盖全部 scheduled routing rows，不受该上限影响。
完整模式把大张量转为 CPU FP16、ID 转为 INT32、top-k weight 转为 FP32 后保存。

该实验固定面向 Qwen3 MoE，假设所有层都使用 vLLM 标准
`FusedTopKRouter(softmax, renormalize=True)`。训练 parent 从第一个 batch 的
top-k 宽度和 logits 专家维度创建一个 router，所有层共享；trace 不再序列化
router 类型和参数。

### `vllm/model_executor/layers/fused_moe/router/base_router.py`

新增 `trace_fn` 和 `set_trace_fn()`。`_select_experts()` 在完成 logical routing、
尚未执行 EPLB 映射时回调 collector。原有 `capture_fn` 保留，trace 使用独立
接口。

### 两版 GPU model runner

`vllm/v1/worker/gpu/model_runner.py` 和
`vllm/v1/worker/gpu_model_runner.py` 在真实 forward 准备好请求 metadata 后
调用 `moe_trace_collector.begin_forward()`，传入 scheduled/computed token、
prefill 长度和 request IDs。

### `vllm/v1/worker/gpu_worker.py`

模型初始化和 profile 完成后调用 `maybe_attach_moe_trace()`，把 collector
安装到所有兼容的 `MoERunner.router`。

## 5. Serving 侧下一层 Gate 预测

### `vllm/model_executor/layers/fused_moe/next_gate_lora.py`

该文件与根目录训练包有意分离。根目录工具负责训练；此文件只负责 serving
加载和预测。

加载 checkpoint 时校验 A/B 的 rank、hidden size 和 routed-expert 数。快速
路径计算并保存独立 dense delta：

```text
delta_W = scale * B @ A
```

原始 `W_next` 不会被修改。shared-expert gate 不属于 routed LoRA，对应 delta
行补零。

只有相邻 gate 的 input size、dtype、device 和 output dtype 兼容时才安装 fused
predictor；否则保留原 gate projector 加低秩 LoRA 的 fallback。最后一个 MoE
层显式清理 predictor，因为它没有下一层需要预测。

### `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`

`MoERunner` 新增 predictor 的安装、清理、读取和执行接口。兼容路径在当前
gate 位置计算 current logits 与 predicted-next logits，并调用下一层 router
的 `_compute_routing()`。不兼容路径可在独立 CUDA stream 上执行，并通过 event
同步。

当前预测是 side channel：当前层真实路由不变；到达下一层时，仍使用真实
`h_(i+1)` 和原始 `W_next` 重新计算真实 gate。预测 top-k 只为未来 expert
prefetch/EPLB 策略保留，目前不改变 expert dispatch。

### 共享 logits workspace

`vllm/model_executor/kernels/linear/dual_gate_lora.py` 按以下 key 维护共享 flat
workspace：

```text
(CUDA device, output dtype, DBO micro-batch slot)
```

所有层共享 current/predicted 两个 flat buffer。每层取所需前缀并 reshape 成
contiguous `[M,E]`；容量不足时扩容，安装 predictor 时为常见小 batch 预留
16-token 容量。DBO slot 分离，避免两个 micro-batch 同时覆盖。

`get_last_next_gate_prediction()` 中的 predicted logits 是临时 workspace view，
下一层复用同一 slot 后可能被覆盖；predicted top-k IDs 是独立输出。

### Worker 生命周期

`gpu_worker.py` 在模型加载后安装 serving predictor，并在 `reload_weights()` 后
重新安装，保证 delta 和 gate 引用对应新权重。开关和目录分别使用
`VLLM_SC_EPLB`、`VLLM_SC_EPLB_LORA_DIR`。

## 6. Dual-gate Kernel

### CuTeDSL 实验路径

`vllm/model_executor/kernels/linear/cute_dsl/_ll_bf16_dual_gate.py` 新增
`LLBf16DualGate`，复用 `LLBf16Dotprod` 的向量化 BF16 load、FP32 accumulator、
warp/shared-memory reduction 和 PDL。

一个 launch 的逻辑是：

```text
current output column:       X @ W_current.T
predicted-next output column: X @ W_next.T + X @ delta_W.T
```

`cute_dsl/ll_bf16.py` 新增 host wrapper、编译缓存和能力检测。当前 CuTeDSL
资格为 `M<=16`、BF16 inputs/weights、FP32 logits、SM90+、K 可被 8 整除、
contiguous 且 CuTeDSL 可用。

该 kernel 是实验实现，尚未在当前工作区执行 CuTeDSL JIT 或 GB200 数值/性能
验证。

### Triton fallback

`dual_gate_lora.py` 当前仍保留 Triton dual-gate kernel。CuTeDSL 条件不满足时
dispatcher 选择 Triton。刚开始的“移除 Triton、只保留 CuTeDSL/cuBLAS”重构
已经按用户要求暂停，没有继续。

## 7. 性能 Benchmark

`benchmarks/kernels/benchmark_dual_gate_lora.py` 比较四种模式：

| 模式 | 定义 |
| --- | --- |
| `single` | 只计算当前层 gate |
| `dual` | 单个 dual kernel 计算 current 与 next+delta |
| `parallel` | 两个 CUDA stream 同时发射 current gate 与 next+delta |
| `concat` | 预合并 next+delta，再拼接 current/next 权重做一次 GEMM |

默认形状包括 Qwen `(2048,128)`、DeepSeek `(7168,256)` 和 Kimi
`(7168,384)`，默认 token 数是 `1,2,4,8,16`。计时前 warmup，并用 PyTorch
FP32 reference 检查 current/predicted logits。输出微秒、相对 single、logical
TFLOPS、最大误差、额外显存和所选 backend。

LoRA rank 只影响加载阶段生成 `delta_W`；在线 kernel 使用 dense delta，所以
rank 不改变推理 FLOPs。

## 8. 测试修改

`tests/model_executor/test_next_gate_lora.py` 覆盖拼接 correctness baseline、
delta 不修改 base weight，以及共享 workspace 跨形状复用和 contiguous。

`test_trainer.py` 使用可计数的标准 fake router，断言每个 batch 的
baseline/LoRA 两条路径都调用 `_compute_routing()`，并且所有层共享一个
router 实例，不再直接调用 `torch.topk(logits)`。

`tests/kernels/test_ll_bf16_gemm.py` 新增 dual-gate CUDA 测试，检查在 SM90+、
BF16、FP32、`M<=16` 时选择 `cutedsl`，并比较 shape、dtype、有限值、cosine
similarity 和 PyTorch FP32 reference。

用户最新要求的“新 CuTeDSL kernel 与现有 `ll_bf16_gemm` baseline 直接比较”
尚未实现。当前比较对象是 PyTorch FP32 reference；该项因暂停实现列为后续
工作。

## 9. 当前验证状态

已经执行并通过：

```text
py_compile
ruff-format
ruff-check
git diff --check
```

当前 `.venv` 未安装 PyTorch，导入结果为
`ModuleNotFoundError: No module named 'torch'`。因此 pytest、Triton JIT、
CuTeDSL JIT、CUDA graph、GB200 数值/性能和 DBO 并发压力测试均未执行。

## 10. 已知限制与后续工作

1. 在 GB200 上直接比较新 CuTeDSL dual kernel 与三次现有
   `ll_bf16_gemm` baseline，并记录 cosine similarity、max error 和 top-k
   一致率。
2. 决定最终是否删除 Triton fallback，或为大 batch 保留 cuBLAS/其他路径。
3. 调优 GB200 上 CTA threads、M/K specialization 和预测分片的带宽。
4. 验证 shared-expert gate、不同 expert 数和不同 output dtype。
5. workspace 目前以 device/dtype/DBO slot 为 key；同一进程同一 GPU 同时运行
   多个 engine 时，需要增加 engine/workspace owner 隔离。
6. predicted logits 是临时 view；长期保存必须由消费者显式复制。默认设计只
   在线消费 top-k，不保存全部 logits。
7. trace 最多记录 4096 token 且要求 eager mode；更大 trace 或 CUDA graph
   内采集需要另行设计。

## 11. 建议的 GB200 验证命令

```bash
.venv/bin/python -m pytest \
  tests/model_executor/test_next_gate_lora.py -v

.venv/bin/python -m pytest \
  tests/kernels/test_ll_bf16_gemm.py \
  -k dual_gate_lora -v

.venv/bin/python benchmarks/kernels/benchmark_dual_gate_lora.py \
  --shapes qwen deepseek kimi \
  --tokens 1 2 4 8 16 32 64 128 \
  --ranks 4 8 16 \
  --warmup 50 \
  --repetitions 500
```

## 12. 工作区说明

`.codegraph/` 是代码索引目录，不属于上述功能修改，不应作为 MoE/gate-LoRA
实现提交。本文列出的新功能文件目前仍是未跟踪文件，提交前需要人工逐行
审查并决定纳入范围。
