# v0.20.2 / v0.20.2rc1 升级说明

本次同时升级实际源码、运行依赖、适配层、测试与设计文档。

| 组件 | 固定版本 / 提交 |
|---|---|
| vLLM | v0.20.2 / bc150f50299199599673614f80d12a196f377655 |
| vLLM-Ascend | v0.20.2rc1 / 367b8e62da799870a7476ce34f5f7658589a8aad |
| Mooncake | fd72779c2ee8ec28379bc6487f2113e3efe418d8，保持原版本 |
| CPU PyTorch | 2.11.0+cpu |
| Transformers | 5.5.3 |

## 行为变化

- 旧 `AscendScheduler` 已被上游移除。`--engine ascend` 现在隔离执行 rc1 原始 `BalanceScheduler`，保持上游默认均衡关闭，实际调度进入 vLLM 父类。
- 同步调度仍显式固定为 `async_scheduling=False`。`async_full` 指异步 KV 加载，不代表启用 vLLM AsyncScheduler。
- `configs/ascend-prefill-first.json` 替换为 `configs/ascend-unchunked.json`；它只关闭 chunked prefill，不模拟已移除的旧 Ascend prefill-first 策略。
- vLLM 新的 `skipped_waiting` 队列由上游维护，轨迹单独记录。short_io 仍仅重排 waiting。

## 接口适配

| 接口 | 新版适配 |
|---|---|
| Scheduler | 显式传入 block_size、hash_block_size；补齐观测、PCP、EC 配置 |
| SchedulerConfig | 显式指定 is_encoder_decoder，关闭异步调度 |
| CacheConfig | 移除已删除的 swap_space 参数 |
| FullAttentionSpec | 使用具名参数，避免字段顺序变化 |
| Request | 移除 eos_token_id 构造参数；保留 SamplingParams 长度停止规则 |
| Hash | 从 vllm.utils.hashing 导入 sha256；继续验证与 Conductor C++ 一致 |
| KVConnector | 提供 KVTransferConfig 与真实 KVCacheConfig，通过原始基类初始化 |
| ModelRunnerOutput | 使用具名字段回传 token 和异步 KV 接收完成信息 |

## Ascend 加载边界

`vendor/vllm_ascend/patch/platform/patch_balance_schedule.py` 保留完整原文件。`lab/ascend_loader.py` 仅执行原始 import、`_balance_scheduling_enabled` 与 `BalanceScheduler` AST 节点，不修改方法体；不执行该文件里的 EngineCore 进程启动替换。附带的 platform.py 与 patch/platform/__init__.py 用于核查版本和选择逻辑，不在 CPU 仿真中加载。

当前没有启用/验证 DP 均衡、dynamic batch、profiling chunk、recompute、AsyncScheduler 或 NPU runner。默认类走父类路径是上游该配置的真实行为。

vLLM CPU requirements 固定 torch 2.11.0；Ascend NPU requirements 固定 torch/torch-npu 2.10.0。本包是 CPU 控制层环境，不应把完整 Ascend 插件装入此环境。真实 NPU 环境需另建。

## 使用

在新目录解压，运行 `bash scripts/setup.sh`。安装脚本创建不继承系统包的 Python 3.12 环境，编译 Conductor 并运行测试。随后按 README 运行各场景。源码原始哈希和提交见 `UPSTREAM.lock.json`，实际验证结果见 `VERIFICATION.md`。

计算及 IO 参数仍未做 NPU 实测校准；`timing_calibrated=false`。
