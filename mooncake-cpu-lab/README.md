# Mooncake + vLLM / Ascend CPU 调度实验台

这是一份已经运行验证的**真实控制代码 + 虚拟执行**起步工程。主设计见 [DESIGN.zh-CN.md](DESIGN.zh-CN.md)。

实际运行的是 Mooncake 的 C++ `PrefixCacheTable`、vLLM V1 `Scheduler` 或 vLLM-Ascend `BalanceScheduler`、真实 `KVCacheManager` / `BlockPool` / 请求状态机。全局路由策略、虚拟时钟、计算和共享存储模型由本工程实现。

**边界：Conductor 在这里提供官方前缀索引，不是论文完整的全局调度器；本工程没有启动完整 vLLM EngineCore、NPUWorker、torch_npu、CANN、HCCL 或真实 Mooncake Store。** 默认性能系数是演示值，输出中 `timing_calibrated=false`。这是可直接搭建、继续修改的研究起点，不能作为已验证的 NPU 性能预测器。

版本固定：**vLLM v0.20.2、vLLM-Ascend v0.20.2rc1、CPU torch 2.11.0+cpu**。完整提交见 `UPSTREAM.lock.json`。Ascend 类通过 `lab/ascend_loader.py` 隔离载入，方法保持原样；均衡、dynamic batch、profiling chunk、recompute 均未启用。

## 快速搭建

已在 Linux x86_64、Python 3.12、g++ 13、OpenSSL 3 上验证。Windows 请使用 WSL2 Ubuntu；macOS 不在当前安装脚本验证范围。

```bash
unzip mooncake-cpu-lab.zip
cd mooncake-cpu-lab

# Ubuntu 系统依赖；如果已有则跳过。
sudo apt-get update
sudo apt-get install -y g++ libssl-dev python3-pip unzip
python3 -m pip install --user uv

# 使用 Python 3.12；uv 会使用或获取对应解释器。
bash scripts/setup.sh

# 真实 vLLM Scheduler + 真实 Mooncake 前缀索引 + 按层 IO 仿真。
.venv/bin/python -m lab.run --out results/vllm-layerwise

# Ascend rc1 原始 BalanceScheduler；默认均衡关闭，调用真实 vLLM 父类调度。
.venv/bin/python -m lab.run --engine ascend --out results/ascend-layerwise

# 验证关闭 chunked prefill 的配置。
.venv/bin/python -m lab.run --engine ascend \
  --config configs/ascend-unchunked.json --out results/ascend-unchunked

# 整段异步加载，走真实 WAITING_FOR_REMOTE_KVS 路径。
.venv/bin/python -m lab.run --load-mode async_full --out results/vllm-async

# 对比全局路由 / 本地等待队列策略。
.venv/bin/python scripts/compare.py
```

首次使用请新建目录解压，避免旧版本环境/源码残留。第一次安装需要联网获取 Python 包。安装 vLLM 的 wheel 是为了获得准确的版本元数据和依赖文件；显式 `--no-deps` 避免拉取整套 CUDA 运行库。这里的 Python 依赖子集仅支持控制层实验，不构成一个可提供真实推理的 vLLM 安装。

不要在本 CPU 环境安装完整 vllm-ascend / torch-npu；Ascend rc1 的 NPU 环境依赖 torch 2.10.0，与这里的 CPU 控制层环境分开。

安装脚本会运行 tests；依赖下载失败可以重新执行。源码已经随包提供，不需要克隆最新 `main`。`vendor/vllm` 会优先于 wheel 被导入。

## 代码在哪里改

| 目的 | 位置 |
|---|---|
| 改 vLLM batch、chunk、抢占、running/waiting 调度 | `vendor/vllm/vllm/v1/core/sched/scheduler.py` |
| 改 vLLM KV 分配、block pool | `vendor/vllm/vllm/v1/core/kv_cache_manager.py`、`block_pool.py` |
| 改 Ascend BalanceScheduler 真实调度代码 | `vendor/vllm_ascend/patch/platform/patch_balance_schedule.py` |
| 改官方 Conductor 索引 / hash 查询 | `vendor/Mooncake/mooncake-conductor/src/prefixindex/` |
| 改本工程的全局路由、观测状态和 cost | `lab/run.py`：`arrive()`、`estimated_io()`、`sample()` |
| 改本工程的等待队列排序示例 | `lab/engine.py`：`Engine.step()` |
| 改仿真 KVConnector / 完成事件回传 | `lab/engine.py`：`SimConnector`、`Engine` |
| 改 batch 每层计算和 IO barrier | `lab/engine.py`：`Batch` |
| 改盘 / Path / NIC 共享带宽与排队 | `lab/storage.py` |
| 改计算系数 | `configs/demo.json` 的 `compute` 与 `model` |

Python 修改后直接重新运行；C++ 修改后运行 `bash scripts/build_conductor.sh`。`UPSTREAM.lock.json` 保存原始提交和原始文件哈希，用于识别你后续做了哪些改动。

## 输出

每次运行生成 `summary.json`、`config.json`、`events.jsonl`、`requests.jsonl`。包含虚拟 makespan、端到端 TTFT（含排队）、TTFT SLO 满足率、goodput、各 worker 的 compute/stall/idle 时间，以及逐 batch 调度和逐层计算 / IO 事件。

`traces/demo.jsonl` 是 12 个请求的合成正确性样例，**不是 Mooncake trace**。正式实验请用实际 trace 和校准过的计算 / IO 模型。

## Mooncake trace 转换

官方 FAST25 trace 的 `timestamp` 单位为毫秒，`hash_ids` 对应 512-token block；没有原始 token。转换器保留时间、长度和完整块前缀相等关系，生成明确标记的代理 token，不能恢复原始文本。

```bash
.venv/bin/python scripts/convert_mooncake.py \
  /path/to/conversation_trace.jsonl traces/mooncake-100.jsonl \
  --timestamp-unit ms --warmup-requests 50 --limit 100
```

转换器还生成 `traces/mooncake-100.pool.jsonl`。复制 demo 配置，设置 `initial_pool_file` 为该文件，并按输出提示增大 `max_model_len`、`num_blocks`；建议设置 `block_size=512`，避免把未知的子块信息当成已知。保持 `chunked_prefill=true`。warmup 请求被假定在测量前已经完整完成；测量区间共享池仍为只读，不会自动加入后续生成的 KV。

## 已覆盖 / 待扩展

当前覆盖：双层策略修改入口、真实本地调度状态机、共享带宽竞争、按层 batch barrier、异步 KV 完成、可延迟的存储遥测、固定初始共享池。

待扩展：完整 PD 分离及 KV 交接、动态共享池写回 / 淘汰、HBM prefix caching 事件发布、真实 Conductor HTTP/ZMQ 服务链路、NPU 算子 / 通信 / 图执行时延校准、TP/PP/EP、推测解码。详细切分、接口和验收条件见设计文档。

第三方源码许可证见 `licenses/`。`native/logshim/glog/logging.h` 仅把诊断日志输出到 stderr；上游索引与 hash 源码保持原样。
