# 仿真环境符合性检视与 vLLM v0.20.2 升级评估

| 项目 | 约定 |
|---|---|
| 检视对象 | v0.11.0 基线工程（检视时名为 `mooncake-cpu-lab/`，现归档于 `archive/mooncake-cpu-lab-v0.11/`；提交基线 `bfc1874`，UPSTREAM 锁定 vLLM v0.11.0 / vllm-ascend v0.11.0 / Mooncake `fd72779`） |
| 需求依据 | 用户原始需求、`ChatGPT-全局调度仿真方案-20260922-2147.md`、`Mooncake_vLLM_CPU_Simulation_Design.zh-CN.md`、`docs/` 两份调度研究设计（20260913、20260917） |
| 检视方法 | 逐文件阅读 lab 代码与 vendored 上游代码；对照测试与运行结果；对 vLLM v0.20.2（tag `bc150f5`）与 vllm-ascend v0.20.2rc1（tag `367b8e6`）实际拉取源码逐文件 diff，关键破坏点已逐一复核 |
| 日期 | 2026-09-22 |

**结论速览**

1. **符合性**：工程满足"控制面用真实代码、执行面仿真"的地基性需求——vLLM V1 Scheduler / KVCacheManager / BlockPool / 请求状态机、vllm-ascend `AscendScheduler`（含 prefill-first 分支）、Mooncake Conductor 的 C++ 前缀索引确实以原始代码在 CPU 上运行，mock 边界清晰且有 16 项测试背书。但有三条边界必须认清：(a) 公开 Conductor 本身只是索引器而非论文级全局路由器，全局路由是 lab 新增代码；(b) `docs/` 中的研究策略（SS/LS/SL/LL 分组、B0/B1 带宽、P0/P1 选路、S1 组批错峰、KV 写回）**均未实现**，本工程是研究起点而非研究成果；(c) 计时系数未校准（`timing_calibrated=false`）。
2. **升级评估**：vLLM v0.11.0 → v0.20.2 是**中等规模、机械但面广的适配**，模块路径全部幸存、核心调度/状态机语义兼容，但 `lab/engine.py` 依赖的 7 处构造点全部破坏（已逐条定位）。最大的硬点是 **vllm-ascend 在 v0.20.2rc1 删除了 `vllm_ascend/core/scheduler.py`**，prefill-first 调度语义没有上游继任者，必须以 lab 维护 fork 的方式移植。Mooncake prefixindex 零漂移，无需动作。总工作量：仅 vLLM 主线约 **3–5 人日**；含 AscendScheduler prefill-first 移植约 **10–15 人日（2–3 周）**。

---

## 第一部分：符合性检视

### 1.1 需求逐条对照

用户原始需求："调度器用真实 Mooncake Conductor 代码；推理引擎层用真实 vLLM 代码（含 vllm-ascend 适配）；需要 NPU 实跑或接存储的全部 mock 成仿真代码；调度策略修改完全基于真实代码进行。"

| # | 需求 | 状态 | 证据 |
|---|---|---|---|
| R1 | Mooncake Conductor 真实代码 | **部分满足（受上游限制）** | `native/conductor_bridge.cpp` 编译执行原始 C++ `PrefixCacheTable`/`HashStrategy`（Register/StoreShared/RemoveShared/Query），`vllm_v1/sha256/0/low64_be` hash profile 与 vLLM block hash 逐字节一致（`tests/test_lab.py:26-34`）。但公开 Conductor 只有索引器，"全局路由决策"不在其中——lab 的 RR/least_tokens/storage_aware 路由（`run.py:104-119`）是自研策略挂载点。设计文档 §1 已如实声明此边界 |
| R2 | vLLM 引擎层真实代码 | **满足（控制面范围）** | 真实 `Scheduler.schedule()/update_from_output()`（`engine.py:177,158`）、真实 `Request`/`SamplingParams`/停止检查、真实 `KVCacheManager`/`BlockPool`/抢占释放、真实 `WAITING_FOR_REMOTE_KVS` 异步状态机。vendored vLLM 从 0.11.0 wheel 原样拷贝无补丁（`scripts/prepare_vllm_source.py`），执行文件 SHA-256 有记录（`VERIFICATION.md:13`）。未含 EngineCore/NPUWorker/tokenizer（设计文档 §12 给出后续集成方案） |
| R3 | vllm-ascend 适配真实代码 | **满足（调度器范围）** | 原始 `AscendScheduler` 按文件路径直接加载避免 NPU 插件链（`engine.py:73-81`）；chunked prefill 开启时走父类、关闭时执行真实 prefill-first 分支与 watermark 检查；`results/ascend-*` 有运行输出。注意：同版本下两引擎默认轨迹相同是代码事实（父类委托），非接线遗漏 |
| R4 | NPU/存储 mock 成仿真 | **满足** | 计算用解析公式（`run.py:74-84`），存储用全局共享流体模型（盘/Path/NIC 三类资源 max-min，`storage.py:20-34,87-96`），虚拟事件时钟（`events.py`）驱动，`time.monotonic` 等真实时间被替换为虚拟时钟（`engine.py:110-113`） |
| R5 | 调度策略修改基于真实代码 | **满足（地基），策略未实现** | README"代码在哪里改"表 + 设计文档 §10 给出分层修改点。但目前只有 3 个全局路由示例和 1 个 waiting 重排示例（`short_io`）；`docs/` 研究设计中的策略族（见 §1.4）尚未落地 |

### 1.2 "真实代码"边界的精确刻画

检视确认的真实执行链路（请求生命周期）：

```
trace 到达 → run.py arrive()【lab：全局路由】
  → engine.py add() 构造真实 Request → 真实 Scheduler.add_request() 入真实 FCFS waiting 队列
  → 真实 Scheduler.schedule()【vLLM 0.11.0 原始代码】
      · 真实 token budget / chunked prefill / max_num_seqs / 抢占
      · 真实 kv_cache_manager.get_computed_blocks()
      · connector.get_num_new_matched_tokens() → lab SimConnector
          → ctypes → 真实 Mooncake C++ PrefixCacheTable.Query【Mooncake 原始代码】
      · 真实 allocate_slots(delay_cache_blocks=…) / WAITING_FOR_REMOTE_KVS（async_full）
  → lab Batch/storage 仿真执行（逐层 IO barrier / 整段异步传输）
  → 真实 Scheduler.update_from_output(so, ModelRunnerOutput)【vLLM 原始代码】
      · 真实停止检查 / token 追加 / block 释放回真实 BlockPool
```

真实/仿真分界与设计文档 §2 声明完全一致，未发现文档与代码不符之处。两点检视中值得强调的实现细节：

- **多实例**：`run.py:44` 在一个进程内构造 `workers` 个各自独立的真实 `Scheduler`+`BlockPool`，共享一个虚拟时钟与一个全局存储模型——跨 NPU 的 IO 干扰（一个 worker 的新 IO 推迟其他 worker 的流）真实存在于模型中，有测试覆盖（`test_bandwidth_changes_when_flow_arrives`）。
- **观测与真值分离**：策略只能读延迟送达的 snapshot（`run.py:121-135`），路由有 `pending_dispatch` 补账防惊群——这满足了 `docs/` 设计中"普通策略不可见真值"的实验方法论要求（信息边界已就位，尚无噪声/版本化）。

### 1.3 mock 层的抽象质量

| 模拟层 | 当前抽象 | 对研究有效性的影响 |
|---|---|---|
| 计算时延 | `C(B)=α+(βΣq+γΣq(c+(q+1)/2))/η(B)`，层间均分 | 公式形状与 `docs/` synthetic-v1 式(3)同构（常数/系数可配置替换）；上下文含复用 KV，未犯"命中即免 attention"错误。但 MoE/异构层、图模式、通信重叠未建模，未校准 |
| 存储 | 盘×Path×NIC 多资源 max-min 流体模型；一致性哈希环（64 vnode/盘）+ 确定性 Path 选择 | 与 `docs/` 的 B0 规则（组基础份额+空闲均分）**不是同一分配器**；Path 由 hash 决定而非"最短队列"选路。当前模型是通用公平共享，用于复现 B0/P0/B1/P1 需要改造 `storage.py` |
| 遥测 | 周期采样 + 传递延迟 | 满足"状态时效"研究的最小要求；无噪声、无版本号、无置信度 |
| KV 池 | 初始只读种子（t=0 StoreShared 后不变） | 无写回/淘汰/迁移闭环——`docs/` 设计的写负载、F_persist、缓存反馈实验全部依赖此扩展 |

### 1.4 与 `docs/` 研究设计的差距清单（研究落地所需的全部增量）

这是"是否满足需求"的核心：工程提供的是真实代码地基，`docs/`（20260917 设计）中的研究机制需要以下增量才能实验。按依赖排序：

| # | `docs/` 设计要求 | lab 现状 | 增量位置 | 规模 |
|---|---|---|---|---|
| G1 | SS/LS/SL/LL 分类（h≤80000/u≤512）与 Path 分组 96/96/32/32 | 无分类概念；Path 由 hash(worker,block) 决定 | `storage.py`：分组目录 + `classify(h,u)` | 小 |
| G2 | B0 带宽规则（组基础额+活跃余量均分，式 6/7）与 B1（组保底+权重借用，式 18） | 通用 max-min，无组概念 | `storage.py` 分配器参数化为可选策略 | 中 |
| G3 | P0/P1 选路（最短队列/预计完成时间） | 无选路策略（确定性 hash） | `storage.py` 提交入口 | 小–中 |
| G4 | 中央唯一等待队列 + 空闲实例原子领取完整 prefill 批（S0/S1 组批、错峰启动） | 请求到达即路由进各实例 vLLM waiting 队列；组批/预算由真实 vLLM schedule() 决定 | **架构性改动**，见下文"模型张力" | 大 |
| G5 | KV 写回（新 KV 逐层写、写暂存池、读写竞争 40 GB/s、F_persist） | 完全没有（池只读） | `storage.py`+`engine.py`+Conductor 发布时机 | 大 |
| G6 | X∈{2,4,8} 卡/盘配比、P 卡/实例 | worker 无卡数概念 | `configs`+`run.py` 系数缩放 | 小 |
| G7 | 四目标（M/SLO/平均 TTFT/归一化 TTFT）+ 分组损害报告 | M/TTFT/SLO/goodput 已有；归一化 TTFT 需 isolated 基线（trace 可选提供）；分组报告无 | `run.py` 指标层 | 小 |
| G8 | 计算画像替换（synthetic-v1 → 实测 profile 表） | 解析公式（系数可配） | `run.py compute_time()` 换查表 | 中（依赖实测） |
| G9 | 观测噪声/版本/控制开销、E26–E31 实验矩阵 | 无 | 逐步补 | 中 |

**模型张力（必须先决策）**：`docs/` 的形式化模型是"中央队列 + 完整 prefill 批 + 批内成员到末层不变 + 单实例单活动批"，这与 vLLM 原生的 per-instance continuous batching / chunked prefill / running·waiting 竞争**不是同一调度模型**。两条实现路线：

- **路线 A（贴近 docs 模型）**：在真实 `Scheduler.schedule()` 内实现中央组批/完整批/领取语义——即把 S1 策略写进真实调度器（修改 waiting 选择、token 预算、启动时序的代码位置，设计文档 §10.2 已列钩子表）。保留真实代码的价值在于：KV 分配/抢占/状态机仍是 vLLM 真实实现，但调度主干被策略化改写，改写幅度大。
- **路线 B（贴近 vLLM 原生语义）**：保留真实 continuous batching 语义，把"组批/错峰"转译为路由时延（到达后暂存中央队列、按策略投放实例）+ waiting 重排 + chunk/budget 参数。更少侵入真实代码，但与 `docs/` 形式化模型不再逐条对齐，需重推导理论结论的适用条件。

无论哪条路线，G1–G3、G5–G7 都要先做；G4 决定上层实验怎么解释。建议先在当前 0.11.0 基线上完成 G1–G3 + G7（纯 `storage.py`/指标层，零升级风险），再做 G4 决策。

### 1.5 测试与可信度边界

- 16 项测试通过（干净虚拟环境重装验证），覆盖：C++ hash 与 vLLM 一致、层 barrier、异步 KV 未到不调度、chunk 中间轮不出 token、真实抢占/重算、block 无泄漏、确定性回放、时间守恒。
- 未测/未证：Ascend prefill-first 与 stock 的**排序差异**（只测了分支可执行）；遥测延迟/补账逻辑的独立单测；Ascend 路径的抢占与 watermark；性能真实性（`timing_calibrated=false`）。
- 运行环境边界：已在 Linux x86_64 + Python 3.12 + g++13 + OpenSSL3 验证；**macOS 不在验证范围**（C++ 桥需 Linux 工具链编译，本机检视时 `build/` 尚未构建）。

### 1.6 符合性结论

工程**成立且质量高于典型"起步工程"**：真实/仿真边界诚实、有哈希审计与确定性回放、研究方法论（观测分离、消融矩阵）已内建到架构。需求满足度的准确表述是：**"基于真实代码做调度策略研究"的地基 100% 就位；研究策略本身 0% 实现；性能结论 0% 可信（未校准）**。这不构成否定——设计文档对三点的声明与代码事实一致，没有夸大。

---

## 第二部分：升级评估——vLLM v0.20.2 / vllm-ascend v0.20.2rc1

### 2.1 版本事实

| 项 | 数据（已逐一核实） |
|---|---|
| vLLM v0.20.2 | tag `bc150f50299199599673614f80d12a196f377655`，发布 2026-05-10；基线 v0.11.0 发布 2025-10-02，跨度约 7.5 个月 |
| 调度器变更强度 | `vllm/v1/core/sched/scheduler.py` 期间 **234 个提交**；文件 1296→2308 行（diff +1370/−469） |
| vllm-ascend v0.20.2rc1 | tag `367b8e62da799870a7476ce34f5f7658589a8aad`；官方 Dockerfile `ARG VLLM_TAG=v0.20.2`，与 vLLM v0.20.2 精确配对；构建 pin torch 2.10.0 / torch-npu 2.10.0 / transformers 5.5.3，Python 3.11 + CANN 9.0.0 基础镜像 |
| Python 要求 | vLLM v0.20.2 `requires-python >=3.10,<3.15`；lab 用 3.12，**满足** |
| V0 引擎 | 已彻底移除（`vllm/core/` 目录不存在）；`SchedulerConfig.scheduler_cls` 默认 `None`（V1） |
| Mooncake | `fd72779...HEAD` 仅 2 提交 28 文件，**prefixindex 零改动**；vendored API（PrefixCacheTable/HashStrategy 等）无漂移 |

### 2.2 逐模块漂移与破坏点

对 lab 实际 import 的 9 个 vLLM 模块做了 vendored 0.11.0 vs v0.20.2 全文件 diff：

| 模块 | diff 行数 | 关键结论 |
|---|---|---|
| `v1/core/sched/scheduler.py` | 2676 | `schedule()`/`update_from_output()` 签名不变；`__init__` **新增必填第 4 参 `block_size`**（+可选 `hash_block_size`）；读取 `vllm_config.observability_config` |
| `v1/core/kv_cache_manager.py` | 656 | 常用方法幸存；`create_empty_block_list` 改名 `create_kv_cache_blocks`；`__init__` 新增必填 `hash_block_size`/`max_num_batched_tokens` |
| `v1/core/block_pool.py` | 411 | 方法集稳定；`__init__` 新增 `hash_block_size` |
| `v1/core/kv_cache_utils.py` | 2133 | **`get_request_block_hasher`/`init_none_hash` 签名完全不变**（Mooncake 桥一致性风险低） |
| `v1/request.py` | 400 | `Request.__init__` **删除 `eos_token_id`**；`WAITING_FOR_REMOTE_KVS` **幸存**（已核实 v0.20.2 `request.py:309`）；新增 `WAITING_FOR_STREAMING_REQ` 等状态 |
| `v1/outputs.py` | 307 | `ModelRunnerOutput`/`KVConnectorOutput` 字段兼容（新增可选字段） |
| `v1/kv_cache_interface.py` | 848 | 三个 Spec 类幸存但改为 **`kw_only=True`**，位置构造报错；新增可选字段 |
| `kv_connector/v1/base.py` | 626 | 生命周期方法名全部幸存（`get_num_new_matched_tokens` 等）；**`__init__` 在 `kv_transfer_config=None` 时直接 raise**（已核实 v0.20.2 `base.py:196-201`） |
| `vllm.config`（包） | — | `from vllm.config import CacheConfig, SchedulerConfig` 仍可用；`CacheConfig` **删除 `swap_space`**；`SchedulerConfig` 的 `is_encoder_decoder` 变为必填 InitVar；删除 `num_lookahead_slots`/`cuda_graph_sizes`/`send_delta_data` |
| `SchedulerOutput`（`sched/output.py`） | — | `grammar_bitmask` 等迁出到新 `GrammarOutput`；`CachedRequestData.resumed_from_preemption` → `resumed_req_ids: set[str]`；新增 `preempted_req_ids`/`prefill_token_ids` 等 |

**`lab/engine.py` 逐条破坏清单**（升级时按下表修改，全部经源码复核）：

| # | 位置 | 0.11.0 写法 | v0.20.2 破坏 | 修法 |
|---|---|---|---|---|
| B1 | `engine.py:90` | `CacheConfig(..., swap_space=0, ...)` | `swap_space` 已删除 → TypeError | 删参 |
| B2 | `engine.py:85-89` | `SchedulerConfig(...)` | `is_encoder_decoder` 必填 → TypeError | 补 `is_encoder_decoder=False` |
| B3 | `engine.py:99` | `FullAttentionSpec(bs,1,1,torch.float32,False)` | kw_only → TypeError | 改关键字构造 |
| B4 | `engine.py:108` | `Scheduler(vc, kv, NoStructuredOutput(), log_stats=False)` | 缺 `block_size` → TypeError；且读 `observability_config` → AttributeError | 补 `block_size`；SimpleNamespace VllmConfig 增加 `observability_config` |
| B5 | `engine.py:131-133` | `Request(..., eos_token_id=None, ...)` | 参数已删 → TypeError | 删参 |
| B6 | `engine.py:45-51` | `SimConnector` 以 `kv_transfer_config=None` 走父类 | 父类 raise ValueError | 构造最小真实 `KVTransferConfig`（或给 SimpleNamespace 填充） |
| B7 | `engine.py:174-176` | 把 `scheduler.waiting` 当 deque `.clear()/.extend()` | 0.20.2 队列实现待复核（若仍为 `FCFSRequestQueue` deque 子类则无害） | 升级时验证 |
| B8 | `engine.py:93-96` | SimpleNamespace 伪造 VllmConfig | 任何新读取字段都会 AttributeError | 对照 `Scheduler.__init__` 增补字段 |

周边配套改动：`scripts/prepare_vllm_source.py:7` 与 `scripts/setup.sh:11` 的 `0.11.0` 硬 pin；`requirements.lock` 全量重锁（0.20.2 控制面 import 链依赖集不同，torch CPU 版本下限需实测）；`UPSTREAM.lock.json` 1391 条 vLLM 哈希全部失效需重生成（`audit_sources.py` 流程不变）；`VERIFICATION.md:13` 的 scheduler SHA-256 基线更新；`run.py:162-166` 自动记录无需改。

### 2.3 vllm-ascend：本次升级的唯一结构性难点

**v0.20.2rc1 已删除 `vllm_ascend/core/scheduler.py`**（已核实 tag 内 `vllm_ascend/core/` 只剩 `recompute_scheduler.py`、`scheduler_dynamic_batch.py`、`scheduler_profiling_chunk.py`，全部为 opt-in，经 `platform.py` 按 `ascend_config` 开关，默认走上游 vLLM V1 `Scheduler`）。vendored 0.11.0 的 `AscendScheduler`（587 行，完整 `schedule()` override，prefill-first 策略、watermark 门控、PD phase）**没有上游继任者**。

三个处置选项：

| 选项 | 做法 | 工作量 | 代价 |
|---|---|---|---|
| A1：fork 移植（保留 prefill-first） | 以 lab 维护 fork 方式，把 587 行 override rebase 到 0.20.2 基类：用 0.20.2 的 `schedule()` 重写主干再移植 prefill-first 分支，处理 `SchedulerOutput` 字段迁移（`resumed_req_ids`/`GrammarOutput`/`preempted_req_ids`）与内部 helper 改名 | **5–10 人日** | 长期承担与上游的 rebase 维护；但这正是"Ascend 适配层真实代码"的研究对象，fork 有独立价值 |
| A2：放弃 prefill-first | 直接用 0.20.2 上游 V1 Scheduler（即 `--engine vllm` 路径）；如需 PD 场景可评估 opt-in `RecomputeScheduler` | 1–2 人日 | 失去 Ascend 特有调度语义，R3 需求退化 |
| A3：延迟升级 | 先在 0.11.0 基线做 G1–G3/G7 增量研究，升级行动推迟到策略抽象层就绪 | 0 | 版本漂移持续累积；若现网目标版本是 0.20.x，最终逃不掉 |

建议：**若 prefill-first 是研究问题的一部分选 A1，否则选 A2**；无论选哪个，都建议先把策略代码抽成版本无关模块（设计文档 §16 的建议），把 `engine.py` 里的构造耦合集中到一个 `compat.py` 适配层，未来 0.20→0.2x 再升级时只改一处。

### 2.4 工作量与实施顺序汇总

| 阶段 | 内容 | 估时（熟悉者） |
|---|---|---|
| P1 | vLLM v0.20.2 主线适配：B1–B8 修复 + pin/lock/哈希重生成 + 16 项测试全绿 + 结果重跑 | 3–5 人日 |
| P2 | AscendScheduler 处置（A1 或 A2） | A1：5–10 人日；A2：1–2 人日 |
| P3 | 行为对拍：0.11.0 vs 0.20.2 在相同 trace/配置下的调度轨迹 diff（234 个调度器提交可能改变行为本身——这既是升级成本也是升级目的），更新 VERIFICATION | 1–2 人日 |
| 合计 | 仅 A2：**4–7 人日**；含 A1：**9–17 人日（约 2–3 周）** | |

风险项：v0.20.2 wheel 的控制面 import 链比 0.11.0 重（transformers 5.x 等），`--no-deps` 安装的依赖子集需要重新裁剪实测；rc1 是候选版，若生产用需关注其正式版节奏；升级后必须重跑 Mooncake C++ hash 对拍（风险低，两端 hash 接口均未变）。

### 2.5 升级决策建议

- **若现网/目标部署版本是 vLLM 0.20.x + vllm-ascend 0.20.2rc1**：升级是必须的（设计文档 §3 明确要求对齐现网版本），按 P1→P2(A1)→P3 执行。
- **若当前是纯机制研究、无现网对齐压力**：建议 A3——先在 0.11.0 基线完成 G1–G3/G7（存储分组/带宽规则/选路/指标，零升级风险、与研究核心直接相关），同时把策略代码与 vLLM 构造点解耦，再择机一次性升级到目标版本。
- 不建议的做法：只改 `pip install` 版本号不改适配（文档 §3 已警告）；以及升级与策略开发混在同一批变更（无法归因行为差异）。

---

## 附录：证据索引

- vLLM v0.20.2 tag：`gh api repos/vllm-project/vllm/git/ref/tags/v0.20.2` → `bc150f5…`；vllm-ascend v0.20.2rc1 → `367b8e6…`（2026-09-22 核实）
- v0.20.2 源码抽查记录：`Scheduler.__init__` 含 `block_size`（v0.20.2 `scheduler.py:68-78`）；`KVConnectorBase_V1.__init__` 对 `kv_transfer_config=None` raise（`base.py:196-201`）；`Request.__init__` 无 `eos_token_id`、`block_hasher` 幸存（`request.py:59-78`）；`WAITING_FOR_REMOTE_KVS` 幸存（`request.py:309`）
- vllm-ascend v0.20.2rc1 树中 `vllm_ascend/core/` 文件清单（无 `scheduler.py`）；其 `platform.py` opt-in 接线、Dockerfile `VLLM_TAG=v0.20.2` 配对
- Mooncake `fd72779...HEAD` compare：2 提交，prefixindex 路径零改动
- lab 侧证据见正文 file:line 引用；测试事实见 `archive/mooncake-cpu-lab-v0.11/VERIFICATION.md`
