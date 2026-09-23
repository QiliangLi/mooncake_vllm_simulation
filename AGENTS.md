# AGENTS.md

基于真实控制代码（vLLM / vLLM-Ascend 调度器 + Mooncake Conductor C++ 前缀索引）+ 虚拟执行的 CPU 调度仿真研究仓库。

## 目录索引

| 路径 | 说明 |
|---|---|
| `mooncake-cpu-lab/` | **当前主力工程**（vLLM v0.20.2 / vllm-ascend v0.20.2rc1 / Mooncake `fd72779`）。入口文档：[README](mooncake-cpu-lab/README.md) · [DESIGN](mooncake-cpu-lab/DESIGN.zh-CN.md) · [VERIFICATION](mooncake-cpu-lab/VERIFICATION.md) · [UPGRADE（版本适配与行为变化）](mooncake-cpu-lab/UPGRADE.zh-CN.md) |
| `mooncake-cpu-lab/lab/` | 仿真驱动层：`engine`（真实调度器接入/虚拟时钟/SimConnector）、`run`（trace/路由/指标）、`storage`（盘/Path/NIC 流体模型）、`conductor`（ctypes 桥）、`ascend_loader`（AST 隔离加载 BalanceScheduler） |
| `mooncake-cpu-lab/vendor/` | 锁定的上游源码（`UPSTREAM.lock.json` 哈希审计） |
| `Mooncake_vLLM_CPU_Simulation_Design.zh-CN.md` | 工程总体设计（当前 v0.20.2 版） |
| `docs/共享KV多ASU多Path下的组批错峰与带宽协同调度设计-20260917.md` | 研究策略目标（SS/LS/SL/LL 分组、B0/B1 带宽、P0/P1 选路、S1 组批、E26–E31） |
| `docs/共享KV统一队列下的Batch效率与错峰调度形式化分析-20260913.md` | 形式化分析（前序） |
| `docs/仿真环境符合性检视与vLLM-v0.20.2升级评估-20260922.md` | 符合性检视 + G1–G9 研究增量清单 + 升级破坏点 B1–B8 |
| `docs/v0.20代码落地核查与升级再评估-20260923.md` | v0.20.2 落地核查证据链 + 升级路线 |
| `archive/` | 历史归档（v0.11 基线工程与旧版设计文档，只读，见 [archive/README](archive/README.md)） |
| `ChatGPT-全局调度仿真方案-20260922-2147.md` | 需求沟通原始记录 |

## 修改规则

1. **文档同步**：任何代码/配置改动，必须同批更新相关文档（工程内 README / DESIGN / VERIFICATION / UPGRADE，或 `docs/` 下对应分析），并同步维护本索引；新增文档要在上表登记。
2. **vendor 纪律**：`vendor/` 是哈希锁定的上游源码，禁止直接修改；`UPSTREAM.lock.json` 与 `results/source-audit.json` 必须保持一致（详见 UPGRADE 文档）。
3. **验证**：改动 `mooncake-cpu-lab/` 后在 Linux 环境跑 `tests/test_lab.py`（macOS 不在验证范围）；结果产物标记 `timing_calibrated=false`，不得当作性能结论。
4. **归档纪律**：`archive/` 只读；如需对照 0.11 行为，复制后在别处实验。

## Git 工作流（自动提交）

- 每完成一轮自洽的修改（代码 + 文档 + 测试通过）后，**无需询问，直接执行**：`git add -A && git commit && git push origin main`（远端 https://github.com/QiliangLi/mooncake_vllm_simulation，main 分支）。
- 提交信息用 Conventional Commits（`feat`/`fix`/`docs`/`chore` + 中文描述），一个提交对应一个完整改动。
- 已知坑：`vendor/…/model_executor/models/registry.py` 中的模型类名 `Mistral3ForConditionalGeneration` 会触发 GitHub push protection 误报（上游公开源码，非真实密钥）；该误报已按 false_positive 关闭并放行后续推送，若未来出现**新的**拦截，处理方法见 `docs/v0.20代码落地核查与升级再评估-20260923.md` 附录。
