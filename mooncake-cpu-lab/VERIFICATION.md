# 交付验证记录

日期：2026-09-23。环境：Linux x86_64，Python 3.12.14，g++ 13.3.0，OpenSSL 3，torch 2.11.0+cpu。

## 实际执行

- 编译原始 Mooncake C++ hash / PrefixCacheTable，仅日志依赖替换为 stderr 适配。
- 使用 vLLM 0.20.2 原始 scheduler、request、cache manager、block pool。
- 使用 vLLM-Ascend 0.20.2rc1 原始 BalanceScheduler（默认均衡关闭）。
- 在不继承宿主 Python site-packages 的新虚拟环境执行安装脚本和测试。
- 新环境结果：`17 passed, 16 warnings in 21.09s`（warnings 来自第三方弃用接口）。测试自身运行时间随 CPU 变化，不是仿真的 NPU 时延。
- 运行真实 vLLM / Ascend × layerwise / async_full；另运行 Ascend 非 chunked 配置。
- 原始 wheel 中 scheduler 文件与固定 vLLM tag 文件逐字节相同，SHA-256 为 `ff208267973b16f1cde384cb4d86090b950f35d0e8bcd97433c642551867572d`。

## 测试覆盖

| 检查 | 结果 |
|---|---|
| Conductor 连续前缀、断点、移除 | 通过 |
| Conductor C++ hash 与真实 vLLM block hash 一致 | 通过 |
| 多资源 max-min 带宽与流加入后的动态完成时间 | 通过 |
| 两类真实 scheduler、两种 KV 加载路径 | 通过 |
| 每层 batch 等待所有相关请求 KV | 通过 |
| 整段 KV 未加载完不安排请求计算 | 通过 |
| chunk 中间轮不生成首 token | 通过 |
| Ascend 非 chunked 配置；默认配置与 vLLM 轨迹一致，未加载设备/EngineCore | 通过 |
| 输出 token 数正确、KV blocks 无泄漏 | 通过 |
| 真实内存压力引发抢占 / 重计算 | 通过 |
| 虚拟事件确定性回放 | 通过 |
| Compute / STALL / Idle 时间守恒，扣除首到达前空闲 | 通过 |
| 存储感知路由与等待队列策略可执行 | 通过 |

## 验证范围

这些检查证明起步工程能够执行真实控制代码并维持已覆盖场景的状态和事件约束。没有在真实 NPU、RDMA 网络或 ASU/CMS 存储上校准；没有运行真实完整 EngineCore / NPUWorker / Conductor HTTP 服务；没有证明生产性能误差或策略收益。所有演示结果都标记 `timing_calibrated=false`。

依赖下载时 PyPI 索引曾超时；vLLM 使用从官方 PyPI 文件地址下载的 0.20.2 wheel，按 `--no-deps` 装入新环境后完成安装脚本验证。交付脚本会优先尝试已缓存的 CPU torch，缓存缺失时正常联网下载；首次使用仍需可访问 PyPI 和 PyTorch CPU wheel 源。

## 版本源码核对

- vLLM：2,163 个包内文件的 Git blob hash 与固定提交一致；另有 103 个 wheel 自带文件（含版本文件与随 wheel 打包的代码）逐字节核对 wheel。
- Ascend：本包采用的 3 个原始文件与固定提交 Git blob hash 一致；调度文件 SHA-256：`13e4407040d82ba29a35e12901551c94da0961439b23e507e73ccfe124e42ba5`。
- vLLM 官方 wheel SHA-256：`22a7dd06eb03371298e13d6100f3dedbf307352342aaf08e87c929c60aae9b4d`。
- `scripts/audit_sources.py` 输出 `modified_upstream_files: []`。
- 重新生成 9 组示例结果，所有 summary 指向 vLLM v0.20.2 / Ascend v0.20.2rc1，且 `timing_calibrated=false`。
- Ascend 采用 AST 隔离加载，方法体保持原样；详细边界及 CPU/NPU 依赖区别见 `UPGRADE.zh-CN.md`。
