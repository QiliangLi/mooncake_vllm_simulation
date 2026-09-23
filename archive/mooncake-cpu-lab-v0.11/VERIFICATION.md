# 交付验证记录

日期：2026-09-22。环境：Linux x86_64，Python 3.12.14，g++ 13.3.0，OpenSSL 3，torch 2.8.0+cpu。

## 实际执行

- 编译原始 Mooncake C++ hash / PrefixCacheTable，仅日志依赖替换为 stderr 适配。
- 使用 vLLM 0.11.0 原始 scheduler、request、cache manager、block pool。
- 使用 vLLM-Ascend 0.11.0 原始 AscendScheduler。
- 在不继承宿主 Python site-packages 的新虚拟环境执行安装脚本和测试。
- 新环境结果：`16 passed in 9.38s`。测试自身运行时间随 CPU 变化，不是仿真的 NPU 时延。
- 运行真实 vLLM / Ascend × layerwise / async_full；另运行 Ascend 非 chunked prefill-first 分支。
- 原始 wheel 中 scheduler 文件与固定 vLLM tag 文件逐字节相同，SHA-256 为 `357ac414a5b5da8c3423984dd1fdde860584a0d417c30ac2ee25874ea9e5a04c`。

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
| Ascend prefill-first 独有分支 | 通过 |
| 输出 token 数正确、KV blocks 无泄漏 | 通过 |
| 真实内存压力引发抢占 / 重计算 | 通过 |
| 虚拟事件确定性回放 | 通过 |
| Compute / STALL / Idle 时间守恒，扣除首到达前空闲 | 通过 |
| 存储感知路由与等待队列策略可执行 | 通过 |

## 验证范围

这些检查证明起步工程能够执行真实控制代码并维持已覆盖场景的状态和事件约束。没有在真实 NPU、RDMA 网络或 ASU/CMS 存储上校准；没有运行真实完整 EngineCore / NPUWorker / Conductor HTTP 服务；没有证明生产性能误差或策略收益。所有演示结果都标记 `timing_calibrated=false`。

依赖下载时曾遇到网络超时，使用下载缓存后完成了新环境安装。交付脚本会优先尝试已缓存的 CPU torch，缓存缺失时正常联网下载；首次使用仍需可访问 PyPI 和 PyTorch CPU wheel 源。
