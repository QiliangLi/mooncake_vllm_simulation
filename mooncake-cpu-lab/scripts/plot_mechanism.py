"""Plot the mechanism-experiment figures from results/mech-20260923/metrics.json.

Analysis-only dependency (matplotlib); figures land in <repo>/docs/figures/.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
M = json.loads((ROOT / 'results/mech-20260923/metrics.json').read_text())
E = M['experiments']
OUT = REPO / 'docs/figures'
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.sans-serif': ['PingFang SC', 'Hiragino Sans GB', 'Arial Unicode MS', 'Heiti TC'],
    'axes.unicode_minus': False, 'figure.dpi': 150, 'savefig.bbox': 'tight',
})
C = {'p50': '#2563eb', 'p95': '#dc2626', 'slo': '#16a34a', 'stall': '#f59e0b', 'compute': '#2563eb'}


def fig_e1():
    variants = ['sym/round_robin', 'sym/least_tokens', 'sym/storage_aware',
                'asym/round_robin', 'asym/least_tokens', 'asym/storage_aware']
    labels = ['RR', 'least\ntokens', 'storage\naware', 'RR', 'least\ntokens', 'storage\naware']
    d = E['e1_routing']
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    x = range(len(variants))
    ax[0].bar([i - 0.2 for i in x], [d[v]['p50_ttft_s'] * 1e3 for v in variants], 0.4, label='p50', color=C['p50'])
    ax[0].bar([i + 0.2 for i in x], [d[v]['p95_ttft_s'] * 1e3 for v in variants], 0.4, label='p95', color=C['p95'])
    ax[0].set_xticks(list(x), labels)
    ax[0].axvline(2.5, color='gray', ls=':', lw=1)
    ax[0].text(1, 880, '对称 NIC', ha='center', fontsize=9, color='gray')
    ax[0].text(4, 880, '非对称 NIC (1.2G / 0.3G)', ha='center', fontsize=9, color='gray')
    ax[0].set_ylabel('TTFT (ms)')
    ax[0].set_title('(a) TTFT：p50 / p95')
    ax[0].legend()
    slo = [d[v]['slo_attainment'] * 100 for v in variants]
    ax[1].bar(list(x), slo, 0.55, color=C['slo'])
    for i, v in enumerate(slo):
        ax[1].text(i, v + 1, f'{v:.0f}%', ha='center', fontsize=8)
    ax[1].set_xticks(list(x), labels)
    ax[1].set_ylabel('SLO 满足率 (%)')
    ax[1].set_title('(b) SLO 满足率')
    fig.suptitle('E1 全局路由策略：真实调度器下的决策质量（对称 vs 非对称 NIC）', y=1.04)
    fig.savefig(OUT / 'fig_mech_e1_routing.png')
    plt.close(fig)


def fig_e2():
    d = E['e2_loadmode']
    fig = plt.figure(figsize=(11, 3.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1.4])
    ax0 = fig.add_subplot(gs[0])
    for i, mode in enumerate(('layerwise', 'async_full')):
        reqs = [json.loads(s) for s in (ROOT / f'results/mech-20260923/e2-loadmode/{mode}/requests.jsonl').read_text().splitlines()]
        ttft = [(r['first_token_s'] - r['arrival_s']) * 1e3 for r in reqs]
        ax0.scatter([i + (j % 8) * 0.045 - 0.16 for j in range(len(ttft))], ttft, s=14, alpha=0.65,
                    color=C['p50'] if mode == 'layerwise' else C['p95'])
        ax0.hlines(d[mode]['p50_ttft_s'] * 1e3, i - 0.25, i + 0.25, color='black', lw=1.5)
        ax0.hlines(d[mode]['p95_ttft_s'] * 1e3, i - 0.25, i + 0.25, color='black', lw=1.5, ls='--')
    ax0.set_xticks([0, 1], ['layerwise\n(逐层 barrier)', 'async_full\n(整段异步)'])
    ax0.set_ylabel('TTFT (ms)')
    ax0.set_title('(a) 每请求 TTFT 分布')

    ax1 = fig.add_subplot(gs[1])
    modes = ['layerwise', 'async_full']
    comp = [d[m]['compute_share'] * 100 for m in modes]
    stal = [d[m]['stall_share'] * 100 for m in modes]
    idl = [100 - c - s for c, s in zip(comp, stal)]
    ax1.bar(modes, comp, 0.5, label='compute', color=C['compute'])
    ax1.bar(modes, stal, 0.5, bottom=comp, label='stall', color=C['stall'])
    ax1.bar(modes, idl, 0.5, bottom=[c + s for c, s in zip(comp, stal)], label='idle', color='#d1d5db')
    ax1.set_ylabel('时间占比 (%)')
    ax1.set_title('(b) worker 时间构成')
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[2])
    for row, mode in enumerate(('layerwise', 'async_full')):
        evs = [json.loads(s) for s in (ROOT / f'results/mech-20260923/e2-loadmode/{mode}/events.jsonl').read_text().splitlines()]
        states = [e for e in evs if e['event'] == 'state' and e['worker'] == 0]
        end = max(e['t'] for e in evs) * 1.0
        segs, prev_t, prev_s = [], 0.0, 'idle'
        for e in states + [{'t': min(end, d[mode]['makespan_s']), 'state': 'end'}]:
            if e['t'] > prev_t:
                segs.append((prev_t, e['t'], prev_s))
            prev_t, prev_s = e['t'], e['state']
        colors = {'compute': C['compute'], 'stall': C['stall'], 'idle': '#d1d5db'}
        for t0, t1, s in segs:
            ax2.broken_barh([(t0 * 1e3, (t1 - t0) * 1e3)], (row - 0.32, 0.64), color=colors.get(s, 'gray'))
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=c, label=s) for s, c in colors.items()]
    ax2.legend(handles=handles, fontsize=7, loc='upper right', ncol=3)
    ax2.set_yticks([0, 1], ['layerwise', 'async_full'])
    ax2.set_xlabel('虚拟时间 (ms)')
    ax2.set_title('(c) worker 0 状态时间线')
    fig.suptitle('E2 KV 加载语义：逐层 barrier vs 整段异步（num_blocks=512）', y=1.04)
    fig.savefig(OUT / 'fig_mech_e2_loadmode.png')
    plt.close(fig)


def fig_e3():
    d = E['e3_bandwidth']
    fs = sorted(d, key=float, reverse=True)
    x = range(len(fs))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(list(x), [d[f]['p50_ttft_s'] * 1e3 for f in fs], 'o-', color=C['p50'], label='p50 TTFT')
    ax.plot(list(x), [d[f]['p95_ttft_s'] * 1e3 for f in fs], 's--', color=C['p95'], label='p95 TTFT')
    ax.set_yscale('log')
    ax.set_xticks(list(x), [f'×{f}' for f in fs])
    ax.set_xlabel('共享带宽缩放因子（盘与 NIC 同步缩放）')
    ax.set_ylabel('TTFT (ms, 对数轴)')
    ax2 = ax.twinx()
    ax2.plot(list(x), [d[f]['slo_attainment'] * 100 for f in fs], '^:', color=C['slo'], label='SLO 满足率')
    ax2.set_ylabel('SLO 满足率 (%)')
    ax.set_title('E3 共享带宽压力：从“IO 被掩盖”到“IO 主导”')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8)
    fig.savefig(OUT / 'fig_mech_e3_bandwidth.png')
    plt.close(fig)


def fig_e4():
    d = E['e4_prefetch']
    ws = sorted(d, key=int)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(ws, [d[w]['p95_ttft_s'] * 1e3 for w in ws], 'o-', color=C['p95'], label='p95 TTFT (左)')
    ax.set_xlabel('逐层预取窗口 prefetch_layers（模型共 8 层）')
    ax.set_ylabel('p95 TTFT (ms)')
    ax2 = ax.twinx()
    ax2.plot(ws, [d[w]['stall_share'] * 100 for w in ws], 's--', color=C['stall'], label='stall 占比 (右)')
    ax2.set_ylabel('stall 时间占比 (%)')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8)
    ax.set_title('E4 预取窗口扫描：本负载下非瓶颈旋钮')
    fig.savefig(OUT / 'fig_mech_e4_prefetch.png')
    plt.close(fig)


def fig_e5():
    d = E['e5_capacity']
    bs = sorted(d, key=int)
    x = range(len(bs))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(list(x), [d[b]['p95_ttft_s'] * 1e3 for b in bs], 'o-', color=C['p95'], label='p95 TTFT (左)')
    ax.set_xticks(list(x), bs)
    ax.set_xlabel('KV 池容量 num_blocks（块 × 16 token）')
    ax.set_ylabel('p95 TTFT (ms)')
    ax2 = ax.twinx()
    ax2.bar(list(x), [d[b]['recompute_token_overhead'] for b in bs], 0.5, alpha=0.45, color=C['compute'], label='重计算开销 token (右)')
    ax2.set_ylabel('重计算 token 开销')
    onset = next(i for i, b in enumerate(bs) if d[b]['recompute_token_overhead'] > 0)
    ax2.annotate('真实抢占/重计算出现', xy=(onset, d[bs[onset]]['recompute_token_overhead']),
                 xytext=(onset + 1.4, max(r['recompute_token_overhead'] for r in d.values()) * 0.72),
                 arrowprops=dict(arrowstyle='->', lw=1), fontsize=9)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper left')
    ax.set_title('E5 KV 容量压力：调度器真实抢占与重计算行为')
    fig.savefig(OUT / 'fig_mech_e5_capacity.png')
    plt.close(fig)


fig_e1(); fig_e2(); fig_e3(); fig_e4(); fig_e5()
print('figures written to', OUT)
