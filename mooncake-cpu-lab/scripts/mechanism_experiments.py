"""Run the mechanism-validation experiment matrix on the real-code simulator.

All variants share one deterministic trace (traces/exp-mech.jsonl) and the
demo compute coefficients; every run executes the real vLLM v0.20.2 Scheduler
and the real Mooncake prefix index. Results land in results/mech-20260923/.
Plotting is separate (scripts/plot_mechanism.py, needs matplotlib).
"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.run import Simulation

OUT = ROOT / 'results/mech-20260923'
BASE = json.loads((ROOT / 'configs/demo.json').read_text())
TRACE = [json.loads(s) for s in (ROOT / 'traces/exp-mech.jsonl').read_text().splitlines() if s.strip()]


def isolated_baselines(cfg):
    """TTFT of each job alone on worker 0 with the same hardware model."""
    out = {}
    for job in TRACE:
        single = copy.deepcopy(cfg)
        single['workers'] = 1
        single['storage']['nic_Bps'] = [cfg['storage']['nic_Bps'][0]]
        sim = Simulation(single, [job], 'vllm')
        sim.run()
        out[job['id']] = sim.jobs[job['id']]['first_token_s'] - job['arrival_s']
    return out


def recompute_overhead(events):
    """Scheduled tokens beyond the per-request minimum (miss + decode - 1)."""
    minimal = {j['id']: len(j['tokens']) - j.get('warm_prefix_tokens', 0) + j['output_tokens'] - 1
               for j in TRACE}
    actual = {}
    for e in events:
        if e['event'] == 'schedule':
            for rid, n in e['tokens'].items():
                actual[rid] = actual.get(rid, 0) + n
    return sum(actual.get(rid, 0) - m for rid, m in minimal.items())


def run_variant(tag, cfg, engine='vllm'):
    jobs = copy.deepcopy(TRACE)
    for j in jobs:
        j['isolated_ttft_s'] = ISO[j['id']]
    sim = Simulation(cfg, jobs, engine)
    result = sim.run()
    result['recompute_token_overhead'] = recompute_overhead(sim.events)
    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    (d / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    (d / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in sim.events))
    (d / 'requests.jsonl').write_text(
        ''.join(json.dumps({k: v for k, v in j.items() if k != 'tokens'}) + '\n' for j in sim.jobs.values()))
    keys = ('mean_ttft_s', 'p50_ttft_s', 'p95_ttft_s', 'slo_attainment', 'goodput_rps',
            'makespan_s', 'mean_normalized_ttft_vs_isolated', 'recompute_token_overhead')
    row = {k: result.get(k) for k in keys}
    row['stall_share'] = (sum(t['stall'] for t in result['worker_time_s']) /
                          sum(sum(t.values()) for t in result['worker_time_s']))
    row['compute_share'] = (sum(t['compute'] for t in result['worker_time_s']) /
                            sum(sum(t.values()) for t in result['worker_time_s']))
    return row


ISO = isolated_baselines(BASE)
metrics = {'isolated_ttft_s': ISO, 'experiments': {}}

# E0 determinism: identical config twice must give identical summaries.
a = run_variant('e0-determinism/a', copy.deepcopy(BASE))
b = run_variant('e0-determinism/b', copy.deepcopy(BASE))
metrics['experiments']['e0_determinism'] = {'a': a, 'b': b, 'identical': a == b}

# E1 routing policies, symmetric and asymmetric NICs.
e1 = {}
for nic_tag, nics in (('sym', BASE['storage']['nic_Bps']), ('asym', [1.2e9, 0.3e9])):
    for policy in ('round_robin', 'least_tokens', 'storage_aware'):
        cfg = copy.deepcopy(BASE); cfg['router_policy'] = policy
        cfg['storage']['nic_Bps'] = list(nics)
        e1[f'{nic_tag}/{policy}'] = run_variant(f'e1-routing/{nic_tag}-{policy}', cfg)
metrics['experiments']['e1_routing'] = e1

# E2 KV load semantics. num_blocks raised to 512: async_full reserves matched
# blocks per waiting request and deadlocks below ~384 blocks on this trace
# (see E2b); 512 removes that confound so the two modes compare fairly.
e2 = {}
for mode in ('layerwise', 'async_full'):
    cfg = copy.deepcopy(BASE); cfg['load_mode'] = mode
    cfg['engine']['num_blocks'] = 512
    e2[mode] = run_variant(f'e2-loadmode/{mode}', cfg)
metrics['experiments']['e2_loadmode'] = e2

# E2b async_full admission deadlock threshold under KV-pool pressure.
e2b = {}
for blocks in (256, 320, 384, 512):
    cfg = copy.deepcopy(BASE); cfg['load_mode'] = 'async_full'
    cfg['engine']['num_blocks'] = blocks
    cfg['event_limit'] = 200000
    try:
        row = run_variant(f'e2b-deadlock/b{blocks}', cfg)
        row['deadlock'] = False
    except RuntimeError:
        row = {'deadlock': True, 'note': 'event limit hit; requests stuck in '
               'WAITING_FOR_REMOTE_KS with finished transfers and exhausted pool'}
    e2b[str(blocks)] = row
metrics['experiments']['e2b_deadlock'] = e2b

# E3 shared-bandwidth pressure (disk and all NICs scaled together).
e3 = {}
for f in (8, 4, 2, 1, 0.5, 0.25):
    cfg = copy.deepcopy(BASE)
    cfg['storage']['disk_Bps'] = BASE['storage']['disk_Bps'] * f
    cfg['storage']['nic_Bps'] = [b * f for b in BASE['storage']['nic_Bps']]
    e3[str(f)] = run_variant(f'e3-bandwidth/f{f}', cfg)
metrics['experiments']['e3_bandwidth'] = e3

# E4 layer-prefetch window.
e4 = {}
for w in (1, 2, 3, 4, 6, 8):
    cfg = copy.deepcopy(BASE); cfg['prefetch_layers'] = w
    e4[str(w)] = run_variant(f'e4-prefetch/w{w}', cfg)
metrics['experiments']['e4_prefetch'] = e4

# E5 KV capacity pressure (real preemption / recompute appears at the cliff).
e5 = {}
for blocks in (256, 192, 160, 128, 112, 96, 88):
    cfg = copy.deepcopy(BASE); cfg['engine']['num_blocks'] = blocks
    e5[str(blocks)] = run_variant(f'e5-capacity/b{blocks}', cfg)
metrics['experiments']['e5_capacity'] = e5

(OUT / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
print(json.dumps({k: 'ok' for k in metrics['experiments']}, indent=2))
