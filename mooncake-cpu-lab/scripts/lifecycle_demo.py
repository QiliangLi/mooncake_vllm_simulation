"""Regenerate the request-lifecycle walkthrough data (docs/仿真架构说明-20260923.md §4).

Runs the first four demo-trace jobs on a single worker in layerwise mode and
writes results/lifecycle-demo/ (summary + events + requests). Deterministic.
"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.run import Simulation

BASE = json.loads((ROOT / 'configs/demo.json').read_text())
TRACE = [json.loads(s) for s in (ROOT / 'traces/demo.jsonl').read_text().splitlines() if s.strip()][:4]

cfg = copy.deepcopy(BASE)
cfg['workers'] = 1
cfg['storage']['nic_Bps'] = [BASE['storage']['nic_Bps'][0]]

sim = Simulation(cfg, TRACE, 'vllm')
result = sim.run()
out = ROOT / 'results/lifecycle-demo'
out.mkdir(parents=True, exist_ok=True)
(out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
(out / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
(out / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in sim.events))
(out / 'requests.jsonl').write_text(
    ''.join(json.dumps({k: v for k, v in j.items() if k != 'tokens'}) + '\n' for j in sim.jobs.values()))
print(f'{len(TRACE)} jobs, {len(sim.events)} events, makespan {result["makespan_s"]*1e3:.2f} ms -> {out}')
