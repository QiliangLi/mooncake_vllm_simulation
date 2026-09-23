"""Deterministic synthetic trace for mechanism experiments (NOT a real workload).

Three shared-prefix families create cross-request hits on the same blocks (and
therefore the same disks), plus per-job suffixes; a mid-run burst raises
concurrency. Lengths/SLOs are chosen so routing/bandwidth/capacity effects are
visible with the demo compute coefficients.
"""
import json
from pathlib import Path
root = Path(__file__).resolve().parents[1]

def interarrival(i, n):
    # base stream plus one burst in the middle third
    if n // 3 <= i < 2 * n // 3:
        return 0.00008
    return 0.00040

rows = []
for i in range(48):
    fam = i % 3
    warm = [768, 512, 256][fam] if i % 4 != 3 else 0
    suffix = [192, 96, 320, 160][i % 4]
    prompt_len = max(warm + suffix, 64)
    tokens = [fam * 100000 + j for j in range(warm)] + \
             [500000 + i * 4096 + j for j in range(prompt_len - warm)]
    rows.append({
        'id': f'e{i:02d}',
        'arrival_s': round(sum(interarrival(k, 48) for k in range(i)), 6),
        'tokens': tokens,
        'warm_prefix_tokens': warm,
        'output_tokens': 4 + i % 5,
        'slo_s': 0.02 + (i % 6) * 0.01,
    })
(root / 'traces').mkdir(exist_ok=True)
(root / 'traces/exp-mech.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
print(f'wrote {len(rows)} jobs')
