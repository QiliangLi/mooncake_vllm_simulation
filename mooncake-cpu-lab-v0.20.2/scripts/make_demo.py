"""Tiny synthetic protocol fixture, NOT a Mooncake workload or performance claim."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
rows=[]
for i in range(12):
    warm=[256,64,128,0][i%4]
    length=warm+[96,160,32,192][i%4]
    # Disjoint prompts make the initial hit length explicit.
    rows.append({'id':f'r{i:02d}','arrival_s':i*0.0004,
                 'tokens':[1000+i*2048+j for j in range(length)],
                 'warm_prefix_tokens':warm,'output_tokens':4+i%3,'slo_s':0.15})
(root/'traces').mkdir(exist_ok=True)
(root/'traces/demo.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
