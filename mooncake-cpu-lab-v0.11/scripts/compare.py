"""Ablate global routing and local waiting order under identical inputs."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--engine',choices=['vllm','ascend'],default='vllm')
p.add_argument('--config',default='configs/demo.json');p.add_argument('--trace',default='traces/demo.jsonl')
args=p.parse_args()
for router,local in [('round_robin','fcfs'),('least_tokens','fcfs'),('storage_aware','fcfs'),('storage_aware','short_io')]:
    out=f'results/{args.engine}-{router}-{local}'
    subprocess.run([sys.executable,'-m','lab.run','--engine',args.engine,'--config',args.config,
                    '--trace',args.trace,'--router',router,'--local-policy',local,'--out',out],cwd=root,check=True,
                   stdout=subprocess.DEVNULL)
    s=json.loads((root/out/'summary.json').read_text())
    print(router,local,'TTFT(ms)=',round(s['mean_ttft_s']*1000,4),'M(ms)=',round(s['makespan_s']*1000,4))
print('Analytic demo coefficients; these are NOT calibrated NPU results.')
