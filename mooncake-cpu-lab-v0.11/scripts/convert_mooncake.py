"""Convert anonymized Mooncake FAST25 traces to deterministic surrogate token IDs.

Preserves arrival/length and full 512-token prefix equality. Does NOT recover
real tokens or claim a measured cache hit rate. Optional warm-up requests are
assumed fully completed before the measured run; the starter pool is read-only.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

def surrogate_tokens(row):
    n=int(row['input_length']); ids=row['hash_ids']; tokens=[]
    if len(ids)!=math.ceil(n/512): raise ValueError('hash_ids/input_length/512 mismatch')
    for position,h in enumerate(ids):
        seed=hashlib.sha256(f'{position}:{h}'.encode()).digest()
        # 512 int32 ids; same prefix-block id creates the same token block.
        data=hashlib.shake_256(seed).digest(512*4)
        tokens.extend(int.from_bytes(data[i:i+4],'little')%32000 for i in range(0,len(data),4))
    return tokens[:n]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('input');p.add_argument('output')
    p.add_argument('--timestamp-unit',choices=['ms','s'],required=True)
    p.add_argument('--limit',type=int,default=100)
    p.add_argument('--warmup-requests',type=int,default=0)
    p.add_argument('--slo-s',type=float,default=1.0)
    args=p.parse_args()
    if args.limit<1 or args.warmup_requests<0:raise ValueError('invalid counts')
    selected=[]
    with open(args.input) as f:
        for line in f:
            if line.strip():selected.append(json.loads(line))
            if len(selected)>=args.limit+args.warmup_requests:break
    if len(selected)<=args.warmup_requests:raise ValueError('no measured requests')
    if any(b['timestamp']<a['timestamp'] for a,b in zip(selected,selected[1:])):raise ValueError('trace must be ordered')
    scale=.001 if args.timestamp_unit=='ms' else 1.0
    offset=selected[args.warmup_requests]['timestamp']
    pool=[]; measured=[]
    for i,row in enumerate(selected):
        tokens=surrogate_tokens(row)
        if i<args.warmup_requests:
            pool.append({'id':f'warmup-{i}','tokens':tokens[:len(tokens)//512*512]});continue
        measured.append({'id':f'trace-{i}', 'arrival_s':(row['timestamp']-offset)*scale,
            'tokens':tokens,'output_tokens':int(row['output_length']),
            'warm_prefix_tokens':0,'slo_s':args.slo_s,
            'token_provenance':'surrogate_from_mooncake_hash_ids'})
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(''.join(json.dumps(r)+'\n' for r in measured))
    pool_out=out.with_suffix('.pool.jsonl')
    pool_out.write_text(''.join(json.dumps(r)+'\n' for r in pool))
    maximum=max(len(j['tokens'])+j['output_tokens'] for j in measured)
    print(json.dumps({'requests':len(measured),'initial_pool_file':str(pool_out),
        'required_max_model_len':maximum,'trace_block_size':512,
        'warning':'surrogate tokens; static warm-up pool; generated KV writes are not modeled'},indent=2))

if __name__=='__main__':main()
