import argparse
import copy
import csv
import importlib.metadata
import hashlib
import inspect
import json
import math
import platform
from pathlib import Path

from .conductor import Conductor
from .events import Clock
from .storage import Storage

def quantile(values, q):
    values = sorted(values)
    i = (len(values)-1)*q; lo = int(i); hi = math.ceil(i)
    return values[lo]+(values[hi]-values[lo])*(i-lo)

class Simulation:
    def __init__(self, cfg, jobs, kind='vllm'):
        from .engine import Engine
        self.cfg = copy.deepcopy(cfg); self.clock = Clock(); self.events = []
        self.jobs = {j['id']:copy.deepcopy(j) for j in jobs}
        if not jobs or len(self.jobs) != len(jobs): raise ValueError('empty trace or duplicate ids')
        self.validate()
        self.kind = kind
        self.conductor = Conductor(cfg['engine']['block_size'], [f'w{i}' for i in range(cfg['workers'])])
        if cfg.get('initial_pool_file'):
            for line in Path(cfg['initial_pool_file']).read_text().splitlines():
                if line.strip():
                    item=json.loads(line)
                    self.conductor.seed(item['tokens'],item['id'])
        # Explicit initial warm pool, set by trace. No future generated prefix
        # is inserted. A job may also hit other jobs' declared initial prefixes.
        for job in jobs:
            if job.get('warm_prefix_tokens', 0):
                self.conductor.seed(job['tokens'][:job['warm_prefix_tokens']], job['id'])
        self.storage = Storage(self.clock, cfg['storage'], self.emit)
        m = cfg['model']
        self.bytes_per_token_layer = 2*m['kv_heads']*m['head_dim']*m['dtype_bytes']
        self.bytes_per_block_layer = self.bytes_per_token_layer*cfg['engine']['block_size']
        self.engines = [Engine(self, i, kind) for i in range(cfg['workers'])]
        self.observed_storage = self.storage.snapshot()
        self.observed_workers = [{'queue_tokens':0} for _ in self.engines]
        self.pending_dispatch_tokens = [0]*cfg['workers']
        self.last_arrival = max(j['arrival_s'] for j in jobs)
        for job in self.jobs.values():
            self.clock.at(job['arrival_s'], lambda j=job: self.arrive(j), priority=10)
        self.clock.at(0, self.sample, priority=5)

    def validate(self):
        c=self.cfg; e=c['engine']; s=c['storage']
        if c['workers']<1 or len(s['nic_Bps'])!=c['workers']: raise ValueError('worker/NIC mismatch')
        if c['load_mode'] not in ('async_full','layerwise'): raise ValueError('unsupported load_mode')
        if c['router_policy'] not in ('round_robin','least_tokens','storage_aware'): raise ValueError('router policy')
        if c['local_policy'] not in ('fcfs','short_io'): raise ValueError('local policy')
        if min(s['disks'],s['paths_per_disk'],s['disk_Bps'],s.get('path_Bps',s['disk_Bps']),*s['nic_Bps'],c['prefetch_layers'],c['telemetry_interval_s'])<=0: raise ValueError('positive capacities/interval required')
        if s['latency_s']<0 or c['telemetry_delay_s']<0: raise ValueError('negative latency')
        if c['prefetch_layers']>c['model']['layers']: raise ValueError('prefetch exceeds layers')
        if min(c['model'].values())<=0: raise ValueError('model dimensions must be positive')
        for j in self.jobs.values():
            n=len(j['tokens']); out=j['output_tokens']; warm=j.get('warm_prefix_tokens',0)
            if n<1 or out<1 or n+out>e['max_model_len']: raise ValueError(f"invalid lengths: {j['id']}")
            if not 0<=warm<n or warm%e['block_size']: raise ValueError('warm prefix must be whole blocks shorter than prompt')
            if j['arrival_s']<0 or j.get('slo_s',1)<=0: raise ValueError('invalid arrival/SLO')
            if (n+out+e['block_size']-1)//e['block_size']>=e['num_blocks']: raise ValueError('one request cannot fit KV pool')
            if not e['chunked_prefill'] and n>e['max_num_batched_tokens']: raise ValueError('unchunked prompt exceeds token budget')

    def emit(self, event, **values):
        self.events.append(dict(t=self.clock.now, event=event, **values))

    def compute_time(self, engine, so):
        # Analytic demonstration only. q is this batch's new-token count,
        # c is its preceding attention context, including reused remote KV.
        p=self.cfg['compute']; total_q=0; attention=0
        for rid,q in so.num_scheduled_tokens.items():
            r=engine.scheduler.requests[rid]
            context=max(0,r.num_computed_tokens-q)
            total_q+=q; attention+=q*(context+(q+1)/2)
        efficiency=min(1.0,max(p['min_efficiency'],total_q/p['saturation_tokens']))
        return p['batch_overhead_s']+(p['linear_s_per_token']*total_q+
              p['attention_s_per_pair']*attention)/efficiency

    def estimated_io(self, tokens, wid):
        bs=self.cfg['engine']['block_size']
        hit=min(self.conductor.query(tokens,f'w{wid}'), (len(tokens)-1)//bs*bs)
        hashes=self.conductor.hashes(tokens[:hit])
        disk_bytes=[0]*self.cfg['storage']['disks']
        for h in hashes: disk_bytes[self.storage.disk_for(h)]+=self.bytes_per_block_layer*self.cfg['model']['layers']
        observed=self.observed_storage
        s=self.cfg['storage']
        disk_time=max(((b+observed['disk_queued_bytes'][i])/s['disk_Bps']
                       for i,b in enumerate(disk_bytes) if b),default=0)
        nic_time = (sum(disk_bytes)+observed['worker_queued_bytes'][wid])/s['nic_Bps'][wid] if hit else 0
        return max(disk_time,nic_time)+ (s['latency_s'] if hit else 0)

    def local_score(self, request, wid):
        # Deliberately a simple research policy, not an official Mooncake policy.
        age=self.clock.now-request.arrival_time
        return self.estimated_io(request.prompt_token_ids,wid)-self.cfg['aging_weight']*age

    def arrive(self, job):
        policy=self.cfg['router_policy']
        if policy=='round_robin': wid=sum('worker' in j for j in self.jobs.values())%len(self.engines)
        else:
            scores=[]
            for i in range(len(self.engines)):
                queue=self.observed_workers[i]['queue_tokens']+self.pending_dispatch_tokens[i]
                score=queue*self.cfg['compute']['linear_s_per_token']
                if policy=='storage_aware': score+=self.estimated_io(job['tokens'],i)
                scores.append((score,i))
            wid=min(scores)[1]
        job['worker']=wid
        self.pending_dispatch_tokens[wid]+=len(job['tokens'])
        self.emit('dispatch',request=job['id'],worker=wid,
                  storage_sample_time=self.observed_storage['sample_time'])
        self.engines[wid].add(job)

    def sample(self):
        snapshot=self.storage.snapshot()
        # Snapshot request ids too, so requests dispatched after this sampling
        # point are not forgotten when delayed telemetry arrives.
        known={rid for e in self.engines for rid in e.scheduler.requests}
        workers=[{'queue_tokens':e.active_batch_tokens+sum(max(0,r.num_tokens-r.num_computed_tokens)
                      for r in e.scheduler.requests.values())} for e in self.engines]
        def deliver():
            self.observed_storage=snapshot; self.observed_workers=workers
            self.pending_dispatch_tokens=[sum(len(j['tokens']) for rid,j in self.jobs.items()
                if j.get('worker')==wid and rid not in known and 'finish_s' not in j)
                for wid in range(len(self.engines))]
        self.clock.after(self.cfg['telemetry_delay_s'],deliver,priority=5)
        if self.clock.now < self.last_arrival or any(e.scheduler.get_num_unfinished_requests() for e in self.engines):
            self.clock.after(self.cfg['telemetry_interval_s'],self.sample,priority=5)

    def run(self):
        self.clock.run(self.cfg.get('event_limit',1000000))
        unfinished=[j['id'] for j in self.jobs.values() if 'finish_s' not in j]
        if unfinished: raise RuntimeError(f'no progress, unfinished: {unfinished[:10]}')
        start=min(j['arrival_s'] for j in self.jobs.values())
        end=max(j['finish_s'] for j in self.jobs.values())
        times=[]
        for e in self.engines:
            durations=dict(e.times)
            # Use the last request completion as horizon, excluding trailing telemetry.
            durations[e.state]+=end-e.since
            durations['idle']-=start
            times.append(durations)
        ttft=[j['first_token_s']-j['arrival_s'] for j in self.jobs.values()]
        slos=[j['first_token_s']-j['arrival_s']<=j.get('slo_s',self.cfg['default_slo_s']) for j in self.jobs.values()]
        result={
            'execution':'real-vllm-scheduler + real-mooncake-prefix-index + simulated-hardware',
            'engine':self.kind,'timing_calibrated':False,
            'router_policy':self.cfg['router_policy'],'local_policy':self.cfg['local_policy'],
            'load_mode':self.cfg['load_mode'],'requests':len(ttft),
            'makespan_s':end-start,'mean_ttft_s':sum(ttft)/len(ttft),
            'p50_ttft_s':quantile(ttft,.5),'p95_ttft_s':quantile(ttft,.95),
            'slo_attainment':sum(slos)/len(slos),'request_throughput_rps':len(ttft)/(end-start),
            'goodput_rps':sum(slos)/(end-start),'worker_time_s':times,
            'storage_bytes':self.storage.bytes_transferred,'event_count':self.clock.events,
            'versions':{'python':platform.python_version(),'vllm':importlib.metadata.version('vllm'),
                        'torch':importlib.metadata.version('torch')},
        }
        scheduler_file=Path(inspect.getfile(type(self.engines[0].scheduler)))
        result['scheduler_source_sha256']=hashlib.sha256(scheduler_file.read_bytes()).hexdigest()
        root=Path(__file__).resolve().parents[1]
        lock=json.loads((root/'UPSTREAM.lock.json').read_text())
        result['upstream']=lock['upstream']
        result['scheduler_class']=type(self.engines[0].scheduler).__name__
        result['scheduler_source']=str(scheduler_file.relative_to(root))
        result['base_scheduler_source_sha256']=hashlib.sha256(
            (root/'vendor/vllm/vllm/v1/core/sched/scheduler.py').read_bytes()).hexdigest()
        result['ascend_features']={'balance_scheduling':False,'dynamic_batch':False,
                                  'profiling_chunk':False,'recompute_scheduler':False}
        for i,t in enumerate(times):
            result.setdefault('worker_utilization',[]).append({k:v/(end-start) for k,v in t.items()})
        if all('isolated_ttft_s' in j for j in self.jobs.values()):
            if any(j['isolated_ttft_s']<=0 for j in self.jobs.values()): raise ValueError('invalid isolated baseline')
            normalized=[(j['first_token_s']-j['arrival_s'])/j['isolated_ttft_s'] for j in self.jobs.values()]
            result['mean_normalized_ttft_vs_isolated']=sum(normalized)/len(normalized)
        self.conductor.close()
        return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/demo.json')
    p.add_argument('--trace',default='traces/demo.jsonl')
    p.add_argument('--engine',choices=['vllm','ascend'],default='vllm')
    p.add_argument('--router',choices=['round_robin','least_tokens','storage_aware'])
    p.add_argument('--local-policy',choices=['fcfs','short_io'])
    p.add_argument('--load-mode',choices=['async_full','layerwise'])
    p.add_argument('--out',default='results/run')
    args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    if args.router:cfg['router_policy']=args.router
    if args.local_policy:cfg['local_policy']=args.local_policy
    if args.load_mode:cfg['load_mode']=args.load_mode
    jobs=[json.loads(s) for s in Path(args.trace).read_text().splitlines() if s.strip()]
    sim=Simulation(cfg,jobs,args.engine); result=sim.run()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (out/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in sim.events))
    (out/'requests.jsonl').write_text(''.join(json.dumps({k:v for k,v in j.items() if k!='tokens'})+'\n' for j in sim.jobs.values()))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
