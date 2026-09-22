import copy
import json
from pathlib import Path

import pytest

from lab.conductor import Conductor
from lab.events import Clock
from lab.storage import Storage, max_min_rates

ROOT=Path(__file__).resolve().parents[1]

def config(): return json.loads((ROOT/'configs/demo.json').read_text())
def jobs(): return [json.loads(x) for x in (ROOT/'traces/demo.jsonl').read_text().splitlines()]

def test_conductor_contiguous_prefix_and_remove():
    c=Conductor(16,['w0','w1']); prefix=list(range(64))
    assert c.query(prefix,'w0')==0
    c.seed(prefix,'object')
    assert c.query(prefix,'w0')==64
    assert c.query(prefix[:32]+[999]*32,'w1')==32
    c.seed(prefix,'object',remove=True)
    assert c.query(prefix,'w0')==0
    c.close()

def test_conductor_hash_matches_real_vllm():
    from lab.engine import Request, SamplingParams, get_request_block_hasher, init_none_hash, sha256
    init_none_hash(sha256)
    tokens=list(range(64))
    r=Request('hash',tokens,SamplingParams(max_tokens=1),None,None,
              block_hasher=get_request_block_hasher(16,sha256))
    c=Conductor(16,['w0'])
    assert c.hashes(tokens)==[int.from_bytes(h[-8:],'big') for h in r.block_hashes]
    c.close()

def test_multi_resource_max_min():
    # One flow constrained by its NIC; the other recovers spare disk capacity.
    rates=max_min_rates({0:{'disk','slow'},1:{'disk','fast'}},
                       {'disk':100,'slow':20,'fast':100})
    assert rates=={0:20,1:80}

def test_bandwidth_changes_when_flow_arrives():
    c=Clock(); completed=[]
    s=Storage(c,dict(disks=1,paths_per_disk=16,disk_Bps=100,nic_Bps=[100,100],latency_s=0),lambda *a,**k:None)
    # Force independent path queues so the two flows share disk bandwidth.
    h1,h2=1,2
    from lab.storage import hash64
    while hash64(f'0:{h1}')%16==hash64(f'1:{h2}')%16: h2+=1
    s.submit(0,[h1],100,lambda:completed.append(('a',c.now)),'a')
    c.at(.5,lambda:s.submit(1,[h2],100,lambda:completed.append(('b',c.now)),'b'))
    c.run()
    assert completed==[('a',pytest.approx(1.5)),('b',pytest.approx(2.0))]
    assert s.bytes_transferred==pytest.approx(200)

@pytest.mark.parametrize('kind',['vllm','ascend'])
@pytest.mark.parametrize('mode',['layerwise','async_full'])
def test_real_scheduler_completion_and_conservation(kind,mode):
    from lab.run import Simulation
    c=config(); c['load_mode']=mode
    sim=Simulation(c,jobs(),kind); result=sim.run()
    assert result['requests']==12
    for job in sim.jobs.values():
        assert job['arrival_s']<=job['first_token_s']<=job['finish_s']
        assert len(job['token_times_s'])==job['output_tokens']
    for e in sim.engines:
        assert e.scheduler.get_num_unfinished_requests()==0
        assert e.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()==c['engine']['num_blocks']-1
    for times in result['worker_time_s']:
        assert min(times.values())>=-1e-10
        assert sum(times.values())==pytest.approx(result['makespan_s'])
    expected=sum(j['warm_prefix_tokens'] for j in jobs())*sim.bytes_per_token_layer*c['model']['layers']
    assert result['storage_bytes']==pytest.approx(expected)

def test_layer_barrier_waits_for_every_request():
    from lab.run import Simulation
    sim=Simulation(config(),jobs()); sim.run()
    done={}
    for ev in sim.events:
        if ev['event']=='io_done': done[ev['label']]=ev['t']
        if ev['event']=='layer_start':
            for rid in ev['requests']:
                if sim.jobs[rid]['warm_prefix_tokens']:
                    assert done[f"{rid}:L{ev['layer']}"]<=ev['t']

def test_async_full_no_schedule_before_receive():
    from lab.run import Simulation
    c=config(); c['load_mode']='async_full'
    sim=Simulation(c,jobs()); sim.run(); done={}
    for ev in sim.events:
        if ev['event']=='io_done':done[ev['label']]=ev['t']
        if ev['event']=='schedule':
            for rid in ev['tokens']:
                if sim.jobs[rid]['warm_prefix_tokens']:assert done[rid+':all']<=ev['t']

def test_ascend_prefill_first_branch():
    from lab.run import Simulation
    cfg=json.loads((ROOT/'configs/ascend-prefill-first.json').read_text())
    sim=Simulation(cfg,jobs(),'ascend'); result=sim.run()
    assert result['requests']==12
    assert not sim.engines[0].scheduler.scheduler_config.chunked_prefill_enabled

def test_no_partial_chunk_generates_output():
    from lab.run import Simulation
    c=config(); c['engine']['chunk_size']=16
    j=jobs()[3]; j['arrival_s']=0; j['output_tokens']=1
    sim=Simulation(c,[j]); sim.run()
    scheduled=[e for e in sim.events if e['event']=='schedule' and j['id'] in e['tokens']]
    assert len(scheduled)==len(j['tokens'])//16
    assert len(sim.jobs[j['id']]['token_times_s'])==1

def test_deterministic_event_replay():
    from lab.run import Simulation
    a=Simulation(config(),jobs()); ra=a.run()
    b=Simulation(config(),jobs()); rb=b.run()
    assert a.events==b.events and ra==rb

def test_storage_aware_and_local_policy_execute():
    from lab.run import Simulation
    c=config(); c['router_policy']='storage_aware'; c['local_policy']='short_io'
    sim=Simulation(c,jobs()); assert sim.run()['requests']==12

def test_kv_pressure_exercises_real_preemption():
    from lab.run import Simulation
    c=config(); c['engine'].update(num_blocks=40,max_num_batched_tokens=256,chunk_size=64)
    js=jobs()[3:4]*1
    js=[dict(js[0],id=f'p{i}',arrival_s=0,output_tokens=16) for i in range(4)]
    c['workers']=1;c['storage']['nic_Bps']=[1.2e9]
    sim=Simulation(c,js); result=sim.run()
    # Repeated NewRequestData admission after memory pressure requires real
    # preemption/recomputation; request counter must never be patched by us.
    executed=sum(sum(e['tokens'].values()) for e in sim.events if e['event']=='schedule')
    minimal=sum(len(j['tokens'])+j['output_tokens']-1 for j in js)
    assert executed>minimal
    assert result['requests']==4

def test_arrival_offset_not_charged_to_utilization():
    from lab.run import Simulation
    c=config();js=jobs()
    for j in js:j['arrival_s']+=.1
    sim=Simulation(c,js);r=sim.run()
    for t in r['worker_time_s']:assert sum(t.values())==pytest.approx(r['makespan_s'])
