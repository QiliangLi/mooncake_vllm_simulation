"""Original vLLM / Ascend schedulers, driven without EngineCore or device workers."""
import copy
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS

os.environ.setdefault('VLLM_PLUGINS', '')
os.environ.setdefault('PYTHONHASHSEED', '0')
os.environ.setdefault('VLLM_LOGGING_LEVEL', 'ERROR')

import torch
from vllm.config import CacheConfig, SchedulerConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.sampling_params import SamplingParams
from vllm.utils import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import ModelRunnerOutput, KVConnectorOutput
from vllm.v1.request import Request

ROOT = Path(__file__).resolve().parents[1]

class NoStructuredOutput:
    # Text-only workload contract; no grammar or tokenizer is initialized.
    def grammar_init(self, request):
        if request.use_structured_output: raise NotImplementedError('structured outputs')
    def should_advance(self, request): return False

@dataclass
class TransferMeta(KVConnectorMetadata):
    loads: dict

class SimConnector(KVConnectorBase_V1):
    def __init__(self, cfg, conductor, worker_name, mode):
        super().__init__(cfg, KVConnectorRole.SCHEDULER)
        self.conductor, self.worker_name, self.mode = conductor, worker_name, mode
        self.pending = {}
        self.local_tokens = {}

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        matched = self.conductor.query(request.prompt_token_ids, self.worker_name)
        # Only whole reusable blocks, retaining one token to produce logits.
        bs = self.conductor.block_size
        matched = min(matched, (request.num_prompt_tokens - 1) // bs * bs)
        count = max(0, matched-num_computed_tokens)
        return count, count > 0 and self.mode == 'async_full'

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens:
            bs = self.conductor.block_size
            total = min(self.conductor.query(request.prompt_token_ids, self.worker_name),
                        (request.num_prompt_tokens-1)//bs*bs)
            start = total-num_external_tokens
            hashes = self.conductor.hashes(request.prompt_token_ids[:total])[start//bs:]
            self.pending[request.request_id] = hashes

    def build_connector_meta(self, scheduler_output):
        value = TransferMeta(self.pending)
        self.pending = {}
        return value

    # Physical worker methods must not accidentally be called by this harness.
    def start_load_kv(self, *args, **kwargs): raise RuntimeError('handled by virtual executor')
    def wait_for_layer_load(self, *args, **kwargs): raise RuntimeError('handled by virtual executor')
    def save_kv_layer(self, *args, **kwargs): raise RuntimeError('read-only pool in starter')
    def wait_for_save(self): raise RuntimeError('read-only pool in starter')

def scheduler_class(name):
    if name == 'vllm': return Scheduler
    if name != 'ascend': raise ValueError(name)
    path = ROOT / 'vendor/vllm_ascend/core/scheduler.py'
    spec = importlib.util.spec_from_file_location('lab_upstream_ascend_scheduler', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.AscendScheduler

def make_scheduler(cfg, kind):
    bs, blocks = cfg['block_size'], cfg['num_blocks']
    sc = SchedulerConfig(max_num_seqs=cfg['max_num_seqs'],
        max_num_batched_tokens=cfg['max_num_batched_tokens'],
        max_model_len=cfg['max_model_len'], enable_chunked_prefill=cfg['chunked_prefill'],
        policy='fcfs', async_scheduling=False,
        long_prefill_token_threshold=cfg.get('chunk_size', 0))
    cc = CacheConfig(block_size=bs, gpu_memory_utilization=0.9,
        swap_space=0, cache_dtype='auto', enable_prefix_caching=False)
    cc.num_gpu_blocks = blocks
    vc = NS(scheduler_config=sc, cache_config=cc, lora_config=None, kv_events_config=None,
            parallel_config=NS(data_parallel_rank=0, decode_context_parallel_size=1, pipeline_parallel_size=1),
            model_config=NS(is_encoder_decoder=False, is_multimodal_model=False),
            speculative_config=None, kv_transfer_config=None)
    kv = KVCacheConfig(num_blocks=blocks, kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(['logical-layer'],
            FullAttentionSpec(bs, 1, 1, torch.float32, False))])
    # Each logical block represents storage allocated across all model layers;
    # no tensors or model weights are allocated. Physical byte size is in config.
    return scheduler_class(kind)(vc, kv, NoStructuredOutput(), log_stats=False), vc

class Engine:
    def __init__(self, sim, wid, kind):
        self.sim, self.wid = sim, wid
        self.clock, self.cfg = sim.clock, sim.cfg
        self.scheduler, vc = make_scheduler(self.cfg['engine'], kind)
        # Replace module-local clock references, never the process-wide time module.
        virtual_time = NS(time=lambda: self.clock.now, monotonic=lambda: self.clock.now)
        for module_name in ('vllm.v1.request', 'vllm.v1.core.sched.scheduler',
                            'lab_upstream_ascend_scheduler'):
            if module_name in sys.modules: sys.modules[module_name].time = virtual_time
        self.connector = SimConnector(vc, sim.conductor, f'w{wid}', self.cfg['load_mode'])
        self.scheduler.connector = self.connector
        self.busy = False; self.kick_event = None; self.last_output = None
        self.recv = set(); self.inflight_recv = set()
        self.active_batch_tokens = 0
        self.state = 'idle'; self.since = 0.0
        self.times = dict(idle=0.0, compute=0.0, stall=0.0)
        init_none_hash(sha256)
        self.hasher = get_request_block_hasher(self.cfg['engine']['block_size'], sha256)

    def transition(self, state):
        if state != self.state:
            self.times[self.state] += self.clock.now-self.since
            self.sim.emit('state', worker=self.wid, state=state)
            self.state, self.since = state, self.clock.now

    def add(self, job):
        request = Request(job['id'], job['tokens'], SamplingParams(
            max_tokens=job['output_tokens'], ignore_eos=True, temperature=0),
            pooling_params=None, eos_token_id=None, arrival_time=job['arrival_s'], block_hasher=self.hasher)
        self.scheduler.add_request(request)
        self.kick()

    def kick(self):
        if self.busy or self.kick_event is not None: return
        self.kick_event = self.clock.after(0, self.step, priority=20)

    def runner_output(self, so, completed=()):
        ids = list(so.num_scheduled_tokens)
        sampled = []
        for rid in ids:
            r = self.scheduler.requests.get(rid)
            sampled.append([7] if r and r.num_computed_tokens >= r.num_tokens else [])
        return ModelRunnerOutput(ids, {r:i for i,r in enumerate(ids)}, sampled,
            None, {}, [], KVConnectorOutput(finished_recving=set(completed)) if completed else None)

    def consume(self, so, output):
        result = self.scheduler.update_from_output(so, output)
        for bundle in result.values():
            for item in bundle.outputs:
                job = self.sim.jobs[item.request_id]
                if item.new_token_ids:
                    job.setdefault('first_token_s', self.clock.now)
                    job.setdefault('token_times_s', []).append(self.clock.now)
                if item.finish_reason is not None:
                    job['finish_s'] = self.clock.now
                    self.sim.emit('request_done', request=item.request_id, worker=self.wid)

    def step(self):
        self.kick_event = None
        if self.busy: return
        if self.recv:
            assert self.last_output is not None
            empty = copy.copy(self.last_output)
            empty.num_scheduled_tokens = {}; empty.total_num_scheduled_tokens = 0
            self.consume(empty, self.runner_output(empty, self.recv))
            self.recv.clear()
        # Optional experimental policy: reorder WAITING only, keeping all
        # upstream scheduling/allocation/preemption/continuation code intact.
        if self.cfg['local_policy'] == 'short_io':
            waiting = list(self.scheduler.waiting)
            waiting.sort(key=lambda r: (self.sim.local_score(r, self.wid), r.arrival_time, r.request_id))
            self.scheduler.waiting.clear(); self.scheduler.waiting.extend(waiting)
        so = self.scheduler.schedule()
        self.last_output = so
        meta = so.kv_connector_metadata
        loads = meta.loads if meta else {}
        self.sim.emit('schedule', worker=self.wid, tokens=dict(so.num_scheduled_tokens),
            running=[r.request_id for r in self.scheduler.running],
            waiting=[r.request_id for r in self.scheduler.waiting],
            free_blocks=self.scheduler.kv_cache_manager.block_pool.get_num_free_blocks())
        if self.cfg['load_mode'] == 'async_full':
            for rid, hashes in loads.items():
                self.inflight_recv.add(rid)
                def done(rid=rid):
                    self.inflight_recv.remove(rid); self.recv.add(rid); self.kick()
                self.sim.storage.submit(self.wid, hashes, self.sim.bytes_per_block_layer*self.cfg['model']['layers'],
                                        done, f'{rid}:all')
            loads = {}
        if not so.num_scheduled_tokens:
            self.consume(so, self.runner_output(so))
            self.transition('stall' if self.scheduler.get_num_unfinished_requests() else 'idle')
            return
        self.busy = True
        self.active_batch_tokens = so.total_num_scheduled_tokens
        Batch(self, so, loads).start()

    def complete(self, so):
        self.consume(so, self.runner_output(so, self.recv))
        self.recv.clear(); self.busy = False
        self.active_batch_tokens = 0
        self.transition('stall' if self.scheduler.get_num_unfinished_requests() else 'idle')
        self.kick()

class Batch:
    """One shared batch, each layer waits for all of that batch's required KV."""
    def __init__(self, engine, so, loads):
        self.e, self.so, self.loads = engine, so, loads
        self.sim, self.clock = engine.sim, engine.clock
        self.layers = self.sim.cfg['model']['layers']
        self.ready = [not loads]*self.layers
        self.submitted = set(); self.layer = 0; self.computing = False
        self.layer_time = self.sim.compute_time(engine, so) / self.layers

    def submit_layer(self, layer):
        if layer >= self.layers or layer in self.submitted: return
        self.submitted.add(layer)
        if not self.loads: return
        left = [len(self.loads)]
        def done():
            left[0] -= 1
            if not left[0]:
                self.ready[layer] = True
                if layer == self.layer: self.try_compute()
        for rid, hashes in self.loads.items():
            self.sim.storage.submit(self.e.wid, hashes, self.sim.bytes_per_block_layer,
                                    done, f'{rid}:L{layer}')

    def start(self):
        for layer in range(min(self.layers, self.sim.cfg['prefetch_layers'])): self.submit_layer(layer)
        self.try_compute()

    def try_compute(self):
        if self.computing: return
        if not self.ready[self.layer]: self.e.transition('stall'); return
        self.e.transition('compute'); self.computing = True
        self.sim.emit('layer_start', worker=self.e.wid, layer=self.layer,
                      requests=list(self.so.num_scheduled_tokens))
        self.clock.after(self.layer_time, self.layer_done, priority=0)

    def layer_done(self):
        self.computing = False; self.layer += 1
        if self.layer == self.layers: self.e.complete(self.so); return
        # Refill the bounded prefetch window as computation consumes a layer.
        self.submit_layer(self.layer+self.sim.cfg['prefetch_layers']-1)
        self.try_compute()
