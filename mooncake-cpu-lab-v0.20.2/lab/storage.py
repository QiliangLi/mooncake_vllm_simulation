"""Shared disks + per-path FIFO + shared worker NIC, in virtual seconds."""
import bisect
import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass

def hash64(value):
    return int.from_bytes(hashlib.sha256(str(value).encode()).digest()[:8], 'big')

@dataclass
class Flow:
    fid: int
    worker: int
    disk: int
    path: int
    remaining: float
    done: object
    rate: float = 0.0

def max_min_rates(constraints, capacities):
    """Unweighted progressive filling. Every flow consumes all named resources."""
    rates = {i: 0.0 for i in constraints}
    unfixed = set(rates)
    left = dict(capacities)
    while unfixed:
        counts = {r: sum(r in constraints[i] for i in unfixed) for r in capacities}
        step = min(left[r] / n for r, n in counts.items() if n)
        for i in unfixed: rates[i] += max(0.0, step)
        for r, n in counts.items(): left[r] -= step * n
        tight = {r for r in capacities if counts[r] and left[r] <= max(1e-3, capacities[r]*1e-12)}
        frozen = {i for i in unfixed if constraints[i] & tight}
        if not frozen: raise RuntimeError('water filling failed to make progress')
        unfixed -= frozen
    return rates

class Storage:
    def __init__(self, clock, cfg, emit):
        self.clock, self.cfg, self.emit = clock, cfg, emit
        self.queues = defaultdict(deque)
        self.last = 0.0; self.serial = 0; self.timer = None
        self.bytes_transferred = 0.0
        self.ring = sorted((hash64(f'disk:{d}:vnode:{v}'), d)
                           for d in range(cfg['disks']) for v in range(64))
        self.ring_points = [x[0] for x in self.ring]

    def disk_for(self, block_hash):
        return self.ring[bisect.bisect_left(self.ring_points, hash64(block_hash)) % len(self.ring)][1]

    def heads(self):
        return [q[0] for q in self.queues.values() if q]

    def advance(self):
        dt = self.clock.now - self.last
        for f in self.heads():
            moved = min(f.remaining, dt * f.rate)
            f.remaining -= moved; self.bytes_transferred += moved
        self.last = self.clock.now

    def submit(self, worker, hashes, bytes_per_block, done, label):
        # Transport latency precedes queue admission; no fixed bandwidth per request.
        groups = defaultdict(float)
        for h in hashes:
            disk = self.disk_for(h)
            path = hash64(f'{worker}:{h}') % self.cfg['paths_per_disk']
            groups[disk, path] += bytes_per_block
        if not groups:
            self.clock.after(0, done); return
        count = [len(groups)]
        def part_done():
            count[0] -= 1
            if not count[0]:
                self.emit('io_done', worker=worker, label=label)
                done()
        def enqueue():
            self.advance()
            for (disk, path), size in groups.items():
                self.serial += 1
                self.queues[disk, path].append(Flow(self.serial, worker, disk, path, size, part_done))
            self.emit('io_submit', worker=worker, label=label, bytes=sum(groups.values()))
            self.rebalance()
        self.clock.after(self.cfg['latency_s'], enqueue, priority=0)

    def rebalance(self):
        if self.timer: self.timer.cancel()
        heads = self.heads()
        if not heads: return
        capacities, constraints = {}, {}
        for f in heads:
            disk, nic, path = ('disk', f.disk), ('nic', f.worker), ('path', f.disk, f.path)
            capacities[disk] = self.cfg['disk_Bps']
            capacities[nic] = self.cfg['nic_Bps'][f.worker]
            capacities[path] = self.cfg.get('path_Bps', self.cfg['disk_Bps'])
            constraints[f.fid] = {disk, nic, path}
        rates = max_min_rates(constraints, capacities)
        for f in heads: f.rate = rates[f.fid]
        self.timer = self.clock.after(min(f.remaining/f.rate for f in heads), self.finish, priority=0)

    def finish(self):
        self.advance()
        callbacks = []
        for q in self.queues.values():
            if q and q[0].remaining < 1e-3:
                callbacks.append(q.popleft().done)
        self.rebalance()
        for cb in callbacks: cb()

    def snapshot(self):
        self.advance()
        return {'sample_time': self.clock.now,
                'worker_queued_bytes': [sum(f.remaining for q in self.queues.values() for f in q
                                            if f.worker == w) for w in range(len(self.cfg['nic_Bps']))],
                'disk_queued_bytes': [sum(f.remaining for (disk, _), q in self.queues.items()
                                          if disk == d for f in q) for d in range(self.cfg['disks'])],
                'active_paths': [sum(bool(q) for (disk, _), q in self.queues.items() if disk == d)
                                 for d in range(self.cfg['disks'])]}
