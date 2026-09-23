import heapq
import itertools
from dataclasses import dataclass

@dataclass
class Handle:
    cancelled: bool = False
    def cancel(self): self.cancelled = True

class Clock:
    """Virtual seconds. No sleeps. Stable same-time event order."""
    def __init__(self):
        self.now = 0.0
        self.queue = []
        self.serial = itertools.count()
        self.events = 0

    def at(self, when, fn, priority=10):
        if when < self.now - 1e-12: raise ValueError('event scheduled in the past')
        h = Handle()
        heapq.heappush(self.queue, (max(when, self.now), priority, next(self.serial), h, fn))
        return h

    def after(self, dt, fn, priority=10):
        if dt < 0: raise ValueError('negative duration')
        return self.at(self.now + dt, fn, priority)

    def run(self, limit=1000000):
        while self.queue:
            t, _, _, h, fn = heapq.heappop(self.queue)
            if h.cancelled: continue
            self.now = t; self.events += 1
            if self.events > limit: raise RuntimeError('event limit: possible livelock')
            fn()
