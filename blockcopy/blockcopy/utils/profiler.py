import time
import torch

from contextlib import contextmanager
from collections import defaultdict

class Timings():
    def __init__(self, level=0):
        self.level = level
        self.average = True
        self.cnt = 0
        self.reset()
    
    def add_cnt(self, cnt=1):
        if self.level >= 0:
            self.cnt += cnt

    def set_level(self, level):
        self.level = level

    def reset(self):
        self.records = defaultdict(float)
        self.starts  = defaultdict(float)
        self.counts  = defaultdict(int)
        self.cnt = 0

    # start/stop
    def start(self, name, level=0):
        if level <= self.level:
            torch.cuda.synchronize()
            self.starts[name] = time.perf_counter()
            self.counts[name] += 1

    def stop(self, name, level=0):
        if name in self.starts:
            torch.cuda.synchronize()
            self.records[name] += time.perf_counter() - self.starts[name]

    def __repr__(self):
        s = ''
        if self.cnt > 0:
            s += f"### Profiler (images: {self.cnt})###\n"
            for name in sorted(self.records):
                val = self.records[name]*1000
                val_avg = val/self.cnt
                val_per_run = val/self.counts[name]
                
                s += f'# {name:20}: {val_avg:4.3f} ms per image (number of calls: {self.counts[name]}, per call: {val_per_run:4.3f} ms) \n'
        elif self.cnt == 0:
            s = '## Profiler: no batches registered'
        else:
            s = '## Profiler: disabled'
        return s
    
    @contextmanager
    def env(self, name, level=0):
        self.start(name, level)
        yield
        self.stop(name)

timings = Timings(level=0)
