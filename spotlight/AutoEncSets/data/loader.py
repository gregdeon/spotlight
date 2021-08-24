from __future__ import print_function
import numpy as np
import os
import time

from multiprocessing import Process, Queue, Value


def _worker_fn(batches_remaining, index_queue, sampler):
    while batches_remaining.value > 0:
        indices = sampler()
        index_queue.put(indices)
        batches_remaining.value = batches_remaining.value - 1


class IndexIterator(object):
    def __init__(self, iters_per_epoch, sampler, n_workers=0, epochs=1):
        self.n_workers = n_workers
        self.iters_per_epoch = iters_per_epoch
        self.items_returned = 0
        self.epochs = epochs
        self.iters_left = len(self)
        self.sampler = sampler
        if n_workers > 0:
            self.batches_remaining = Value('i', len(self))
            self.index_queue = Queue()
            self.workers = [Process(target=_worker_fn, args=(self.batches_remaining,
                                                             self.index_queue,
                                                             self.sampler))
                            for i in range(n_workers)]
            for p in self.workers:
                p.start()
        
    def __len__(self):
        n = self.iters_per_epoch * self.epochs
        return n
        
    def __iter__(self):
        return self
    
    def _get_batch(self):
        if self.n_workers > 0:
            return self.index_queue.get()
        else:
            return self.sampler()
        
    def __next__(self):
        if self.iters_left > 0:
            sampled = self._get_batch()
            self.iters_left -= 1
            return sampled
        else:
            self._shutdown_workers()
            raise StopIteration()
    
    next = __next__
        
    def _shutdown_workers(self):
        if self.n_workers > 0:
            for p in self.workers:
                if p is not None:
                    p.terminate()
    
    def __del__(self):
        if self.n_workers > 0:
            self._shutdown_workers()



if __name__ == '__main__':
    import recsys
    from samplers import UniformSampler
    
    data = recsys.ml100k(0.)
    sampler = UniformSampler(80000, data)
    ave = 0
    n = 5
    sleep_time = 0.2
    for epoch in range(1):
        iterator = IndexIterator(1, sampler, n_workers=1, epochs=n)
        t = time.time()
        for i, idx in enumerate(iterator):
            print(i)
            time.sleep(sleep_time)
            pass
        extra = time.time() - t
        ave += extra - (i+1) * sleep_time
    print("average time per epoch: %1.3f" % (ave / n))
