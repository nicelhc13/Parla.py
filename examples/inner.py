"""
A simple inner product implemented in Parla.

This is probably the most basic example of Parla.
"""
import numpy as np

from parla import Parla
from parla.array import copy, storage_size
from parla.cuda import gpu
from parla.cpu import cpu
from parla.ldevice import LDeviceSequenceBlocked
from parla.tasks import *
import time
import os

def main():
    divisions = 10
    mapper = LDeviceSequenceBlocked(divisions)

    async def inner(a, b):
        a_part = mapper.partition_tensor(a)
        b_part = mapper.partition_tensor(b)
        # Create array to store partial sums from each logical device
        partial_sums = np.empty(divisions)
        # Start a block of tasks that much all complete before leaving the block.
        async with finish():
            # For each logical device, perform the local inner product using the numpy multiply operation, @.
            for i in range(divisions):
                @spawn(placement=[a_part[i], b_part[i]], memory=storage_size(a_part[i], b_part[i]))
                def inner_local():
                    copy(partial_sums[i:i+1], a_part[i] @ b_part[i])
        # Reduce the partial results (sequentially)
        res = 0.
        for i in range(divisions):
            res += partial_sums[i]
        return res

    @spawn()
    async def main_task():
        n = 3*1000
        a = np.random.rand(n)
        b = np.random.rand(n)
        print("Starting.", a.shape, b.shape)
        res = await inner(a, b)
        assert np.allclose(np.inner(a, b), res)
        print("Success.", res)


if __name__ == '__main__':
    with Parla():
        main()
