"""
A naive implementation of blocked Cholesky using Numba kernels on CPUs.
"""

import numpy as np
# import cupy
from numba import jit, void, float64
import math
import time

from parla import Parla, get_all_devices
from parla.array import copy, clone_here

from parla.cuda import gpu
from parla.cpu import cpu

from parla.function_decorators import specialized
from parla.tasks import *

import cupy as cp
from cupy.cuda import cublas
from cupy.cuda import device
from cupy.linalg import _util

from scipy import linalg

loc = gpu

gpu_arrs = []

#EXEC_SOL = True
EXEC_SOL = False
EXEC_MULT_GPU = True
#EXEC_MULT_GPU = False

@specialized
@jit(float64[:,:](float64[:,:]), nopython=True, nogil=True)
def cholesky(a):
    """
    Naive version of dpotrf. Write results into lower triangle of the input array.
    """
    if a.shape[0] != a.shape[1]:
        raise ValueError("A square array is required.")
    for j in range(a.shape[0]):
        a[j,j] = math.sqrt(a[j,j] - (a[j,:j] * a[j,:j]).sum())
        for i in range(j+1, a.shape[0]):
            a[i,j] -= (a[i,:j] * a[j,:j]).sum()
            a[i,j] /= a[j,j]
    return a


@cholesky.variant(gpu)
def choleksy_gpu(a):
    a = cp.linalg.cholesky(a)
    return a

@specialized
@jit(float64[:,:](float64[:,:], float64[:,:]), nopython=True, nogil=True)
def ltriang_solve(a, b):
    """
    This is a naive version of dtrsm. The result is written over the input array `b`.
    """
    b = b.T
    if a.shape[0] != b.shape[0]:
        raise ValueError("Input array shapes are not compatible.")
    if a.shape[0] != a.shape[1]:
        raise ValueError("Array for back substitution is not square.")
    # For the implementation here, just assume lower triangular.
    for i in range(a.shape[0]):
        b[i] /= a[i,i]
        b[i+1:] -= a[i+1:,i:i+1] * b[i:i+1]
    return b.T

#comments would repack the data to column-major
def cupy_trsm_wrapper(a, b):
    cublas_handle = device.get_cublas_handle()
    trsm = cublas.dtrsm
    uplo = cublas.CUBLAS_FILL_MODE_LOWER

    a = cp.array(a, dtype=np.float64, order='F')
    b = cp.array(b, dtype=np.float64, order='F')
    trans = cublas.CUBLAS_OP_T
    side = cublas.CUBLAS_SIDE_RIGHT

    #trans = cublas.CUBLAS_OP_T
    #side = cublas.CUBLAS_SIDE_LEFT

    diag = cublas.CUBLAS_DIAG_NON_UNIT
    m, n = (b.side, 1) if b.ndim == 1 else b.shape
    trsm(cublas_handle, side, uplo, trans, diag, m, n, 1.0, a.data.ptr, m, b.data.ptr, m)
    return b

@ltriang_solve.variant(gpu)
def ltriang_solve_gpu(a, b):
    b = cupy_trsm_wrapper(a, b)
    return b

def update_kernel(a, b, c):
    c -= a @ b.T
    return c

@specialized
def update(a, b, c):
    c = update_kernel(a, b, c)
    return c

@update.variant(gpu)
def update_gpu(a, b, c):
    c = update_kernel(a, b, c)
    #c = cupy_gemm_wrapper(a, b, c)
    return c

def cholesky_blocked_inplace(a, num_gpus):
    """
    This is a less naive version of dpotrf with one level of blocking.
    Blocks are currently assumed to evenly divide the axes lengths.
    The input array 4 dimensional. The first and second index select
    the block (row first, then column). The third and fourth index
    select the entry within the given block.
    """
    if a.shape[0] * a.shape[2] != a.shape[1] * a.shape[3]:
        raise ValueError("A square matrix is required.")
    if a.shape[0] != a.shape[1]:
        raise ValueError("Non-square blocks are not supported.")

    # Define task spaces
    gemm1 = TaskSpace("gemm1")        # Inter-block GEMM
    subcholesky = TaskSpace("subcholesky")  # Cholesky on block
    gemm2 = TaskSpace("gemm2")        # Inter-block GEMM
    solve = TaskSpace("solve")        # Triangular solve

    for j in range(a.shape[0]):
        for k in range(j):
            # Inter-block GEMM
            @spawn(gemm1[j, k], [solve[j, k]], placement=[gpu(j%num_gpus)])
            #@spawn(gemm1[j, k], [solve[j, k]], placement=loc)
            def t1():
                if EXEC_SOL:
                  out = clone_here(a[j,j])  # Move data to the current device
                  rhs = clone_here(a[j,k])
                  out = update(rhs, rhs, out)
                  copy(a[j,j], out)  # Move the result to the global array

                if EXEC_MULT_GPU:
                  out = get_gpu_memory(j, j, num_gpus)
                  rhs = get_gpu_memory(j, k, num_gpus)
                  out = update(rhs, rhs, out)
                  set_gpu_memory_from_gpu(j, j, num_gpus, out)

                if EXEC_SOL and EXEC_MULT_GPU:
                  print (out, " and ", a[j,j])

        # Cholesky on block
        @spawn(subcholesky[j], [gemm1[j, 0:j]], placement=[gpu(j%num_gpus)])
        def t2():
            if EXEC_SOL:
              dblock = clone_here(a[j, j])
              dblock = cholesky(dblock)
              copy(a[j, j], dblock)

            if EXEC_MULT_GPU:
              dblock = get_gpu_memory(j, j, num_gpus) 
              dblock = cholesky(dblock)
              set_gpu_memory_from_gpu(j, j, num_gpus, dblock)

            if EXEC_SOL and EXEC_MULT_GPU:
              print (dblock, " and ", a[j,j])

        for i in range(j+1, a.shape[0]):
            for k in range(j):
                # Inter-block GEMM
                @spawn(gemm2[i, j, k], [solve[j, k], solve[i, k]], placement=[gpu(i%num_gpus)])
                def t3():
                    if EXEC_SOL:
                      out = clone_here(a[i,j])  # Move data to the current device
                      rhs1 = clone_here(a[i,k])
                      rhs2 = clone_here(a[j,k])
                      out = update(rhs1, rhs2, out)
                      copy(a[i,j], out)  # Move the result to the global array

                    if EXEC_MULT_GPU:
                      cur_id = i % num_gpus
                      out = get_gpu_memory(i, j, num_gpus)
                      rhs1 = get_gpu_memory(i, k, num_gpus)
                      rhs2 = copy_gpu_memory(cur_id, j, k, num_gpus)
                      out = update(rhs1, rhs2, out)
                      set_gpu_memory_from_gpu(i, j, num_gpus, out)
                    if EXEC_SOL and EXEC_MULT_GPU:
                      print (out, " and ", a[i,j])

            # Triangular solve
            @spawn(solve[i, j], [gemm2[i, j, 0:j], subcholesky[j]], placement=[gpu(i%num_gpus)])
            def t4():
                if EXEC_SOL:
                  factor = clone_here(a[j, j])
                  panel = clone_here(a[i, j])
                  out = ltriang_solve(factor, panel)
                  copy(a[i, j], out)

                if EXEC_MULT_GPU:
                  cur_id = i % num_gpus
                  factor = copy_gpu_memory(cur_id, j, j, num_gpus)
                  panel  = get_gpu_memory(i, j, num_gpus)
                  out = ltriang_solve(factor, panel)
                  set_gpu_memory_from_gpu(i, j, num_gpus, out)

                if EXEC_SOL and EXEC_MULT_GPU:
                  print (out, " and ", a[i,j])

    return subcholesky[a.shape[0]-1]

def set_device(i:int):
    cudevice = cp.cuda.Device(i)
    try:
        cudevice.compute_capability
    except cp.cuda.runtime.CUDARuntimeError:
        print("Fail to set device:"+str(i))
    return cudevice

def allocate_gpu_memory(i:int, r:int, n:int, b:int):
    with cp.cuda.Device(i):
      prealloced = cp.ndarray([r, n // b, b, b])
      gpu_arrs.append(prealloced)

def copy_gpu_memory(cur_id:int, i:int, j:int, num_gpus:int):
    dev_id   = i % num_gpus
    local_id = i // num_gpus
    with cp.cuda.Device(cur_id):
      return cp.array(gpu_arrs[dev_id][local_id][j], copy=True)

def get_gpu_memory(i:int, j:int, num_gpus:int):
    dev_id   = i % num_gpus
    local_id = i // num_gpus
    return gpu_arrs[dev_id][local_id][j]

def set_gpu_memory_from_gpu(i:int, j:int, num_gpus:int, v):
    dev_id   = i % num_gpus
    local_id = i // num_gpus
    gpu_arrs[dev_id][local_id][j] = v

def set_gpu_memory_from_cpu(a, num_gpus):
    for j in range(a.shape[0]):
      dev_id   = j % num_gpus 
      local_id = j // num_gpus 
      with cp.cuda.Device(dev_id):
        gpu_arrs[dev_id][local_id] = cp.array(a[j], copy=True)

def main():
    num_gpus = cp.cuda.runtime.getDeviceCount()
    @spawn(placement=cpu)
    async def test_blocked_cholesky():
        # Configure environment
        block_size = 32*5
        n = block_size*16
        #block_size = 2
        #n = block_size * 7 
        assert not n % block_size

        if EXEC_MULT_GPU:
          for d in range(num_gpus):
            row_size = n // (block_size * num_gpus)
            if d < ((n / block_size) % num_gpus):
              row_size += 1
              """
            elif d != 0 and d == ((n / block_size) % num_gpus):
              row_size += 1
              print("2 add")
              """
            if row_size > 0:
              allocate_gpu_memory(d, row_size, n, block_size)

        np.random.seed(10)
        # Construct input data
        a = np.random.rand(n, n)
        a = a @ a.T

        # Copy and layout input
        a1 = a.copy()
        ap = a1.reshape(n // block_size, block_size, n // block_size, block_size).swapaxes(1,2)
        start = time.perf_counter()
        if EXEC_MULT_GPU:
          set_gpu_memory_from_cpu(ap, num_gpus)

        # Call Parla Cholesky result and wait for completion
        await cholesky_blocked_inplace(ap, num_gpus)

        end = time.perf_counter()
        print(end - start, "seconds")

        if EXEC_MULT_GPU:
          for d in range(num_gpus):
            gpu_arrs[d] = cp.swapaxes(gpu_arrs[d], 2, 1)

          cpu_arrs = cp.asnumpy(gpu_arrs[0][0][0])
          for r_num in range(n // block_size):
            dev_id   = r_num % num_gpus 
            local_id = r_num // num_gpus 
            cpu_sub_arr = cp.asnumpy(gpu_arrs[dev_id][local_id])
            loop = True
            for cpu_sub_sub_arr in cpu_sub_arr:
              if dev_id == 0 and local_id == 0 and loop:
                loop = False
                continue
              cpu_arrs = np.concatenate((cpu_arrs, cpu_sub_sub_arr))
          cpu_arrs = cpu_arrs.reshape(n, n)
          """
          for s in range(1, len(gpu_arrs)):
            cpu_arrs = np.concatenate((cpu_arrs, cp.asnumpy(gpu_arrs[s])))
            """
          #cpu_arrs = cpu_arrs.swapaxes(1,2).reshape(n, n)

        print("Truth", linalg.cholesky(a).T)

        # Check result
        if EXEC_MULT_GPU:
          computed_L = np.tril(cpu_arrs)
        else:
          computed_L = np.tril(a1)
        print("Soln", computed_L)
        error = np.max(np.absolute(a-computed_L @ computed_L.T))
        print("Error", error)
        assert(error < 1E-8)

if __name__ == '__main__':
    with Parla():
        main()
