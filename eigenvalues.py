#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT
import numpy as np
import time
from utils import get_info

try:
    import skcuda.magma as magma
    HAS_SKCUDA = True
except:
    HAS_SKCUDA = False
from scipy.linalg import eig

np.random.seed(1234)

from scipy.linalg import eig

typedict = {"s": np.float32, "d": np.float64, "c": np.complex64, "z": np.complex128}
typedict_ = {v: k for k, v in typedict.items()}


def eig_magma(a):
    """
    Eigenvalue solver using Magma GPU library (variants of Lapack's geev).

    Note:
    - not a generalized eigenvalue solver
    """

    if len(a.shape) != 2:
        raise ValueError("M needs to be a rank 2 square array for eig.")

    magma.magma_init()

    dtype = type(a[0, 0])
    t = typedict_[dtype]
    N = a.shape[0]

    # Set up output buffers:
    if t in ["s", "d"]:
        wr = np.zeros((N,), dtype)  # eigenvalues
        wi = np.zeros((N,), dtype)  # eigenvalues
    elif t in ["c", "z"]:
        w = np.zeros((N,), dtype)  # eigenvalues

    vl = np.zeros((1, 1), dtype)
    jobvl = "N"
    vr = np.zeros((N, N), dtype)
    jobvr = "V"

    # Set up workspace:
    if t == "s":
        nb = magma.magma_get_sgeqrf_nb(N, N)
    if t == "d":
        nb = magma.magma_get_dgeqrf_nb(N, N)
    if t == "c":
        nb = magma.magma_get_cgeqrf_nb(N, N)
    if t == "z":
        nb = magma.magma_get_zgeqrf_nb(N, N)

    lwork = N * (1 + 2 * nb)
    work = np.zeros((lwork,), dtype)
    if t in ["c", "z"]:
        rwork = np.zeros((2 * N,), dtype)

    # Compute:
    gpu_time = time.time()
    if t == "s":
        status = magma.magma_sgeev(
            jobvl,
            jobvr,
            N,
            a.ctypes.data,
            N,
            wr.ctypes.data,
            wi.ctypes.data,
            vl.ctypes.data,
            N,
            vr.ctypes.data,
            N,
            work.ctypes.data,
            lwork,
        )
    if t == "d":
        status = magma.magma_dgeev(
            jobvl,
            jobvr,
            N,
            a.ctypes.data,
            N,
            wr.ctypes.data,
            wi.ctypes.data,
            vl.ctypes.data,
            N,
            vr.ctypes.data,
            N,
            work.ctypes.data,
            lwork,
        )
    if t == "c":
        status = magma.magma_cgeev(
            jobvl,
            jobvr,
            N,
            a.ctypes.data,
            N,
            w.ctypes.data,
            vl.ctypes.data,
            N,
            vr.ctypes.data,
            N,
            work.ctypes.data,
            lwork,
            rwork.ctypes.data,
        )
    if t == "z":
        status = magma.magma_zgeev(
            jobvl,
            jobvr,
            N,
            a.ctypes.data,
            N,
            w.ctypes.data,
            vl.ctypes.data,
            N,
            vr.ctypes.data,
            N,
            work.ctypes.data,
            lwork,
            rwork.ctypes.data,
        )

    if t in ["s", "d"]:
        w_gpu = wr + 1j * wi
    else:
        w_gpu = w

    magma.magma_finalize()

    return w_gpu, vr


if __name__ == "__main__":
    
    ## just to init
    matrix = np.random.random((3, 3))
    W, V = eig(matrix)
    if HAS_SKCUDA:
        W, V = eig_magma(matrix)

    matsize = [250, 500, 1000, 2000, 4000, 8000]
    benchmark_timing = dict(magma=[], scipy=[])

    benchmark_info = dict(magma=[], scipy=[])
    
    cases = ["scipy"]
    if HAS_SKCUDA:
        cases += ["magma"]

    for N in matsize:
        print("\n")
        print("#"*40)
        print("matrix size = ", N)
        matrix = np.random.random((N, N))
        for case in cases:
            print(">> case: ", case)

            elapsed_time = time.time()
            if case == "magma":
                W, V = eig_magma(matrix)
            else:
                W, V = eig(matrix)
            elapsed_time = time.time() - elapsed_time
            print("   elapsed time = ", elapsed_time)
            info = get_info(verbose=True)
            benchmark_info[case].append(info)

    np.savez(
        "benchmark.npz",
        matsize=matsize,
        benchmark_timing=benchmark_timing,
        benchmark_info=benchmark_info,
    )
    
