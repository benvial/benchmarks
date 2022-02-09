#!/usr/bin/env python
import os
import time
import sys

nt = int(sys.argv[1])
os.environ["OMP_NUM_THREADS"] = f"{nt}"
# os.environ["OPENBLAS_NUM_THREADS"] = f"{nt}"
# os.environ["MKL_NUM_THREADS"] =  f"{nt}"
# os.environ["VECLIB_MAXIMUM_THREADS"] =  f"{nt}"
# os.environ["NUMEXPR_NUM_THREADS"] =  f"{nt}"

#
# import mkl
# mkl.set_num_threads(nt)


import numpy as np


N = 2000
M = np.random.rand(N,N)
t = -time.time()
a,b = np.linalg.eig(M)
t += time.time()
print(f"lapsed time {t}")
