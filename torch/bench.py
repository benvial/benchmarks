#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import time
import numpy as np
import torch
import scipy.linalg
import scipy.fftpack
import sys

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# N = 800

# matsize=[10,100,200,400,800,1600,3200]
#
# for N in matsize:
#     print("#"*50)
#     print(f"matrix size = {N}")
#
#     # N = int(sys.argv[1])
#
#     A = torch.rand(N, N) + 1j *torch.rand(N, N)
#
#     Adev = A.to(device)
#
#
#     print("------- eig ---------")
#
#     t_eig_numpy =-time.time()
#     w,v =  np.linalg.eig(A)
#     t_eig_numpy += time.time()
#     print(f"numpy time {t_eig_numpy:0.4}s")
#
#
#     t_eig_scipy =-time.time()
#     w,v =  scipy.linalg.eig(A)
#     t_eig_scipy += time.time()
#     print(f"scipy time {t_eig_scipy:0.4}s")
#
#     t_eig_torch =-time.time()
#     w,v =  torch.linalg.eig(Adev)
#     t_eig_torch+=time.time()
#     print(f"torch time {t_eig_torch:0.4}s")
#
#     print(f"speedup numpy {t_eig_numpy/t_eig_torch:0.4}")
#     print(f"speedup scipy {t_eig_scipy/t_eig_torch:0.4}")
#
#
#     print("------- inv ---------")
#
#
#     t_inv_numpy =-time.time()
#     _ =  np.linalg.inv(A)
#     t_inv_numpy += time.time()
#     print(f"numpy time {t_inv_numpy:0.4}s")
#
#
#     t_inv_scipy =-time.time()
#     _ =  scipy.linalg.inv(A)
#     t_inv_scipy += time.time()
#     print(f"scipy time {t_inv_scipy:0.4}s")
#
#     t_inv_torch =-time.time()
#     _ =  torch.linalg.inv(Adev)
#     t_inv_torch+=time.time()
#     print(f"torch time {t_inv_torch:0.4}s")
#
#     print(f"speedup numpy {t_inv_numpy/t_inv_torch:0.4}")
#     print(f"speedup scipy {t_inv_scipy/t_inv_torch:0.4}")
#

matsize=[2,8,9,10,11,12]


for n in matsize:

    N=2**(n)
    print("#"*50)
    print(f"matrix size = {N}")


    print("------- fft ---------")


    A = torch.rand(N, N) +1j *torch.rand(N, N)

    Adev = A.to(device)

    t_fft_numpy =-time.time()
    _ =  np.fft.fft2(A)
    t_fft_numpy += time.time()
    print(f"numpy time {t_fft_numpy:0.4}s")


    t_fft_scipy =-time.time()
    _ =  scipy.fftpack.fft2(np.array(A))
    t_fft_scipy += time.time()
    print(f"scipy time {t_fft_scipy:0.4}s")

    t_fft_torch =-time.time()
    _ =  torch.fft.fft2(Adev)
    t_fft_torch+=time.time()
    print(f"torch time {t_fft_torch:0.4}s")


    print(f"speedup numpy {t_fft_numpy/t_fft_torch:0.4}")
    print(f"speedup scipy {t_fft_scipy/t_fft_torch:0.4}")
