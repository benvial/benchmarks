#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import time
import numpy as np
import torch
import scipy.linalg
import scipy.fftpack

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

N=800
import sys
N=int(sys.argv[1])

A = torch.rand(N, N) +1j *torch.rand(N, N)

Adev = A.to(device)


print("------- eig ---------")


t_eig_numpy =-time.time()
w,v =  np.linalg.eig(A)
t_eig_numpy += time.time()
print(f"numpy time {t_eig_numpy:0.4}s")


t_eig_scipy =-time.time()
w,v =  scipy.linalg.eig(A)
t_eig_scipy += time.time()
print(f"scipy time {t_eig_scipy:0.4}s")

t_eig_torch =-time.time()
w,v =  torch.linalg.eig(Adev)
t_eig_torch+=time.time()
print(f"torch time {t_eig_torch:0.4}s")

print(f"speedup {t_eig_numpy/t_eig_torch:0.4}")


print("------- inv ---------")


t_eig_numpy =-time.time()
_ =  np.linalg.inv(A)
t_eig_numpy += time.time()
print(f"numpy time {t_eig_numpy:0.4}s")


t_eig_scipy =-time.time()
_ =  scipy.linalg.inv(A)
t_eig_scipy += time.time()
print(f"scipy time {t_eig_scipy:0.4}s")

t_eig_torch =-time.time()
_ =  torch.linalg.inv(Adev)
t_eig_torch+=time.time()
print(f"torch time {t_eig_torch:0.4}s")

print(f"speedup {t_eig_numpy/t_eig_torch:0.4}")



print("------- fft ---------")

N=2**(int(sys.argv[2]))

A = torch.rand(N, N) +1j *torch.rand(N, N)

Adev = A.to(device)

t_eig_numpy =-time.time()
_ =  np.fft.fft2(A)
t_eig_numpy += time.time()
print(f"numpy time {t_eig_numpy:0.4}s")


t_eig_scipy =-time.time()
_ =  scipy.fftpack.fft2(np.array(A))
t_eig_scipy += time.time()
print(f"scipy time {t_eig_scipy:0.4}s")

t_eig_torch =-time.time()
_ =  torch.fft.fft2(Adev)
t_eig_torch+=time.time()
print(f"torch time {t_eig_torch:0.4}s")

print(f"speedup {t_eig_numpy/t_eig_torch:0.4}")
