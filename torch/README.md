
# Benchmarks


Install requirements

```bash
module load miniconda
conda create -n torch python=3.9
conda activate torch
conda install pytorch magma cudatoolkit=11.3 -c pytorch
conda install numpy scipy
```

Run benchmark

```bash
conda activate torch
export OMP_NUM_THREADS=1
python benchmark.py
```
