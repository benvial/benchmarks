
# Benchmarks


Install requirements

```bash
module load miniconda
conda create -n benchmarks python=3.9
conda activate benchmarks
conda install -c conda-forge pycuda magma
pip install -r requirements.txt
```

Run benchmark

```bash
conda activate benchmarks
export OMP_NUM_THREADS=1
python eigenvalues.py 
```
