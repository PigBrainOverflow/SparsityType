import numpy as np
import scipy.sparse as sp
import time
import matplotlib.pyplot as plt

N = 5000  
densities = np.logspace(-4, 0, 10)  
dense_times = []
sparse_times = []

for d in densities:
    print(f"\nTesting density = {d:.5f}")

    A_sparse = sp.random(N, N, density=d, format="csr")
    I_sparse = sp.identity(N, format="csr")

    start = time.perf_counter()
    _ = I_sparse @ A_sparse
    sparse_t = time.perf_counter() - start
    sparse_times.append(sparse_t)

    if d >= 1e-3:  
        A_dense = A_sparse.toarray()
        I_dense = np.eye(N)
        start = time.perf_counter()
        _ = I_dense @ A_dense
        dense_t = time.perf_counter() - start
    else:
        dense_t = np.nan
    dense_times.append(dense_t)

    print(f"Sparse time: {sparse_t:.4f}s | Dense time: {dense_t:.4f}s")

for d, t_d, t_s in zip(densities, dense_times, sparse_times):
    if not np.isnan(t_d) and t_d < t_s:
        print(f"\n⚡ Dense becomes faster than Sparse at density ≈ {d:.4f}")
        break