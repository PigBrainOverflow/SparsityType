import sys, os
sys.path.insert(0, os.getcwd())

import spring as spr
import numpy as np
import scipy.sparse as sp
import time


def benchmark(N: int, density: float, repeat: int = 100, tol: float = 1e-5):
    desc = f"spmatmul with size {N}x{N} and density {density}"
    print(f"\n=== {desc} ===")

    sp_times = []
    np_times = []
    spr_times = []

    for _ in range(repeat):
        # create random sparse matrices
        sp_a = sp.random(N, N, density=density, format="csr", dtype=np.float32)
        sp_b = sp.random(N, N, density=density, format="csr", dtype=np.float32)
        np_a = sp_a.todense()
        np_b = sp_b.todense()
        spr_a = spr.NDArray.from_sparse(sp_a)
        spr_b = spr.NDArray.from_sparse(sp_b)

        # run and time
        start = time.time()
        spr_c = spr_a @ spr_b
        spr_times.append(time.time() - start)

        start = time.time()
        sp_c = sp_a @ sp_b
        sp_times.append(time.time() - start)

        start = time.time()
        np_c = np_a @ np_b
        np_times.append(time.time() - start)

        # verify
        spr_c_dense = spr_c.to_dense()
        if np.linalg.norm(spr_c_dense - np_c) > tol * N * N:
            raise ValueError("Results do not match!")

    print(f"Spring avg time: {np.mean(spr_times):.6f} s")
    print(f"SciPy avg time: {np.mean(sp_times):.6f} s")
    print(f"NumPy avg time: {np.mean(np_times):.6f} s")


if __name__ == "__main__":
    benchmark(N=1000, density=0.01, repeat=10)
    benchmark(N=2000, density=0.01, repeat=10)
    benchmark(N=5000, density=0.01, repeat=10)

    benchmark(N=1000, density=0.005, repeat=10)
    benchmark(N=2000, density=0.005, repeat=10)
    benchmark(N=5000, density=0.005, repeat=10)

    benchmark(N=1000, density=0.001, repeat=10)
    benchmark(N=2000, density=0.001, repeat=10)
    benchmark(N=5000, density=0.001, repeat=10)