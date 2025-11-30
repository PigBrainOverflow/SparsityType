import sys, os
sys.path.insert(0, os.getcwd())

import spring as spr
import numpy as np
import scipy.sparse as sp
import time

def benchmark(N: int, density: float, length: int, repeat: int = 100, tol: float = 1e-5):
    desc = f"spaddchain with size {N}x{N}, density {density} and length {length}"
    print(f"\n=== {desc} ===")

    sp_times = []
    spr_times = []
    np_times = []

    for _ in range(repeat):
        # create random sparse matrices
        sp_matrices = [sp.random(N, N, density=density, format="csr", dtype=np.float32) for _ in range(length)]
        np_matrices = [m.todense() for m in sp_matrices]
        spr_matrices = [spr.NDArray.from_sparse(m) for m in sp_matrices]

        # run and time Spring
        start = time.time()
        spr_result = spr_matrices[0]
        for mat in spr_matrices[1:]:
            spr_result = spr_result + mat
        spr_times.append(time.time() - start)

        # run and time SciPy
        start = time.time()
        sp_result = sp_matrices[0]
        for mat in sp_matrices[1:]:
            sp_result = sp_result + mat
        sp_times.append(time.time() - start)

        # run and time NumPy
        start = time.time()
        np_result = np_matrices[0]
        for mat in np_matrices[1:]:
            np_result = np_result + mat
        np_times.append(time.time() - start)

        # verify
        spr_result_dense = spr_result.to_dense()
        if np.linalg.norm(spr_result_dense - np_result) > tol * N * N:
            raise ValueError("Results do not match!")

    print(f"Spring avg time: {np.mean(spr_times):.6f} s")
    print(f"SciPy avg time: {np.mean(sp_times):.6f} s")
    print(f"NumPy avg time: {np.mean(np_times):.6f} s")


if __name__ == "__main__":
    benchmark(N=100, density=0.01, length=100, repeat=10)
    benchmark(N=200, density=0.01, length=100, repeat=10)
    benchmark(N=500, density=0.01, length=100, repeat=10)

    benchmark(N=100, density=0.005, length=100, repeat=10)
    benchmark(N=200, density=0.005, length=100, repeat=10)
    benchmark(N=500, density=0.005, length=100, repeat=10)

    benchmark(N=100, density=0.001, length=100, repeat=10)
    benchmark(N=200, density=0.001, length=100, repeat=10)
    benchmark(N=500, density=0.001, length=100, repeat=10)