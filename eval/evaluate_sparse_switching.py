import numpy as np
import scipy.sparse as sp
import time
import psutil
import matplotlib.pyplot as plt

N = 2000
densities = np.linspace(0.01, 1.0, 10)
results = {"numpy": [], "scipy": []}

def measure_time_and_memory(func):
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1e6
    start = time.time()
    func()
    end = time.time()
    mem_after = process.memory_info().rss / 1e6
    return end - start, mem_after - mem_before

for d in densities:
    A = np.random.rand(N, N)
    A[A > d] = 0 
    I = np.eye(N)
    t, mem = measure_time_and_memory(lambda: I @ A)
    results["numpy"].append((t, mem))

for d in densities:
    A = sp.random(N, N, density=d, format="csr")
    I = sp.identity(N, format="csr")
    t, mem = measure_time_and_memory(lambda: I @ A)
    results["scipy"].append((t, mem))

plt.figure(figsize=(8, 5))
plt.plot(densities, [r[0] for r in results["numpy"]], label="NumPy Dense")
plt.plot(densities, [r[0] for r in results["scipy"]], label="SciPy Sparse")
plt.xlabel("Matrix Density")
plt.ylabel("Time (s)")
plt.title("Multiplication Time vs Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("eval/sparse_vs_dense_time.png")

plt.figure(figsize=(8, 5))
plt.plot(densities, [r[1] for r in results["numpy"]], label="NumPy Dense")
plt.plot(densities, [r[1] for r in results["scipy"]], label="SciPy Sparse")
plt.xlabel("Matrix Density")
plt.ylabel("Memory Change (MB)")
plt.title("Memory Usage vs Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("eval/sparse_vs_dense_memory.png")

print("Evaluation completed. Results saved in eval/ directory.")