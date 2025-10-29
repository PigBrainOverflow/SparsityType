import numpy as np
import scipy.sparse as sp
import time

N = 10000

I = np.eye(N)
A = np.random.rand(N, N)

start_time = time.time()
C = I @ A
end_time = time.time()
print(f"Time taken for multiplication: {end_time - start_time} seconds")
print(f"Result shape: {C.shape}")

I = sp.identity(n=N, format="csr")
A = sp.random(N, N, density=1, format="csr")
start_time = time.time()
C = I @ A
end_time = time.time()
print(f"Time taken for sparse multiplication: {end_time - start_time} seconds")
print(f"Result shape: {C.shape}")