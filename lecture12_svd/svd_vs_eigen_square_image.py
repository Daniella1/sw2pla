import numpy as np
import time

A = np.random.randn(1000, 1000) # gives an image-like 2D matrix

# Both work!
start_time_eig = time.perf_counter_ns()
eig_vals, eig_vecs = np.linalg.eig(A)          # ✓ Works
end_time_eig = time.perf_counter_ns()

start_time_svd = time.perf_counter_ns()
U, s, Vt = np.linalg.svd(A)                    # ✓ Works
end_time_svd = time.perf_counter_ns()

print(f"Time for EIG with non-square matrix: {(end_time_eig - start_time_eig) / 1_000_000} ms")
print(f"Time for SVD with non-square matrix: {(end_time_svd - start_time_svd) / 1_000_000} ms")