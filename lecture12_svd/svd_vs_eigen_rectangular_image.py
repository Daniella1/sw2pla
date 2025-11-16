import numpy as np
import time

A = np.random.randn(1920, 1080) # gives an image-like 2D matrix

# Eigendecomposition: Doesn't work directly!
#eig_vals, eig_vecs = np.linalg.eig(A)          # ✗ ERROR: Last 2 dimensions of the array must be square

# Would need to compute A^T A or AA^T first:
start_time_eig = time.perf_counter_ns()
eig_vals, eig_vecs = np.linalg.eig(A.T @ A)    # ✓ Works but slower
end_time_eig = time.perf_counter_ns()

# SVD: Just works
start_time_svd = time.perf_counter_ns()
U, s, Vt = np.linalg.svd(A)                    # ✓ Works directly!
end_time_svd = time.perf_counter_ns()

print(f"Time for EIG with non-square matrix: {(end_time_eig - start_time_eig) / 1_000_000} ms")
print(f"Time for SVD with non-square matrix: {(end_time_svd - start_time_svd) / 1_000_000} ms")