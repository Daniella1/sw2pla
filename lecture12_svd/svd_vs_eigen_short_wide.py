import numpy as np
import time

X = np.random.randn(100, 20000)

# For PCA: eigendecomposition of covariance
start_time_eig = time.perf_counter_ns()
Cov = X.T @ X                                  # 20,000×20,000 (HUGE!)
eig_vals, eig_vecs = np.linalg.eig(Cov)        # ✗ Very slow!
end_time_eig = time.perf_counter_ns()

# SVD directly
start_time_svd = time.perf_counter_ns()
U, s, Vt = np.linalg.svd(X, full_matrices=False)  # ✓ Much faster!
end_time_svd = time.perf_counter_ns()

print(f"Time for EIG with non-square matrix: {(end_time_eig - start_time_eig) / 1_000_000_000} seconds")
print(f"Time for SVD with non-square matrix: {(end_time_svd - start_time_svd) / 1_000_000_000} seconds")