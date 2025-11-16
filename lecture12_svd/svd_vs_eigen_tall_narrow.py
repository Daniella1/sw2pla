import numpy as np
import time

X = np.random.randn(10000, 50)

# For PCA: eigendecomposition of covariance
start_time_eig = time.perf_counter_ns()
Cov = X.T @ X                                  # 50×50 (small, good!)
eig_vals, eig_vecs = np.linalg.eig(Cov)        # ✓ Fast
end_time_eig = time.perf_counter_ns()

# SVD directly
start_time_svd = time.perf_counter_ns()
U, s, Vt = np.linalg.svd(X, full_matrices=False)  # ✓ Also fast
end_time_svd = time.perf_counter_ns()

print(f"Time for EIG with non-square matrix: {(end_time_eig - start_time_eig) / 1_000_000} ms")
print(f"Time for SVD with non-square matrix: {(end_time_svd - start_time_svd) / 1_000_000} ms")