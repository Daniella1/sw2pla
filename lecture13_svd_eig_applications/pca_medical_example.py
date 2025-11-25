"""
Step-by-Step PCA Implementation: Medical Blood Test Example
Following the exact 5 steps from the textbook
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(42)


# ============================================================================
# GENERATE SYNTHETIC MEDICAL DATA
# ============================================================================

n_patients = 100
n_tests = 5

# Create realistic correlations between blood tests
# Tests: Inflammation_A, Inflammation_B, Cholesterol, Glucose, Enzyme
mean = np.array([25, 48, 180, 95, 42])

# Covariance structure: inflammation markers correlated, glucose & cholesterol somewhat correlated
cov = np.array([
    [20,  35,   5,   3,   8],   # Inflammation A
    [35,  80,   8,   5,  15],   # Inflammation B (highly correlated with A)
    [ 5,   8, 400,  50,  10],   # Cholesterol
    [ 3,   5,  50, 100,   5],   # Glucose (somewhat correlated with Cholesterol)
    [ 8,  15,  10,   5,  30]    # Enzyme
])

# Generate data
X = np.random.multivariate_normal(mean, cov, n_patients)

# Create DataFrame for better visualization
feature_names = ['Inflammation_A (mg/L)', 'Inflammation_B (U/L)', 
                 'Cholesterol (mg/dL)', 'Glucose (mg/dL)', 'Enzyme (U/L)']
df = pd.DataFrame(X, columns=feature_names)

print(f"\nGenerated data for {n_patients} patients with {n_tests} blood tests")
print("\nFirst 5 patients:")
print(df.head())
print("\nData statistics:")
print(df.describe())

# ============================================================================
# STEP 1: MEAN-CENTER THE DATA AND COMPUTE COVARIANCE MATRIX
# ============================================================================
print("\n" + "="*80)
print("STEP 1: Mean-center data and compute covariance matrix C = X^T X")
print("="*80)

# Mean-center the data
X_mean = X.mean(axis=0)
X_centered = X - X_mean

print("\nOriginal means:")
for i, name in enumerate(feature_names):
    print(f"  {name}: {X_mean[i]:.2f}")

print("\nAfter centering, means should be ~0:")
for i, name in enumerate(feature_names):
    print(f"  {name}: {X_centered.mean(axis=0)[i]:.6f}")

# Compute covariance matrix
C = X_centered.T @ X_centered

print(f"\nCovariance matrix C shape: {C.shape}")
print("Covariance matrix C:")
print(C)

# ============================================================================
# STEP 2: EIGENDECOMPOSITION OF COVARIANCE MATRIX
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Eigendecomposition - Solve CW = WΛ")
print("="*80)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(C)

print(f"\nFound {len(eigenvalues)} eigenvalues (one per feature)")
print("\nEigenvalues (unsorted):")
for i, val in enumerate(eigenvalues):
    print(f"  λ_{i+1} = {val:.2f}")

print("\nEigenvectors W (unsorted):")
print("Each column is an eigenvector (principal component)")
print(eigenvectors)

# Verify the eigenvalue equation: Cw = λw for first eigenvector
w1 = eigenvectors[:, 0]
Cw1 = C @ w1
lambda1_w1 = eigenvalues[0] * w1
print("\nVerification: Does Cw₁ = λ₁w₁?")
print(f"Cw₁:      {Cw1}")
print(f"λ₁w₁:     {lambda1_w1}")
print(f"Difference: {np.linalg.norm(Cw1 - lambda1_w1):.10f} (should be ~0)")

# ============================================================================
# STEP 3: SORT BY EIGENVALUE MAGNITUDE (DESCENDING)
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Sort eigenvalues (descending) and eigenvectors accordingly")
print("="*80)

# Sort indices by eigenvalue magnitude (largest first)
sorted_indices = np.argsort(eigenvalues)[::-1]

# Sort eigenvalues and eigenvectors
eigenvalues_sorted = eigenvalues[sorted_indices]
eigenvectors_sorted = eigenvectors[:, sorted_indices]

print("\nEigenvalues (sorted, largest to smallest):")
for i, val in enumerate(eigenvalues_sorted):
    print(f"  λ_{i+1} = {val:.2f}")

print("\nSorted eigenvectors W:")
print("Column 1 = PC1 (largest variance)")
print("Column 2 = PC2 (2nd largest variance)")
print("etc.")
print(eigenvectors_sorted)

print("\nInterpretation of PC1 (first eigenvector):")
print("PC1 weights:")
for i, (name, weight) in enumerate(zip(feature_names, eigenvectors_sorted[:, 0])):
    print(f"  {weight:+.3f} × {name}")

# ============================================================================
# STEP 4: COMPUTE COMPONENT SCORES (PROJECT DATA ONTO PCs)
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Compute component scores - Project data onto principal components")
print("="*80)

# Component scores = X_centered @ W
# Each column = scores for one PC
component_scores = X_centered @ eigenvectors_sorted

print(f"\nComponent scores shape: {component_scores.shape}")
print(f"({n_patients} patients × {n_tests} principal components)")

print("\nFirst 5 patients' component scores:")
scores_df = pd.DataFrame(component_scores, 
                         columns=[f'PC{i+1}' for i in range(n_tests)])
print(scores_df.head())

print("\nExample calculation for Patient 1, PC1:")
print(f"Patient 1 centered values: {X_centered[0]}")
print(f"PC1 weights (w₁): {eigenvectors_sorted[:, 0]}")
print(f"PC1 score = weighted sum:")
calculation = ""
for i in range(n_tests):
    calculation += f"({X_centered[0, i]:.2f} × {eigenvectors_sorted[i, 0]:.3f})"
    if i < n_tests - 1:
        calculation += " + "
print(f"  {calculation}")
print(f"  = {component_scores[0, 0]:.2f}")


# ============================================================================
# STEP 5: CONVERT TO PERCENT VARIANCE EXPLAINED
# ============================================================================
print("\n" + "="*80)
print("STEP 5: Convert eigenvalues to percent variance explained")
print("="*80)

# Total variance = sum of all eigenvalues
total_variance = eigenvalues_sorted.sum()

# Percent variance explained by each PC
variance_explained = eigenvalues_sorted / total_variance * 100

# Cumulative variance explained
cumulative_variance = np.cumsum(variance_explained)

print("\nVariance explained by each principal component:")
print(f"{'PC':<6} {'Eigenvalue':<15} {'Variance %':<15} {'Cumulative %':<15}")
print("-" * 60)
for i in range(n_tests):
    print(f"PC{i+1:<4} {eigenvalues_sorted[i]:<15.2f} {variance_explained[i]:<15.2f} {cumulative_variance[i]:<15.2f}")

print(f"\nTotal variance (sum of eigenvalues): {total_variance:.2f}")

print("\nInterpretation:")
print(f"  • PC1 captures {variance_explained[0]:.1f}% of all variation")
print(f"  • PC1 + PC2 capture {cumulative_variance[1]:.1f}% of all variation")
print(f"  • PC1 + PC2 + PC3 capture {cumulative_variance[2]:.1f}% of all variation")

# How many PCs to keep 95% of variance?
n_components_95 = np.argmax(cumulative_variance >= 95) + 1
print(f"\nTo retain 95% of variance, keep {n_components_95} principal components")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("Creating visualizations...")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Original data (first 2 features)
ax1 = axes[0, 0]
ax1.scatter(X[:, 0], X[:, 1], alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax1.set_xlabel(feature_names[0], fontsize=10)
ax1.set_ylabel(feature_names[1], fontsize=10)
ax1.set_title('Original Data: First 2 Features\n(Before centering)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Plot 2: Centered data (first 2 features)
ax2 = axes[0, 1]
ax2.scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax2.axhline(y=0, color='r', linewidth=1, linestyle='--', alpha=0.5)
ax2.axvline(x=0, color='r', linewidth=1, linestyle='--', alpha=0.5)
# Draw PC directions
scale = 40
ax2.arrow(0, 0, eigenvectors_sorted[0, 0]*scale, eigenvectors_sorted[1, 0]*scale,
         head_width=1, head_length=1.5, fc='green', ec='green', linewidth=2, 
         alpha=0.7, label='PC1')
ax2.arrow(0, 0, eigenvectors_sorted[0, 1]*scale, eigenvectors_sorted[1, 1]*scale,
         head_width=1, head_length=1.5, fc='orange', ec='orange', linewidth=2, 
         alpha=0.7, label='PC2')
ax2.set_xlabel(feature_names[0] + ' (centered)', fontsize=10)
ax2.set_ylabel(feature_names[1] + ' (centered)', fontsize=10)
ax2.set_title('Centered Data with PC Directions', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

# Plot 3: Scree plot (eigenvalues)
ax3 = axes[0, 2]
ax3.bar(range(1, n_tests + 1), eigenvalues_sorted, alpha=0.7, edgecolor='black', linewidth=1.5)
ax3.set_xlabel('Principal Component', fontsize=10, fontweight='bold')
ax3.set_ylabel('Eigenvalue (Variance)', fontsize=10, fontweight='bold')
ax3.set_title('Scree Plot: Eigenvalues', fontsize=12, fontweight='bold')
ax3.set_xticks(range(1, n_tests + 1))
ax3.grid(True, alpha=0.3, axis='y')
for i, val in enumerate(eigenvalues_sorted):
    ax3.text(i + 1, val, f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 4: Variance explained
ax4 = axes[1, 0]
ax4.bar(range(1, n_tests + 1), variance_explained, alpha=0.7, color='steelblue', 
       edgecolor='black', linewidth=1.5)
ax4.set_xlabel('Principal Component', fontsize=10, fontweight='bold')
ax4.set_ylabel('Variance Explained (%)', fontsize=10, fontweight='bold')
ax4.set_title('Variance Explained by Each PC', fontsize=12, fontweight='bold')
ax4.set_xticks(range(1, n_tests + 1))
ax4.grid(True, alpha=0.3, axis='y')
for i, val in enumerate(variance_explained):
    ax4.text(i + 1, val, f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 5: Cumulative variance
ax5 = axes[1, 1]
ax5.plot(range(1, n_tests + 1), cumulative_variance, 'o-', linewidth=2, markersize=8, 
        color='darkgreen')
ax5.axhline(y=95, color='r', linewidth=2, linestyle='--', alpha=0.7, label='95% threshold')
ax5.axhline(y=90, color='orange', linewidth=2, linestyle='--', alpha=0.7, label='90% threshold')
ax5.set_xlabel('Number of Principal Components', fontsize=10, fontweight='bold')
ax5.set_ylabel('Cumulative Variance Explained (%)', fontsize=10, fontweight='bold')
ax5.set_title('Cumulative Variance Explained', fontsize=12, fontweight='bold')
ax5.set_xticks(range(1, n_tests + 1))
ax5.set_ylim([0, 105])
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)
for i, val in enumerate(cumulative_variance):
    ax5.text(i + 1, val + 2, f'{val:.1f}%', ha='center', fontsize=9)

# Plot 6: Component scores (PC1 vs PC2)
ax6 = axes[1, 2]
ax6.scatter(component_scores[:, 0], component_scores[:, 1], alpha=0.6, s=50, 
           edgecolors='black', linewidth=0.5)
ax6.axhline(y=0, color='orange', linewidth=1, linestyle='--', alpha=0.5)
ax6.axvline(x=0, color='green', linewidth=1, linestyle='--', alpha=0.5)
ax6.set_xlabel(f'PC1 ({variance_explained[0]:.1f}% variance)', fontsize=10, fontweight='bold')
ax6.set_ylabel(f'PC2 ({variance_explained[1]:.1f}% variance)', fontsize=10, fontweight='bold')
ax6.set_title('Projected Data: PC1 vs PC2', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.text(0.5, 0.05, f'Visualizing first 2 PCs (80.6% of variance). To capture >90% we need 3 PCs',
        transform=ax6.transAxes, ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.suptitle('PCA Step-by-Step: Medical Blood Test Analysis', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('pca_step_by_step.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved!")

plt.show()

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"""
✓ Step 1: Centered data and computed covariance matrix C ({n_tests}×{n_tests})
✓ Step 2: Found {n_tests} eigenvalues and eigenvectors (CW = WΛ)
✓ Step 3: Sorted by eigenvalue magnitude (largest = most variance)
✓ Step 4: Computed component scores (projected {n_patients} patients onto PCs)
✓ Step 5: Converted to variance explained (PC1 = {variance_explained[0]:.1f}%)

Key Result:
  • {n_components_95} principal components capture {cumulative_variance[n_components_95-1]:.1f}% of variance
  • Reduced from {n_tests} features to {n_components_95} features!
  • Lost only {100 - cumulative_variance[n_components_95-1]:.1f}% of information
""")