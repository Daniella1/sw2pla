"""
LDA Visualization with Marginal Distributions
Medical Blood Test Example: Show how LDA finds separating direction
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

plt.rcParams.update({'font.size': 11})

# Set random seed
np.random.seed(42)

# ============================================================================
# GENERATE SYNTHETIC MEDICAL DATA WITH TWO CLASSES
# ============================================================================

n_samples_per_class = 100

# Class 0: Healthy patients
mean_healthy = [25, 48]
cov_healthy = [[18, 12],   # Slightly reduced variance
               [12, 25]]
X_healthy = np.random.multivariate_normal(mean_healthy, cov_healthy, n_samples_per_class)
y_healthy = np.zeros(n_samples_per_class)

# Class 1: Sick patients (shifted mean further, slightly different covariance)
mean_sick = [34, 60]   # Increased separation (was [32, 58])
cov_sick = [[20, 10],  # Slightly reduced variance
            [10, 28]]
X_sick = np.random.multivariate_normal(mean_sick, cov_sick, n_samples_per_class)
y_sick = np.ones(n_samples_per_class)

# Combine data
X = np.vstack([X_healthy, X_sick])
y = np.hstack([y_healthy, y_sick])

# Center the data for visualization
X_mean = X.mean(axis=0)
X_centered = X - X_mean

# Separate centered data by class
X_centered_healthy = X_centered[y == 0]
X_centered_sick = X_centered[y == 1]

# ============================================================================
# COMPUTE LDA
# ============================================================================

lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(X_centered, y)

# Get LDA direction (the linear discriminant)
lda_direction = lda.coef_[0]
lda_direction = lda_direction / np.linalg.norm(lda_direction)  # normalize

# Project data onto LDA direction
projections_lda = X_centered @ lda_direction

print("="*70)
print("LDA Analysis: Medical Blood Test Data")
print("="*70)
print(f"\nLDA Direction (normalized): {lda_direction}")
print(f"  Interpretation: {lda_direction[0]:.3f} × Marker A + {lda_direction[1]:.3f} × Marker B")
print(f"\nClass means (centered):")
print(f"  Healthy: [{X_centered_healthy.mean(axis=0)[0]:.2f}, {X_centered_healthy.mean(axis=0)[1]:.2f}]")
print(f"  Sick:    [{X_centered_sick.mean(axis=0)[0]:.2f}, {X_centered_sick.mean(axis=0)[1]:.2f}]")
print(f"\nSeparation on LDA direction:")
print(f"  Healthy mean: {projections_lda[y==0].mean():.2f}")
print(f"  Sick mean:    {projections_lda[y==1].mean():.2f}")
print(f"  Difference:   {abs(projections_lda[y==1].mean() - projections_lda[y==0].mean()):.2f}")

# ============================================================================
# CREATE VISUALIZATION
# ============================================================================

fig = plt.figure(figsize=(18, 12))

# ========== MAIN PLOT: 2D Data with LDA Direction ==========
ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)

# Plot data points
ax_main.scatter(X_centered_healthy[:, 0], X_centered_healthy[:, 1], 
               c='blue', alpha=0.6, s=50, edgecolors='black', linewidth=0.5, 
               label='Healthy', marker='o')
ax_main.scatter(X_centered_sick[:, 0], X_centered_sick[:, 1], 
               c='red', alpha=0.6, s=50, edgecolors='black', linewidth=0.5, 
               label='Sick', marker='s')

# Plot class means
ax_main.scatter(X_centered_healthy.mean(axis=0)[0], X_centered_healthy.mean(axis=0)[1],
               c='darkblue', s=300, marker='*', edgecolors='yellow', linewidth=2,
               label='Healthy mean', zorder=5)
ax_main.scatter(X_centered_sick.mean(axis=0)[0], X_centered_sick.mean(axis=0)[1],
               c='darkred', s=300, marker='*', edgecolors='yellow', linewidth=2,
               label='Sick mean', zorder=5)

# Draw LDA direction
scale = 20
arrow = FancyArrowPatch((0, 0), 
                       (lda_direction[0]*scale, lda_direction[1]*scale),
                       arrowstyle='->', mutation_scale=30, linewidth=4,
                       color='green', alpha=0.8, zorder=4)
ax_main.add_patch(arrow)
ax_main.text(lda_direction[0]*scale*0.5, lda_direction[1]*scale*0.5 + 8, 
            'LDA Direction', fontsize=13, fontweight='bold', color='green',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# Draw original axes
# ax_main.axhline(y=0, color='gray', linewidth=1.5, linestyle='--', alpha=0.5)
# ax_main.axvline(x=0, color='gray', linewidth=1.5, linestyle='--', alpha=0.5)

ax_main.set_xlabel('Biomarker A: Inflammation (mg/L) [centered]', fontsize=12, fontweight='bold')
ax_main.set_ylabel('Biomarker B: Enzyme Activity (U/L) [centered]', fontsize=12, fontweight='bold')
ax_main.set_title('LDA Finds Direction That Maximally Separates Classes', 
                 fontsize=14, fontweight='bold')
ax_main.legend(loc='upper left', fontsize=11)
ax_main.grid(True, alpha=0.3)
ax_main.set_aspect('equal', adjustable='box')

# Add annotation
# ax_main.text(0.5, 0.02, 'Classes are separable, but NOT on either axis alone!', 
#             transform=ax_main.transAxes, ha='center', fontsize=12,
#             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# ========== TOP RIGHT: Marginal Distribution (Feature 1 / x-axis) ==========
ax_top = plt.subplot2grid((3, 3), (0, 2))

# Histogram along x-axis (Feature 1)
bins = np.linspace(X_centered[:, 0].min(), X_centered[:, 0].max(), 20)
ax_top.hist(X_centered_healthy[:, 0], bins=bins, alpha=0.6, color='blue', 
           edgecolor='black', label='Healthy')
ax_top.hist(X_centered_sick[:, 0], bins=bins, alpha=0.6, color='red', 
           edgecolor='black', label='Sick')
ax_top.axvline(X_centered_healthy[:, 0].mean(), color='darkblue', linewidth=2, 
              linestyle='--', label='Healthy mean')
ax_top.axvline(X_centered_sick[:, 0].mean(), color='darkred', linewidth=2, 
              linestyle='--', label='Sick mean')

ax_top.set_xlabel('Biomarker A (centered)', fontsize=11, fontweight='bold')
ax_top.set_ylabel('Count', fontsize=11, fontweight='bold')
ax_top.set_title('Marginal Distribution:\nFeature 1 (x-axis) ALONE', 
                fontsize=12, fontweight='bold', color='darkred')
ax_top.legend(fontsize=9)
ax_top.grid(True, alpha=0.3, axis='y')
ax_top.text(0.5, 0.95, 'Classes OVERLAP! ✗', 
           transform=ax_top.transAxes, ha='center', va='top', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
           fontweight='bold')

# ========== MIDDLE RIGHT: Marginal Distribution (Feature 2 / y-axis) ==========
ax_right = plt.subplot2grid((3, 3), (1, 2))

# Histogram along y-axis (Feature 2)
bins = np.linspace(X_centered[:, 1].min(), X_centered[:, 1].max(), 20)
ax_right.hist(X_centered_healthy[:, 1], bins=bins, alpha=0.6, color='blue', 
             edgecolor='black', label='Healthy')
ax_right.hist(X_centered_sick[:, 1], bins=bins, alpha=0.6, color='red', 
             edgecolor='black', label='Sick')
ax_right.axvline(X_centered_healthy[:, 1].mean(), color='darkblue', linewidth=2, 
                linestyle='--', label='Healthy mean')
ax_right.axvline(X_centered_sick[:, 1].mean(), color='darkred', linewidth=2, 
                linestyle='--', label='Sick mean')

ax_right.set_xlabel('Biomarker B (centered)', fontsize=11, fontweight='bold')
ax_right.set_ylabel('Count', fontsize=11, fontweight='bold')
ax_right.set_title('Marginal Distribution:\nFeature 2 (y-axis) ALONE', 
                  fontsize=12, fontweight='bold', color='darkred')
ax_right.legend(fontsize=9)
ax_right.grid(True, alpha=0.3, axis='y')
ax_right.text(0.5, 0.95, 'Classes OVERLAP! ✗', 
             transform=ax_right.transAxes, ha='center', va='top', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
             fontweight='bold')

# ========== BOTTOM: Projection onto LDA Direction ==========
ax_bottom = plt.subplot2grid((3, 3), (2, 0), colspan=3)

# Histogram of projections onto LDA direction
bins_lda = np.linspace(projections_lda.min(), projections_lda.max(), 25)
ax_bottom.hist(projections_lda[y == 0], bins=bins_lda, alpha=0.6, color='blue', 
              edgecolor='black', label='Healthy')
ax_bottom.hist(projections_lda[y == 1], bins=bins_lda, alpha=0.6, color='red', 
              edgecolor='black', label='Sick')
ax_bottom.axvline(projections_lda[y == 0].mean(), color='darkblue', linewidth=3, 
                 linestyle='--', label='Healthy mean')
ax_bottom.axvline(projections_lda[y == 1].mean(), color='darkred', linewidth=3, 
                 linestyle='--', label='Sick mean')

# Draw decision boundary (midpoint between means)
decision_boundary = (projections_lda[y == 0].mean() + projections_lda[y == 1].mean()) / 2
ax_bottom.axvline(decision_boundary, color='black', linewidth=3, linestyle='-',
                 label='Decision boundary', alpha=0.8)

ax_bottom.set_xlabel('Projection onto LDA Direction', fontsize=12, fontweight='bold')
ax_bottom.set_ylabel('Count', fontsize=12, fontweight='bold')
ax_bottom.set_title('Distribution on LDA Direction: Classes WELL SEPARATED!', 
                   fontsize=13, fontweight='bold', color='darkgreen')
ax_bottom.legend(fontsize=11, loc='upper right')
ax_bottom.grid(True, alpha=0.3, axis='y')
ax_bottom.text(0.5, 0.95, 'Classes SEPARATED! ✓', 
              transform=ax_bottom.transAxes, ha='center', va='top', fontsize=13,
              bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
              fontweight='bold')

# Add annotation showing separation
separation = abs(projections_lda[y==1].mean() - projections_lda[y==0].mean())
ax_bottom.annotate('', xy=(projections_lda[y==1].mean(), 25), 
                  xytext=(projections_lda[y==0].mean(), 25),
                  arrowprops=dict(arrowstyle='<->', color='green', lw=3))
ax_bottom.text((projections_lda[y==0].mean() + projections_lda[y==1].mean())/2, 27,
              f'Separation: {separation:.1f}', ha='center', fontsize=11,
              fontweight='bold', color='green')

plt.suptitle('LDA: Finding Directions That Maximize Class Separation\nMedical Blood Test Example', 
            fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('lda_marginal_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)
print("\n✗ Looking at Marker A alone: Classes overlap significantly")
print("✗ Looking at Marker B alone: Classes overlap significantly")
print("✓ Looking at LDA direction: Classes are WELL SEPARATED!")
print("\nThis is why LDA is powerful for classification:")
print("  • Finds the optimal linear combination of features")
print("  • Maximizes between-class separation")
print("  • Makes classification much easier!")
print("\n✓ Visualization saved!")