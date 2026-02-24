"""
Statistics Fundamentals - Part 2: Distributions and Correlation
================================================================

This module covers:
- Normal distribution
- Other distributions
- Correlation and covariance
- Probability calculations
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# ============================================================================
# 1. NORMAL DISTRIBUTION
# ============================================================================

print("=" * 70)
print("1. NORMAL DISTRIBUTION")
print("=" * 70)

# Generate normal distribution
mu = 100  # mean
sigma = 15  # standard deviation
data = np.random.normal(mu, sigma, 10000)

print(f"Generated {len(data)} samples from N({mu}, {sigma})")
print(f"Sample mean: {np.mean(data):.2f}")
print(f"Sample std: {np.std(data):.2f}")

# Probability calculations
print(f"\nProbability calculations:")
print(f"P(X < 100) = {stats.norm.cdf(100, mu, sigma):.4f}")
print(f"P(X > 115) = {1 - stats.norm.cdf(115, mu, sigma):.4f}")
print(f"P(85 < X < 115) = {stats.norm.cdf(115, mu, sigma) - stats.norm.cdf(85, mu, sigma):.4f}")

# Z-scores
print(f"\nZ-scores:")
print(f"Z-score for X=115: {(115 - mu) / sigma:.2f}")
print(f"Z-score for X=85: {(85 - mu) / sigma:.2f}")

# ============================================================================
# 2. STANDARD NORMAL DISTRIBUTION
# ============================================================================

print("\n" + "=" * 70)
print("2. STANDARD NORMAL DISTRIBUTION (Z-distribution)")
print("=" * 70)

# Standard normal (mean=0, std=1)
z_scores = np.linspace(-4, 4, 100)
probabilities = stats.norm.pdf(z_scores)

print(f"P(Z < 0) = {stats.norm.cdf(0):.4f}")
print(f"P(Z < 1) = {stats.norm.cdf(1):.4f}")
print(f"P(Z < 2) = {stats.norm.cdf(2):.4f}")
print(f"P(Z < 3) = {stats.norm.cdf(3):.4f}")

print(f"\nEmpirical rule:")
print(f"P(-1 < Z < 1) = {stats.norm.cdf(1) - stats.norm.cdf(-1):.4f} ≈ 68%")
print(f"P(-2 < Z < 2) = {stats.norm.cdf(2) - stats.norm.cdf(-2):.4f} ≈ 95%")
print(f"P(-3 < Z < 3) = {stats.norm.cdf(3) - stats.norm.cdf(-3):.4f} ≈ 99.7%")

# ============================================================================
# 3. OTHER DISTRIBUTIONS
# ============================================================================

print("\n" + "=" * 70)
print("3. OTHER DISTRIBUTIONS")
print("=" * 70)

# Uniform distribution
uniform_data = np.random.uniform(0, 10, 1000)
print(f"Uniform(0, 10):")
print(f"  Mean: {np.mean(uniform_data):.2f}")
print(f"  Std: {np.std(uniform_data):.2f}")

# Exponential distribution
exp_data = np.random.exponential(2, 1000)
print(f"\nExponential(λ=0.5):")
print(f"  Mean: {np.mean(exp_data):.2f}")
print(f"  Std: {np.std(exp_data):.2f}")

# Binomial distribution
binom_data = np.random.binomial(n=10, p=0.5, size=1000)
print(f"\nBinomial(n=10, p=0.5):")
print(f"  Mean: {np.mean(binom_data):.2f}")
print(f"  Std: {np.std(binom_data):.2f}")

# Poisson distribution
poisson_data = np.random.poisson(lam=3, size=1000)
print(f"\nPoisson(λ=3):")
print(f"  Mean: {np.mean(poisson_data):.2f}")
print(f"  Std: {np.std(poisson_data):.2f}")

# ============================================================================
# 4. CORRELATION
# ============================================================================

print("\n" + "=" * 70)
print("4. CORRELATION")
print("=" * 70)

# Perfect positive correlation
x = np.array([1, 2, 3, 4, 5])
y_perfect_pos = 2 * x + 1

# Positive correlation
y_pos = 2 * x + np.random.randn(5) * 0.5

# No correlation
y_no_corr = np.random.randn(5)

# Negative correlation
y_neg = -x + np.random.randn(5) * 0.5

# Calculate correlations
corr_perfect = np.corrcoef(x, y_perfect_pos)[0, 1]
corr_pos = np.corrcoef(x, y_pos)[0, 1]
corr_no = np.corrcoef(x, y_no_corr)[0, 1]
corr_neg = np.corrcoef(x, y_neg)[0, 1]

print(f"Perfect positive correlation: {corr_perfect:.4f}")
print(f"Positive correlation: {corr_pos:.4f}")
print(f"No correlation: {corr_no:.4f}")
print(f"Negative correlation: {corr_neg:.4f}")

# ============================================================================
# 5. COVARIANCE
# ============================================================================

print("\n" + "=" * 70)
print("5. COVARIANCE")
print("=" * 70)

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

cov = np.cov(x, y)
print(f"Covariance matrix:")
print(cov)

print(f"\nCovariance(X, Y): {cov[0, 1]:.4f}")
print(f"Variance(X): {cov[0, 0]:.4f}")
print(f"Variance(Y): {cov[1, 1]:.4f}")

# ============================================================================
# 6. PRACTICAL EXAMPLE: Test Scores Correlation
# ============================================================================

print("\n" + "=" * 70)
print("6. PRACTICAL EXAMPLE: Test Scores Correlation")
print("=" * 70)

# Study hours vs test scores
study_hours = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
test_scores = np.array([50, 55, 60, 65, 70, 75, 80, 85, 90])

correlation = np.corrcoef(study_hours, test_scores)[0, 1]
print(f"Study hours: {study_hours}")
print(f"Test scores: {test_scores}")
print(f"\nCorrelation: {correlation:.4f}")
print(f"Interpretation: Strong positive correlation")

# ============================================================================
# 7. VISUALIZATION
# ============================================================================

print("\n" + "=" * 70)
print("7. VISUALIZATION")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Normal distribution
x = np.linspace(-4, 4, 100)
axes[0, 0].plot(x, stats.norm.pdf(x), linewidth=2)
axes[0, 0].fill_between(x, stats.norm.pdf(x), alpha=0.3)
axes[0, 0].set_title('Normal Distribution')
axes[0, 0].grid(True, alpha=0.3)

# Uniform distribution
x_uniform = np.linspace(0, 10, 100)
axes[0, 1].plot(x_uniform, stats.uniform.pdf(x_uniform, 0, 10), linewidth=2)
axes[0, 1].fill_between(x_uniform, stats.uniform.pdf(x_uniform, 0, 10), alpha=0.3)
axes[0, 1].set_title('Uniform Distribution')
axes[0, 1].grid(True, alpha=0.3)

# Exponential distribution
x_exp = np.linspace(0, 10, 100)
axes[0, 2].plot(x_exp, stats.expon.pdf(x_exp, scale=2), linewidth=2)
axes[0, 2].fill_between(x_exp, stats.expon.pdf(x_exp, scale=2), alpha=0.3)
axes[0, 2].set_title('Exponential Distribution')
axes[0, 2].grid(True, alpha=0.3)

# Positive correlation
x = np.random.randn(100)
y_pos = 2 * x + np.random.randn(100) * 0.5
axes[1, 0].scatter(x, y_pos, alpha=0.5)
axes[1, 0].set_title(f'Positive Correlation (r={np.corrcoef(x, y_pos)[0, 1]:.2f})')
axes[1, 0].grid(True, alpha=0.3)

# No correlation
y_no = np.random.randn(100)
axes[1, 1].scatter(x, y_no, alpha=0.5)
axes[1, 1].set_title(f'No Correlation (r={np.corrcoef(x, y_no)[0, 1]:.2f})')
axes[1, 1].grid(True, alpha=0.3)

# Negative correlation
y_neg = -2 * x + np.random.randn(100) * 0.5
axes[1, 2].scatter(x, y_neg, alpha=0.5)
axes[1, 2].set_title(f'Negative Correlation (r={np.corrcoef(x, y_neg)[0, 1]:.2f})')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# 8. PRACTICAL EXAMPLE: Confidence Intervals
# ============================================================================

print("\n" + "=" * 70)
print("8. PRACTICAL EXAMPLE: Confidence Intervals")
print("=" * 70)

# Sample data
sample = np.random.normal(100, 15, 100)
sample_mean = np.mean(sample)
sample_std = np.std(sample, ddof=1)
n = len(sample)

# 95% confidence interval
confidence = 0.95
alpha = 1 - confidence
t_critical = stats.t.ppf(1 - alpha/2, n - 1)
margin_error = t_critical * (sample_std / np.sqrt(n))

ci_lower = sample_mean - margin_error
ci_upper = sample_mean + margin_error

print(f"Sample mean: {sample_mean:.2f}")
print(f"Sample std: {sample_std:.2f}")
print(f"Sample size: {n}")
print(f"\n95% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}]")
print(f"Margin of error: ±{margin_error:.2f}")
