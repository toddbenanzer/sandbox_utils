from distribution_fitter import DistributionFitter
from distribution_sampler import DistributionSampler
from main_module import main  # Assuming the main function is located in a file named main_module.py
from plot_distribution_fit import plot_distribution_fit
import numpy as np


# Example 1: Fit distribution to a normal dataset
data_normal = np.random.normal(loc=0, scale=1, size=1000)
fitter_normal = DistributionFitter(data_normal)
best_fit_normal = fitter_normal.fit_distribution()
print(f"Best fit for normal data: {best_fit_normal}")

# Example 2: Fit distribution to an exponential dataset
data_expon = np.random.exponential(scale=1.0, size=1000)
fitter_expon = DistributionFitter(data_expon)
best_fit_expon = fitter_expon.fit_distribution()
print(f"Best fit for exponential data: {best_fit_expon}")

# Example 3: Fit distribution to uniform data with missing values
data_uniform_with_nans = np.array([0.1, 0.2, np.nan, 0.4, 0.5, 0.6, np.nan, 0.8, 0.9])
fitter_uniform_nans = DistributionFitter(data_uniform_with_nans)
best_fit_uniform_nans = fitter_uniform_nans.fit_distribution()
print(f"Best fit for uniform data with NaNs: {best_fit_uniform_nans}")

# Example 4: Fit distribution to a dataset with constant values (will raise an error)
try:
    data_constant = np.full(1000, 2.5)
    fitter_constant = DistributionFitter(data_constant)
    best_fit_constant = fitter_constant.fit_distribution()
except ValueError as e:
    print(f"Constant data error: {e}")



# Example 1: Generate samples from a normal distribution
normal_sampler = DistributionSampler("norm", {"loc": 0, "scale": 1})
normal_samples = normal_sampler.generate_samples(100)
print("Normal distribution samples:", normal_samples[:5])

# Example 2: Generate samples from an exponential distribution
expon_sampler = DistributionSampler("expon", {"scale": 1.0})
expon_samples = expon_sampler.generate_samples(50)
print("Exponential distribution samples:", expon_samples[:5])

# Example 3: Generate samples from a uniform distribution with specified range
uniform_sampler = DistributionSampler("uniform", {"loc": 0, "scale": 1})
uniform_samples = uniform_sampler.generate_samples(200)
print("Uniform distribution samples:", uniform_samples[:5])

# Example 4: Attempt to generate samples with an unrecognized distribution (will raise an error)
try:
    unknown_sampler = DistributionSampler("unknown", {})
    unknown_samples = unknown_sampler.generate_samples(10)
except ValueError as e:
    print(f"Error: {e}")



# Example 1: Plot normal distribution fit
data_normal = np.random.normal(loc=0, scale=1, size=1000)
plot_distribution_fit(data_normal, 'norm', {'loc': 0, 'scale': 1})

# Example 2: Plot exponential distribution fit
data_expon = np.random.exponential(scale=1.0, size=1000)
plot_distribution_fit(data_expon, 'expon', {'loc': 0, 'scale': 1})

# Example 3: Plot uniform distribution fit
data_uniform = np.random.uniform(low=0.0, high=1.0, size=1000)
plot_distribution_fit(data_uniform, 'uniform', {'loc': 0, 'scale': 1})

# Example 4: Attempt to plot with an unsupported distribution (should raise an error)
try:
    plot_distribution_fit(data_normal, 'invalid', {'loc': 0, 'scale': 1})
except ValueError as e:
    print(f"Error: {e}")



# Example 1: Analyze and plot a normally distributed dataset
data_normal = np.random.normal(loc=0, scale=1, size=1000)
result_normal = main(data_normal)
print("Normal Distribution Result:", result_normal)

# Example 2: Analyze and plot an exponentially distributed dataset
data_expon = np.random.exponential(scale=1.0, size=1000)
result_expon = main(data_expon)
print("Exponential Distribution Result:", result_expon)

# Example 3: Analyze and plot a uniformly distributed dataset
data_uniform = np.random.uniform(low=0.0, high=1.0, size=1000)
result_uniform = main(data_uniform)
print("Uniform Distribution Result:", result_uniform)

# Example 4: Analyze a dataset with missing values (NaNs)
data_with_nans = np.array([1, 2, np.nan, 4, 5, 6, 7, np.nan, 9])
result_with_nans = main(data_with_nans)
print("Distribution Result with NaNs:", result_with_nans)
