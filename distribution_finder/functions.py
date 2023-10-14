umpy as np
from scipy.stats import norm, expon, gamma

def check_zero_variance(data):
    """
    Check for zero variance in the input data.

    Parameters:
        data (np.ndarray): Input data as a numpy array.

    Returns:
        bool: True if the data has zero variance, False otherwise.
    """
    return np.var(data) == 0

def check_constant_values(data):
    """
    Check for constant values in the input data.

    Parameters:
        data (numpy.ndarray): Input data array.

    Returns:
        bool: True if constant values are found, False otherwise.
    """
    unique_values = np.unique(data)
    return len(unique_values) == 1

def handle_missing_values(data):
    """
    Handle missing values in the input data by replacing them with np.nan.

    Parameters:
        data (numpy.ndarray): Input data array.

    Returns:
        numpy.ndarray: Data array with missing values replaced.
    """
    data = np.where(np.isnan(data), np.nan, data)
    
    return data

def handle_zeroes(data):
    """
    Handle zeroes in the input data by replacing them with a small non-zero value.

    Parameters:
        data (numpy.ndarray): Input data array.

    Returns:
        numpy.ndarray: Data array with zeroes replaced.
    """
    # Copy the input data to avoid modifying the original array
    data = np.copy(data)

    # Find indices of the zero values
    zero_indices = np.where(data == 0)

    # Replace the zero values with a small non-zero value
    data[zero_indices] = np.finfo(data.dtype).eps

    return data

def calculate_statistics(data):
    """
    Function to calculate various statistical measures (mean, median, mode, etc.) of the input data.

    Parameters:
        data (numpy array): Input data

    Returns:
        statistics (dict): Dictionary containing the calculated statistics
    """
    # Check for missing values and replace them with NaN
    data = np.where(np.isnan(data), np.nan, data)

    # Calculate mean
    mean = np.nanmean(data)

    # Calculate median
    median = np.nanmedian(data)

    # Calculate mode
    mode = stats.mode(data, nan_policy='omit').mode[0]

    # Calculate standard deviation
    std_dev = np.nanstd(data)

    # Calculate variance
    variance = np.nanvar(data)

    # Create dictionary to store the calculated statistics
    statistics = {
        'mean': mean,
        'median': median,
        'mode': mode,
        'standard_deviation': std_dev,
        'variance': variance
    }

    return statistics

def fit_distribution(data):
    """
    Fit various statistical distributions (normal, exponential, gamma, etc.) to the input data.

    Parameters:
        data (numpy.ndarray): Input data array.

    Returns:
        str: Name of the best-fitting distribution.
    """
    # Check if data contains missing values
    if np.isnan(data).any():
        # Remove missing values from data
        data = data[~np.isnan(data)]
    
    # Check for zero variance and constant values
    if np.var(data) == 0:
        return "Constant"
    
    # Fit normal distribution to data
    normal_params = norm.fit(data)
    
    # Fit exponential distribution to data
    exponential_params = expon.fit(data)
    
    # Fit gamma distribution to data
    gamma_params = gamma.fit(data)
    
    # Calculate log-likelihoods for each distribution
    normal_log_likelihood = np.sum(norm.logpdf(data, *normal_params))
    exponential_log_likelihood = np.sum(expon.logpdf(data, *exponential_params))
    gamma_log_likelihood = np.sum(gamma.logpdf(data, *gamma_params))
    
    # Determine the distribution with maximum log-likelihood
    max_log_likelihood = max(normal_log_likelihood, exponential_log_likelihood, gamma_log_likelihood)
    
    if max_log_likelihood == normal_log_likelihood:
        return "Normal"
    elif max_log_likelihood == exponential_log_likelihood:
        return "Exponential"
    else:
        return "Gamma"

def select_best_distribution(data):
    """
    Select the best-fitting distribution based on goodness-of-fit tests.

    Parameters:
        data (numpy.ndarray): Input data array.

    Returns:
        str: Name of the best-fitting distribution.
    """
    # Remove missing values
    data = data[~np.isnan(data)]

    # Check for zero variance or constant values
    if np.var(data) == 0:
        return "Constant Distribution"
    
    # Check for non-zero values
    if np.all(data == 0):
        return "Zero Distribution"

    # Perform goodness-of-fit tests
    ks_test = kstest(data, 'norm')
    ad_test = anderson(data, 'norm')

    if ks_test.pvalue > ad_test.significance_level:
        return "Normal Distribution"
    else:
        return "Other Distribution"

def generate_random_samples(data):
    """
    Generate random samples from the selected distribution.

    Parameters:
        data (numpy.ndarray): Input data array.

    Returns:
        numpy.ndarray: Random samples from the selected distribution.
        
    Raises:
        ValueError: If the data has zero variance or constant values, or no valid data points.
    """
    # Check if the data has zero variance or constant values
    if np.var(data) == 0 or np.unique(data).size == 1:
        raise ValueError("Data has zero variance or constant values.")

    # Remove missing values from the data
    data = data[~np.isnan(data)]

    # Check if there are still elements in the data
    if len(data) == 0:
        raise ValueError("No valid data points.")

    # Determine the best statistical distribution for the data
    best_distribution = None
    best_params = None
    best_sse = np.inf

    # Fit normal distribution and calculate the sum of squared errors (SSE)
    params_normal = norm.fit(data)
    sse_normal = np.sum((data - norm.pdf(data, *params_normal)) ** 2)

    if sse_normal < best_sse:
        best_distribution = "normal"
        best_params = params_normal
        best_sse = sse_normal

    # Fit exponential distribution and calculate SSE
    params_exponential = expon.fit(data)
    sse_exponential = np.sum((data - expon.pdf(data, *params_exponential)) ** 2)

    if sse_exponential < best_sse:
        best_distribution = "exponential"
        best_params = params_exponential
        best_sse = sse_exponential

    # Fit gamma distribution and calculate SSE
    params_gamma = gamma.fit(data)
    sse_gamma = np.sum((data - gamma.pdf(data, *params_gamma)) ** 2)

    if sse_gamma < best_sse:
        best_distribution = "gamma"
        best_params = params_gamma

    # Generate random samples from the selected distribution
    if best_distribution == "normal":
        samples = norm.rvs(*best_params, size=len(data))
    elif best_distribution == "exponential":
        samples = expon.rvs(*best_params, size=len(data))
    elif best_distribution == "gamma":
        samples = gamma.rvs(*best_params, size=len(data))

    return samples

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, gamma

def plot_histogram(data):
    """
    Plot a histogram of the input data.

    Parameters:
        data (numpy.ndarray): Input data array.
    """
    # Plot histogram of data
    plt.hist(data, bins='auto', alpha=0.7, density=True)
    plt.xlabel('Data')
    plt.ylabel('Density')
    plt.title('Histogram of Input Data')
    plt.show()

def plot_density(data, distribution):
    """
    Plot a density plot of the fitted distribution.

    Parameters:
        data (numpy.ndarray): Input data array.
        distribution (scipy.stats.rv_continuous): Fitted distribution object.
    """
    # Plot density plot of fitted distribution
    x = np.linspace(np.min(data), np.max(data), 100)
    y = distribution.pdf(x)
    
    plt.plot(x, y)
    plt.xlabel('Data')
    plt.ylabel('Density')
    plt.title('Density Plot of Fitted Distribution')
    plt.show()

# Example usage:
data = np.random.randn(1000)  # Input data
distribution = fit_distribution(data)  # Fit distribution
if distribution is not None:
    plot_histogram(data)  # Plot histogram of input data
    plot_density(data, distribution)  # Plot density plot of fitted distributio