from distribution_fitter import DistributionFitter
from distribution_sampler import DistributionSampler
from plot_distribution_fit import plot_distribution_fit
from scipy import stats
from scipy.stats import norm, expon, uniform
from typing import Dict
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np


class DistributionFitter:
    """
    A class to fit a statistical distribution to the provided data.
    """

    def __init__(self, data: np.ndarray):
        """
        Initializes the DistributionFitter with the provided data.

        Args:
            data (np.ndarray): An array of data to fit the distribution to.
        """
        self.data = data
        self._validate_input()
        self._handle_missing_values()
        self._check_zero_variance()

    def _validate_input(self):
        """
        Validates the input data to ensure it is suitable for analysis.
        
        Raises:
            ValueError: If the data is empty or contains non-numeric values.
        """
        if self.data.size == 0:
            raise ValueError("Input data array is empty.")
        if not np.issubdtype(self.data.dtype, np.number):
            raise ValueError("Input data contains non-numeric values.")

    def _handle_missing_values(self):
        """
        Handles missing values within the data, e.g., by imputing with the mean.
        """
        if np.any(np.isnan(self.data)):
            self.data = np.nan_to_num(self.data, nan=np.nanmean(self.data))

    def _check_zero_variance(self):
        """
        Checks for zero variance or constant values, handling them appropriately.
        
        Raises:
            ValueError: If the data has zero variance.
        """
        if np.var(self.data) == 0:
            raise ValueError("Input data array has zero variance.")

    def fit_distribution(self) -> str:
        """
        Fits a statistical distribution to the data and returns the name of the best fit.

        Returns:
            str: The name of the distribution that best fits the data.
        """
        distributions = ['norm', 'expon', 'uniform']
        return self._determine_best_fit(distributions)

    def _determine_best_fit(self, distributions: List[str]) -> str:
        """
        Determines which distribution best fits the data from the provided list.

        Args:
            distributions (List[str]): A list of distribution names to evaluate.

        Returns:
            str: The name of the best-fitting distribution.
        """
        best_distribution = None
        best_p_value = -np.inf

        for distribution_name in distributions:
            distribution = getattr(stats, distribution_name)
            params = distribution.fit(self.data)
            
            # Perform a goodness-of-fit test
            _, p_value = stats.kstest(self.data, distribution_name, args=params)
            
            if p_value > best_p_value:
                best_p_value = p_value
                best_distribution = distribution_name

        if not best_distribution:
            raise RuntimeError("Failed to determine a best-fitting distribution.")

        return best_distribution



class DistributionSampler:
    """
    A class to generate random samples from a specified statistical distribution.
    """

    def __init__(self, distribution_name: str, params: Dict):
        """
        Initializes the DistributionSampler with the specified distribution and parameters.

        Args:
            distribution_name (str): The name of the distribution for sample generation.
            params (Dict): The parameters for the distribution (e.g., mean and std for normal).
        
        Raises:
            ValueError: If the distribution name is not recognized by scipy.stats.
        """
        self.distribution_name = distribution_name
        self.params = params

        # Validate that the distribution exists in scipy.stats
        if not hasattr(stats, self.distribution_name):
            raise ValueError(f"Distribution '{self.distribution_name}' is not recognized.")

    def generate_samples(self, size: int) -> np.ndarray:
        """
        Generates random samples from the specified distribution.

        Args:
            size (int): The number of random samples to generate.

        Returns:
            np.ndarray: An array of generated random samples.

        Raises:
            ValueError: If size is not a positive integer.
            RuntimeError: If sample generation fails due to incorrect parameters.
        """
        if size <= 0:
            raise ValueError("Size must be a positive integer.")

        try:
            distribution = getattr(stats, self.distribution_name)
            samples = distribution.rvs(size=size, **self.params)
        except Exception as e:
            raise RuntimeError(f"Failed to generate samples: {e}")

        return samples



def plot_distribution_fit(data: np.ndarray, distribution_name: str, params: Dict):
    """
    Visualizes the fit of the specified statistical distribution over the given data.

    Args:
        data (np.ndarray): The input data for plotting.
        distribution_name (str): The name of the distribution to fit and plot.
        params (Dict): Parameters for the distribution, such as mean and std for a normal distribution.

    Raises:
        ValueError: If the distribution name is not recognized.
    """
    # Validate distribution name
    if distribution_name not in ['norm', 'expon', 'uniform']:
        raise ValueError(f"Distribution '{distribution_name}' is not supported for plotting.")

    # Plot the data histogram
    plt.hist(data, bins=30, density=True, alpha=0.6, color='g', label='Data Histogram')

    # Plot the distribution using the params
    x = np.linspace(min(data), max(data), 1000)
    if distribution_name == 'norm':
        y = norm.pdf(x, loc=params.get('loc', 0), scale=params.get('scale', 1))
    elif distribution_name == 'expon':
        y = expon.pdf(x, loc=params.get('loc', 0), scale=params.get('scale', 1))
    elif distribution_name == 'uniform':
        y = uniform.pdf(x, loc=params.get('loc', 0), scale=params.get('scale', 1))

    plt.plot(x, y, 'r-', label=f'{distribution_name.capitalize()} Distribution')

    # Add labels and legend
    plt.title(f'Fit of {distribution_name.capitalize()} Distribution')
    plt.xlabel('Data Value')
    plt.ylabel('Frequency')
    plt.legend()

    # Show plot
    plt.show()



def main(data: np.ndarray) -> Dict:
    """
    Main function to fit a distribution, generate samples, and plot the distribution fit.

    Args:
        data (np.ndarray): The input data array for statistical analysis.

    Returns:
        Dict: A dictionary with details on the best fit distribution, its parameters, and generated samples.
    """
    # Step 1: Fit the distribution to the data
    fitter = DistributionFitter(data)
    best_distribution = fitter.fit_distribution()
    
    # Step 2: Get parameters for the best distribution
    distribution_params = {}
    if best_distribution == 'norm':
        distribution_params = {'loc': np.mean(data), 'scale': np.std(data)}
    elif best_distribution == 'expon':
        distribution_params = {'loc': np.min(data), 'scale': np.mean(data)}
    elif best_distribution == 'uniform':
        distribution_params = {'loc': np.min(data), 'scale': np.ptp(data)}

    # Step 3: Generate samples from the best-fit distribution
    sampler = DistributionSampler(best_distribution, distribution_params)
    samples = sampler.generate_samples(size=1000)

    # Step 4: Plot the distribution fit
    plot_distribution_fit(data, best_distribution, distribution_params)

    # Compile and return the results
    return {
        "best_distribution": best_distribution,
        "params": distribution_params,
        "samples": samples,
    }
