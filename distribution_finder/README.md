# DistributionFitter Class Documentation

## Overview
The `DistributionFitter` class provides functionality to fit a statistical distribution to a given dataset. It handles data validation, missing values, and zero variance cases, returning the best-fitting distribution from a predefined set.

## Initialization

### `__init__(self, data: np.ndarray)`
Initializes the `DistributionFitter` with the provided data.

#### Parameters
- `data` (np.ndarray): An array of numeric data to fit the distribution to.

#### Raises
- `ValueError`: If the input data is empty or contains non-numeric values.

## Methods

### `_validate_input(self)`
Validates the input data to ensure it is suitable for analysis.

#### Raises
- `ValueError`: If the data is empty or contains non-numeric values.

### `_handle_missing_values(self)`
Handles missing values in the dataset by imputing them with the mean of the non-missing values.

### `_check_zero_variance(self)`
Checks for zero variance or constant values in the data.

#### Raises
- `ValueError`: If the data has zero variance.

### `fit_distribution(self) -> str`
Fits a statistical distribution to the data and returns the name of the best fit.

#### Returns
- `str`: The name of the distribution that best fits the data.

### `_determine_best_fit(self, distributions: List[str]) -> str`
Determines which distribution from the provided list best fits the data.

#### Parameters
- `distributions` (List[str]): A list of statistical distribution names to evaluate (e.g., 'norm', 'expon', 'uniform').

#### Returns
- `str`: The name of the best-fitting distribution.

#### Raises
- `RuntimeError`: If no best-fitting distribution can be determined.


# DistributionSampler Class Documentation

## Overview
The `DistributionSampler` class provides functionality to generate random samples from various statistical distributions. It allows users to specify the distribution type and necessary parameters for sample generation.

## Initialization

### `__init__(self, distribution_name: str, params: Dict)`
Initializes the `DistributionSampler` with the specified distribution name and parameters.

#### Parameters
- `distribution_name` (str): The name of the statistical distribution for sample generation (e.g., 'norm', 'expon').
- `params` (Dict): A dictionary containing the parameters required for the specified distribution (e.g., mean and standard deviation for normal distribution).

#### Raises
- `ValueError`: If the distribution name provided is not recognized by `scipy.stats`.

## Methods

### `generate_samples(self, size: int) -> np.ndarray`
Generates random samples from the specified distribution.

#### Parameters
- `size` (int): The number of random samples to generate.

#### Returns
- `np.ndarray`: An array of generated random samples from the specified distribution.

#### Raises
- `ValueError`: If the size provided is not a positive integer.
- `RuntimeError`: If sample generation fails due to incorrect parameters or other reasons related to the specified distribution.


# plot_distribution_fit Function Documentation

## Overview
The `plot_distribution_fit` function visualizes how well a specified statistical distribution fits a given dataset by plotting the data's histogram and overlaying the probability density function (PDF) of the distribution.

## Parameters

### `data: np.ndarray`
- **Description:** The input data to be plotted, represented as a one-dimensional NumPy array.
- **Example:** `np.random.normal(0, 1, 1000)`

### `distribution_name: str`
- **Description:** The name of the statistical distribution to fit and plot. Supported distributions include:
  - `'norm'`: Normal distribution
  - `'expon'`: Exponential distribution
  - `'uniform'`: Uniform distribution

### `params: Dict`
- **Description:** A dictionary containing the parameters needed for the specified distribution. The necessary parameters vary by distribution:
  - For the normal distribution:
    - `loc`: Mean of the distribution (default is 0)
    - `scale`: Standard deviation of the distribution (default is 1)
  - For the exponential distribution:
    - `loc`: Location parameter (default is 0)
    - `scale`: Scale parameter (default is 1)
  - For the uniform distribution:
    - `loc`: Lower bound of the distribution (default is 0)
    - `scale`: Width of the distribution (default is 1)

## Raises
- **ValueError:** If the specified `distribution_name` is not recognized or supported.

## Usage
This function is typically called after generating or collecting a dataset to visualize how well a theoretical distribution fits this data. The resulting plot shows the histogram of the data alongside the PDF of the specified distribution, providing insights into the fitting accuracy.

## Example


# main Function Documentation

## Overview
The `main` function serves as the primary interface for analyzing a given dataset by fitting a statistical distribution, generating random samples from that distribution, and visualizing the fit via a plot.

## Parameters

### `data: np.ndarray`
- **Description:** A one-dimensional NumPy array containing the dataset for statistical analysis. The data may include numerical values and may have missing values.
- **Example:** `np.random.normal(loc=0, scale=1, size=1000)`

## Returns

### `Dict`
A dictionary containing the results of the analysis:
- **best_distribution**: `str`
  - The name of the statistical distribution that best fits the input data (e.g., 'norm', 'expon', 'uniform').
  
- **params**: `Dict`
  - A dictionary of parameters for the best-fitting distribution:
    - For `norm`: `{'loc': mean, 'scale': standard deviation}`
    - For `expon`: `{'loc': minimum value, 'scale': mean value}`
    - For `uniform`: `{'loc': minimum value, 'scale': range of the data}`
    
- **samples**: `np.ndarray`
  - An array containing random samples generated from the best-fit distribution, with a size of 1000.

## Functionality
1. **Fit Distribution**: Utilizes the `DistributionFitter` class to assess the input data and determine the best-fitting statistical distribution.
2. **Extract Parameters**: Based on the identified distribution, calculates the necessary parameters for that distribution.
3. **Generate Samples**: Creates random samples from the fitted distribution using the `DistributionSampler` class.
4. **Plot Fit**: Visualizes the fit of the statistical distribution to the input data using the `plot_distribution_fit` function.

## Usage
This function is typically called with a dataset to perform a comprehensive distribution analysis, making it useful for data scientists and analysts seeking to understand underlying data patterns.

## Example
