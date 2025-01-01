from calculate_statistics import calculate_statistics
from data_handler import DataHandler
from descriptive_statistics import DescriptiveStatistics
from detect_outliers import detect_outliers
from estimate_distribution import estimate_likely_distribution
from visualize_distribution import visualize_distribution
import numpy as np
import pandas as pd


# Example 1: Basic Central Tendency Calculation
data_series = pd.Series([10, 20, 20, 40, 50, np.nan])
stats = DescriptiveStatistics(data_series)
central_tendency = stats.compute_central_tendency()
print("Central Tendency:", central_tendency)

# Example 2: Calculating Dispersion Measures
dispersion = stats.compute_dispersion()
print("Dispersion Measures:", dispersion)

# Example 3: Detecting Outliers with Z-score Method
outliers_z_score = stats.detect_outliers(method='z-score')
print("Outliers (Z-score method):", outliers_z_score)

# Example 4: Detecting Outliers with IQR Method
outliers_iqr = stats.detect_outliers(method='IQR')
print("Outliers (IQR method):", outliers_iqr)

# Example 5: Estimating Statistical Distribution
estimated_distribution = stats.estimate_distribution()
print("Estimated Distribution:", estimated_distribution)



# Example 1: Handle missing values using mean strategy
data_series = pd.Series([1, 2, np.nan, 4, np.nan])
handler = DataHandler(data_series)
handled_data = handler.handle_missing_values(strategy='mean')
print("Handled Data (Mean Strategy):")
print(handled_data)

# Example 2: Handle missing values using drop strategy
handled_data_drop = handler.handle_missing_values(strategy='drop')
print("\nHandled Data (Drop Strategy):")
print(handled_data_drop)

# Example 3: Handle infinite values using median strategy
data_with_infinity = pd.Series([1, np.inf, 3, -np.inf, 5])
handler_infinity = DataHandler(data_with_infinity)
handled_infinity = handler_infinity.handle_infinite_values(strategy='median')
print("\nHandled Data with Infinity (Median Strategy):")
print(handled_infinity)

# Example 4: Check if data is null or trivial
trivial_data = pd.Series([5, 5, np.nan, 5])
handler_trivial = DataHandler(trivial_data)
is_trivial = handler_trivial.check_null_trivial()
print("\nIs the data trivial:", is_trivial)



# Example 1: Calculate statistics for non-trivial data
data_series = pd.Series([10, 20, 30, 40, 50, 60])
statistics = calculate_statistics(data_series)
print("Statistics for Non-Trivial Data:")
print(statistics)

# Example 2: Calculate statistics when data has missing and infinite values
data_with_missing_infinite = pd.Series([5, 15, np.nan, 25, np.inf, -np.inf])
statistics_with_missing = calculate_statistics(data_with_missing_infinite)
print("\nStatistics with Missing and Infinite Values:")
print(statistics_with_missing)

# Example 3: Calculate statistics for trivial data
trivial_data = pd.Series([7, 7, 7, 7])
trivial_statistics = calculate_statistics(trivial_data)
print("\nStatistics for Trivial Data:")
print(trivial_statistics)

# Example 4: Calculate statistics for empty data
empty_data = pd.Series([])
empty_statistics = calculate_statistics(empty_data)
print("\nStatistics for Empty Data:")
print(empty_statistics)



# Example 1: Visualize distribution of a small numeric data set
data_series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
visualize_distribution(data_series)

# Example 2: Visualize distribution including negative values and zeros
data_with_negatives = pd.Series([-5, 0, 5, 10, 15, 20])
visualize_distribution(data_with_negatives)

# Example 3: Visualize distribution for normally distributed random numbers
data_random = pd.Series(pd.np.random.normal(loc=0, scale=1, size=1000))
visualize_distribution(data_random)

# Example 4: Edge case with empty data
empty_data = pd.Series([])
visualize_distribution(empty_data)

# Example 5: Edge case with non-numeric data
non_numeric_data = pd.Series(['a', 'b', 'c'])
visualize_distribution(non_numeric_data)



# Example 1: Detecting outliers in a small dataset with no outliers
data_series = pd.Series([10, 12, 14, 16, 18])
outliers = detect_outliers(data_series)
print("Outliers:", outliers)

# Example 2: Detecting outliers in a dataset with a clear outlier
data_with_outlier = pd.Series([10, 12, 14, 16, 100])
outliers = detect_outliers(data_with_outlier)
print("Outliers:", outliers)

# Example 3: Attempt to detect outliers in an empty dataset
empty_data = pd.Series([])
outliers = detect_outliers(empty_data)
print("Outliers:", outliers)

# Example 4: Handling non-numeric data
non_numeric_data = pd.Series(['apple', 'banana', 'cherry'])
outliers = detect_outliers(non_numeric_data)
print("Outliers:", outliers)



# Example 1: Estimating distribution for normal data
normal_data = pd.Series(np.random.normal(loc=0, scale=1, size=1000))
print("Estimated Distribution for Normal Data:", estimate_likely_distribution(normal_data))

# Example 2: Estimating distribution for uniform data
uniform_data = pd.Series(np.random.uniform(low=0, high=10, size=1000))
print("Estimated Distribution for Uniform Data:", estimate_likely_distribution(uniform_data))

# Example 3: Estimating distribution for exponential data
exponential_data = pd.Series(np.random.exponential(scale=1, size=1000))
print("Estimated Distribution for Exponential Data:", estimate_likely_distribution(exponential_data))

# Example 4: Handling empty data
empty_data = pd.Series([])
print("Estimated Distribution for Empty Data:", estimate_likely_distribution(empty_data))

# Example 5: Handling non-numeric data
non_numeric_data = pd.Series(['apple', 'banana', 'cherry'])
print("Estimated Distribution for Non-Numeric Data:", estimate_likely_distribution(non_numeric_data))
