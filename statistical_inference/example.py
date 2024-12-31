from your_module_name import BooleanTest
from your_module_name import CategoricalTest
from your_module_name import NumericTest
from your_module_name import StatisticalTest
from your_module_name import compute_descriptive_statistics
from your_module_name import impute_missing_values
from your_module_name import validate_input_data
from your_module_name import visualize_results
import numpy as np


# Example 1: Basic Initialization and Zero Variance Check
data1 = np.array([1, 1, 1, 1])
data2 = np.array([2, 2, 2, 3])
test_params = {'missing_values_method': 'remove'}
stat_test = StatisticalTest(data1, data2, test_params)

# Check for zero variance
stat_test.check_zero_variance()

# Example 2: Handling Missing Values by Removal
data1 = np.array([1, np.nan, 3, 4])
data2 = np.array([5, 6, np.nan, 8])
stat_test = StatisticalTest(data1, data2, test_params)

# Handle missing values by removing them
stat_test.handle_missing_values()
print("Cleaned data1:", stat_test.data1)
print("Cleaned data2:", stat_test.data2)

# Example 3: Handling Missing Values by Imputation
test_params = {'missing_values_method': 'impute'}
stat_test = StatisticalTest(data1, data2, test_params)

# Handle missing values by imputing
stat_test.handle_missing_values()
print("Imputed data1:", stat_test.data1)
print("Imputed data2:", stat_test.data2)

# Example 4: Handling Constant Values
data1 = np.array([4, 4, 4, 4])
data2 = np.array([5, 7, 9, 9])
test_params = {}
stat_test = StatisticalTest(data1, data2, test_params)

# Handle constant values
stat_test.handle_constant_values()



# Example 1: Performing a t-test
data1 = np.array([2.5, 3.0, 2.8, 3.6, 3.0])
data2 = np.array([3.1, 3.5, 3.2, 4.0, 3.8])
test_params = {}
numeric_test = NumericTest(data1, data2, test_params)

t_statistic, p_value = numeric_test.perform_t_test()
print("T-Test - t-statistic:", t_statistic, ", p-value:", p_value)

# Example 2: Performing an ANOVA
data1 = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
data2 = np.array([2.0, 2.5, 3.0, 3.5, 4.0])
numeric_test = NumericTest(data1, data2, test_params)

f_statistic, p_value = numeric_test.perform_anova()
print("ANOVA - F-statistic:", f_statistic, ", p-value:", p_value)

# Example 3: T-test with Missing Values
data1 = np.array([1.2, 2.3, np.nan, 3.8, 4.5])
data2 = np.array([2.1, np.nan, 3.3, 3.9, 4.2])
test_params = {'missing_values_method': 'remove'}
numeric_test = NumericTest(data1, data2, test_params)

t_statistic, p_value = numeric_test.perform_t_test()
print("T-Test with Missing Values - t-statistic:", t_statistic, ", p-value:", p_value)

# Example 4: ANOVA with Constant Values
data1 = np.array([2.0, 2.0, 2.0, 2.0])
data2 = np.array([3.0, 3.5, 4.0, 4.5, 5.0])
numeric_test = NumericTest(data1, data2, test_params)

f_statistic, p_value = numeric_test.perform_anova()
print("ANOVA with Constant Values - F-statistic:", f_statistic, ", p-value:", p_value)



# Example 1: Basic Chi-squared Test
data1 = np.array([10, 20, 30])
data2 = np.array([20, 30, 10])
test_params = {}
categorical_test = CategoricalTest(data1, data2, test_params)

chi2_statistic, p_value, degrees_of_freedom, expected_frequencies = categorical_test.perform_chi_squared_test()
print(f"Chi-squared Statistic: {chi2_statistic}, p-value: {p_value}")

# Example 2: Chi-squared Test with Missing Values
data1 = np.array([15, np.nan, 25])
data2 = np.array([35, 25, np.nan])
test_params = {'missing_values_method': 'remove'}
categorical_test = CategoricalTest(data1, data2, test_params)

chi2_statistic, p_value, degrees_of_freedom, expected_frequencies = categorical_test.perform_chi_squared_test()
print(f"Chi-squared Statistic with Missing Values: {chi2_statistic}, p-value: {p_value}")

# Example 3: Chi-squared Test with Constant Values
data1 = np.array([50, 50, 50])
data2 = np.array([10, 20, 10])
categorical_test = CategoricalTest(data1, data2, test_params)

chi2_statistic, p_value, degrees_of_freedom, expected_frequencies = categorical_test.perform_chi_squared_test()
print(f"Chi-squared Statistic with Constant Values: {chi2_statistic}, p-value: {p_value}")



# Example 1: Basic Proportions Test
data1 = np.array([True, True, False, True, False])
data2 = np.array([False, True, True, False, False])
test_params = {}
boolean_test = BooleanTest(data1, data2, test_params)

z_statistic, p_value = boolean_test.perform_proportions_test()
print(f"Z-Statistic: {z_statistic}, p-value: {p_value}")

# Example 2: Proportions Test with Missing Values
data1 = np.array([True, False, np.nan, True, False])
data2 = np.array([True, np.nan, True, False, True])
test_params = {'missing_values_method': 'remove'}
boolean_test = BooleanTest(data1, data2, test_params)

z_statistic, p_value = boolean_test.perform_proportions_test()
print(f"Proportions Test with Missing Values - Z-Statistic: {z_statistic}, p-value: {p_value}")

# Example 3: Proportions Test with Constant Values
data1 = np.array([True, True, True, True, True])
data2 = np.array([False, False, False, False, False])
boolean_test = BooleanTest(data1, data2, test_params)

z_statistic, p_value = boolean_test.perform_proportions_test()
print(f"Proportions Test with Constant Values - Z-Statistic: {z_statistic}, p-value: {p_value}")



# Example 1: Basic usage with a simple dataset
data = np.array([1, 2, 3, 4, 5, 5, 6, 7, 8, 9])
statistics = compute_descriptive_statistics(data)
print("Descriptive Statistics:", statistics)

# Example 2: Handling NaN and infinite values
data_with_nan_inf = np.array([1, 2, np.nan, 4, np.inf, 6, -np.inf, 8, 9])
statistics_with_nan_inf = compute_descriptive_statistics(data_with_nan_inf)
print("Statistics with NaN and Inf:", statistics_with_nan_inf)

# Example 3: Single value dataset
single_value_data = np.array([10])
statistics_single_value = compute_descriptive_statistics(single_value_data)
print("Statistics for Single Value:", statistics_single_value)

# Example 4: Dataset with all identical values
identical_values_data = np.array([5, 5, 5, 5, 5])
identical_values_statistics = compute_descriptive_statistics(identical_values_data)
print("Statistics for Identical Values:", identical_values_statistics)



# Example 1: Visualizing a Histogram for Data Distribution
data_results = {
    'data': np.random.normal(loc=0, scale=1, size=100)
}
visualize_results(data_results)

# Example 2: Visualizing Statistical Test Metrics using a Bar Chart
stat_results = {
    't_statistic': 2.5,
    'p_value': 0.03
}
visualize_results(stat_results)

# Example 3: Visualizing Data Spread using a Box Plot
box_plot_data = {
    'box_data': [
        np.random.normal(loc=0, scale=1, size=100),
        np.random.normal(loc=1, scale=2, size=100)
    ]
}
visualize_results(box_plot_data)

# Example 4: Visualizing Category Proportions using a Pie Chart
category_props = {
    'category_proportions': {'Category A': 40, 'Category B': 35, 'Category C': 25}
}
visualize_results(category_props)



# Example 1: Valid datasets
data1 = np.array([1.0, 2.1, 3.5, 4.7])
data2 = np.array([4.5, 3.2, 2.1, 1.0])
if validate_input_data(data1, data2):
    print("Datasets are valid for analysis.")

# Example 2: One dataset is empty
try:
    empty_data = np.array([])
    validate_input_data(empty_data, data2)
except ValueError as e:
    print("Validation Error:", e)

# Example 3: Datasets of different lengths
try:
    short_data = np.array([1, 2, 3])
    validate_input_data(short_data, data2)
except ValueError as e:
    print("Validation Error:", e)

# Example 4: One dataset with too many missing values
try:
    data_with_nans = np.array([np.nan, np.nan, 1, 2])
    validate_input_data(data_with_nans, data2)
except ValueError as e:
    print("Validation Error:", e)

# Example 5: Lists as input
data_list1 = [1, 2, 3, 4]
data_list2 = [4, 3, 2, 1]
if validate_input_data(data_list1, data_list2):
    print("List datasets are valid for analysis.")



# Example 1: Mean Imputation
data_with_nans = np.array([1.0, np.nan, 2.5, np.nan, 3.0])
imputed_data_mean = impute_missing_values(data_with_nans, method='mean')
print("Mean Imputed Data:", imputed_data_mean)

# Example 2: Median Imputation
imputed_data_median = impute_missing_values(data_with_nans, method='median')
print("Median Imputed Data:", imputed_data_median)

# Example 3: Mode Imputation
data_with_mode_nans = np.array([1, np.nan, 2, 2, np.nan])
imputed_data_mode = impute_missing_values(data_with_mode_nans, method='mode')
print("Mode Imputed Data:", imputed_data_mode)

# Example 4: Zero Imputation
imputed_data_zero = impute_missing_values(data_with_nans, method='zero')
print("Zero Imputed Data:", imputed_data_zero)
