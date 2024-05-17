# Functionality Documentation

## Overview
This package provides a set of functions for calculating various descriptive statistics and performing statistical tests on numeric and categorical data.

## Usage
To use this package, first import the necessary modules:
```python
import numpy as np
from scipy import stats
```

### calculate_mean(numeric_array)
Calculate the mean of a numeric array.
```python
mean = calculate_mean(numeric_array)
```

### calculate_median(data)
Calculate the median of a numeric array.
```python
median = calculate_median(data)
```

### calculate_mode(categorical_array)
Calculate the mode of a categorical array.
```python
mode = calculate_mode(categorical_array)
```

### calculate_variance(data)
Calculate the variance of a numeric array.
```python
variance = calculate_variance(data)
```

### calculate_standard_deviation(data)
Calculate the standard deviation of a numeric array.
```python
std_dev = calculate_standard_deviation(data)
```

### calculate_range(data)
Calculate the range of a numeric array.
```python
range = calculate_range(data)
```

### calculate_skewness(data)
Calculate the skewness of a numeric array.
```python
skewness = calculate_skewness(data)
```

### calculate_kurtosis(data)
Calculate the kurtosis of a numeric array.
```python
kurtosis = calculate_kurtosis(data)
```

### independent_t_test(data1, data2)
Perform an independent t-test on two numeric arrays.
```python
t_statistic, p_value, degrees_of_freedom = independent_t_test(data1, data2)
```

### paired_t_test(data1, data2)
Perform a paired t-test on two numeric arrays.
```python
t_statistic, p_value = paired_t_test(data1, data2)
```

### one_sample_t_test(data, popmean)
Perform a one-sample t-test on a numeric array.
```python
result = one_sample_t_test(data, popmean)
```

### perform_chi_square_test(data1, data2)
Perform a chi-square test on two categorical arrays.
```python
result = perform_chi_square_test(data1, data2)
```

### contingency_table_analysis(array1, array2)
Create a contingency table from two categorical arrays.
```python
contingency_table = contingency_table_analysis(array1, array2)
```

### perform_anova_test(arrays)
Perform an ANOVA test on multiple numeric arrays.
```python
f_value, p_value = perform_anova_test(arrays)
```

### perform_kruskal_wallis_test(*arrays)
Perform a Kruskal-Wallis test on multiple numeric arrays.
```python
statistic, p_value = perform_kruskal_wallis_test(*arrays)
```

### perform_mann_whitney_u_test(data1, data2)
Perform a Mann-Whitney U test on two numeric arrays.
```python
u_statistic, p_value = perform_mann_whitney_u_test(data1, data2)
```

### check_zero_variance(data)
Check if a numeric array has zero variance.
```python
has_zero_variance = check_zero_variance(data)
```

### check_constant_values(array)
Check if a numeric array has constant values.
```python
has_constant_values = check_constant_values(array)
```

### remove_missing_values(array)
Remove missing values from a numeric array.
```python
clean_array = remove_missing_values(array)
```

### replace_zeros(array, operation='division')
Replace zeros in a numeric array for division or logarithm operations.
```python
updated_array = replace_zeros(array, operation='division')
```

### calculate_confidence_interval(data1, data2, alpha=0.05)
Calculate the confidence interval for the mean difference between two numeric arrays.
```python
confidence_interval = calculate_confidence_interval(data1, data2, alpha=0.05)
```

### calculate_effect_size(data1, data2)
Calculate the effect size for t-tests between two numeric arrays.
```python
effect_size = calculate_effect_size(data1, data2)
```

### calculate_correlation(array1, array2)
Calculate the correlation coefficient between two numeric arrays.
```python
correlation = calculate_correlation(array1, array2)
```

### generate_random_samples(distribution, size)
Generate random samples from a given distribution.
```python
samples = generate_random_samples('uniform', size=100)
```

### plot_distribution(data, bins=None, kde=False)
Plot the distribution of a numeric array using histograms or kernel density estimation (KDE).
```python
plot_distribution(data, bins=10, kde=True)
```

### plot_boxplots(arrays, labels)
Plot boxplots for multiple arrays and compare their distributions visually.
```python
plot_boxplots(arrays, labels=['Array 1', 'Array 2'])
```

### plot_bar_charts(data1, data2)
Plot bar charts for two sets of categorical data and compare their frequencies visually.
```python
plot_bar_charts(data1, data2)
```

### power_analysis(test, data, alpha=0.05, power=0.8)
Perform power analysis to determine the sample size needed for a statistical test.
```python
sample_size = power_analysis('t-Test', data, alpha=0.05, power=0.8)
```

### fit_distribution(data)
Fit a probability distribution (e.g., normal distribution) to the parameters of input data.
```python
parameters = fit_distribution(data)
```

### calculate_p_value(test_statistics, degrees_of_freedom)
Calculate the p-value from test statistics and degrees of freedom.
```python
p_value = calculate_p_value(test_statistics, degrees_of_freedom)
```

### calculate_effect_size(data1, data2)
Calculate the effect size for two sets of data using Cohen's formula.
```python
effect_size = calculate_effect_size(data1, data2)
```

### bayesian_hypothesis_testing(target_variables, first_set_of_data, second_set_of_data)
Perform Bayesian hypothesis testing on two sets of data.
```python
p_value = bayesian_hypothesis_testing(target_variables, first_set_of_data, second_set_of_data)
```

## Examples

### Example 1: Calculate the mean of a numeric array
```python
import numpy as np

numeric_array = np.array([1, 2, 3, 4, 5])
mean = calculate_mean(numeric_array)
print(mean) # Output: 3.0
```

### Example 2: Perform an independent t-test on two numeric arrays
```python
import numpy as np

data1 = np.array([1, 2, 3, 4, 5])
data2 = np.array([6, 7, 8, 9, 10])
t_statistic, p_value, degrees_of_freedom = independent_t_test(data1, data2)
print(t_statistic) # Output: -5.916079783099617
print(p_value) # Output: 0.0019986825497599387
print(degrees_of_freedom) # Output: 8
```

### Example 3: Create a contingency table from two categorical arrays
```python
import numpy as np

array1 = np.array(['A', 'B', 'A', 'B', 'A'])
array2 = np.array(['X', 'Y', 'X', 'Y', 'X'])
contingency_table = contingency_table_analysis(array1, array2)
print(contingency_table)
```

### Example 4: Perform power analysis for a t-test
```python
import numpy as np

data = np.array([1, 2, 3, 4, 5])
sample_size = power_analysis('t-Test', data, alpha=0.05, power=0.8)
print(sample_size) # Output: 14
```

## Conclusion
This package provides a comprehensive set of functions for analyzing and performing statistical tests on numeric and categorical data. It is designed to be easy to use and provides accurate results for a wide range of statistical analyses.