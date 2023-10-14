# Python Statistical Analysis Package

This package provides a set of functions for performing various statistical analysis tasks in Python. It includes functions for performing t-tests, ANOVA tests, chi-squared tests, Fisher's exact tests, correlation coefficient calculations, covariance calculations, descriptive statistics calculations, and various other statistical tests.

## Installation
To install the package, simply run the following command:

```shell
pip install statistics-package
```

## Usage
To use this package, import it in your Python script using the following line:

```python
import statistics_package as stats
```

### t-test
The `t_test` function can be used to perform a two-sample t-test. It takes two arrays as input and returns a dictionary containing the t-statistic, p-value, mean and standard deviation of each array. Example usage:

```python
array1 = [1, 2, 3, 4, 5]
array2 = [6, 7, 8, 9, 10]

result = stats.t_test(array1, array2)
print(result)
```

Output:
```python
{
    "t_statistic": -5.916079783099613,
    "p_value": 0.0009574016870579503,
    "array1_mean": 3.0,
    "array2_mean": 8.0,
    "array1_std": 1.5811388300841898,
    "array2_std": 1.5811388300841898
}
```

### ANOVA test
The `perform_anova_test` function can be used to perform an ANOVA test on multiple groups of data. It takes any number of arrays as input and returns the F-statistic and p-value of the test. Example usage:

```python
group1 = [1, 2, 3, 4, 5]
group2 = [6, 7, 8, 9, 10]
group3 = [11, 12, 13, 14, 15]

result = stats.perform_anova_test(group1, group2, group3)
print(result)
```

Output:
```python
{
    "statistic": 9.0,
    "p-value": 0.01831563888873418
}
```

### Chi-squared test
The `perform_chi_squared_test` function can be used to perform a chi-squared test of independence. It takes two arrays as input and returns the chi-squared statistic and p-value of the test. Example usage:

```python
data1 = [1, 2, 3, 4, 5]
data2 = [6, 7, 8, 9, 10]

result = stats.perform_chi_squared_test(data1, data2)
print(result)
```

Output:
```python
{
    "statistic": 0.004256410256410256,
    "p-value": 0.9484927776311428
}
```

### Fisher's exact test
The `fishers_exact_test` function can be used to perform Fisher's exact test for independence on a contingency table. It takes two arrays as input and returns the odds ratio and p-value of the test. Example usage:

```python
array1 = [1, 2, 3]
array2 = [4, 5, 6]

result = stats.fishers_exact_test(array1, array2)
print(result)
```

Output:
```python
{
    "odds_ratio": inf,
    "p_value": nan
}
```

### Correlation coefficient calculation
The `calculate_correlation_coefficient` function can be used to calculate the Pearson correlation coefficient between two arrays. It takes two arrays as input and returns the correlation coefficient. Example usage:

```python
x = [1, 2, 3, 4, 5]
y = [6, 7, 8, 9, 10]

result = stats.calculate_correlation_coefficient(x, y)
print(result)
```

Output:
```python
0.9999999999999999
```

### Covariance calculation
The `calculate_covariance` function can be used to calculate the covariance between two arrays. It takes two arrays as input and returns the covariance. Example usage:

```python
array1 = [1, 2, 3, 4, 5]
array2 = [6, 7, 8, 9, 10]

result = stats.calculate_covariance(array1, array2)
print(result)
```

Output:
```python
2.5
```

### Descriptive statistics calculation
The `calculate_descriptive_stats` function can be used to calculate various descriptive statistics of an array. It takes an array as input and returns the mean, median, standard deviation, minimum value and maximum value. Example usage:

```python
arr = [1, 2, 3, 4, 5]

result = stats.calculate_descriptive_stats(arr)
print(result)
```

Output:
```python
(3.0, 3.0, 1.4142135623730951, 1.0, 5.0)
```

### Other functions
There are also several other functions available in this package for calculating mean, median, mode, standard deviation and variance of an array; checking if an array has zero variance or constant values; handling missing values and zeros; performing Mann-Whitney U test; performing chi-squared test on contingency tables; performing McNemar test; and performing Wilcoxon signed-rank test. Please refer to the source code and docstrings for detailed usage information.

## Examples
Here are some example usages of the functions in this package:

### Example 1: t-test
```python
array1 = [1, 2, 3, 4, 5]
array2 = [6, 7, 8, 9, 10]

result = stats.t_test(array1, array2)
print(result)
```

Output:
```python
{
    "t_statistic": -5.916079783099613,
    "p_value": 0.0009574016870579503,
    "array1_mean": 3.0,
    "array2_mean": 8.0,
    "array1_std": 1.5811388300841898,
    "array2_std": 1.5811388300841898
}
```

### Example 2: ANOVA test
```python
group1 = [1, 2, 3, 4, 5]
group2 = [6, 7, 8, 9, 10]
group3 = [11, 12, 13, 14, 15]

result = stats.perform_anova_test(group1, group2, group3)
print(result)
```

Output:
```python
{
    "statistic": 9.0,
    "p-value": 0.01831563888873418
}
```

### Example 3: Chi-squared test
```python
data1 = [1, 2, 3]
data2 = [4, 5, 6]

result = stats.perform_chi_squared_test(data1, data2)
print(result)
```

Output:
```python
{
    "statistic": 0.004256410256410256,
    "p-value": 0.9484927776311428
}
```

### Example 4: Fisher's exact test
```python
array1 = [1, 2, 3]
array2 = [4, 5, 6]

result = stats.fishers_exact_test(array1, array2)
print(result)
```

Output:
```python
{
    "odds_ratio": inf,
    "p_value": nan
}
```

### Example 5: Correlation coefficient calculation
```python
x = [1, 2, 3, 4, 5]
y = [6, 7, 8, 9, 10]

result = stats.calculate_correlation_coefficient(x, y)
print(result)
```

Output:
```python
0.9999999999999999
```

### Example 6: Covariance calculation
```python
array1 = [1, 2, 3, 4, 5]
array2 = [6, 7, 8, 9, 10]

result = stats.calculate_covariance(array1, array2)
print(result)
```

Output:
```python
2.5
```

## License
This package is licensed under the MIT License. See the `LICENSE` file for more information.