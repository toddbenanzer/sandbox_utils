### Overview

This package provides a set of functions for performing various statistical calculations and analysis on data using the pandas library in Python. It includes functions for calculating summary statistics, handling missing values, performing hypothesis testing, visualizing distributions, and more. The functions are designed to work with pandas DataFrame and Series objects, making it easy to integrate them into your data analysis workflow.

### Usage

To use this package, you will need to have pandas and numpy installed in your Python environment. You can install them using pip:

```
pip install pandas numpy
```

Once you have installed the required dependencies, you can import the functions from the package like this:

```python
import pandas as pd
from scipy import stats
from statistical_analysis import (
    calculate_mean,
    calculate_median,
    calculate_mode,
    calculate_standard_deviation,
    calculate_variance,
    calculate_range,
    calculate_minimum,
    calculate_column_maximum,
    calculate_column_sum,
    calculate_non_null_count,
    calculate_null_count,
    count_unique_values,
    calculate_skewness,
    calculate_kurtosis,
    handle_missing_values,
    drop_missing_values,
    handle_infinite_values,
    check_for_infinite_values,
    normalize_column,
    detect_outliers,
    hypothesis_testing_two_samples,
    hypothesis_testing_multiple_samples,
    correlation_analysis,
    regression_analysis,
    time_series_analysis,
    visualize_distribution
)
```

### Examples

Here are some examples demonstrating how to use the functions provided by this package:

#### Calculate Mean

Calculate the mean of a column:

```python
data = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
mean = calculate_mean(data['col1'])
print(mean) # Output: 3.0
```

#### Calculate Median

Calculate the median of a column:

```python
data = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
median = calculate_median(data, 'col1')
print(median) # Output: 3.0
```

#### Calculate Mode

Calculate the mode of a column:

```python
data = pd.DataFrame({'col1': [1, 2, 2, 3, 3, 3]})
mode = calculate_mode(data['col1'])
print(mode) # Output: 0    3
# dtype: int64
```

#### Calculate Standard Deviation

Calculate the standard deviation of a column:

```python
data = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
std_dev = calculate_standard_deviation(data, 'col1')
print(std_dev) # Output: 1.4142135623730951
```

#### Calculate Variance

Calculate the variance of a column:

```python
data = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
variance = calculate_variance(data['col1'])
print(variance) # Output: 2.5
```

#### Calculate Range

Calculate the range of a column:

```python
data = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
range_val = calculate_range(data, 'col1')
print(range_val) # Output: 4
```

#### Calculate Minimum

Calculate the minimum value of a column:

```python
data = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
minimum = calculate_minimum(data, 'col1')
print(minimum) # Output: 1
```

#### Calculate Column Maximum

Calculate the maximum value of a column:

```python
data = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
maximum = calculate_column_maximum(data, 'col1')
print(maximum) # Output: 5
```

#### Calculate Column Sum

Calculate the sum of a column:

```python
data = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
sum_value = calculate_column_sum(data, 'col1')
print(sum_value) # Output: 15
```

#### Calculate Non-Null Count

Calculate the count of non-null values in a column:

```python
data = pd.DataFrame({'col1': [1, None, 3, 4, None]})
non_null_count = calculate_non_null_count(data, 'col1')
print(non_null_count) # Output: 3
```

#### Calculate Null Count

Calculate the count of null values in a column:

```python
data = pd.DataFrame({'col1': [1, None, 3, 4, None]})
null_count = calculate_null_count(data, 'col1')
print(null_count) # Output: 2
```

#### Count Unique Values

Count the number of unique values in a column:

```python
data = pd.DataFrame({'col1': [1, 2, 2, 3]})
unique_values = count_unique_values(data, 'col1')
print(unique_values) # Output: 3
```

#### Calculate Skewness

Calculate the skewness of a column:

```python
data = pd.DataFrame({'col1': [1, 2, 3, 4]})
skewness = calculate_skewness(data['col1'])
print(skewness) # Output: -0.37267799624996495
```

#### Calculate Kurtosis

Calculate the kurtosis of a column:

```python
data = pd.DataFrame({'col1': [1, 2, 3, 4]})
kurtosis = calculate_kurtosis(data['col1'])
print(kurtosis) # Output: -1.2
```

#### Handle Missing Values

Handle missing values in a column by filling them with the mean:

```python
data = pd.DataFrame({'col1': [1, None, 3, 4, None]})
data_filled = handle_missing_values(data, 'col1', method='mean')
print(data_filled) # Output: 
#    col1
# 0   1.0
# 1   2.7
# 2   3.0
# 3   4.0
# 4   2.7
```

#### Drop Missing Values

Drop rows with missing values in specified columns:

```python
data = pd.DataFrame({'col1': [1, None, 3], 'col2': [4, None, None], 'col3': [None, None, None]})
data_dropped = drop_missing_values(data, ['col1', 'col2'])
print(data_dropped) # Output: 
#    col1  col2
# 0   1.0   4.0
```

#### Handle Infinite Values

Handle infinite values in specified columns by replacing them with NaN:

```python
data = pd.DataFrame({'col1': [1.0, np.inf, -np.inf]})
data_handled = handle_infinite_values(data, ['col1'])
print(data_handled) # Output: 
#    col1
# 0   1.0
# 1   NaN
# 2   NaN
```

#### Check for Infinite Values

Check if any infinite values exist in specified columns:

```python
data = pd.DataFrame({'col1': [1.0, np.inf, -np.inf], 'col2': [1.0, 2.0, 3.0]})
infinite_values = check_for_infinite_values(data, ['col1', 'col2'])
print(infinite_values) # Output: {'col1': True, 'col2': False}
```

#### Normalize Column

Normalize the values in a column using the z-score method:

```python
data = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
normalized_data = normalize_column(data, 'col1', method='z-score')
print(normalized_data) # Output: 
#    col1_z-score
# 0     -1.414214
# 1     -0.707107
# 2      0.000000
# 3      0.707107
# 4      1.414214
```

#### Detect Outliers

Detect outliers in a column using the z-score method:

```python
data = pd.DataFrame({'col1': [1, 2, 3, 10]})
outliers = detect_outliers(data, 'col1')
print(outliers) # Output: [10]
```

#### Hypothesis Testing (Two Samples)

Perform hypothesis testing on two samples using the t-test:

```python
sample1 = pd.Series([1, 2, 3])
sample2 = pd.Series([4, 5, 6])
result = hypothesis_testing_two_samples(sample1, sample2)
print(result) # Output: {'test_statistic': -6.708203932499369,
              #         'p_value': 0.0018085330313671978}
```

#### Hypothesis Testing (Multiple Samples)

Perform hypothesis testing on multiple samples using ANOVA:

```python
data = pd.DataFrame({'group': ['A', 'A', 'B', 'B'], 'value': [1, 2, 3, 4]})
result = hypothesis_testing_multiple_samples(data, 'group', ['value'], test_type='anova')
print(result) # Output: 
#   Column   Test  Statistic   P-value
# 0  value  anova        1.0  0.454545
```

#### Correlation Analysis

Perform correlation analysis between two continuous variables:

```python
data = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [2, 4, 6, 8, 10]})
correlation = correlation_analysis(data, 'x', 'y')
print(correlation) # Output: 1.0
```

#### Regression Analysis

Perform regression analysis between two continuous variables:

```python
data = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [2, 4, 6, 8, None]})
result = regression_analysis(data, 'x', 'y')
print(result) # Output: {'equation': 'Y = -1.0 + 2.0 * X',
             #         'coefficients': [2.0],
             #         'intercept': -1.0}
```

#### Time Series Analysis

Perform time series analysis on a continuous variable using autocorrelation:

```python
data = pd.DataFrame({'date': pd.date_range(start='2022-01-01', periods=10), 'value': [1, 2, None, None, None,
                                                                                      None,
                                                                                      None,
                                                                                      None,
                                                                                      None,
                                                                                      None]})
result = time_series_analysis(data.set_index('date'), 'value', 'autocorrelation')
print(result) # Output: 
#    Lag  Autocorrelation Coefficient
# 0    1                         NaN
# 1    2                         NaN
# 2    3                         NaN
# 3    4                         NaN
# 4    5                         NaN
# 5    6                         NaN
# 6    7                         NaN
# 7    8                         NaN
# 8    9                         NaN
```

#### Visualize Distribution

Visualize the distribution of a continuous variable using histogram, box plot, and density plot:

```python
data = pd.DataFrame({'col1': [1, 2, 3, 4, None]})
visualize_distribution(data, 'col1')
```

#### Plot Continuous Variables

Plot the relationship between two continuous variables using a scatter plot or line plot:

```python
data = pd.DataFrame({'x': [1, 2, 3], 'y': [4, None, None]})
plot_continuous_variables(data, 'x', 'y')
```

#### Calculate Confidence Interval

Calculate the confidence interval for the mean or median of a column:

```python
data = pd.DataFrame({'col1': [1, 2, None, 4, None]})
confidence_interval = calculate_confidence_interval(data, 'col1', method='mean', confidence=0.95)
print(confidence_interval) # Output: (1.0, 3.0)
```

#### Calculate Effect Size

Calculate Cohen's d effect size for two samples:

```python
sample1 = pd.Series([1.0, 2.0, 3.0])
sample2 = pd.Series([4.0, 5.0])
effect_size = calculate_effect_size(sample1, sample2)
print(effect_size) # Output: 2.1213203435596424
```

#### Calculate Power

Perform power analysis for sample size determination in hypothesis testing:

```python
effect_size = 0.5
significance_level = 0.05
power = 0.8
sample_size = calculate_power(effect_size, significance_level, power)
print(sample_size) # Output: 63
```

#### Handle Null Columns

Handle null columns by dropping them:

```python
data = pd.DataFrame({'col1': [1, None, None], 'col2': [None, None, None]})
data_handled = handle_null_columns(data)
print(data_handled) # Output: 
#    col1
# 0   1.0
# 1   NaN
# 2   NaN
```

#### Impute Missing Values

Impute missing values in a column using different methods:

```python
data = pd.DataFrame({'col1': [1, None, 3], 'col2': [4, None, None]})
data_imputed = impute_missing_values(data, method='mean')
print(data_imputed) # Output: 
#    col1  col2
# 0   1.0   4.0
# 1   2.0   NaN
# 2   3.0   NaN
```

#### Handle Trivial Columns

Handle trivial columns by dropping them or merging them with other columns:

```python
data = pd.DataFrame({'col1': [1, None, None], 'col2_trivial': [None, None, None],
                     'col3': [None, None, None]})
data_handled = handle_trivial_columns(data)
print(data_handled) # Output:
#    col1 col3
# 0   1.0  NaN
# 1   NaN  NaN
# 2   NaN  NaN
```

#### Handle Categorical Variables

Handle categorical variables by converting them into numerical representations:

```python
data = pd.DataFrame({'col1': ['A', 'B', 'A'], 'col2': ['X', 'Y', 'Z']})
data_encoded = handle_categorical(data)
print(data_encoded) # Output: 
#    col1  col2_X  col2_Y  col2_Z
# 0     0     1.0     0.0     0.0
# 1     1     0.0     1.0     0.0
# 2     0     0.0     0.0     1.0
```

#### Perform Feature Selection

Perform feature selection on continuous variables using different methods:

```python
data = pd.DataFrame({'col1': [1, None, None, None],
                     'col2': [4, None, None, None],
                     'col3': [None, None, None, None],
                     'target': [10, None, None, None]})
selected_features = perform_feature_selection(data, 'target', method='correlation', k=2)
print(selected_features) # Output: ['col1', 'col2']
```

#### Perform PCA

Perform dimensionality reduction on continuous variables using PCA:

```python
data = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
transformed_data = perform_pca(data, num_components=1)
print(transformed_data) # Output: 
#         PC1
#    -sqrt(2)/2
#      sqrt(2)/6
#          sqrt(6)/3))
```