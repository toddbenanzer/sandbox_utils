# Package Name

The package is called `Python_Stats_Utils`.

## Overview

This package provides a collection of statistical functions and data handling operations for analyzing and manipulating data using Python. It includes functions for calculating various statistical measures such as mean, median, mode, variance, standard deviation, skewness, and kurtosis. Additionally, it offers functionality for handling missing or infinite data, dropping null or trivial columns, and performing column operations such as adding or calculating the difference, product, or quotient of two columns. The package also includes functions for calculating range, interquartile range, coefficient of variation, and z-scores. Hypothesis testing can be performed using functions for independent t-test, paired t-test, ANOVA, chi-square test, and correlation analysis. Various plotting functions are also provided, including histogram, boxplot, scatter plot, line graph, and bar chart. Additionally, the package includes calculations for sample size and power analysis.

## Usage

To use this package in your Python project or script:

1. Install the required packages:
```shell
$ pip install pandas numpy scipy statsmodels scikit-learn factor_analyzer matplotlib
```

2. Import the necessary modules:
```python
import pandas as pd
import numpy as np
from scipy.stats import skew, ttest_ind, ttest_rel, chi2_contingency
from statsmodels.stats.anova import AnovaRM
from statsmodels.formula.api import ols
from statsmodels.api import Logit, add_constant
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from factor_analyzer import FactorAnalyzer
import statsmodels.api as sm
import matplotlib.pyplot as plt
```

3. Import the desired functions from the package:
```python
from Python_Stats_Utils import calculate_mean, calculate_median,
    calculate_mode,
    calculate_variance,
    calculate_std,
    calculate_skewness,
    calculate_kurtosis,
    handle_missing_data,
    handle_infinite_data,
    check_null_trivial,
    drop_null_or_trivial_columns,
    add_column_sum,
    add_column_difference,
    calculate_product,
    calculate_quotient,
    calculate_range,
    calculate_interquartile_range,
    calculate_coefficient_of_variation,
    calculate_zscore,
    independent_t_test,
    paired_t_test,
    perform_anova,
    chi_square_test,
    correlation_analysis,
    plot_histogram,
    plot_boxplot,
    scatter_plot,
    plot_line_graph,
    plot_bar_chart
```

4. Use the imported functions as needed.

## Examples

### Statistical Functions

```python
# Calculate the mean of a dataframe
mean = calculate_mean(dataframe)

# Calculate the median of a dataframe
median = calculate_median(dataframe)

# Calculate the mode of a dataframe
mode = calculate_mode(dataframe)

# Calculate the variance of a dataframe
variance = calculate_variance(dataframe)

# Calculate the standard deviation of a dataframe
std = calculate_std(dataframe)

# Calculate the skewness of a dataframe
skewness = calculate_skewness(dataframe)

# Calculate the kurtosis of a dataframe
kurtosis = calculate_kurtosis(dataframe)
```

### Data Handling Functions

```python
# Handle missing data in a dataframe
df = handle_missing_data(df)

# Handle infinite data in a dataframe
df = handle_infinite_data(df, replace_value=np.nan)
```

### Column Check and Drop Functions

```python
# Check if a column has null or trivial values
is_null_trivial = check_null_trivial(column)

# Drop columns with only null or trivial values from a dataframe
df = drop_null_or_trivial_columns(dataframe)
```

### Column Operations Functions

```python
# Add the sum of two columns and create a new column
df = add_column_sum(df, column1, column2, new_column)

# Calculate the difference between two columns and create a new column
df = add_column_difference(df, column1, column2, new_column_name)

# Calculate the product of two columns and create a new column
df = calculate_product(df, column1, column2, new_column_name)

# Calculate the quotient of two columns and create a new column
df = calculate_quotient(df, column1, column2, new_column_name)
```

### Range Calculation Functions

```python
# Calculate the range of each column in a dataframe
ranges = calculate_range(dataframe)

# Calculate the interquartile range of each column in a dataframe
iqr_df = calculate_interquartile_range(dataframe)
```

### Coefficient and Z-score Calculation Functions

```python
# Calculate the coefficient of variation for each column in a dataframe
coefficient_of_variation = calculate_coefficient_of_variation(dataframe)

# Calculate the z-scores for a specific column in a dataframe
z_scores = calculate_zscore(df, column_name)
```

### Hypothesis Testing Functions

```python
# Perform an independent t-test between two columns in a dataframe
t_statistic, p_value = independent_t_test(df,column1,column2)

# Perform a paired t-test between two columns in a dataframe
t_statistic, p_value = paired_t_test(df,column1,column2)

# Perform an ANOVA on multiple columns grouped by a categorical variable in a dataframe
anova_result = perform_anova(df,categorical_var ,columns)

# Perform a chi-square test between two columns in a dataframe
chi_square , p_value = chi_square_test(df,column1,column2)

# Perform correlation analysis between two columns in a dataframe using the specified method (default: Pearson)
correlation = correlation_analysis(df,column_ ,column_ ,method='pearson')
```

### Plotting Functions

```python
# Plot a histogram of each numeric column in a dataframe
plot_histogram(df)

# Plot a boxplot of each numeric column in a dataframe
plot_boxplot(df)

# Create a scatter plot between two columns in a dataframe
scatter_plot(df,x_col,y_col)

# Create a line graph for each column in a dataframe
plot_line_graph(data)

# Create a bar chart for each column in a dataframe
plot_bar_chart(data)
```

### Sample Size and Power Calculations

```python
# Calculate the required sample size given the margin of error and confidence level
sample_size = calculate_sample_size(margin_of_error, confidence_level)

# Calculate the z-score corresponding to a given confidence level
z_score = get_z_score(confidence_level)

# Calculate the statistical power given the effect size, sample size, and significance level
power = calculate_power(effect_size, sample_size, significance_level)
```

### One Sample T-Test

```python
# Perform a one-sample t-test on a specific column in a dataframe against a specified population mean
result = one_sample_ttest(data,colum,population_mean)
```

### One-Way ANOVA

```python
# Perform a one-way ANOVA on multiple columns grouped by a categorical variable in a dataframe
results = one_way_anova(data,categorical_var)
```

### Two-Way ANOVA

```python
# Perform a two-way ANOVA on multiple columns grouped by two categorical variables in a dataframe
results = two_way_anova(data,categorical_var1,categorical_var2)
```

Please refer to the function definitions for more details on input parameters and return values.

## Contribution

Contributions to this package are welcome. If you find any issues or have suggestions for new features, please feel free to open an issue or submit a pull request on GitHub.

## License

This package is licensed under the MIT license. See the [LICENSE](LICENSE) file for more information.