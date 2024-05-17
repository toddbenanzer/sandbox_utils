# Overview

This package provides several functions for calculating descriptive statistics on numeric columns in a pandas DataFrame. The functions included are:

- `calculate_mean`: calculates the mean of a numeric column.
- `calculate_median`: calculates the median of a numeric column.
- `calculate_mode`: calculates the mode of a numeric column.
- `calculate_quartiles`: calculates the quartiles of a numeric column.
- `calculate_range`: calculates the range of a numeric column.
- `calculate_standard_deviation`: calculates the standard deviation of a numeric column.
- `calculate_variance`: calculates the variance of a numeric column.
- `calculate_skewness`: calculates the skewness of a numeric column.

# Usage

To use this package, you will need to have pandas and numpy installed in your environment. You can install them using pip:

```
pip install pandas numpy
```

Once you have installed the required dependencies, you can import the package and use the provided functions. Here is an example:

```python
import pandas as pd
import numpy as np
from descriptive_stats import calculate_mean, calculate_median

# Create a sample dataframe
data = {'A': [1, 2, 3, 4, 5], 'B': [6, 7, 8, 9, 10]}
df = pd.DataFrame(data)

# Calculate the mean and median of column 'A'
mean = calculate_mean(df, 'A')
median = calculate_median(df, 'A')

print("Mean:", mean)
print("Median:", median)
```

# Examples

Here are some examples demonstrating how to use each function:

```python
import pandas as pd
from descriptive_stats import calculate_mean, calculate_median, calculate_mode, calculate_quartiles, calculate_range,
                               calculate_standard_deviation, calculate_variance, calcuate_skewness

# Create a sample dataframe
data = {'A': [1, 2, 3, 4, 5], 'B': [6, 7, 8, 9, 10]}
df = pd.DataFrame(data)

# Calculate the mean of column 'A'
mean = calculate_mean(df, 'A')
print("Mean:", mean)

# Calculate the median of column 'A'
median = calculate_median(df, 'A')
print("Median:", median)

# Calculate the mode of column 'A'
mode = calculate_mode(df, 'A')
print("Mode:", mode)

# Calculate the quartiles of column 'A'
quartiles = calculate_quartiles(df['A'])
print("Quartiles:", quartiles)

# Calculate the range of column 'A'
range_ = calculate_range(df['A'])
print("Range:", range_)

# Calculate the standard deviation of column 'A'
std_deviation = calculate_standard_deviation(df, 'A')
print("Standard Deviation:", std_deviation)

# Calculate the variance of column 'A'
variance = calculate_variance(df, 'A')
print("Variance:", variance)

# Calculate the skewness of column 'A'
skewness = calcuate_skewness(df['A'])
print("Skewness:", skewness)
```

These are just a few examples of how to use the functions in this package. For more information on each function and their parameters, refer to the docstrings provided in the code.