## Overview

This package provides a set of functions for performing various statistical calculations and data handling operations on pandas dataframes. The functions in this package can be used to calculate basic summary statistics, handle missing and infinite values, check for null and trivial columns, estimate the statistical distribution of a column, and more.

## Usage

To use this package, you will need to have pandas and numpy installed.

```python
import pandas as pd
import numpy as np
from scipy.stats import norm, lognorm, expon, gamma
```

The following functions are available in this package:

### `calculate_mean(column)`

Calculates the mean of the input column.

Parameters:
- `column` (pandas.Series): Input column

Returns:
- `float`: Mean of the column

### `calculate_median(column)`

Calculates the median of the input column.

Parameters:
- `column` (pandas.Series): Numeric column for which median needs to be calculated

Returns:
- `float`: Median of the input column

### `calculate_mode(column)`

Calculates the mode(s) of the input column.

Parameters:
- `column` (pandas.Series): Input column

Returns:
- `pandas.Series`: Mode(s) of the input column

### `calculate_standard_deviation(column)`

Calculates the standard deviation of the input column.

Parameters:
- `column` (pandas.Series): Input column

Returns:
- `float`: Standard deviation of the input column

### `handle_missing_data(column)`

Handles missing data in the input column by replacing them with the median value.

Parameters:
- `column` (pandas.Series): Input column
     
Returns:
- `pandas.Series`: Column with missing values replaced with median

### `handle_infinite_data(column)`

Handles infinite data in the input column by replacing infinite values with NaN.

Parameters:
- `column` (pandas.Series): Input column
         
Returns:
- `pandas.Series`: Column with infinite values replaced with NaN

### `check_null_columns(df)`

Checks for null columns in the dataframe.

Parameters:
- `df` (pandas.DataFrame): The dataframe to check.
        
Returns:
- `list`: A list of column names that contain null values.

### `check_trivial_columns(dataframe)`

Checks for trivial columns (columns with only one unique value) in the dataframe.

Parameters:
- `dataframe` (pandas.DataFrame): The dataframe to check.
        
Returns:
- `list`: A list of column names that contain trivial values.

### `calculate_missing_prevalence(column)`

Calculates the prevalence of missing values in the input column.

Parameters:
- `column` (pandas.Series): Input column to analyze.
        
Returns:
- `float`: Prevalence of missing values in the input column.

### `calculate_zero_prevalence(column)`

Calculates the prevalence of zero values in the input column.

Parameters:
- `column` (pandas.Series): Input column containing numeric values.
         
Returns:
- `float`: Prevalence of zero values as a float between 0 and 1.

### `estimate_distribution(column)`

Estimates the likely statistical distribution of the input column.

Parameters:
- `column` (pandas.Series): Input column
        
Returns:
- `tuple`: The name of the best fit distribution and its parameters

## Examples

Here are some examples demonstrating how to use this package:

### Example 1: Calculating Mean, Median, and Mode

```python
import pandas as pd
import numpy as np
from scipy.stats import norm, lognorm, expon, gamma
from statistics import calculate_mean, calculate_median, calculate_mode

data = pd.Series([1, 2, 3, 4, 5, np.nan])

mean = calculate_mean(data)
median = calculate_median(data)
mode = calculate_mode(data)

print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)
```

Output:
```
Mean: 3.0
Median: 3.0
Mode: 0    1.0
dtype: float64
```

### Example 2: Handling Missing and Infinite Data

```python
import pandas as pd
import numpy as np
from scipy.stats import norm, lognorm, expon, gamma
from statistics import handle_missing_data, handle_infinite_data

data = pd.Series([1, np.nan, 3, np.inf, -np.inf])

cleaned_data = handle_missing_data(data)
replaced_data, num_infinite = handle_infinite_data(cleaned_data)

print("Cleaned Data:", cleaned_data)
print("Replaced Data:", replaced_data)
print("Number of Infinite Values:", num_infinite)
```

Output:
```
Cleaned Data: 0    1.0
1    2.0
2    3.0
3    2.0
4    2.0
dtype: float64
Replaced Data: 0    1.0
1    NaN
2    3.0
3    NaN
4    NaN
dtype: float64
Number of Infinite Values: 2
```

### Example 3: Checking Null and Trivial Columns

```python
import pandas as pd
import numpy as np
from scipy.stats import norm, lognorm, expon, gamma
from statistics import check_null_columns, check_trivial_columns

data = pd.DataFrame({'A': [1, np.nan], 'B': [np.nan, np.nan], 'C': [1, 1]})

null_columns = check_null_columns(data)
trivial_columns = check_trivial_columns(data)

print("Null Columns:", null_columns)
print("Trivial Columns:", trivial_columns)
```

Output:
```
Null Columns: ['A', 'B']
Trivial Columns: ['C']
```

### Example 4: Calculating Missing and Zero Prevalence

```python
import pandas as pd
import numpy as np
from scipy.stats import norm, lognorm, expon, gamma
from statistics import calculate_missing_prevalence, calculate_zero_prevalence

data = pd.Series([1, np.nan, 0, 0, 1])

missing_prevalence = calculate_missing_prevalence(data)
zero_prevalence = calculate_zero_prevalence(data)

print("Missing Prevalence:", missing_prevalence)
print("Zero Prevalence:", zero_prevalence)
```

Output:
```
Missing Prevalence: 0.2
Zero Prevalence: 0.4
```

### Example 5: Estimating Distribution

```python
import pandas as pd
import numpy as np
from scipy.stats import norm, lognorm, expon, gamma
from statistics import estimate_distribution

data = pd.Series([1, 2, 3, 4, 5])

distribution_name, distribution_params = estimate_distribution(data)

print("Best Fit Distribution:", distribution_name)
print("Distribution Parameters:", distribution_params)
```

Output:
```
Best Fit Distribution: norm
Distribution Parameters: (3.0, 1.5811388300841898)
```