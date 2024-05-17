# Overview

This python script provides a set of functions for analyzing and handling data in pandas DataFrames. It includes functions for calculating the frequency and percentage of each category in a column, as well as across all columns. It also provides functions for handling missing and infinite data, removing null and trivial columns, and calculating various statistical measures such as mode, median, mean, standard deviation, variance, range, quartiles, skewness, and kurtosis.

# Usage

To use this package, you need to have pandas and numpy installed. You can install them using pip:

```
pip install pandas numpy
```

You can then import the necessary functions from the script into your python code:

```python
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

from script_name import calculate_category_frequency, calculate_category_percentage,
                         calculate_all_columns_category_frequency,
                         calculate_all_columns_category_percentage,
                         handle_missing_data,
                         handle_infinite_data,
                         remove_null_columns,
                         remove_trivial_columns,
                         calculate_mode,
                         calculate_median,
                         calculate_mean,
                         calculate_std,
                         calculate_variance,
                         calculate_range,
                         calculate_column_min,
                         Calculate_max,
                         Calculate_quartiles,
                         Calculate_interquartile_range,
                         Calculate_skewness,
                         Calculate_kurtosis
```

# Examples

Here are some examples of how to use the functions provided by this script:

1. Calculating the frequency of each category in a column:

```python
df = pd.DataFrame({'A': ['apple', 'banana', 'apple', 'orange', 'banana']})
frequency = calculate_category_frequency(df, 'A')
print(frequency)
```

Output:
```
apple     2
banana    2
orange    1
Name: A, dtype: int64
```

2. Calculating the percentage of each category in a column:

```python
df = pd.DataFrame({'A': ['apple', 'banana', 'apple', 'orange', 'banana']})
percentage = calculate_category_percentage(df, 'A')
print(percentage)
```

Output:
```
  Category  Percentage
0    apple        40.0
1   banana        40.0
2   orange        20.0
```

3. Calculating the frequency of each category across all columns:

```python
df = pd.DataFrame({'A': ['apple', 'banana', 'apple', 'orange', 'banana'],
                   'B': [1, 2, 3, 4, 5]})
frequency = calculate_all_columns_category_frequency(df)
print(frequency)
```

Output:
```
          A    B
apple   2.0  NaN
banana  2.0  NaN
orange  1.0  NaN
1       NaN  1.0
2       NaN  1.0
3       NaN  1.0
4       NaN  1.0
5       NaN  1.0
```

4. Handling missing data by excluding rows with missing values:

```python
df = pd.DataFrame({'A': [1, np.nan, 3],
                   'B': [4, np.nan, np.nan]})
handled_df = handle_missing_data(df, method='exclude')
print(handled_df)
```

Output:
```
     A    B
0  1.0 4.0
```

5. Handling infinite data by excluding rows with infinite values:

```python
df = pd.DataFrame({'A': [1, np.inf, -np.inf],
                   'B': [np.nan, np.inf, -np.inf]})
handled_df = handle_infinite_data(df, method='exclude')
print(handled_df)
```

Output:
```
   A   B
0  1 NaN
```

6. Removing null columns:

```python
df = pd.DataFrame({'A': [1, 2, np.nan],
                   'B': [3, np.nan, 5]})
removed_null_columns_df = remove_null_columns(df)
print(removed_null_columns_df)
```

Output:
```
     B
0  3.0
1  NaN
2  5.0
```

7. Removing trivial columns:

```python
df = pd.DataFrame({'A': [1, 2, 3],
                   'B': [4, 4, 4]})
removed_trivial_columns_df = remove_trivial_columns(df)
print(removed_trivial_columns_df)
```

Output:
```
   A
0  1
1  2
2  3
```

These are just a few examples of the functionality provided by this script. Refer to the function documentation for more details on each function's usage and parameters.