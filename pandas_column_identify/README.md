**README.md**

# Data Type Analyzer

A Python package for analyzing and handling data types in a pandas DataFrame.

## Overview

The Data Type Analyzer is a Python package designed to help users analyze and handle different data types in a pandas DataFrame. It provides functions for checking the data type of a column, handling missing or infinite data, determining the most likely data type of a column, and handling mixed data.

## Usage

To use the Data Type Analyzer package, follow these steps:

1. Install the package by running `pip install datatype-analyzer`.
2. Import the necessary modules:

```python
import pandas as pd
import numpy as np
from datatype_analyzer import *
```

3. Use the available functions to analyze and handle data types in your DataFrame.

### Available Functions

- `check_column_string(df, column_name)`: Checks if all values in a column are strings.
- `check_column_numeric(column)`: Checks if all values in a column are numeric.
- `check_column_integer(df, column_name)`: Checks if all values in a column are integers.
- `check_column_float(column)`: Checks if all values in a column are floats.
- `check_column_boolean(column)`: Checks if all values in a column are boolean.
- `check_column_date(column)`: Checks if all values in a column are dates.
- `check_column_datetime(column)`: Checks if all values in a column are datetimes.
- `handle_missing_data(column)`: Handles missing data by replacing empty or NaN values with NaN.
- `handle_infinite_data(column)`: Handles infinite data by replacing infinity values with NaN.
- `check_null_or_empty(column)`: Checks if a column contains any null or empty values.
- `is_trivial_column(column)`: Checks if a column contains only one unique value or binary 0s and 1s.
- `determine_most_likely_data_type(df, column_name)`: Determines the most likely data type of a column.
- `check_null_column(column)`: Checks if all values in a column are null.
- `is_trivial_column(column)`: Checks if a column contains only one unique value or binary 0s and 1s.
- `check_categorical(column)`: Checks if a column is categorical.
- `is_boolean_column(column)`: Checks if all values in a column are boolean.
- `is_categorical(column)`: Checks if a column is categorical.
- `handle_missing_data(column, method='impute', value=None)`: Handles missing data by either imputing or removing NaN values.
- `handle_infinite_data(data, remove_values=True)`: Handles infinite data by replacing infinity values with NaN, optionally removing them.
- `determine_data_type(column)`: Determines the data type of a column.
- `handle_mixed_data(df, column_name)`: Handles mixed data types in a column.

## Examples

Here are some examples of how to use the Data Type Analyzer package:

### Checking Data Types

```python
df = pd.DataFrame({
    'col1': ['apple', 'banana', 'cherry'],
    'col2': [1, 2, 3],
    'col3': [1.1, 2.2, 3.3],
    'col4': [True, False, True],
    'col5': ['2021-01-01', '2022-01-01', '2023-01-01'],
    'col6': pd.to_datetime(['2021-01-01', '2022-01-01', '2023-01-01']),
})

check_column_string(df, 'col1')  # True
check_column_numeric(df['col2'])  # True
check_column_integer(df, 'col2')  # True
check_column_float(df['col3'])  # True
check_column_boolean(df['col4'])  # True
check_column_date(df['col5'])  # True
check_column_datetime(df['col6'])  # True
```

### Handling Missing Data

```python
column = pd.Series(['', 'NA', 'N/A', 'nan', 'NaN'])
handle_missing_data(column)
# Output: pd.Series([])

column = pd.Series([1, np.inf, -np.inf, 2])
handle_infinite_data(column)
# Output: pd.Series([1, nan, nan, 2])

column = pd.Series([None, '', np.nan])
check_null_or_empty(column)
# Output: True
```

### Determining Most Likely Data Type

```python
df = pd.DataFrame({
    'col1': ['apple', 'banana', 'cherry'],
    'col2': [True, False, True],
    'col3': ['2021-01-01', '2022-01-01', '2023-01-01'],
    'col4': pd.to_datetime(['2021-01-01', '2022-01-01', '2023-01-01']),
    'col5': [1.1, 2.2, 3.3],
})

determine_most_likely_data_type(df, 'col1')  # String
determine_most_likely_data_type(df, 'col2')  # Boolean
determine_most_likely_data_type(df, 'col3')  # Date
determine_most_likely_data_type(df, 'col4')  # Datetime
determine_most_likely_data_type(df, 'col5')  # Float
```

### Handling Mixed Data

```python
df = pd.DataFrame({
    'col1': ['apple', 1, True, pd.NA],
    'col2': [1, 2, 3, None],
})

handle_mixed_data(df, 'col1')
# Output:
#   col1  col2
# 0   NaN     1
# 1   NaN     2
# 2   NaN     3
# 3   NaN  <NA>
```

## License

This package is licensed under the MIT License.