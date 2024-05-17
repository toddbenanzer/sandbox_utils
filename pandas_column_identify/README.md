# Functionality Documentation

## Overview
This package provides a set of functions for handling and analyzing data in pandas DataFrames. The functions cover a range of tasks such as checking for missing or trivial data, handling missing or infinite values, and calculating various statistics on the data.

## Usage
To use this package, you need to have pandas and numpy installed. You can import the necessary modules using the following code:

```python
import pandas as pd
import numpy as np
```

The package includes the following functions:

1. `is_column_null_or_empty(df, column_name)`: Checks if a column is null or empty in a pandas DataFrame.

2. `is_trivial(column)`: Checks if a column contains only trivial data (e.g. all values are the same).

3. `handle_missing_data(column)`: Handles missing data in a pandas Series by filling it with appropriate values based on its type.

4. `handle_infinite_values(column)`: Handles infinite values in a pandas Series by replacing them with NaN.

5. `check_boolean_data(column)`: Checks if a column contains boolean data.

6. `is_categorical(column)`: Checks if a column contains categorical data.

7. `is_string_column(column)`: Checks if a column contains string data.

8. `check_numeric_data(column)`: Checks if a column contains numeric data.

9. `convert_string_to_numeric(df, column)`: Converts a string column to numeric type if possible.

10. `convert_string_to_date(df, column_name)`: Converts a string column to datetime type if possible.

11. `convert_boolean_column(column)`: Converts a boolean column to appropriate boolean type (True/False or 1/0).

12. `convert_categorical_column(df, column_name)`: Converts a categorical column to category type.

13. `calculate_missing_percentage(column)`: Calculates the percentage of missing values in a column.

14. `calculate_infinite_percentage(column)`: Calculates the percentage of infinite values in a column.

15. `calculate_frequency_distribution(dataframe, column)`: Calculates the frequency distribution of values in a column.

16. `calculate_unique_values(column)`: Calculates the unique values in a column.

17. `calculate_non_null_values(column)`: Calculates the number of non-null values in a column.

18. `calculate_average(column)`: Calculates the average numeric value in a column.

19. `calculate_numeric_sum(column)`: Calculates the sum of numeric values in a column.

20. `calculate_min_value(dataframe, column)`: Calculates the minimum numeric value in a column.

21. `calculate_max_value(dataframe, column)`: Calculates the maximum numeric value in a column.

22. `calculate_numeric_range(column)`: Calculates the range of numeric values in a column.

23. `calculate_median(column)`: Calculates the median numeric value in a column.

24. `calculate_mode(dataframe, column)`: Calculates the mode categorical value in a column.

25. `calculate_earliest_date(dataframe, column)`: Calculates the earliest date value in a datetime column.

26. `calculate_latest_date(column)`: Calculates the latest date value in a datetime column.

27. `calculate_time_range(column)`: Calculates the time range (difference between min and max values) of a datetime column.


## Examples
Here are some examples demonstrating how to use the functions:

1. Checking if a column is null or empty:
```python
df = pd.DataFrame({'A': [1, 2, np.nan], 'B': ['', 'hello', 'world']})
is_column_null_or_empty(df, 'A')  # Output: False
is_column_null_or_empty(df, 'B')  # Output: False
```

2. Checking if a column contains only trivial data:
```python
column = pd.Series([1, 1, 1])
is_trivial(column)  # Output: True

column = pd.Series([1, 2, 3])
is_trivial(column)  # Output: False
```

3. Handling missing data in a column:
```python
column = pd.Series([1, np.nan, 3])
handle_missing_data(column)  # Output: [1, 0, 3]
```

4. Handling infinite values in a column:
```python
column = pd.Series([np.inf, -np.inf, 1])
handle_infinite_values(column)  # Output: [NaN, NaN, 1]
```

5. Checking if a column contains boolean data:
```python
column = pd.Series([True, False, True])
check_boolean_data(column)  # Output: True

column = pd.Series([True, False, 'foo'])
check_boolean_data(column)  # Output: False
```

6. Checking if a column contains categorical data:
```python
column = pd.Series(['cat', 'dog', 'cat'], dtype='category')
is_categorical(column)  # Output: True

column = pd.Series(['cat', 'dog', 'cat'])
is_categorical(column)  # Output: False
```

7. Checking if a column contains string data:
```python
column = pd.Series(['foo', 'bar', 'baz'])
is_string_column(column)  # Output: True

column = pd.Series([1, 2, 3])
is_string_column(column)  # Output: False
```

8. Checking if a column contains numeric data:
```python
column = pd.Series([1, 2, 3])
check_numeric_data(column)  # Output: True

column = pd.Series(['foo', 'bar', 'baz'])
check_numeric_data(column)  # Output: False
```

These are just a few examples, there are many more functions available in this package for handling and analyzing data in pandas DataFrames.