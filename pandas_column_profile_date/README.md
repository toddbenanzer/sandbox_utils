# Python Date Utilities Package

This package provides various utility functions for working with date columns in pandas dataframes. It includes functions for calculating mean, median, mode, standard deviation, variance and other statistical measures. There are also functions for handling missing and infinite values, as well as checking for null or trivial columns.

## Installation

To install the package, use the following command:

```
pip install date-utils
```

## Usage

To use the package, import it into your python script as follows:

```python
import date_utils
```

The package provides the following functions:

### calculate_mean_date(dataframe: pd.DataFrame, date_column: str) -> pd.Timestamp

Calculate the mean of the date column.

Parameters:
- `dataframe` (pd.DataFrame): The input dataframe.
- `date_column` (str): The name of the date column.

Returns:
- `pd.Timestamp`: The mean value of the date column.

Raises:
- `ValueError`: If the specified column does not exist in the dataframe.
- `TypeError`: If the specified column is not a valid date column.

### calculate_median_date(dataframe: pd.DataFrame, date_column: str) -> pd.Timestamp

Calculate the median of the date column.

Parameters:
- `dataframe` (pd.DataFrame): The input dataframe.
- `date_column` (str): The name of the date column.

Returns:
- `pd.Timestamp`: The median value of the date column.

Raises:
- `ValueError`: If the specified column does not exist in the dataframe.

### calculate_mode_date(dataframe: pd.DataFrame, date_column: str) -> pd.Series

Calculate the mode of the date column.

Parameters:
- `dataframe` (pd.DataFrame): The input dataframe.
- `date_column` (str): The name of the date column.

Returns:
- `pd.Series`: The mode(s) of the date column.

Raises:
- `ValueError`: If the specified column does not exist in the dataframe.

### calculate_standard_deviation_date(column: pd.Series) -> float

Calculate the standard deviation of a date column.

Parameters:
- `column` (pd.Series): The input date column.

Returns:
- `float`: The standard deviation of the dates.

### calculate_variance_date(date_column: pd.Series) -> float

Calculate the variance of a date column.

Parameters:
- `date_column` (pd.Series): The input date column.

Returns:
- `float`: The variance of the date column.

### calculate_minimum_date(dataframe: pd.DataFrame, column_name: str) -> pd.Timestamp

Calculate the minimum value of a date column in a pandas dataframe.

Parameters:
- `dataframe` (pd.DataFrame): The input dataframe.
- `column_name` (str): The name of the date column.

Returns:
- `pd.Timestamp`: The minimum value of the date column.

Raises:
- `ValueError`: If the specified column does not exist in the dataframe or if it is not a valid date column.

### calculate_maximum_date(column: pd.Series) -> pd.Timestamp

Calculate the maximum value of a date column.

Parameters:
- `column` (pd.Series): The input date column.

Returns:
- `pd.Timestamp`: The maximum value of the date column.

Raises:
- `ValueError`: If the input column is not a valid date column.

### calculate_range_of_dates(column: pd.Series) -> pd.Timedelta

Calculate the range of values in a date column.

Parameters:
- `column` (pd.Series): The input date column.

Returns:
- `pd.Timedelta`: The range of values in the date column.

### count_non_null_dates(dataframe: pd.DataFrame, column_name: str) -> int

Calculates the count of non-null values in a date column.

Parameters:
- `dataframe` (pd.DataFrame): The input dataframe.
- `column_name` (str): The name of the date column.

Returns:
- `int`: The count of non-null values in the date column.

### count_null_values(date_column: pd.Series) -> int

Calculates the count of null values in a date column.

Parameters:
- `date_column` (pd.Series): The input date column.

Returns:
- `int`: The count of null values in the date column.

### count_unique_dates(dataframe: pd.DataFrame, date_column: str) -> int

Calculate the count of unique values in a date column of a pandas dataframe.

Parameters:
- `dataframe` (pd.DataFrame): The input dataframe.
- `date_column` (str): The name of the date column.

Returns:
- `int`: The count of unique values in the date column.

### calculate_empty_values(dataframe: pd.DataFrame, date_column: str) -> int

Calculate the count of empty values in a date column of a pandas dataframe.

Parameters:
- `dataframe` (pd.DataFrame): The input dataframe.
- `date_column` (str): The name of the date column.

Returns:
- `int`: The count of empty values in the date column.

### check_null_or_trivial(date_column: pd.Series) -> bool

Function to check if a given date column is null or trivial.

Parameters:
- `date_column` (pd.Series): Date column to be checked.

Returns:
- `bool`: True if the date column is null or trivial, False otherwise.

### handle_missing_data_remove(dataframe: pd.DataFrame, column: str) -> pd.DataFrame

Handle missing data in a pandas dataframe by removing rows with missing values.

Parameters:
- `dataframe` (pd.DataFrame): The input dataframe.
- `column` (str): The name of the column to handle missing data for.

Returns:
- `pd.DataFrame`: The modified dataframe with rows containing missing values removed.

### handle_infinite_data(dataframe: pd.DataFrame, column_name: str) -> pd.DataFrame

Handle infinite data in a pandas dataframe by replacing infinite values with appropriate values.

Parameters:
- `dataframe` (pd.DataFrame): The input dataframe.
- `column_name` (str): The name of the column to handle infinite data for.

Returns:
- `pd.DataFrame`: The modified dataframe with infinite values replaced.

### calculate_missing_values(dataframe: pd.DataFrame, column_name: str) -> float

Calculate the prevalence of missing values in a date column.

Parameters:
- `dataframe` (pd.DataFrame): The input dataframe.
- `column_name` (str): The name of the date column.

Returns:
- `float`: The prevalence of missing values in the date column.

Raises:
- `ValueError`: If the specified column does not exist in the dataframe.

### check_null_values(date_column: pd.Series) -> bool

Function to check if any null values exist in the date column.

Parameters:
- `date_column` (pd.Series): The date column to be checked.

Returns:
- `bool`: True if there are null values, False otherwise.

### calculate_unique_dates(column: pd.Series) -> int

Function to calculate the number of unique dates in a column.

Parameters:
- `column` (pd.Series): The input date column.

Returns:
- `int`: The number of unique dates in the column.

### calculate_date_frequency(column: pd.Series) -> pd.DataFrame

Function to calculate the frequency distribution of dates in a column.

Parameters:
- `column` (pd.Series): The input date column.

Returns:
- `pd.DataFrame`: The frequency distribution of dates in the column.

### handle_missing_data(dataframe: pd.DataFrame, column: str) -> pd.DataFrame

Handle missing data by replacing it with appropriate values.

Parameters:
- `dataframe` (pd.DataFrame): The input dataframe.
- `column` (str): The name of the column to handle missing data for.

Returns:
- `pd.DataFrame`: The modified dataframe with missing values replaced.

### handle_infinite_values(dataframe: pd.DataFrame, column_name: str) -> pd.DataFrame

Handle infinite values in a pandas dataframe by replacing them with appropriate values.

Parameters:
- `dataframe` (pd.DataFrame): The input dataframe.
- `column_name` (str): The name of the column to handle infinite values for.

Returns:
- `pd.DataFrame`: The modified dataframe with infinite values replaced.

## Examples

Here are some examples of how to use the package:

```python
import pandas as pd
import date_utils

# Create a sample dataframe
df = pd.DataFrame({'date': ['2021-01-01', '2021-01-02', '2021-01-03'], 'value': [1, 2, 3]})

# Calculate the mean date
mean_date = date_utils.calculate_mean_date(df, 'date')
print(mean_date)

# Calculate the range of dates
date_range = date_utils.calculate_range_of_dates(df['date'])
print(date_range)
```

Output:
```
2021-01-02 00:00:00
2 days 00:00:00
```

## Contributing

Contributions are welcome! If you have any suggestions or find any issues, please create a new issue or submit a pull request.