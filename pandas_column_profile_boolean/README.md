# BooleanDescriptiveStats Class Documentation

## Overview
The `BooleanDescriptiveStats` class provides methods to analyze a boolean column in a pandas DataFrame, calculating various descriptive statistics such as mean, count of True/False values, prevalence of missing values, and more.

## Initialization

### `__init__(self, dataframe: pd.DataFrame, column_name: str)`
Initializes the `BooleanDescriptiveStats` object.

#### Args:
- `dataframe` (pd.DataFrame): The DataFrame containing the boolean column.
- `column_name` (str): The name of the boolean column to be analyzed.

#### Raises:
- `ValueError`: If the specified column is invalid or not boolean.

## Methods

### `validate_column(self) -> bool`
Validates if the specified column exists in the DataFrame and is of boolean type.

#### Returns:
- `bool`: True if valid; otherwise, raises a `ValueError`.

### `calculate_mean(self) -> float`
Calculates the mean (proportion of True values) in the boolean column.

#### Returns:
- `float`: The mean of True values.

### `calculate_true_count(self) -> int`
Counts the number of True values in the boolean column.

#### Returns:
- `int`: The count of True values.

### `calculate_false_count(self) -> int`
Counts the number of False values in the boolean column.

#### Returns:
- `int`: The count of False values.

### `find_most_common_value(self)`
Identifies the most common value (mode) in the boolean column.

#### Returns:
- `bool or None`: The most common value or None if no mode is found.

### `calculate_missing_prevalence(self) -> float`
Calculates the prevalence of missing (null) values in the boolean column.

#### Returns:
- `float`: The prevalence of missing values.

### `calculate_empty_prevalence(self) -> float`
Calculates the prevalence of empty values in the boolean column.

#### Returns:
- `float`: The prevalence of empty values.

### `check_trivial_column(self) -> bool`
Checks if the boolean column is trivial (consists entirely of identical values or null).

#### Returns:
- `bool`: True if the column is trivial; otherwise, False.

### `handle_missing_infinite_data(self)`
Processes the column to handle missing or infinite data.

#### Raises:
- `Warning`: If infinite values are detected in the boolean column.


# display_statistics Function Documentation

## Overview
The `display_statistics` function presents descriptive statistics of a boolean column in various formats. This function allows users to view the statistics in either a dictionary, JSON, or tabular format.

## Parameters

### `stats` (dict)
A dictionary containing the descriptive statistics of a boolean column, which must include the following keys:
- `mean`: (float) The mean (proportion of True values).
- `true_count`: (int) The count of True values.
- `false_count`: (int) The count of False values.
- `mode`: (bool or None) The most common value in the column.
- `missing_prevalence`: (float) The prevalence of missing (null) values.
- `empty_prevalence`: (float) The prevalence of empty values.

### `format` (str, optional)
The format in which to display the statistics. Options include:
- `'dictionary'`: Displays the statistics as a plain dictionary.
- `'json'`: Formats the statistics as a JSON string.
- `'table'`: Displays the statistics in a tabular format (requires the `tabulate` library).

#### Default:
- The default value for `format` is `'dictionary'`.

## Raises
- `ValueError`: If the provided `format` is unsupported or if the `stats` dictionary is missing any of the required keys.
- `ImportError`: If the 'table' format is requested but the `tabulate` library is not installed.

## Usage Examples
