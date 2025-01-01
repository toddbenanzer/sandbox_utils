from string_statistics import StringStatistics
from string_statistics import check_trivial_column
from string_statistics import get_descriptive_statistics
from string_statistics import handle_missing_and_infinite
from string_statistics import validate_column
import numpy as np
import pandas as pd


# Example 1: Basic Usage
data = pd.Series(["apple", "banana", "orange", "banana", "apple", ""])
stats = StringStatistics(data)

print("Mode:", stats.calculate_mode())
print("Missing Prevalence:", stats.calculate_missing_prevalence(), "%")
print("Empty Prevalence:", stats.calculate_empty_prevalence(), "%")
print("Min Length:", stats.calculate_min_length())
print("Max Length:", stats.calculate_max_length())
print("Avg Length:", stats.calculate_avg_length())

# Example 2: Handling Missing Values
data_with_nans = pd.Series(["apple", None, "banana", "orange", None, None])
stats_with_nans = StringStatistics(data_with_nans)

print("\nMissing Prevalence with NaNs:", stats_with_nans.calculate_missing_prevalence(), "%")

# Example 3: Working with Uniform Strings
uniform_data = pd.Series(["test", "test", "test", "test"])
uniform_stats = StringStatistics(uniform_data)

print("\nMode for uniform data:", uniform_stats.calculate_mode())
print("Min Length for uniform data:", uniform_stats.calculate_min_length())
print("Max Length for uniform data:", uniform_stats.calculate_max_length())
print("Avg Length for uniform data:", uniform_stats.calculate_avg_length())



# Example 1: Validating a proper string column
try:
    data = pd.Series(["apple", "banana", "cherry"])
    validate_column(data)
    print("Column is valid for analysis.")
except ValueError as e:
    print(e)

# Example 2: Attempting to validate a non-Series input
try:
    invalid_data = ["apple", "banana", "cherry"]
    validate_column(invalid_data)
except ValueError as e:
    print(f"Validation failed: {e}")

# Example 3: Validating a Series with non-string data
try:
    non_string_data = pd.Series([1, 2, 3])
    validate_column(non_string_data)
except ValueError as e:
    print(f"Validation failed: {e}")

# Example 4: Validating an empty Series
try:
    empty_data = pd.Series([])
    validate_column(empty_data)
except ValueError as e:
    print(f"Validation failed: {e}")

# Example 5: Validating a Series with only null and empty values
try:
    null_empty_data = pd.Series([None, "", "   ", None])
    validate_column(null_empty_data)
except ValueError as e:
    print(f"Validation failed: {e}")



# Example 1: Handling NaN values in a DataFrame
data_with_nan = pd.DataFrame({
    "A": [1, np.nan, 3],
    "B": [np.nan, 5, 6]
})
cleaned_data_nan = handle_missing_and_infinite(data_with_nan)
print(cleaned_data_nan)

# Example 2: Handling infinite values in a DataFrame
data_with_inf = pd.DataFrame({
    "A": [1, np.inf, 3],
    "B": [-np.inf, 5, 8]
})
cleaned_data_inf = handle_missing_and_infinite(data_with_inf)
print(cleaned_data_inf)

# Example 3: Handling both NaN and infinite values
data_with_nan_inf = pd.DataFrame({
    "A": [1, np.inf, np.nan, 3],
    "B": [np.nan, -np.inf, 5, 7]
})
cleaned_data_nan_inf = handle_missing_and_infinite(data_with_nan_inf)
print(cleaned_data_nan_inf)

# Example 4: DataFrame without any missing or infinite values
data_no_missing = pd.DataFrame({
    "A": [1, 2, 3],
    "B": [4, 5, 6]
})
cleaned_data_no_missing = handle_missing_and_infinite(data_no_missing)
print(cleaned_data_no_missing)



# Example 1: All identical values
data_identical = pd.Series(["apple", "apple", "apple"])
is_trivial = check_trivial_column(data_identical)
print("Is trivial:", is_trivial)  # Output: True

# Example 2: All empty strings
data_empty = pd.Series(["", "", ""])
is_trivial_empty = check_trivial_column(data_empty)
print("Is trivial:", is_trivial_empty)  # Output: True

# Example 3: Mixed values
data_mixed = pd.Series(["apple", "banana", "cherry"])
is_trivial_mixed = check_trivial_column(data_mixed)
print("Is trivial:", is_trivial_mixed)  # Output: False

# Example 4: Only NaN values
data_nan = pd.Series([None, None, None])
is_trivial_nan = check_trivial_column(data_nan)
print("Is trivial:", is_trivial_nan)  # Output: True

# Example 5: Stripped empty values
data_stripped = pd.Series([" ", "  ", "\t"])
is_trivial_stripped = check_trivial_column(data_stripped)
print("Is trivial:", is_trivial_stripped)  # Output: True



# Example 1: Basic Descriptive Statistics
dataframe = pd.DataFrame({
    "fruits": ["apple", "banana", "apple", "orange", "banana", "banana"]
})
stats = get_descriptive_statistics(dataframe, "fruits")
print("Descriptive Statistics for 'fruits':", stats)

# Example 2: Handling NaN Values
dataframe_with_nan = pd.DataFrame({
    "snacks": ["chips", None, "soda", "cookies", None, "chips"]
})
nan_stats = get_descriptive_statistics(dataframe_with_nan, "snacks")
print("Descriptive Statistics for 'snacks':", nan_stats)

# Example 3: Trivial Column
trivial_dataframe = pd.DataFrame({
    "trivial_column": ["same", "same", "same"]
})
trivial_stats = get_descriptive_statistics(trivial_dataframe, "trivial_column")
print("Descriptive Statistics for 'trivial_column':", trivial_stats)

# Example 4: Column with Mixed Values
mixed_dataframe = pd.DataFrame({
    "words": ["hello", "world", "hello", "py", "py"]
})
mixed_stats = get_descriptive_statistics(mixed_dataframe, "words")
print("Descriptive Statistics for 'words':", mixed_stats)
