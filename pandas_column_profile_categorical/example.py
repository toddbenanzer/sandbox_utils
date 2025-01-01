from your_module import CategoricalStatsCalculator
from your_module import handle_missing_and_infinite
from your_module import validate_input
import numpy as np
import pandas as pd


# Example DataFrame
data = {'Category': ['Cat', 'Dog', 'Cat', 'Fish', None, 'Dog', 'Cat']}
df = pd.DataFrame(data)

# Initialize the calculator
calculator = CategoricalStatsCalculator(df, 'Category')

# Calculate frequency distribution
frequency_distribution = calculator.calculate_frequency_distribution()
print("Frequency Distribution:", frequency_distribution)

# Find the most common values
most_common_value = calculator.find_most_common_values()
print("Most Common Value:", most_common_value)

# Find the top 2 most common values
top_two_common_values = calculator.find_most_common_values(n=2)
print("Top 2 Most Common Values:", top_two_common_values)

# Count unique values
unique_values_count = calculator.count_unique_values()
print("Unique Values Count:", unique_values_count)

# Count missing values
missing_values_count = calculator.count_missing_values()
print("Missing Values Count:", missing_values_count)

# Identify if the column is trivial
is_trivial = calculator.identify_trivial_column()
print("Is column trivial?", is_trivial)



# Example 1: Using a DataFrame with a categorical dtype column
df1 = pd.DataFrame({'Category': pd.Categorical(['A', 'B', 'A', 'C'])})
print(f"'Category' is categorical: {is_categorical(df1, 'Category')}")  # Output: True

# Example 2: Using a DataFrame with an object dtype column with low cardinality
df2 = pd.DataFrame({'Category': ['A', 'B', 'A', 'C', 'A']})
print(f"'Category' is categorical: {is_categorical(df2, 'Category')}")  # Output: True

# Example 3: Using a DataFrame with an object dtype column with high cardinality
df3 = pd.DataFrame({'Category': ['A', 'B', 'C', 'D', 'E', 'F', 'G']})
print(f"'Category' is categorical: {is_categorical(df3, 'Category')}")  # Output: False

# Example 4: Attempting to check a non-existent column
df4 = pd.DataFrame({'Category': ['A', 'B', 'A', 'C']})
try:
    is_categorical(df4, 'NonExistent')
except ValueError as e:
    print(e)  # Output: Column 'NonExistent' does not exist in the DataFrame.



# Example 1: Using 'ignore' method
data = [1, 2, np.nan, np.inf, 4]
print("Ignore method result:", handle_missing_and_infinite(data, method='ignore'))

# Example 2: Using 'remove' method
data = [1, 2, np.nan, np.inf, 5]
print("Remove method result:", handle_missing_and_infinite(data, method='remove'))

# Example 3: Using 'fill' method
data = [1, 2, np.nan, np.inf, 6]
print("Fill method result:", handle_missing_and_infinite(data, method='fill'))

# Example 4: Using a pandas Series with 'fill' method
series_data = pd.Series([1, 2, np.nan, np.inf, 7])
print("Fill method with pandas Series result:", handle_missing_and_infinite(series_data, method='fill'))



# Example 1: Validating a DataFrame with a categorical column
data = {'Category': ['A', 'B', 'B', 'C']}
df = pd.DataFrame(data)
try:
    valid = validate_input(df, 'Category')
    print(f"Validation result for 'Category' column: {valid}")
except Exception as e:
    print(e)

# Example 2: Attempting to validate a non-existent column
try:
    valid = validate_input(df, 'NonExistent')
except Exception as e:
    print(e)

# Example 3: Attempting to validate a non-DataFrame input
data_as_list = ['A', 'B', 'B', 'C']
try:
    valid = validate_input(data_as_list, 'Category')
except Exception as e:
    print(e)

# Example 4: Validating with a non-categorical column
data_numbers = {'Numbers': [1, 2, 3, 4]}
df_numbers = pd.DataFrame(data_numbers)
try:
    valid = validate_input(df_numbers, 'Numbers')
except Exception as e:
    print(e)


# Example 1: Displaying an error message for a missing column
error_message = display_error_message('ERR001')
print(error_message)  # Output: "The specified column does not exist in the DataFrame."

# Example 2: Displaying an error message for an invalid DataFrame input
error_message = display_error_message('ERR002')
print(error_message)  # Output: "The input is not a valid pandas DataFrame."

# Example 3: Displaying an error message for a non-categorical column
error_message = display_error_message('ERR003')
print(error_message)  # Output: "The input column is not of a categorical data type."

# Example 4: Displaying an error message for an unknown error
error_message = display_error_message('ERR_UNKNOWN')
print(error_message)  # Output: "An unknown error has occurred."

# Example 5: Displaying a message for an invalid error code
error_message = display_error_message('ERR_INVALID')
print(error_message)  # Output: "Unknown error code provided."
