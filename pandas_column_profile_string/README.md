# Package Name

This package provides various functions for analyzing string columns in pandas DataFrames. These functions can be used to calculate counts, percentages, and statistical measurements related to missing values, empty strings, unique values, and string lengths.

## Installation

To install the package, you can use pip:

```shell
pip install package-name
```

## Usage

You can import the package and use its functions as shown below:

```python
import package_name as pn
import pandas as pd

# Create a sample DataFrame
data = {'Name': ['John', 'Alice', '', 'David', None],
        'Age': [25, 30, 35, 40, 45]}

df = pd.DataFrame(data)

# Calculate the count of non-null values in the 'Name' column
non_null_count = pn.count_non_null_values(df, 'Name')
print(f"Count of non-null values: {non_null_count}")

# Calculate the count of missing values in the 'Name' column
missing_count = pn.count_missing_values(df['Name'])
print(f"Count of missing values: {missing_count}")

# Calculate the count of empty strings in the 'Name' column
empty_string_count = pn.count_empty_strings(df['Name'])
print(f"Count of empty strings: {empty_string_count}")

# Calculate the count of unique values in the 'Name' column
unique_value_count = pn.count_unique_values(df['Name'])
print(f"Count of unique values: {unique_value_count}")

# Calculate the prevalence of missing values in the 'Name' column
missing_prevalence = pn.calculate_missing_prevalence(df['Name'])
print(f"Missing prevalence: {missing_prevalence}%")

# Calculate the prevalence of empty strings in the 'Name' column
empty_string_prevalence = pn.calculate_empty_string_prevalence(df['Name'])
print(f"Empty string prevalence: {empty_string_prevalence}%")

# Calculate the minimum string length in the 'Name' column
min_length = pn.calculate_min_string_length(df['Name'])
print(f"Minimum string length: {min_length}")

# Calculate the average string length in the 'Name' column
avg_length = pn.calculate_average_string_length(df['Name'])
print(f"Average string length: {avg_length}")

# Calculate the total count of missing values in the 'Name' column
total_missing_count = pn.calculate_total_missing_values(df['Name'])
print(f"Total missing count: {total_missing_count}")

# Calculate the percentage of missing values in the 'Name' column
missing_percentage = pn.calculate_percentage_missing_values(df['Name'])
print(f"Missing percentage: {missing_percentage}%")

# Calculate the percentage of empty strings in the 'Name' column
empty_string_percentage = pn.calculate_percentage_empty_strings(df['Name'])
print(f"Empty string percentage: {empty_string_percentage}%")

# Calculate the count of non-empty strings in the 'Name' column
non_empty_count = pn.count_non_empty_strings(df, 'Name')
print(f"Count of non-empty strings: {non_empty_count}")

# Calculate the percentage of non-empty strings in the 'Name' column
non_empty_percentage = pn.calculate_percentage_non_empty_strings(df['Name'])
print(f"Non-empty string percentage: {non_empty_percentage}%")

# Calculate the percentage of unique values in the 'Name' column
unique_value_percentage = pn.calculate_percentage_unique_values(df, 'Name')
print(f"Unique value percentage: {unique_value_percentage}%")

# Calculate the most common values and their frequencies in the 'Name' column
most_common_values = pn.calculate_most_common_values(df['Name'])
print("Most common values:")
for value in most_common_values:
    print(value)

# Calculate the count and percentage of null values in the 'Name' column
null_count, null_percentage = pn.calculate_null_values(df, 'Name')
print(f"Null count: {null_count}")
print(f"Null percentage: {null_percentage}%")

# Calculate the count and percentage of trivial columns in the DataFrame
trivial_count, trivial_percentage = pn.calculate_trivial_columns(df)
print(f"Trivial column count: {trivial_count}")
print(f"Trivial column percentage: {trivial_percentage}%")

# Calculate additional statistical measurements for strings in the 'Name' column
string_statistics = pn.calculate_string_statistics(df['Name'])
if string_statistics is not None:
    print("String statistics:")
    for key, value in string_statistics.items():
        print(f"{key}: {value}")
else:
    print("No statistics available.")
```

## Examples

Here are some examples to demonstrate the usage of the package functions:

### Example 1: Counting Non-null Values

```python
import package_name as pn
import pandas as pd

data = {'Name': ['John', 'Alice', '', 'David', None],
        'Age': [25, 30, 35, 40, 45]}

df = pd.DataFrame(data)

non_null_count = pn.count_non_null_values(df, 'Name')
print(f"Count of non-null values: {non_null_count}")
```

Output:
```
Count of non-null values: 4
```

### Example 2: Calculating Missing Prevalence

```python
import package_name as pn
import pandas as pd

data = {'Name': ['John', 'Alice', '', 'David', None],
        'Age': [25, 30, 35, 40, 45]}

df = pd.DataFrame(data)

missing_prevalence = pn.calculate_missing_prevalence(df['Name'])
print(f"Missing prevalence: {missing_prevalence}%")
```

Output:
```
Missing prevalence: 20.0%
```

### Example 3: Calculating String Statistics

```python
import package_name as pn
import pandas as pd

data = {'Name': ['John', 'Alice', '', 'David', None],
        'Age': [25, 30, 35, 40, 45]}

df = pd.DataFrame(data)

string_statistics = pn.calculate_string_statistics(df['Name'])
if string_statistics is not None:
    print("String statistics:")
    for key, value in string_statistics.items():
        print(f"{key}: {value}")
else:
    print("No statistics available.")
```

Output:
```
String statistics:
min_length: 3
max_length: 5
avg_length: 3.6
most_common_values: ['John', 'Alice', '', 'David']
most_common_frequencies: [1, 1, 1, 1]
std_deviation: 1.140175425099138
median_length: 4
first_quartile: 3.0
third_quartile: 5.0
```

## Contributing

If you would like to contribute to this package, feel free to submit a pull request or open an issue on the GitHub repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.