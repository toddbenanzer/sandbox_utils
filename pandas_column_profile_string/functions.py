andas as pd
import statistics


def count_non_null_values(df, column_name):
    """
    Calculate the count of non-null values in a string column.

    Parameters:
        - df: pandas DataFrame
            The input DataFrame containing the string column.
        - column_name: str
            The name of the string column.

    Returns:
        - int
            The count of non-null values in the specified column.
    """
    return df[column_name].count()


def count_missing_values(column):
    """
    Calculates the count of missing values in a string column.

    Parameters:
        - column: pandas.Series
            The string column to calculate missing values for.

    Returns:
        - int
            The count of missing values in the column.
    """
    return column.isnull().sum()


def count_empty_strings(column):
    """
    Calculate the count of empty strings in a string column.

    Parameters:
        - column: pandas.Series
            The string column to calculate empty strings for.

    Returns:
        - int
            The count of empty strings in the column.
    """
    return column.str.count('^$').sum()


def count_unique_values(column):
    """
    Calculate the count of unique values in a string column.

    Parameters:
        - column: pandas.Series
            The string column to calculate unique values for.

    Returns:
        - int
            The count of unique values in the column.
    """
    unique_values = column.unique()
    return len(unique_values)


def calculate_most_common_values(column):
    """
    Calculate the most common values in a string column.

    Parameters:
        - column: pandas.Series
            The string column to calculate most common values for.

    Returns:
        - list
            A list of the most common values in the column.
    """
  
   
   
   
   
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
'return most_common_value

def calculate_missing_prevalence(column):
    """
    Calculates the prevalence of missing values in a string column.

    Parameters:
        - column: pandas.Series
            The string column to calculate missing prevalence for.

    Returns:
        - float
            The prevalence of missing values in the column.
    """
    num_missing = column.isnull().sum()
    total_values = len(column)
    prevalence = (num_missing / total_values) * 100
    return prevalence


def calculate_empty_string_prevalence(column):
    """
    Calculates the prevalence of empty strings in a string column.

    Parameters:
        - column: pandas.Series
            The string column to calculate empty string prevalence for.

    Returns:
        - float
            The prevalence of empty strings in the column.
    """
    if not pd.api.types.is_string_dtype(column):
        raise ValueError("The column must be of string type")
    
    num_empty_strings = column.isna().sum() + (column == "").sum()
    
    prevalence = num_empty_strings / len(column)
    
    return prevalence


def calculate_min_string_length(column):
    """
    Calculate the minimum string length in a string column.

    Parameters:
        - column: pandas.Series
            The string column to calculate minimum string length for.

    Returns:
        - int
            The minimum string length in the column.
    """
   
    

    
    
    
    












max_length = column.str.len().max()

return max_length

def calculate_average_string_length(column):
"""
Calculate the average string length in a string column.

Parameters:
- column: pandas.Series
The string column to calculate average string length for.

Returns:
- float
The average string length in the column.
"""
if not pd.api.types.is_string_dtype(column):
raise ValueError("Column must be of string type")

total_length = sum(len(str(val)) for val in column)

num_values = len(column)

average_length = total_length / num_values

return average_length


def calculate_total_missing_values(column):
"""
Calculate the total count of missing values in a string column.

Parameters:
- column: pandas.Series
The string column to calculate total missing values for.

Returns:
- int
The total count of missing values in the column.
"""
missing_count = column.isnull().sum()

return missing_count


def calculate_percentage_missing_values(column):
"""
Calculate the percentage of missing values in a string column.

Parameters:
- column: pandas.Series
The string column to calculate percentage of missing values for.

Returns:
- float
The percentage of missing values in the column.
"""
missing_count = column.isnull().sum()
total_count = len(column)
missing_percentage = (missing_count / total_count) * 100

return missing_percentage


def calculate_percentage_empty_strings(column):
"""
Calculate the percentage of empty strings in a string column.

Parameters:
- column: pandas.Series
The string column to calculate percentage of empty strings for.

Returns:
- float
The percentage of empty strings in the column.
"""
num_empty_strings = column.str.strip().eq('').sum()
total_rows = len(column)
percentage_empty_strings = (num_empty_strings / total_rows) * 100

return percentage_empty_strings


def count_non_empty_strings(dataframe, column_name):
"""
Calculate the count of non-empty strings in a string column.

Parameters:
- dataframe: pandas.DataFrame
The input dataframe containing the string column.
- column_name: str
The name of the string column to analyze.

Returns:
- int
The count of non-empty strings in the specified column.
"""
nonempty_strings = dataframe[column_name].dropna().astype(str).str.strip()
return len(nonempty_strings)


def calculate_percentage_non_empty_strings(column):
"""
Calculate the percentage of non-empty strings in a string column.

Parameters:
- column: pandas.Series
The string column to calculate percentage of non-empty strings for.

Returns:
- float
The percentage of non-empty strings in the column.
"""
non_empty_count = column.str.strip().replace('', pd.NA).count()
total_count = len(column)
non_empty_percentage = (non_empty_count / total_count) * 100

return non_empty_percentage


def calculate_percentage_unique_values(df, column_name):
"""
Calculate the percentage of unique values in a string column.

Parameters:
- df: pandas.DataFrame
The input dataframe containing the string column.
- column_name: str
The name of the string column to analyze.

Returns:
- float
The percentage of unique values in the column.
"""

num_unique_values = len(df[column_name].unique())
total_values = df[column_name].count()
percentage_unique_values = (num_unique_values / total_values) * 100

return percentage_unique_values


def calculate_most_common_values(column):
"""
Calculate the most common values and their frequencies in a string column.

Parameters:
- column: pandas.Series
The string column to calculate most common values for.

Returns:
- pandas.DataFrame
A DataFrame containing the most common values and their frequencies.
"""

value_counts = column.value_counts()
most_common_values = value_counts.index.tolist()

return most_common_values


def calculate_null_values(df, column_name):
"""
Calculate the count and percentage of null values in a string column.

Parameters:
- df: pandas.DataFrame
The input DataFrame containing the string column.
- column_name: str
The name of the string column to analyze.

Returns:
- int
The count of null values in the specified column.
- float
The percentage of null values in the specified column.
"""

if column_name not in df.columns:
raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

null_count = df[column_name].isnull().sum()
total_count = len(df)
null_percentage = (null_count / total_count) * 100

return null_count, null_percentage


def calculate_trivial_columns(dataframe):
"""
Calculate the count and percentage of trivial columns in a dataframe.

Parameters:
- dataframe: pandas.DataFrame
The input DataFrame to analyze.

Returns:
- int
The count of trivial columns.
- float
The percentage of trivial columns.
"""

unique_counts = dataframe.nunique()
trivial_columns = unique_counts[unique_counts == 1]
count = len(trivial_columns)
percentage = (count / len(dataframe.columns)) * 100

return count, percentage


def calculate_string_statistics(column):
"""
Calculate additional statistical measurements for strings in a column.

Parameters:
- column: pandas.Series
The string column to calculate statistics for.

Returns:
- dict or None
A dictionary containing the calculated statistical measurements, or None if the column is empty or contains only missing values.
Possible keys in the dictionary: 'min_length', 'max_length', 'avg_length', 'most_common_values',
'most_common_frequencies', 'std_deviation', 'median_length', 'first_quartile', 'third_quartile'.
"""

if column.isnull().all() or column.empty:
return None

column = column.dropna()

min_length = column.str.len().min()
max_length = column.str.len().max()
avg_length = column.str.len().mean()

most_common_values = column.value_counts().index.tolist()
most_common_frequencies = column.value_counts().tolist()

try:
std_deviation = statistics.stdev(column.str.len())
median_length = statistics.median(column.str.len())
first_quartile = pd.Series.quantile(column.str.len(), q=0.25)
third_quartile = pd.Series.quantile(column.str.len(), q=0.75)

return {
'min_length': min_length,
'max_length': max_length,
'avg_length': avg_length,
'most_common_values': most_common_values,
'most_common_frequencies': most_common_frequencies,
'std_deviation': std_deviation,
'median_length': median_length,
'first_quartile': first_quartile,
'third_quartile': third_quartile
}
except statistics.StatisticsError:
return Non