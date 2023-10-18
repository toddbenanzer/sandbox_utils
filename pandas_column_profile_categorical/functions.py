andas as pd
import numpy as np
from scipy.stats import entropy, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.multicomp as mc
from scipy.stats import f_oneway, pointbiserialr
import math


def is_valid_dataframe(data):
    """
    Function to check if the input is a valid pandas dataframe.
    
    Parameters:
        data (any): The input data to be checked.
        
    Returns:
        bool: True if the input is a valid pandas dataframe, False otherwise.
    """
    return isinstance(data, pd.DataFrame)


def is_categorical_column(column):
    """
    Checks if the input column is a valid categorical column.
    
    Parameters:
        - column (pandas.Series): The input column to be checked.
        
    Returns:
        - bool: True if the column is a valid categorical column, False otherwise.
    """
    return isinstance(column, pd.Series) and column.dtype == 'object'


def calculate_unique_count(column):
    """
    Calculate the count of unique values in a column.

    Parameters:
    column (pandas.Series): The column to analyze.

    Returns:
    int: The count of unique values.
    """
    return len(column.unique())


def calculate_missing_values(column):
    """
    Calculate the prevalence of missing values in a column.

    Parameters:
        column (pandas Series): The column to analyze

    Returns:
        float: The prevalence of missing values in the column
    """

    # Count the number of missing values in the column
    num_missing = column.isnull().sum()

    # Calculate the total number of values in the column
    total_values = len(column)

    # Calculate the prevalence of missing values
    prevalence = num_missing / total_values

    return prevalence


def calculate_empty_prevalence(column):
    """
   Calculate the prevalence of empty values in the column.

   Parameters:
   column (pandas.Series): The input column.

   Returns:
   float: The prevalence of empty values in the column.
   """
    empty_count = column.isnull().sum()
    return empty_count / len(column)


def calculate_non_null_count(column):
    """
    Function to calculate the count of non-null values in the column.

    Parameters:
        column (pandas.Series): The input column

    Returns:
        int: Count of non-null values in the column
    """
    return column.notnull().sum()


def count_null_values(column):
    """
    Calculates the count of null values in a given column of a pandas dataframe.

    Parameters:
        column (pandas.Series): The column to check for null values.

    Returns:
        int: The count of null values in the column.
    """
    return column.isnull().sum()


def calculate_missing_percentage(column):
    """
    Calculate the percentage of missing values in a column.

    Parameters:
        column (pandas.Series): The column to analyze.

    Returns:
        float: The percentage of missing values.
    """
    total_values = len(column)
    missing_values = column.isnull().sum()

    return (missing_values / total_values) * 100


def calculate_empty_percentage(column):
    """
       Calculates the percentage of empty values in a column.

       Parameters:
       - column: pandas Series or DataFrame column to analyze

       Returns:
       - percentage: float, representing the percentage of empty values in the column
       """
    
    # Count the number of empty values in the column
    num_empty = column.isna().sum()

    # Calculate the total number of values in the column
    total_values = len(column)

    # Calculate the percentage of empty values
    percentage = (num_empty / total_values) * 100

    return percentage


def calculate_mode(dataframe, column):
    # Check if the column exists in the dataframe
    if column not in dataframe.columns:
        raise ValueError("Column does not exist in the dataframe")

    # Check if the column is categorical
    if not pd.api.types.is_categorical_dtype(dataframe[column]):
        raise ValueError("Column is not of categorical type")

    # Calculate the mode
    mode = dataframe[column].mode()

    return mode


def calculate_unique_value_counts(column):
    # Count the number of occurrences of each unique value in the column
    value_counts = column.value_counts()

    # Calculate the percentage of each unique value in the column
    percentages = value_counts / len(column) * 100

    return value_counts, percentages


def calculate_entropy(dataframe, column):
    # Check if the column exists in the dataframe
    if column not in dataframe.columns:
        return "Column does not exist in the dataframe."

    # Remove missing and empty values from the column
    clean_column = dataframe[column].dropna().replace('', np.nan).dropna()

    # Check if the column is null or trivial
    if clean_column.empty:
        return "Column is null or trivial."

    # Calculate the entropy of the categorical distribution
    unique_values = clean_column.unique()
    value_counts = clean_column.value_counts(normalize=True)
    entropy_value = entropy(value_counts)

    return entropy_value


def calculate_gini_index(df, column_name):
    """
       Calculates the Gini index of inequality for a categorical distribution in a column.

       Parameters:
           - df (pandas.DataFrame): The input dataframe
           - column_name (str): The name of the column to analyze

       Returns:
           - float: The Gini index of inequality
       """
    
     # Get the counts of each category in the column
        value_counts = df[column_name].value_counts()

        # Calculate the probabilities for each category
        probabilities = value_counts / value_counts.sum()

        # Calculate the Gini index using the formula: 1 - sum(p^2)
        gini_index = 1 - (probabilities ** 2).sum()

        return gini_index


def remove_trivial_columns(df):
    # Create a list to store the column names of trivial columns
    trivial_columns = []

    # Iterate over each column in the dataframe
    for column in df.columns:
        # Check if the number of unique values in the column is equal to 1
        if df[column].nunique() == 1:
            # If yes, add the column name to the trivial_columns list
            trivial_columns.append(column)

    # Remove the trivial columns from the dataframe
    df = df.drop(columns=trivial_columns)

    # Return the updated dataframe
    return df


def handle_missing_data(dataframe, fill_value):
    """
       Function to handle missing data by imputing with a specified value.

       Parameters:
           - dataframe: pandas DataFrame
               Input DataFrame containing categorical column(s).
           - fill_value: str or int or float
               Value to be used for imputing missing values.

       Returns:
           - dataframe_imputed: pandas DataFrame
               DataFrame with missing values replaced by the specified fill value.
       """
    
      dataframe_imputed = dataframe.fillna(fill_value)
      return dataframe_imputed


def handle_missing_data_mode(df, column_name):
    # Check if the column exists in the dataframe
    if column_name not in df.columns:
        return f"Column '{column_name}' does not exist in the dataframe."

    # Get the most frequent value in the column
    most_frequent_value = df[column_name].mode().values[0]

    # Replace missing values with the most frequent value
    df[column_name].fillna(most_frequent_value, inplace=True)

    return df


def impute_missing_with_mode(df, target_column, reference_column):
    """
    Function to handle missing data by imputing with mode value based on another categorical variable.

    Parameters:
    df (pandas.DataFrame): Input dataframe containing the target and reference columns.
    target_column (str): Name of the column with missing values to be imputed.
    reference_column (str): Name of the column used as a reference for imputing missing values.

    Returns:
    pandas.DataFrame: DataFrame with missing values imputed with mode value based on the reference column.
    """

    # Calculate mode value for the reference column
    mode_value = df[reference_column].mode()[0]

    # Impute missing values in the target column with the mode value
    df[target_column] = df[target_column].fillna(mode_value)

    return df


def handle_missing_data_random(df, column_name):
    import random

    def get_random_value(series):
        unique_values = series.dropna().unique()
        return random.choice(unique_values)

        # Check if the column has any missing values

    if df[column_name].isnull().any():
        # Get unique non-null values in the column
        unique_values = df[column_name].dropna().unique()

        # Randomly choose a value from the unique non-null values
        random_value = get_random_value(unique_values)

        # Replace the missing values with the randomly chosen value
        df[column_name].fillna(random_value, inplace=True)

    return df


def handle_missing_data_delete(df):
    import pandas as pd

  # Drop rows with missing values
  df.dropna(inplace=True)
  
  return df


def replace_infinite_values(column, replace_value):
  
  # Replace infinite values with NaN
  column.replace([np.inf, -np.inf], np.nan, inplace=True)
  
  # Replace NaN with specified value
  column.fillna(replace_value, inplace=True)


def delete_rows_with_infinite_values(dataframe):
    # Check if any column in the dataframe contains infinite values
    has_infinite_values = np.any(np.isinf(dataframe.values))

    if has_infinite_values:
        # Drop rows with infinite values
        dataframe = dataframe[~np.any(np.isinf(dataframe), axis=1)]

    return dataframe


def handle_null_data(df, column, replace_value):
    """
    Function to handle null data by replacing it with a specified value.

    Args:
        df (pandas.DataFrame): Input dataframe.
        column (str): Name of the column containing null values.
        replace_value: Value to replace null values with.

    Returns:
        pandas.DataFrame: Updated dataframe with null values replaced.
    """
    
     # Replace null values in the specified column with the specified value
      df[column].fillna(replace_value, inplace=True)

      return df


def handle_null_data_delete(df):
      """
      Function to handle null data by deleting rows with null values from the dataframe.
      
      Args:
          df (pandas.DataFrame): Input dataframe
          
      Returns:
          pandas.DataFrame: DataFrame with null rows removed
      """
      return df.dropna()

def generate_summary_statistics(df, column_name):
    """
       Function to generate summary statistics for a categorical column in a pandas dataframe.

       Parameters:
           - df (pandas.DataFrame): The input dataframe.
           - column_name (str): The name of the categorical column to analyze.

       Returns:
           - stats_dict (dict): A dictionary containing the summary statistics.
       """
    
     # Check if the column exists in the dataframe
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' does not exist in the dataframe.")

        # Get the series for the specified column
        series = df[column_name]

        # Check if the column is null or trivial
        if series.isnull().all() or len(series.unique()) <= 1:
            raise ValueError(f"Column '{column_name}' is null or trivial.")

        # Calculate the most common values and their frequencies
        most_common = series.value_counts()

        # Calculate the prevalence of missing and empty values
        num_missing = series.isnull().sum()
        num_empty = (series == '').sum()

        # Create a dictionary to store the summary statistics
        stats_dict = {
            'most_common_values': most_common,
            'num_missing': num_missing,
            'num_empty': num_empty
        }

        return stats_dict


def generate_bar_plot(dataframe, column):
    """
       Function to generate a bar plot of the frequency distribution of a categorical column.

       Parameters:
       dataframe (pandas.DataFrame): The input dataframe containing the categorical column.
       column (str): The name of the categorical column in the dataframe.

       Returns:
       None
       """
    
      # Calculate the frequency distribution of the categorical column
      value_counts = dataframe[column].value_counts()

      # Generate the bar plot
      value_counts.plot(kind='bar')

      # Set labels and title
      plt.xlabel(column)
      plt.ylabel('Frequency')
      plt.title('Frequency Distribution of ' + column)

      # Show the plot
      plt.show()


def generate_pie_chart(dataframe, column):
    # Count the occurrences of each unique value in the column
    value_counts = dataframe[column].value_counts()

    # Calculate the percentage distribution
    percentages = value_counts / len(dataframe) * 100

    # Generate the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(percentages, labels=percentages.index, autopct='%1.1f%%')
    plt.axis('equal')
    plt.title(f'Percentage Distribution of {column}')

    # Show the chart
    plt.show()


def generate_count_plot(dataframe, column):
    """
       Function to generate a count plot of the occurrence of each unique value in the categorical column.

       Parameters:
       - dataframe: pandas DataFrame containing the data
       - column: name of the categorical column to analyze

       Returns:
       - None (displays the count plot)
       """

      # Check if the column exists in the dataframe
      if column not in dataframe.columns:
          print(f"Column '{column}' does not exist in the provided DataFrame.")
          return

      # Generate count plot
      plt.figure(figsize=(10, 6))
      sns.countplot(x=column, data=dataframe)
      plt.title(f"Count Plot of {column}")
      plt.xlabel(column)
      plt.ylabel("Count")

      # Show the count plot
      plt.show()


def calculate_chi_square(data, variable1, variable2):
    contingency_table = pd.crosstab(data[variable1], data[variable2])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)

    return chi2, p_value


def calculate_cramers_v(var1, var2):
    # Create a contingency table
    contingency_table = pd.crosstab(var1, var2)

    # Calculate chi-square test statistic
    chi2, _, _, _ = chi2_contingency(contingency_table)

    # Number of rows in the contingency table
    n = contingency_table.sum().sum()

    # Calculate Cramer's V
    v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

    return v


def perform_one_way_anova(categorical_column, continuous_column):
    """
       Perform one-way ANOVA on a categorical variable and a continuous variable.

       Parameters:
       categorical_column (pandas Series): Categorical column in the dataframe.
       continuous_column (pandas Series): Continuous column in the dataframe.

       Returns:
       float: F-statistic
       float: p-value
       """

    # Create a DataFrame with the categorical and continuous columns
    data = pd.DataFrame({categorical_column.name: categorical_column,
                         continuous_column.name: continuous_column})

    # Group the data based on unique categories in the categorical column
    grouped_data = data.groupby(categorical_column.name)

    # Extract the continuous values for each group
    groups = [grouped_data.get_group(group)[continuous_column.name] for group in grouped_data.groups]

    # Perform one-way ANOVA test
    f_statistic, p_value = f_oneway(*groups)

    return f_statistic, p_value


def perform_tukey_hsd(df, column_name, group_column):
    # Perform one-way ANOVA
    groups = df[group_column].unique()
    data_groups = [df[column_name][df[group_column] == group] for group in groups]
    fvalue, pvalue = stats.f_oneway(*data_groups)

    if pvalue < 0.05:
        # Perform Tukey's HSD test
        comp = mc.MultiComparison(df[column_name], df[group_column])
        result = comp.tukeyhsd()

        return result
    else:
        return "No significant difference found."


def calculate_point_biserial_correlation(binary_column, categorical_column):
    # Convert the binary column to numeric values (0 and 1)
    binary_values = pd.get_dummies(binary_column, drop_first=True)

    # Calculate the point-biserial correlation coefficient
    correlation_coefficient, p_value = pointbiserialr(binary_values, categorical_column)

    return correlation_coefficient, p_value


def calculate_phi_coefficient(dataframe, variable1, variable2):
    # Create contingency table
    contingency_table = pd.crosstab(dataframe[variable1], dataframe[variable2])

    # Calculate observed frequencies
    observed_freq = contingency_table.values

    # Calculate expected frequencies (assuming independence)
    row_totals = contingency_table.sum(axis=1)
    col_totals = contingency_table.sum(axis=0)
    total = contingency_table.values.sum()

    expected_freq = np.outer(row_totals, col_totals) / total

    # Calculate chi-square statistic
    chi_square = np.sum((observed_freq - expected_freq) ** 2 / expected_freq)

    # Calculate phi coefficient
    phi_coefficient = np.sqrt(chi_square / total)

    return phi_coefficient


def calculate_chi_square(data, variable1, variable2):
    contingency_table = pd.crosstab(data[variable1], data[variable2])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)

    return chi2, p_value


def correspondence_analysis(dataframe, variable1, variable2):
    contingency_table = pd.crosstab(dataframe[variable1], dataframe[variable2])

     # Perform chi-square test of independence to check if the variables are dependent
        chi2, p, dof, expected = chi2_contingency(contingency_table)

        if p < 0.05:
            # Standardize the contingency table
            row_totals = contingency_table.sum(axis=1)
            column_totals = contingency_table.sum(axis=0)
            total = row_totals.sum()

            observed = contingency_table.values
            expected = np.outer(row_totals, column_totals) / total

            # Calculate the standardized residuals
            residuals = (observed - expected) / np.sqrt(expected)

            # Calculate the row and column masses
            row_masses = row_totals / total
            column_masses = column_totals / total

            return residuals, row_masses, column_masses

        else:
            return "Variables are independent"


def perform_factor_analysis(dataframe):
    """
       Perform factor analysis on a set of categorical variables.

       Args:
           dataframe (pandas.DataFrame): The input dataframe containing the categorical variables.

       Returns:
           pandas.DataFrame: A dataframe with the factor analysis results.
       """

    # Check if the input dataframe is empty
    if dataframe.empty:
        raise ValueError("Input dataframe is empty.")

    # Check for null and trivial columns
    null_columns = dataframe.columns[dataframe.isnull().any()].tolist()
    trivial_columns = [col for col in dataframe.columns if len(dataframe[col].unique()) < 2]

    # Remove null and trivial columns from the dataframe
    valid_columns = [col for col in dataframe.columns if col not in null_columns + trivial_columns]

    # Perform factor analysis on the remaining valid columns
    factor_analysis_results = pd.DataFrame()

    for col in valid_columns:
        column_values = dataframe[col]

        # Calculate the count and percentage of each value in the column
        value_counts = column_values.value_counts(dropna=False)
        value_percentages = column_values.value_counts(normalize=True, dropna=False)

        # Create a row for each unique value in the column with its count and percentage
        value_stats = pd.concat([value_counts, value_percentages], axis=1)
        value_stats.columns = ['Count', 'Percentage']

        # Add a row for missing values (NaNs) if present in the column
        if pd.isnull(column_values).any():
            missing_count = column_values.isnull().sum()
            missing_percentage = missing_count / len(column_values)
            missing_row = pd.DataFrame({'Count': missing_count, 'Percentage': missing_percentage}, index=['NaN'])
            value_stats = pd.concat([value_stats, missing_row])

        # Add the column name as an index level in the dataframe
        value_stats.index = pd.MultiIndex.from_product([[col], value_stats.index])

        # Append the value statistics for the current column to the factor analysis results dataframe
        factor_analysis_results = pd.concat([factor_analysis_results, value_stats])

    return factor_analysis_result