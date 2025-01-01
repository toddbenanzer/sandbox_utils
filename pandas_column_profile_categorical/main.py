from collections import Counter
from typing import List, Dict
from typing import Union
import numpy as np
import pandas as pd


class CategoricalStatsCalculator:
    """
    A class to calculate descriptive statistics for a categorical column in a pandas DataFrame.
    """

    def __init__(self, dataframe: pd.DataFrame, column_name: str):
        """
        Initializes the CategoricalStatsCalculator with a DataFrame and the name of the column to analyze.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the categorical data.
            column_name (str): The name of the column to analyze.
        """
        self.dataframe = dataframe
        self.column_name = column_name
        self.column_data = dataframe[column_name]

    def calculate_frequency_distribution(self) -> Dict:
        """
        Calculates the frequency distribution of values in the categorical column.

        Returns:
            dict: A dictionary with values as keys and their frequencies as values.
        """
        return dict(self.column_data.value_counts(dropna=False))

    def find_most_common_values(self, n: int = 1) -> List:
        """
        Finds the most common values in the categorical column.

        Args:
            n (int): The number of top most common values to return.

        Returns:
            list: A list of the n most common values.
        """
        counter = Counter(self.column_data.dropna())
        most_common = counter.most_common(n)
        return [value for value, count in most_common]

    def count_unique_values(self) -> int:
        """
        Counts the number of unique values in the categorical column.

        Returns:
            int: The count of unique values.
        """
        return self.column_data.nunique(dropna=True)

    def count_missing_values(self) -> int:
        """
        Counts the number of missing values in the categorical column.

        Returns:
            int: The number of missing values.
        """
        return self.column_data.isna().sum()

    def identify_trivial_column(self) -> bool:
        """
        Identifies if the column is trivial (contains only one unique value).

        Returns:
            bool: True if the column is trivial, False otherwise.
        """
        return self.count_unique_values() <= 1



def is_categorical(df: pd.DataFrame, column_name: str) -> bool:
    """
    Determines whether a specified column in a pandas DataFrame is of a categorical data type.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to check for categorical data type.

    Returns:
        bool: True if the specified column is categorical, otherwise False.

    Raises:
        ValueError: If the column does not exist in the DataFrame.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    column_data = df[column_name]
    # Checking if the column is explicitly set as category type
    if pd.api.types.is_categorical_dtype(column_data):
        return True

    # Check if the column has a potential categorical data type
    if pd.api.types.is_object_dtype(column_data):
        unique_ratio = column_data.nunique() / len(column_data)
        # Arbitrary threshold: consider as categorical if less than 0.1 of total unique values
        if unique_ratio < 0.1:
            return True

    return False



def handle_missing_and_infinite(data: Union[list, np.ndarray, pd.Series], method: str = 'ignore') -> Union[list, np.ndarray, pd.Series]:
    """
    Handles missing (NaN) and infinite values in the dataset based on the specified method.

    Args:
        data (list, np.ndarray, pd.Series): The dataset containing potential missing or infinite values.
        method (str): The approach to handle missing and infinite data. Options are 'ignore', 'remove', 'fill'.

    Returns:
        The dataset after handling missing and infinite values as specified by the method.

    Raises:
        ValueError: If an unsupported method is provided.
    """
    if isinstance(data, list):
        data = pd.Series(data)
    
    if not isinstance(data, (pd.Series, np.ndarray)):
        raise TypeError("Data should be a list, numpy array, or pandas series.")

    if method == 'ignore':
        return data
    elif method == 'remove':
        return data.dropna().loc[~np.isinf(data)]
    elif method == 'fill':
        if isinstance(data, pd.Series):
            fill_value = data.median()
            return data.replace([np.inf, -np.inf], np.nan).fillna(fill_value)
        else:
            data = pd.Series(data)
            fill_value = data.median()
            return data.replace([np.inf, -np.inf], np.nan).fillna(fill_value).tolist()
    else:
        raise ValueError(f"Unsupported method '{method}'. Choose 'ignore', 'remove', or 'fill'.")



def validate_input(df: pd.DataFrame, column_name: str) -> bool:
    """
    Validates the input DataFrame and column name to ensure they are suitable for categorical operations.

    Args:
        df (pd.DataFrame): The DataFrame to validate.
        column_name (str): The name of the column to validate.

    Returns:
        bool: True if the input is valid, otherwise raises an appropriate exception.

    Raises:
        TypeError: If the input `df` is not a pandas DataFrame.
        ValueError: If `column_name` does not exist in the DataFrame.
        ValueError: If the column is not categorical.
    """
    # Check if `df` is a valid DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The input data must be a pandas DataFrame.")

    # Check if the specified column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    # Check if the column is categorical
    if not is_categorical(df, column_name):
        raise ValueError(f"Column '{column_name}' is not of a categorical data type.")

    return True


def display_error_message(error_code: str) -> str:
    """
    Provides standardized error messages for given error codes.

    Args:
        error_code (str): A unique identifier for different errors.

    Returns:
        str: A human-readable error message corresponding to the error code.
    """
    error_messages = {
        'ERR001': "The specified column does not exist in the DataFrame.",
        'ERR002': "The input is not a valid pandas DataFrame.",
        'ERR003': "The input column is not of a categorical data type.",
        'ERR_UNKNOWN': "An unknown error has occurred."
    }

    return error_messages.get(error_code, "Unknown error code provided.")
