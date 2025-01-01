from string_statistics import StringStatistics, validate_column, handle_missing_and_infinite, check_trivial_column
from typing import Dict
from typing import List, Union
import numpy as np
import pandas as pd


class StringStatistics:
    """
    A class used to calculate various descriptive statistics for a string column in a pandas DataFrame.
    """

    def __init__(self, column: pd.Series):
        """
        Initializes the StringStatistics object with a given string column.

        Args:
            column (pd.Series): The column to be analyzed, expected to be of string type.
        """
        self.column = column

    def calculate_mode(self) -> List[str]:
        """
        Calculates the most common string values (mode) in the column.

        Returns:
            List[str]: A list of the most common string values.
        """
        if not isinstance(self.column, pd.Series):
            raise ValueError("Input must be a pandas Series.")
        return self.column.mode().tolist()

    def calculate_missing_prevalence(self) -> float:
        """
        Calculates the percentage of missing (NaN) values in the column.

        Returns:
            float: The percentage of missing values.
        """
        total = len(self.column)
        missing_count = self.column.isna().sum()
        return (missing_count / total) * 100

    def calculate_empty_prevalence(self) -> float:
        """
        Calculates the percentage of empty strings in the column.

        Returns:
            float: The percentage of empty strings.
        """
        total = len(self.column)
        empty_count = self.column.apply(lambda x: x == "").sum()
        return (empty_count / total) * 100

    def calculate_min_length(self) -> int:
        """
        Determines the minimum length of strings in the column.

        Returns:
            int: The minimum string length.
        """
        return self.column.dropna().apply(len).min()

    def calculate_max_length(self) -> int:
        """
        Determines the maximum length of strings in the column.

        Returns:
            int: The maximum string length.
        """
        return self.column.dropna().apply(len).max()

    def calculate_avg_length(self) -> float:
        """
        Calculates the average length of strings in the column.

        Returns:
            float: The average string length.
        """
        return self.column.dropna().apply(len).mean()



def validate_column(column: pd.Series):
    """
    Validates that the provided column is suitable for analysis.

    Args:
        column (pd.Series): The column to be validated.

    Raises:
        ValueError: If the input is not a pandas Series.
        ValueError: If the series is not of string type.
        ValueError: If the series is empty or contains only null/empty values.
    """
    if not isinstance(column, pd.Series):
        raise ValueError("Input must be a pandas Series.")
    
    if column.dtype != object and not all(isinstance(x, str) for x in column.dropna()):
        raise ValueError("Series must contain string values.")
    
    if column.dropna().apply(lambda x: x.strip() if isinstance(x, str) else '').replace('', pd.NA).isna().all():
        raise ValueError("Series is empty or contains only null/empty values.")



def handle_missing_and_infinite(data: pd.DataFrame) -> pd.DataFrame:
    """
    Processes the dataset to handle missing and infinite values.

    Args:
        data (pd.DataFrame): The dataset containing potential missing or infinite values.

    Returns:
        pd.DataFrame: A DataFrame with missing and infinite values handled.

    Notes:
        - Missing values are filled using forward fill, then backward fill as a backup.
        - Infinite values are replaced with NaN and treated like any other missing value.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    # Replace infinite values with NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Handle missing values by filling them
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    return data



def check_trivial_column(column: pd.Series) -> bool:
    """
    Evaluates whether the provided column is trivial.

    A column is considered trivial if it contains little to no informational value, such as having
    predominantly a single unique value or being composed entirely of empty strings.

    Args:
        column (pd.Series): The column to be evaluated, expected to be a series of strings.

    Returns:
        bool: True if the column is trivial, False otherwise.

    Raises:
        ValueError: If the input is not a pandas Series or not suitable for string evaluation.
    """
    if not isinstance(column, pd.Series):
        raise ValueError("Input must be a pandas Series.")
    
    # Check if column has only one unique non-null, non-empty value
    unique_values = column.dropna().apply(lambda x: x.strip() if isinstance(x, str) else "").unique()
    
    # Evaluate triviality
    return len(unique_values) <= 1 or ("" in unique_values and len(unique_values) == 1)



def get_descriptive_statistics(dataframe: pd.DataFrame, column_name: str) -> Dict:
    """
    Aggregates and returns descriptive statistics for a specified string column.

    Args:
        dataframe (pd.DataFrame): The dataset containing the string column to be analyzed.
        column_name (str): The name of the column within the DataFrame to calculate statistics for.

    Returns:
        Dict: A dictionary containing all computed descriptive statistics for the specified column.

    Raises:
        ValueError: If the input is not a pandas DataFrame or if the column name is invalid.
    """
    # Validate dataframe and column name
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    # Handle missing and infinite data
    dataframe = handle_missing_and_infinite(dataframe)
    
    # Extract the column
    column = dataframe[column_name]
    
    # Validate the column is suitable for analysis
    validate_column(column)
    
    # Check if the column is trivial
    if check_trivial_column(column):
        return {"trivial_column": True}

    # Create a StringStatistics instance
    stats = StringStatistics(column)

    # Calculate descriptive statistics
    statistics = {
        "mode": stats.calculate_mode(),
        "missing_prevalence": stats.calculate_missing_prevalence(),
        "empty_prevalence": stats.calculate_empty_prevalence(),
        "min_length": stats.calculate_min_length(),
        "max_length": stats.calculate_max_length(),
        "avg_length": stats.calculate_avg_length(),
        "trivial_column": False
    }

    return statistics
