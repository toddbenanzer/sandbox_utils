from collections import Counter
from typing import List, Dict, Tuple
import json
import logging
import numpy as np
import os
import pandas as pd


class DateStatisticsCalculator:
    """
    A class to calculate descriptive statistics for a date column in a pandas DataFrame.
    """

    def __init__(self, dataframe: pd.DataFrame, column_name: str):
        """
        Initialize the DateStatisticsCalculator with a DataFrame and a column name.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the date column.
            column_name (str): The name of the date column to analyze.
        """
        if column_name not in dataframe.columns:
            raise ValueError(f"Column {column_name} does not exist in DataFrame.")
        
        self.dataframe = dataframe
        self.column_name = column_name

    def analyze_dates(self) -> Dict:
        """
        Perform a comprehensive analysis of the date column and return all computed statistics.

        Returns:
            dict: A dictionary containing all the computed statistics for the date column.
        """
        date_stats = {
            "date_range": self.calculate_date_range(),
            "distinct_count": self.count_distinct_dates(),
            "most_common_dates": self.find_most_common_dates(),
            "missing_and_empty_count": self.calculate_missing_and_empty_values(),
            "is_trivial": self.check_trivial_column()
        }
        return date_stats

    def calculate_date_range(self) -> Dict[str, pd.Timestamp]:
        """
        Calculate the minimum and maximum dates in the column.

        Returns:
            dict: A dictionary with keys 'min_date' and 'max_date'.
        """
        dates = self.dataframe[self.column_name]
        min_date = dates.min()
        max_date = dates.max()
        return {"min_date": min_date, "max_date": max_date}

    def count_distinct_dates(self) -> int:
        """
        Count the number of distinct dates in the date column.

        Returns:
            int: The count of distinct dates.
        """
        return self.dataframe[self.column_name].nunique()

    def find_most_common_dates(self, top_n: int = 1) -> List[Tuple[pd.Timestamp, int]]:
        """
        Find the most common date(s) in the column.

        Args:
            top_n (int): Number of top common dates to return.

        Returns:
            list: List of tuples containing the most common dates and their counts.
        """
        dates = self.dataframe[self.column_name]
        common_dates = Counter(dates.dropna()).most_common(top_n)
        return common_dates

    def calculate_missing_and_empty_values(self) -> Dict[str, int]:
        """
        Calculate the count of missing and empty values in the date column.

        Returns:
            dict: Dictionary with keys 'missing_values_count' and 'empty_values_count'.
        """
        dates = self.dataframe[self.column_name]
        missing_count = dates.isnull().sum()
        empty_count = (dates == '').sum()
        return {"missing_values_count": missing_count, "empty_values_count": empty_count}

    def check_trivial_column(self) -> bool:
        """
        Check if the column is trivial, containing a single unique date value.

        Returns:
            bool: True if the column is trivial, False otherwise.
        """
        unique_count = self.dataframe[self.column_name].nunique()
        return unique_count == 1



class DataValidator:
    """
    A class to validate and handle missing or infinite values in a DataFrame column.
    """

    def __init__(self, dataframe: pd.DataFrame, column_name: str):
        """
        Initialize the DataValidator with a DataFrame and a column name.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the column to validate.
            column_name (str): The name of the column to validate.
        """
        if column_name not in dataframe.columns:
            raise ValueError(f"Column {column_name} does not exist in DataFrame.")
        
        self.dataframe = dataframe
        self.column_name = column_name

    def validate_date_column(self) -> bool:
        """
        Validate whether the specified column contains valid date data.

        Returns:
            bool: True if the column is valid and contains only dates, False otherwise.
        """
        try:
            pd.to_datetime(self.dataframe[self.column_name])
            return True
        except Exception:
            return False

    def handle_missing_values(self, strategy: str = 'drop', fill_value: pd.Timestamp = pd.NaT) -> pd.Series:
        """
        Handle missing values in the column.

        Args:
            strategy (str): Strategy for handling missing values. Options: 'drop', 'fill'.
            fill_value (pd.Timestamp): Value to fill in missing entries if strategy is 'fill'.

        Returns:
            pandas.Series: The column with missing values handled.
        """
        column = self.dataframe[self.column_name]
        
        if strategy == 'drop':
            column = column.dropna()
        elif strategy == 'fill':
            column = column.fillna(fill_value)
        else:
            raise ValueError(f"Invalid strategy {strategy}. Use 'drop' or 'fill'.")
        
        return column

    def handle_infinite_values(self) -> pd.Series:
        """
        Handle infinite values in the column.

        Returns:
            pandas.Series: The column with infinite values replaced by NaN.
        """
        column = self.dataframe[self.column_name]
        column = column.replace([np.inf, -np.inf], np.nan)
        return column



def load_dataframe(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The file path to the CSV file to be loaded.

    Returns:
        pandas.DataFrame: A DataFrame containing the contents of the CSV file.

    Raises:
        FileNotFoundError: If the file specified by file_path does not exist.
        pd.errors.EmptyDataError: If the CSV file is empty.
        pd.errors.ParserError: If there is a parsing error in the CSV file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    try:
        # Attempt to read the CSV file
        dataframe = pd.read_csv(file_path, parse_dates=True)
    except pd.errors.EmptyDataError:
        raise ValueError("The provided CSV file is empty.")
    except pd.errors.ParserError:
        raise ValueError("Parsing error: Unable to parse the CSV file.")
    
    return dataframe



def save_descriptive_statistics(statistics: dict, output_path: str) -> None:
    """
    Save the calculated descriptive statistics to an output file.

    Args:
        statistics (dict): A dictionary containing the descriptive statistics to save.
        output_path (str): The file path where the descriptive statistics will be saved.

    Raises:
        ValueError: If the statistics is not a dictionary.
        ValueError: If the output_path does not have a supported file extension.
        IOError: If there is an error writing to the file.
    """
    if not isinstance(statistics, dict):
        raise ValueError("Statistics must be provided as a dictionary.")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Determine the file extension
    file_extension = os.path.splitext(output_path)[1].lower()
    
    try:
        if file_extension == '.json':
            # Save as JSON file
            with open(output_path, 'w') as json_file:
                json.dump(statistics, json_file, default=str, indent=4)
        
        elif file_extension == '.csv':
            # Save as CSV file
            df = pd.DataFrame([statistics])
            df.to_csv(output_path, index=False)
        
        else:
            raise ValueError("Unsupported file extension. Please use .json or .csv.")
    
    except IOError as e:
        raise IOError(f"Error writing to file {output_path}: {e}")



def setup_logging(level=logging.INFO) -> None:
    """
    Configure the logging for the module.

    Args:
        level (int): The logging level threshold. Only log messages with this level
                     or higher will be processed. Default is logging.INFO.
    """
    # Check if the root logger already has handlers configured (to prevent duplication)
    if not logging.getLogger().hasHandlers():
        # Set up the basic configuration for logging
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()  # Log to console (stdout)
            ]
        )
