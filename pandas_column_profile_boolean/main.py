from collections import Counter
import json
import pandas as pd


class BooleanDescriptiveStats:
    def __init__(self, dataframe: pd.DataFrame, column_name: str):
        """
        Initializes the BooleanDescriptiveStats object.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the boolean column.
            column_name (str): The name of the boolean column.
        """
        self.dataframe = dataframe
        self.column_name = column_name
        self.stats = {}

        if not self.validate_column():
            raise ValueError(f"Column '{column_name}' is invalid or not boolean.")

    def validate_column(self) -> bool:
        """
        Validates if the specified column is boolean and exists in the DataFrame.

        Returns:
            bool: True if valid, else raises ValueError.
        """
        if self.column_name not in self.dataframe.columns:
            raise ValueError(f"Column '{self.column_name}' not found in DataFrame.")
        
        if not pd.api.types.is_bool_dtype(self.dataframe[self.column_name]):
            raise ValueError(f"Column '{self.column_name}' is not of boolean type.")
        
        if self.check_trivial_column():
            raise ValueError(f"Column '{self.column_name}' is trivial (all null or same value).")
        
        return True
    
    def calculate_mean(self) -> float:
        """
        Calculates the mean (proportion of True values) in the boolean column.

        Returns:
            float: The mean of True values.
        """
        self.stats['mean'] = self.dataframe[self.column_name].mean()
        return self.stats['mean']
    
    def calculate_true_count(self) -> int:
        """
        Counts the number of True values in the boolean column.

        Returns:
            int: The count of True values.
        """
        self.stats['true_count'] = self.dataframe[self.column_name].sum()
        return self.stats['true_count']
    
    def calculate_false_count(self) -> int:
        """
        Counts the number of False values in the boolean column.

        Returns:
            int: The count of False values.
        """
        self.stats['false_count'] = len(self.dataframe) - self.stats.get('true_count', 0) - self.dataframe[self.column_name].isnull().sum()
        return self.stats['false_count']
    
    def find_most_common_value(self):
        """
        Identifies the most common value (mode) in the boolean column.

        Returns:
            bool or None: The most common value or None if no mode.
        """
        mode = self.dataframe[self.column_name].mode()
        self.stats['mode'] = mode[0] if not mode.empty else None
        return self.stats['mode']
    
    def calculate_missing_prevalence(self) -> float:
        """
        Calculates the prevalence of missing (null) values in the boolean column.

        Returns:
            float: The prevalence of missing values.
        """
        self.stats['missing_prevalence'] = self.dataframe[self.column_name].isnull().mean()
        return self.stats['missing_prevalence']
    
    def calculate_empty_prevalence(self) -> float:
        """
        Calculates the prevalence of empty values in the boolean column.

        Returns:
            float: The prevalence of empty values.
        """
        # For a boolean column, empty values are treated as null.
        self.stats['empty_prevalence'] = self.stats.get('missing_prevalence', self.dataframe[self.column_name].isnull().mean())
        return self.stats['empty_prevalence']
    
    def check_trivial_column(self) -> bool:
        """
        Checks if the boolean column is trivial (all identical or null).

        Returns:
            bool: True if trivial, False otherwise.
        """
        non_null_values = self.dataframe[self.column_name].dropna().unique()
        return len(non_null_values) <= 1

    def handle_missing_infinite_data(self):
        """
        Processes the column to handle missing or infinite data accordingly.

        Raises:
            Warning if infinite values are detected.
        """
        # Infinite values in a boolean column don't logically exist, 
        # but if any numerical transformations lead to them, we should address it.
        if self.dataframe[self.column_name].isin([float('inf'), float('-inf')]).any():
            raise Warning("Infinite values detected, which are invalid in a boolean column.")



def display_statistics(stats: dict, format: str = 'dictionary'):
    """
    Displays the descriptive statistics of a boolean column in the specified format.

    Args:
        stats (dict): A dictionary containing the descriptive statistics, such as 'mean', 'true_count',
                      'false_count', 'mode', 'missing_prevalence', 'empty_prevalence'.
        format (str): The format in which to display the statistics. Options: 'dictionary', 'json', 'table'.

    Raises:
        ValueError: If the format is unsupported or if the stats dictionary is invalid.
    """
    if not isinstance(stats, dict):
        raise ValueError("Stats must be provided as a dictionary.")

    required_keys = {'mean', 'true_count', 'false_count', 'mode', 'missing_prevalence', 'empty_prevalence'}
    if not required_keys.issubset(stats.keys()):
        raise ValueError(f"Stats dictionary is missing required keys: {required_keys - stats.keys()}")

    if format == 'dictionary':
        print(stats)
    elif format == 'json':
        print(json.dumps(stats, indent=4))
    elif format == 'table':
        try:
            from tabulate import tabulate
            table = [(key, value) for key, value in stats.items()]
            print(tabulate(table, headers=["Statistic", "Value"], tablefmt="grid"))
        except ImportError:
            raise ImportError("Please install the 'tabulate' library to use the 'table' format.")
    else:
        raise ValueError(f"Unsupported format: {format}. Choose 'dictionary', 'json', or 'table'.")
