from typing import Optional
import numpy as np
import pandas as pd


class DataTypeDetector:
    """
    A class to detect the most likely data type of a specified column in a pandas DataFrame.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initializes the DataTypeDetector with a DataFrame.

        Args:
            dataframe (pd.DataFrame): The DataFrame to analyze.
        """
        self.dataframe = dataframe

    def detect_column_type(self, column_name: str) -> Optional[str]:
        """
        Analyzes the specified column in the DataFrame to determine its most likely data type.

        Args:
            column_name (str): The name of the column to analyze.

        Returns:
            Optional[str]: The detected data type of the column, such as 'string', 'integer', 
                           'float', 'date', 'datetime', 'boolean', 'categorical', or None if 
                           the column is null or trivial.
        """
        column = self.dataframe[column_name]
        
        if column.isnull().all():
            return None  # or "null", depending on how you want to handle completely null columns

        # Placeholders for how each type could be detected
        if self.is_string(column):
            return 'string'
        elif self.is_integer(column):
            return 'integer'
        elif self.is_float(column):
            return 'float'
        elif self.is_date(column):
            return 'date'
        elif self.is_datetime(column):
            return 'datetime'
        elif self.is_boolean(column):
            return 'boolean'
        elif self.is_categorical(column):
            return 'categorical'
        else:
            return None

    def is_string(self, column):
        return column.apply(lambda x: isinstance(x, str) or pd.isna(x)).all()

    def is_integer(self, column):
        return pd.to_numeric(column, errors='coerce').dropna().apply(float.is_integer).all()

    def is_float(self, column):
        return pd.to_numeric(column, errors='coerce').dropna().apply(lambda x: not float.is_integer(x)).all()

    def is_date(self, column):
        # Implement logic to check for date type
        return False

    def is_datetime(self, column):
        # Implement logic to check for datetime type
        return False

    def is_boolean(self, column):
        return column.dropna().apply(lambda x: isinstance(x, bool)).all()

    def is_categorical(self, column):
        return column.dropna().nunique() / len(column.dropna()) < 0.05  # Example threshold



def is_string(column: pd.Series) -> bool:
    """
    Determines if all non-null values in the given pandas Series are strings.

    Args:
        column (pd.Series): The column to evaluate.

    Returns:
        bool: True if all non-null values are strings, otherwise False.
    """
    # Filter out null values and check if all remaining values are strings
    return column.dropna().apply(lambda x: isinstance(x, str)).all()



def is_integer(column: pd.Series) -> bool:
    """
    Determines if all non-null values in the given pandas Series are integers.

    Args:
        column (pd.Series): The column to evaluate.

    Returns:
        bool: True if all non-null values are integers, otherwise False.
    """
    # Drop null values and check if all remaining values are integers
    return pd.to_numeric(column.dropna(), errors='coerce').apply(float.is_integer).all()



def is_float(column: pd.Series) -> bool:
    """
    Determines if all non-null values in the given pandas Series are floats.

    Args:
        column (pd.Series): The column to evaluate.

    Returns:
        bool: True if all non-null values are floats, otherwise False.
    """
    # Drop null values and check if all remaining values can be interpreted as floats
    non_null_values = pd.to_numeric(column.dropna(), errors='coerce')
    return non_null_values.apply(lambda x: isinstance(x, float) and not float.is_integer(x)).all()



def is_date(column: pd.Series) -> bool:
    """
    Determines if all non-null values in the given pandas Series are dates.

    Args:
        column (pd.Series): The column to evaluate.

    Returns:
        bool: True if all non-null values are dates, otherwise False.
    """
    # Drop null values
    non_null_column = column.dropna()

    # Try to convert to datetime; check if conversion is successful and no time component is present
    try:
        parsed_dates = pd.to_datetime(non_null_column, errors='coerce', format='%Y-%m-%d')
        return all(parsed_dates.notna()) and all(parsed_dates.dt.time == pd.Timestamp('00:00:00').time())
    except (ValueError, TypeError):
        return False



def is_datetime(column: pd.Series) -> bool:
    """
    Determines if all non-null values in the given pandas Series are datetime objects.

    Args:
        column (pd.Series): The column to evaluate.

    Returns:
        bool: True if all non-null values are datetime objects, otherwise False.
    """
    # Drop null values
    non_null_column = column.dropna()

    # Try to convert to datetime and check if conversion is successful
    try:
        parsed_datetimes = pd.to_datetime(non_null_column, errors='coerce')
        return all(parsed_datetimes.notna())
    except (ValueError, TypeError):
        return False



def is_boolean(column: pd.Series) -> bool:
    """
    Determines if all non-null values in the given pandas Series are booleans.

    Args:
        column (pd.Series): The column to evaluate.

    Returns:
        bool: True if all non-null values are booleans, otherwise False.
    """
    # Drop null values and check if all remaining values are booleans
    return column.dropna().apply(lambda x: isinstance(x, bool)).all()



def is_categorical(column: pd.Series, threshold: float = 0.05) -> bool:
    """
    Determines if the given pandas Series is categorical based on the ratio of unique values to total values.

    Args:
        column (pd.Series): The column to evaluate.
        threshold (float): The threshold ratio of unique to total values to be considered categorical. Default is 0.05.

    Returns:
        bool: True if the column is considered categorical, otherwise False.
    """
    # Drop null values from the column
    non_null_column = column.dropna()

    # Calculate the ratio of unique values to non-null values
    unique_ratio = non_null_column.nunique() / len(non_null_column)

    # Determine if the column is categorical based on the threshold
    return unique_ratio < threshold



def handle_missing_values(column: pd.Series, strategy: str = 'mean', fill_value=None) -> pd.Series:
    """
    Handles missing values in a given pandas Series using the specified strategy.

    Args:
        column (pd.Series): The column for which missing values need to be managed.
        strategy (str): The strategy to handle missing values. Options include 'mean', 'median',
                        'mode', 'drop', and 'constant'. Default is 'mean'.
        fill_value: The value with which to fill missing values if strategy is 'constant'. Default is None.

    Returns:
        pd.Series: A pandas Series with missing values handled.
    """
    if strategy == 'mean':
        return column.fillna(column.mean())
    elif strategy == 'median':
        return column.fillna(column.median())
    elif strategy == 'mode':
        return column.fillna(column.mode()[0] if not column.mode().empty else column)
    elif strategy == 'drop':
        return column.dropna()
    elif strategy == 'constant':
        if fill_value is None:
            raise ValueError("A fill_value must be provided when using the 'constant' strategy.")
        return column.fillna(fill_value)
    else:
        raise ValueError("Unsupported strategy. Choose from 'mean', 'median', 'mode', 'drop', or 'constant'.")



def handle_infinite_values(column: pd.Series, strategy: str = 'nan', replacement_value: float = None) -> pd.Series:
    """
    Handles infinite values in a given pandas Series using the specified strategy.

    Args:
        column (pd.Series): The column for which infinite values need to be managed.
        strategy (str): The strategy to handle infinite values. Options include 'nan', 'replace', and 'drop'. Default is 'nan'.
        replacement_value (float): The value with which to replace infinite values if strategy is 'replace'. Default is None.

    Returns:
        pd.Series: A pandas Series with infinite values handled.
    """
    if strategy == 'nan':
        return column.replace([np.inf, -np.inf], np.nan)
    elif strategy == 'replace':
        if replacement_value is None:
            raise ValueError("A replacement_value must be provided when using the 'replace' strategy.")
        return column.replace([np.inf, -np.inf], replacement_value)
    elif strategy == 'drop':
        return column[~column.isin([np.inf, -np.inf])]
    else:
        raise ValueError("Unsupported strategy. Choose from 'nan', 'replace', or 'drop'.")



def check_null_trivial_columns(column: pd.Series, uniqueness_threshold: float = 0.01) -> bool:
    """
    Checks if a given pandas Series is null or trivial.

    Args:
        column (pd.Series): The column to evaluate.
        uniqueness_threshold (float): The threshold below which a column is considered trivial based on its 
                                      uniqueness ratio. Default is 0.01 (1%).

    Returns:
        bool: True if the column is null or trivial, otherwise False.
    """
    # Check if the column is completely NULL
    if column.isnull().all():
        return True

    # Check if the column is trivial (low uniqueness)
    non_null_column = column.dropna()
    unique_ratio = non_null_column.nunique() / len(non_null_column)
    if unique_ratio < uniqueness_threshold:
        return True

    return False
