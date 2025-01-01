from typing import List, Any
import numpy as np
import pandas as pd
import random
import string


class DataGenerator:
    def __init__(self, num_records: int):
        """
        Initialize the DataGenerator with the specified number of records.
        
        Args:
            num_records (int): The number of records to generate.
        """
        self.num_records = num_records
        self.data = {}

    def generate_float_column(self, min_value: float = 0.0, max_value: float = 1.0) -> List[float]:
        """
        Generate a column of random float values.

        Args:
            min_value (float): Minimum value of the float range.
            max_value (float): Maximum value of the float range.

        Returns:
            List[float]: A list of random floats.
        """
        return [random.uniform(min_value, max_value) for _ in range(self.num_records)]

    def generate_integer_column(self, min_value: int = 0, max_value: int = 100) -> List[int]:
        """
        Generate a column of random integer values.

        Args:
            min_value (int): Minimum value of the integer range.
            max_value (int): Maximum value of the integer range.

        Returns:
            List[int]: A list of random integers.
        """
        return [random.randint(min_value, max_value) for _ in range(self.num_records)]

    def generate_boolean_column(self, true_probability: float = 0.5) -> List[bool]:
        """
        Generate a column of random boolean values.

        Args:
            true_probability (float): Probability of choosing True.

        Returns:
            List[bool]: A list of random booleans.
        """
        return [random.random() < true_probability for _ in range(self.num_records)]

    def generate_categorical_column(self, categories: List[Any]) -> List[Any]:
        """
        Generate a column of random categorical values.

        Args:
            categories (List[Any]): List of categories to choose from.

        Returns:
            List[Any]: A list of random categorical values.
        """
        return [random.choice(categories) for _ in range(self.num_records)]

    def generate_string_column(self, length: int = 10) -> List[str]:
        """
        Generate a column of random strings.

        Args:
            length (int): The length of each generated string.

        Returns:
            List[str]: A list of random strings.
        """
        return [''.join(random.choices(string.ascii_letters + string.digits, k=length)) for _ in range(self.num_records)]

    def generate_single_value_column(self, value: Any) -> List[Any]:
        """
        Generate a column filled with a single specified value.

        Args:
            value (Any): The value to fill the column with.

        Returns:
            List[Any]: A list where all elements are the specified value.
        """
        return [value for _ in range(self.num_records)]

    def generate_missing_values(self, data: List[Any], percentage: float) -> List[Any]:
        """
        Introduce missing values into a given data column.

        Args:
            data (List): The data column to modify.
            percentage (float): The percentage of values to set as missing.

        Returns:
            List[Any]: The modified column with missing values.
        """
        num_missing = int(self.num_records * percentage)
        missing_indices = random.sample(range(self.num_records), num_missing)
        
        for idx in missing_indices:
            data[idx] = None
        
        return data

    def include_inf_nan(self, data: List[float], inf_percentage: float, nan_percentage: float) -> List[float]:
        """
        Introduce 'inf' and 'nan' values into a given data column.

        Args:
            data (List[float]): The data column to modify.
            inf_percentage (float): The percentage of values to set as 'inf'.
            nan_percentage (float): The percentage of values to set as 'nan'.

        Returns:
            List[float]: The modified column with 'inf' and 'nan' values.
        """
        num_inf = int(self.num_records * inf_percentage)
        inf_indices = random.sample(range(self.num_records), num_inf)
        
        for idx in inf_indices:
            data[idx] = float('inf')
        
        num_nan = int(self.num_records * nan_percentage)
        nan_indices = random.sample(range(self.num_records), num_nan)
        
        for idx in nan_indices:
            data[idx] = float('nan')
        
        return data

    def to_dataframe(self) -> pd.DataFrame:
        """
        Compile all generated columns into a pandas DataFrame.

        Returns:
            pd.DataFrame: The DataFrame containing all generated data.
        """
        return pd.DataFrame(self.data)



def set_seed(seed: int) -> None:
    """
    Set the seed for random number generation to ensure reproducibility.

    Args:
        seed (int): The seed value for the random number generator.
    """
    random.seed(seed)
    np.random.seed(seed)


def validate_parameters(params: dict) -> None:
    """
    Validate the input parameters to ensure they meet the expected criteria.

    Args:
        params (dict): A dictionary with parameter names as keys and parameter values as values.

    Raises:
        ValueError: If a parameter does not meet the expected criteria.
        TypeError: If a parameter is of an incorrect type.
    """
    for key, value in params.items():
        if key == 'num_records':
            if not isinstance(value, int):
                raise TypeError(f"Parameter '{key}' must be an integer.")
            if value <= 0:
                raise ValueError(f"Parameter '{key}' must be greater than zero.")

        elif key in ['min_value', 'max_value']:
            if not isinstance(value, (int, float)):
                raise TypeError(f"Parameter '{key}' must be a number.")
            if 'min_value' in params and 'max_value' in params:
                if params['min_value'] > params['max_value']:
                    raise ValueError("Parameter 'min_value' cannot be greater than 'max_value'.")

        elif key == 'true_probability':
            if not isinstance(value, float):
                raise TypeError(f"Parameter '{key}' must be a float.")
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"Parameter '{key}' must be between 0.0 and 1.0.")

        elif key == 'categories':
            if not isinstance(value, list):
                raise TypeError(f"Parameter '{key}' must be a list.")
            if not value:
                raise ValueError(f"Parameter '{key}' must not be an empty list.")

        elif key == 'length':
            if not isinstance(value, int):
                raise TypeError(f"Parameter '{key}' must be an integer.")
            if value <= 0:
                raise ValueError(f"Parameter '{key}' must be greater than zero.")

        elif key in ['inf_percentage', 'nan_percentage', 'percentage']:
            if not isinstance(value, float):
                raise TypeError(f"Parameter '{key}' must be a float.")
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"Parameter '{key}' must be between 0.0 and 1.0.")

        else:
            raise ValueError(f"Unknown parameter '{key}'.")

