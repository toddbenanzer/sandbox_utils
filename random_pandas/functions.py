andom
import string
import pandas as pd
import numpy as np

def generate_random_float(start, end):
    """
    Generate a random float value within a specified range.

    Args:
        start (float): The start value of the range.
        end (float): The end value of the range.

    Returns:
        float: A random float value within the specified range.
    """
    return random.uniform(start, end)

def generate_random_integer(start, end):
    """
    Generate a random integer value within a specified range.

    Args:
        start (int): The start value of the range.
        end (int): The end value of the range.

    Returns:
        int: A random integer value within the specified range.
    """
    return random.randint(start, end)

def generate_random_boolean():
    """
    Generate a random boolean value.

    Returns:
    bool: Randomly generated boolean value
    """
    return random.choice([True, False])

def generate_random_categorical(categories):
    """
    Generate a random categorical value from a given list of categories.

    Args:
        categories (list): A list of categories.

    Returns:
        str: A randomly selected category from the list.
    """
    return random.choice(categories)

def generate_random_string(length):
    """
    Generate a random string of specified length.

    Parameters:
        length (int): The desired length of the random string.

    Returns:
        str: The randomly generated string.
    """
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for _ in range(length))

def create_trivial_field(data_type, value):
    """
    Create a trivial field with a single value.

    Args:
        data_type (str): The data type of the field. Supported types: 'float', 'int', 'bool', 'str'.
        value: The single value to be used for the field.

    Returns:
        pandas.Series: A pandas Series object representing the trivial field.
    """
    if data_type == 'float':
        return pd.Series(value, dtype=np.float64)
    elif data_type == 'int':
        return pd.Series(value, dtype=np.int64)
    elif data_type == 'bool':
        return pd.Series(value, dtype=bool)
    elif data_type == 'str':
        return pd.Series(value, dtype=str)
    else:
        raise ValueError("Invalid data type. Supported types: 'float', 'int', 'bool', 'str'")

def create_missing_fields(data, null_ratio=0.1):
    """
    Create missing fields with null or None values in the given pandas DataFrame.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        null_ratio (float): The ratio of missing fields to be created. Defaults to 0.1.

    Returns:
        pd.DataFrame: The modified DataFrame with missing fields.
    """
    # Determine the number of missing fields to create
    num_missing_fields = int(len(data.columns) * null_ratio)

    # Randomly select columns to contain missing fields
    cols_with_missing = np.random.choice(data.columns, size=num_missing_fields, replace=False)

    # Create missing fields with null or None values in the selected columns
    for col in cols_with_missing:
        data[col] = np.where(np.random.rand(len(data)) < null_ratio, None, data[col])

    return data

def generate_random_data(n, include_inf_nan=False):
    """
    Generate random data of specified length.

    Args:
        n (int): The desired length of the random data.
        include_inf_nan (bool): Whether to include random inf and nan values. Defaults to False.

    Returns:
        pd.DataFrame: The randomly generated DataFrame.
    """
    # Generate random data
    data = pd.DataFrame({
        'float_data': np.random.rand(n),
        'integer_data': np.random.randint(0, 10, n),
        'boolean_data': np.random.choice([True, False], n),
        'categorical_data': np.random.choice(['A', 'B', 'C'], n),
        'string_data': [''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 5)) for _ in range(n)]
    })

    if include_inf_nan:
        # Include random inf and nan values in the generated data
        data.loc[np.random.choice(range(n), int(n/10)), 'float_data'] = np.inf
        data.loc[np.random.choice(range(n), int(n/10)), 'float_data'] = -np.inf
        data.loc[np.random.choice(range(n), int(n/10)), 'float_data'] = np.nan

    return dat