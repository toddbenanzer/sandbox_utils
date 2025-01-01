# DataGenerator Documentation

## Overview
`DataGenerator` is a Python class designed to facilitate the creation of random data for testing purposes. The class supports generating various data types, including floats, integers, booleans, categorical values, and strings. It also provides functionality to introduce missing values and special numeric values like `inf` and `nan`.

## Class: DataGenerator

### Initialization

#### `__init__(self, num_records: int)`
Initializes the DataGenerator with the specified number of records.

- **Parameters:**
  - `num_records` (int): The number of records to generate.

### Methods

#### `generate_float_column(self, min_value: float = 0.0, max_value: float = 1.0) -> List[float]`
Generates a column of random float values.

- **Parameters:**
  - `min_value` (float): Minimum value of the float range (default is 0.0).
  - `max_value` (float): Maximum value of the float range (default is 1.0).

- **Returns:**
  - List of random floats.

#### `generate_integer_column(self, min_value: int = 0, max_value: int = 100) -> List[int]`
Generates a column of random integer values.

- **Parameters:**
  - `min_value` (int): Minimum value of the integer range (default is 0).
  - `max_value` (int): Maximum value of the integer range (default is 100).

- **Returns:**
  - List of random integers.

#### `generate_boolean_column(self, true_probability: float = 0.5) -> List[bool]`
Generates a column of random boolean values.

- **Parameters:**
  - `true_probability` (float): Probability of choosing True (default is 0.5).

- **Returns:**
  - List of random booleans.

#### `generate_categorical_column(self, categories: List[Any]) -> List[Any]`
Generates a column of random categorical values.

- **Parameters:**
  - `categories` (List[Any]): List of categories to choose from.

- **Returns:**
  - List of random categorical values.

#### `generate_string_column(self, length: int = 10) -> List[str]`
Generates a column of random strings.

- **Parameters:**
  - `length` (int): The length of each generated string (default is 10).

- **Returns:**
  - List of random strings.

#### `generate_single_value_column(self, value: Any) -> List[Any]`
Generates a column filled with a single specified value.

- **Parameters:**
  - `value` (Any): The value to fill the column with.

- **Returns:**
  - List where all elements are the specified value.

#### `generate_missing_values(self, data: List[Any], percentage: float) -> List[Any]`
Introduces missing values into a given data column.

- **Parameters:**
  - `data` (List): The data column to modify.
  - `percentage` (float): The percentage of values to set as missing.

- **Returns:**
  - The modified column with missing values.

#### `include_inf_nan(self, data: List[float], inf_percentage: float, nan_percentage: float) -> List[float]`
Introduces `inf` and `nan` values into a given data column.

- **Parameters:**
  - `data` (List[float]): The data column to modify.
  - `inf_percentage` (float): The percentage of values to set as `inf`.
  - `nan_percentage` (float): The percentage of values to set as `nan`.

- **Returns:**
  - The modified column with `inf` and `nan` values.

#### `to_dataframe(self) -> pd.DataFrame`
Compiles all generated columns into a pandas DataFrame.

- **Returns:**
  - A `pandas.DataFrame` containing all generated data.


# set_seed Documentation

## Overview
The `set_seed` function is used to set the seed for random number generation in both the built-in Python `random` module and the NumPy library. This ensures that the sequence of generated random numbers is reproducible across different runs of the program.

## Function: set_seed

### Parameters

- `seed` (int): 
  - The seed value for the random number generator. This integer sets the initial state of the random number generator, which determines the sequence of subsequent random numbers generated.

### Returns
- None

### Usage
By calling `set_seed` with a fixed seed value, you can guarantee that the same sequence of random numbers will be produced every time the program is run with that seed. This is particularly useful for debugging, testing, and sharing results, where reproducibility is essential.

### Example



# validate_parameters Documentation

## Overview
The `validate_parameters` function is designed to validate a set of input parameters to ensure they adhere to expected types and value constraints. This is crucial for maintaining data integrity and preventing errors in subsequent operations that rely on these parameters.

## Function: validate_parameters

### Parameters

- `params` (dict):
  - A dictionary where keys are parameter names and values are the corresponding parameter values to be validated.

### Raises
- **ValueError**: 
  - Raised when a parameter does not meet the expected criteria, such as being outside the acceptable range or being logically inconsistent with other parameters (e.g., `min_value` greater than `max_value`).
  
- **TypeError**: 
  - Raised when a parameter is of an incorrect type, such as a string instead of an integer or a float.

### Usage
This function should be called before processing any values in order to ensure that the parameters are robust and do not lead to runtime errors. It serves as a safeguard to validate critical information before it is utilized in computations or data generation.

### Example

