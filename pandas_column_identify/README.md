# DataTypeDetector Documentation

## Class: DataTypeDetector

### Description
The `DataTypeDetector` class is designed to determine the most likely data type of a specified column in a pandas DataFrame. It facilitates the identification of data types such as string, integer, float, date, datetime, boolean, or categorical.

### Initialization

#### `__init__(self, dataframe: pd.DataFrame)`
Initializes the `DataTypeDetector` with the provided DataFrame.

- **Parameters:**
  - `dataframe` (`pd.DataFrame`): The DataFrame that will be analyzed for column data types.

### Methods

#### `detect_column_type(self, column_name: str) -> Optional[str]`
Analyzes the specified column in the DataFrame to determine its most likely data type.

- **Parameters:**
  - `column_name` (`str`): The name of the column to analyze.

- **Returns:**
  - `Optional[str]`: 
    - The detected data type of the column, such as:
      - `'string'`
      - `'integer'`
      - `'float'`
      - `'date'`
      - `'datetime'`
      - `'boolean'`
      - `'categorical'`
    - Returns `None` if the column is null or trivial (contains no significant data).

### Data Type Checking Methods

#### `is_string(self, column) -> bool`
Checks if all values in the given column are strings or null.

- **Parameters:**
  - `column`: A pandas Series representing the column to check.
  
- **Returns:** `bool` - True if all values are strings or null, otherwise False.

#### `is_integer(self, column) -> bool`
Checks if all values in the given column are integers.

- **Parameters:**
  - `column`: A pandas Series representing the column to check.

- **Returns:** `bool` - True if all values are integers, otherwise False.

#### `is_float(self, column) -> bool`
Checks if all values in the given column are floats.

- **Parameters:**
  - `column`: A pandas Series representing the column to check.

- **Returns:** `bool` - True if all values are floats, otherwise False.

#### `is_date(self, column) -> bool`
Checks if the values in the column can be classified as dates (logic to be implemented).

- **Parameters:**
  - `column`: A pandas Series representing the column to check.

- **Returns:** `bool` - Placeholder return value, to be implemented for date-type detection.

#### `is_datetime(self, column) -> bool`
Checks if the values in the column can be classified as datetimes (logic to be implemented).

- **Parameters:**
  - `column`: A pandas Series representing the column to check.

- **Returns:** `bool` - Placeholder return value, to be implemented for datetime-type detection.

#### `is_boolean(self, column) -> bool`
Checks if all values in the given column are booleans.

- **Parameters:**
  - `column`: A pandas Series representing the column to check.

- **Returns:** `bool` - True if all values are booleans, otherwise False.

#### `is_categorical(self, column) -> bool`
Checks if the given column can be classified as categorical based on the proportion of unique values.

- **Parameters:**
  - `column`: A pandas Series representing the column to check.

- **Returns:** `bool` - True if the column is categorical based on the defined threshold, otherwise False.


# Function: is_string

## Description
The `is_string` function evaluates a given pandas Series (column) to determine if all non-null values are of the string data type.

## Parameters

- **column** (`pd.Series`): 
  - The pandas Series representing the column to be evaluated.

## Returns

- **bool**: 
  - Returns `True` if all non-null values in the column are strings. Returns `False` otherwise.

## Behavior
- The function filters out null values during evaluation, ensuring only non-null values are checked for their data type.
- It is optimized for use with large Series and aims to provide efficient performance.


# Function: is_integer

## Description
The `is_integer` function evaluates a given pandas Series (column) to determine if all non-null values are of the integer data type.

## Parameters

- **column** (`pd.Series`): 
  - The pandas Series representing the column to be evaluated.

## Returns

- **bool**: 
  - Returns `True` if all non-null values in the column are integers. Returns `False` otherwise.

## Behavior
- The function handles null values by dropping them during evaluation, ensuring only non-null values are checked for their data type.
- It converts values to numeric and checks if they are integers, allowing it to correctly recognize numbers that might be represented as floats but are visually whole numbers.


# Function: is_float

## Description
The `is_float` function evaluates a given pandas Series (column) to determine if all non-null values are of the float data type.

## Parameters

- **column** (`pd.Series`): 
  - The pandas Series representing the column to be evaluated.

## Returns

- **bool**: 
  - Returns `True` if all non-null values in the column are floats. Returns `False` otherwise.

## Behavior
- The function handles null values by dropping them during evaluation, ensuring only non-null values are checked for their data type.
- It accurately identifies values that are strictly floats (i.e., not integers) by checking if they cannot be represented as integers.


# Function: is_date

## Description
The `is_date` function evaluates a given pandas Series (column) to determine if all non-null values can be classified as dates.

## Parameters

- **column** (`pd.Series`): 
  - The pandas Series representing the column to be evaluated.

## Returns

- **bool**: 
  - Returns `True` if all non-null values in the column are recognized as dates. Returns `False` otherwise.

## Behavior
- The function handles null values by dropping them during evaluation, ensuring only non-null values are checked for their data type.
- It attempts to convert values to a date format and checks if they have no time component, thus ensuring that only date entries (and not datetime entries) are considered valid.
- It effectively handles various edge cases, including invalid date strings and empty Series.


# Function: is_datetime

## Description
The `is_datetime` function evaluates a given pandas Series (column) to determine if all non-null values can be classified as datetime objects.

## Parameters

- **column** (`pd.Series`): 
  - The pandas Series representing the column to be evaluated.

## Returns

- **bool**: 
  - Returns `True` if all non-null values in the column are recognized as datetime objects. Returns `False` otherwise.

## Behavior
- The function handles null values by dropping them during evaluation, ensuring only non-null values are checked for their data type.
- It attempts to convert values to datetime objects and checks if the conversion is successful, ensuring that all entries in the Series can be treated as datetimes.
- The function effectively manages various edge cases, including invalid datetime strings and empty Series, maintaining robustness across different input scenarios.


# Function: is_boolean

## Description
The `is_boolean` function evaluates a given pandas Series (column) to determine if all non-null values are of the boolean data type.

## Parameters

- **column** (`pd.Series`): 
  - The pandas Series representing the column to be evaluated.

## Returns

- **bool**: 
  - Returns `True` if all non-null values in the column are booleans. Returns `False` otherwise.

## Behavior
- The function handles null values by dropping them during evaluation, ensuring only non-null values are checked for their data type.
- It accurately detects boolean values, which can be represented as `True` or `False` in Python.
- The function is efficient and maintains performance with large datasets, providing reliable results even for extensive columns.


# Function: is_categorical

## Description
The `is_categorical` function evaluates a given pandas Series (column) to determine if the values in the column can be classified as categorical data. Categorical data typically has a limited set of possible values, often repeated across many rows.

## Parameters

- **column** (`pd.Series`): 
  - The pandas Series representing the column to be evaluated.

- **threshold** (`float`, optional): 
  - The threshold ratio of unique values to total non-null values to be considered categorical. The default value is 0.05.

## Returns

- **bool**: 
  - Returns `True` if the column is identified as categorical based on the unique to total value ratio being below the specified threshold. Returns `False` otherwise.

## Behavior
- The function handles null values by dropping them during evaluation, ensuring only non-null values are used when calculating the unique value ratio.
- It efficiently processes large datasets, maintaining performance for extensive columns.
- The function can be adjusted with a custom threshold to fine-tune the classification of categorical data according to user needs.


# Function: handle_missing_values

## Description
The `handle_missing_values` function processes a given pandas Series (column) to manage and handle missing data entries using specified strategies.

## Parameters

- **column** (`pd.Series`): 
  - The pandas Series representing the column in which missing values need to be managed.

- **strategy** (`str`, optional): 
  - The strategy used to handle missing values. Available options include:
    - `'mean'`: Fill missing values with the mean of the column.
    - `'median'`: Fill missing values with the median of the column.
    - `'mode'`: Fill missing values with the mode of the column.
    - `'drop'`: Remove entries with missing values entirely.
    - `'constant'`: Fill missing values with a specified constant value (requires `fill_value` parameter).
  - The default strategy is `'mean'`.

- **fill_value**: 
  - The value with which to fill missing values when using the `'constant'` strategy. Default is `None`.

## Returns

- **pd.Series**: 
  - A pandas Series with missing values handled according to the specified strategy.

## Behavior
- The function identifies and addresses missing values (typically represented as `NaN` or `None` in a pandas Series).
- It ensures that appropriate strategies are utilized efficiently, handling columns with potentially large datasets.
- The function raises an error if an unsupported strategy is specified or when the `'constant'` strategy is used without a `fill_value`.


# Function: handle_infinite_values

## Description
The `handle_infinite_values` function processes a given pandas Series (column) to manage and handle infinite data entries using specified strategies.

## Parameters

- **column** (`pd.Series`): 
  - The pandas Series representing the column in which infinite values need to be managed.

- **strategy** (`str`, optional): 
  - The strategy used to handle infinite values. Available options include:
    - `'nan'`: Convert infinite values to NaN.
    - `'replace'`: Replace infinite values with a specified finite value (requires `replacement_value`).
    - `'drop'`: Remove entries that contain infinite values.
  - The default strategy is `'nan'`.

- **replacement_value** (`float`, optional): 
  - The value with which to replace infinite values when using the `'replace'` strategy. Default is `None`.

## Returns

- **pd.Series**: 
  - A pandas Series with infinite values handled according to the specified strategy.

## Behavior
- The function identifies infinite values typically represented as `np.inf` or `-np.inf` in the pandas Series.
- It provides flexibility in handling these values, ensuring data integrity while maintaining performance across potentially large datasets.
- If an unsupported strategy is specified or if the `'replace'` strategy is chosen without a `replacement_value`, the function raises a corresponding error.


# Function: check_null_trivial_columns

## Description
The `check_null_trivial_columns` function evaluates a given pandas Series (column) to determine if it is null (contains all missing or null values) or trivial (lacks variety or significant information due to monotonous data).

## Parameters

- **column** (`pd.Series`): 
  - The pandas Series representing the column to be evaluated for null or trivial content.

- **uniqueness_threshold** (`float`, optional): 
  - The threshold below which a column is considered trivial based on its uniqueness ratio. The default value is 0.01 (1%).

## Returns

- **bool**: 
  - Returns `True` if the column is deemed null or trivial. Returns `False` otherwise.

## Behavior
- The function identifies columns where all entries are missing or null, typically represented as `NaN` or `None`.
- It also detects trivial columns, where:
  - All values are identical or nearly identical.
  - There is a lack of variance, providing insignificant or redundant information.
- It is efficient and capable of processing large datasets, offering reliable evaluations for extensive columns.
- The function accommodates a variety of data types, including strings, numbers, and booleans.
