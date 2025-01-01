from your_module_name import DataTypeDetector
import numpy as np
import pandas as pd


# Example 1: Detecting integer types
data = {'integers': [1, 2, 3, 4, 5]}
df = pd.DataFrame(data)
detector = DataTypeDetector(df)
detected_type = detector.detect_column_type('integers')
print(f'Column "integers" detected type: {detected_type}')  # Expected: 'integer'

# Example 2: Detecting string types
data = {'strings': ['apple', 'banana', 'cherry']}
df = pd.DataFrame(data)
detector = DataTypeDetector(df)
detected_type = detector.detect_column_type('strings')
print(f'Column "strings" detected type: {detected_type}')  # Expected: 'string'

# Example 3: Detecting boolean types
data = {'booleans': [True, False, True]}
df = pd.DataFrame(data)
detector = DataTypeDetector(df)
detected_type = detector.detect_column_type('booleans')
print(f'Column "booleans" detected type: {detected_type}')  # Expected: 'boolean'

# Example 4: Detecting null columns
data = {'nulls': [None, None, None]}
df = pd.DataFrame(data)
detector = DataTypeDetector(df)
detected_type = detector.detect_column_type('nulls')
print(f'Column "nulls" detected type: {detected_type}')  # Expected: None

# Example 5: Detecting categorical types (Assuming logic for is_categorical is implemented)
data = {'category': pd.Series(['cat', 'cat', 'dog', 'dog', 'bird'], dtype='category')}
df = pd.DataFrame(data)
detector = DataTypeDetector(df)
detected_type = detector.detect_column_type('category')
print(f'Column "category" detected type: {detected_type}')  # Expected: 'categorical'



# Example 1: All values are strings
column1 = pd.Series(['apple', 'banana', 'cherry'])
print(is_string(column1))  # Expected output: True

# Example 2: Mixed types with at least one non-string
column2 = pd.Series(['apple', 1, 'cherry'])
print(is_string(column2))  # Expected output: False

# Example 3: All strings with some nulls
column3 = pd.Series(['apple', None, 'cherry'])
print(is_string(column3))  # Expected output: True

# Example 4: All null values
column4 = pd.Series([None, None, None])
print(is_string(column4))  # Expected output: False

# Example 5: Empty Series
column5 = pd.Series([])
print(is_string(column5))  # Expected output: True



# Example 1: All values are integers
column1 = pd.Series([10, 20, 30])
print(is_integer(column1))  # Expected output: True

# Example 2: Mixed types with at least one non-integer
column2 = pd.Series([10, 20.5, 30])
print(is_integer(column2))  # Expected output: False

# Example 3: All integers with some nulls
column3 = pd.Series([10, None, 30])
print(is_integer(column3))  # Expected output: True

# Example 4: All null values
column4 = pd.Series([None, None, None])
print(is_integer(column4))  # Expected output: False

# Example 5: Empty Series
column5 = pd.Series([])
print(is_integer(column5))  # Expected output: True



# Example 1: All values are floats
column1 = pd.Series([1.5, 2.3, 3.9])
print(is_float(column1))  # Expected output: True

# Example 2: Mixed types with at least one non-float
column2 = pd.Series([1.5, 2, 3.9])
print(is_float(column2))  # Expected output: False

# Example 3: All floats with some nulls
column3 = pd.Series([1.5, None, 3.9])
print(is_float(column3))  # Expected output: True

# Example 4: All integers
column4 = pd.Series([1, 2, 3])
print(is_float(column4))  # Expected output: False

# Example 5: Empty Series
column5 = pd.Series([])
print(is_float(column5))  # Expected output: True



# Example 1: All values are valid dates
column1 = pd.Series(['2020-01-01', '2021-12-31', '2019-06-30'])
print(is_date(column1))  # Expected output: True

# Example 2: Dates with some null values
column2 = pd.Series(['2020-01-01', None, '2019-06-30'])
print(is_date(column2))  # Expected output: True

# Example 3: Mixed dates and datetimes
column3 = pd.Series(['2020-01-01', '2019-06-30 12:30:00'])
print(is_date(column3))  # Expected output: False

# Example 4: Invalid date strings
column4 = pd.Series(['2020-01-01', 'not a date', '2019-06-30'])
print(is_date(column4))  # Expected output: False

# Example 5: Completely empty Series
column5 = pd.Series([])
print(is_date(column5))  # Expected output: True



# Example 1: All values are valid datetime strings
column1 = pd.Series(['2020-01-01 12:00:00', '2021-12-31 18:45:00', '2019-06-30 06:30:00'])
print(is_datetime(column1))  # Expected output: True

# Example 2: Datetime strings with some null values
column2 = pd.Series(['2020-01-01 12:00:00', None, '2019-06-30 06:30:00'])
print(is_datetime(column2))  # Expected output: True

# Example 3: Mixed date and datetime
column3 = pd.Series(['2020-01-01', '2019-06-30 12:30:00'])
print(is_datetime(column3))  # Expected output: True

# Example 4: Invalid datetime strings
column4 = pd.Series(['2020-01-01 12:00:00', 'not a datetime', '2019-06-30 06:30:00'])
print(is_datetime(column4))  # Expected output: False

# Example 5: Completely empty Series
column5 = pd.Series([])
print(is_datetime(column5))  # Expected output: True



# Example 1: All values are booleans
column1 = pd.Series([True, False, True])
print(is_boolean(column1))  # Expected output: True

# Example 2: Booleans with some null values
column2 = pd.Series([True, None, False])
print(is_boolean(column2))  # Expected output: True

# Example 3: Mixed boolean and integer types
column3 = pd.Series([True, 0, False])
print(is_boolean(column3))  # Expected output: False

# Example 4: Integer representation of booleans
column4 = pd.Series([1, 0, 1])
print(is_boolean(column4))  # Expected output: False

# Example 5: Completely empty Series
column5 = pd.Series([])
print(is_boolean(column5))  # Expected output: True



# Example 1: Column with repeated categories
column1 = pd.Series(['apple', 'apple', 'banana', 'banana', 'cherry'])
print(is_categorical(column1))  # Expected output: True

# Example 2: Column with more unique values
column2 = pd.Series(['apple', 'banana', 'cherry', 'date', 'elderberry'])
print(is_categorical(column2))  # Expected output: False

# Example 3: Column with categories and nulls
column3 = pd.Series(['red', 'red', None, 'blue', 'blue'])
print(is_categorical(column3))  # Expected output: True

# Example 4: Using a custom threshold
column4 = pd.Series(['dog', 'dog', 'cat', 'cat', 'bird', 'bird', 'fish'])
print(is_categorical(column4, threshold=0.4))  # Expected output: True

# Example 5: Empty Series
column5 = pd.Series([])
print(is_categorical(column5))  # Expected output: False



# Example 1: Fill missing values using the mean strategy
column1 = pd.Series([10, None, 30, 40])
print(handle_missing_values(column1, strategy='mean'))

# Example 2: Fill missing values using the median strategy
column2 = pd.Series([10, 20, None, 40])
print(handle_missing_values(column2, strategy='median'))

# Example 3: Fill missing values using the mode strategy
column3 = pd.Series([10, 10, None, 20, 20, 20])
print(handle_missing_values(column3, strategy='mode'))

# Example 4: Drop rows with missing values
column4 = pd.Series([10, None, 30, 40])
print(handle_missing_values(column4, strategy='drop'))

# Example 5: Fill missing values with a constant
column5 = pd.Series([10, None, 30, 40])
print(handle_missing_values(column5, strategy='constant', fill_value=0))



# Example 1: Replace infinite values with NaN
column1 = pd.Series([1, np.inf, 3, -np.inf, 5])
print(handle_infinite_values(column1, strategy='nan'))

# Example 2: Replace infinite values with a specific value
column2 = pd.Series([1, np.inf, 3, -np.inf, 5])
print(handle_infinite_values(column2, strategy='replace', replacement_value=0))

# Example 3: Drop rows with infinite values
column3 = pd.Series([1, np.inf, 3, -np.inf, 5])
print(handle_infinite_values(column3, strategy='drop'))



# Example 1: Column with all null values
column1 = pd.Series([None, None, None, None])
print(check_null_trivial_columns(column1))  # Expected output: True

# Example 2: Column with identical values
column2 = pd.Series([5, 5, 5, 5, 5])
print(check_null_trivial_columns(column2))  # Expected output: True

# Example 3: Column with mixed values and nulls, considered trivial
column3 = pd.Series([None, 1, None, 1, None])
print(check_null_trivial_columns(column3))  # Expected output: True

# Example 4: Column with diverse values, not trivial
column4 = pd.Series([1, 2, 3, 4, 5, 6])
print(check_null_trivial_columns(column4))  # Expected output: False

# Example 5: Using a custom uniqueness threshold
column5 = pd.Series([1, 1, 1, 2, 2, 2])
print(check_null_trivial_columns(column5, uniqueness_threshold=0.5))  # Expected output: False
