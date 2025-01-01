from your_module_name import DataTypeDetector
from your_module_name import check_null_trivial_columns
from your_module_name import handle_infinite_values
from your_module_name import handle_missing_values
from your_module_name import is_boolean
from your_module_name import is_categorical
from your_module_name import is_date
from your_module_name import is_datetime
from your_module_name import is_float
from your_module_name import is_integer
from your_module_name import is_string
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'integers': [1, 2, 3, 4, 5],
        'floats': [1.1, 2.2, 3.3, 4.4, 5.5],
        'strings': ['a', 'b', 'c', 'd', 'e'],
        'booleans': [True, False, True, True, False],
        'dates': pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01']),
        'categorical': pd.Series(['cat', 'dog', 'cat', 'bird', 'dog'], dtype='category'),
        'nulls': [None, None, None, None, None]
    })

def test_detect_integer_type(sample_dataframe):
    detector = DataTypeDetector(sample_dataframe)
    assert detector.detect_column_type('integers') == 'integer'

def test_detect_float_type(sample_dataframe):
    detector = DataTypeDetector(sample_dataframe)
    assert detector.detect_column_type('floats') == 'float'

def test_detect_string_type(sample_dataframe):
    detector = DataTypeDetector(sample_dataframe)
    assert detector.detect_column_type('strings') == 'string'

def test_detect_boolean_type(sample_dataframe):
    detector = DataTypeDetector(sample_dataframe)
    assert detector.detect_column_type('booleans') == 'boolean'

def test_detect_date_type(sample_dataframe):
    detector = DataTypeDetector(sample_dataframe)
    # Assuming is_date and is_datetime methods are correctly implemented
    assert detector.detect_column_type('dates') == 'date'

def test_detect_categorical_type(sample_dataframe):
    detector = DataTypeDetector(sample_dataframe)
    assert detector.detect_column_type('categorical') == 'categorical'

def test_detect_null_column(sample_dataframe):
    detector = DataTypeDetector(sample_dataframe)
    assert detector.detect_column_type('nulls') is None



def test_is_string_all_strings():
    column = pd.Series(['apple', 'banana', 'cherry'])
    assert is_string(column) == True

def test_is_string_with_nulls():
    column = pd.Series(['apple', None, 'cherry'])
    assert is_string(column) == True

def test_is_string_mixed_types():
    column = pd.Series(['apple', 1, 'cherry'])
    assert is_string(column) == False

def test_is_string_all_nulls():
    column = pd.Series([None, None, None])
    assert is_string(column) == False

def test_is_string_empty_series():
    column = pd.Series([])
    assert is_string(column) == True



def test_is_integer_all_integers():
    column = pd.Series([1, 2, 3, 4, 5])
    assert is_integer(column) == True

def test_is_integer_with_nulls():
    column = pd.Series([1, None, 3, 4, 5])
    assert is_integer(column) == True

def test_is_integer_mixed_types():
    column = pd.Series([1, 2, 'a', 4.5, 5])
    assert is_integer(column) == False

def test_is_integer_all_floats():
    column = pd.Series([1.1, 2.2, 3.3])
    assert is_integer(column) == False

def test_is_integer_all_nulls():
    column = pd.Series([None, None, None])
    assert is_integer(column) == False

def test_is_integer_empty_series():
    column = pd.Series([])
    assert is_integer(column) == True



def test_is_float_all_floats():
    column = pd.Series([1.1, 2.2, 3.3])
    assert is_float(column) == True

def test_is_float_with_nulls():
    column = pd.Series([1.1, None, 3.3])
    assert is_float(column) == True

def test_is_float_mixed_types():
    column = pd.Series([1.1, 2, 3.3])
    assert is_float(column) == False

def test_is_float_all_integers():
    column = pd.Series([1, 2, 3])
    assert is_float(column) == False

def test_is_float_all_nulls():
    column = pd.Series([None, None, None])
    assert is_float(column) == False

def test_is_float_empty_series():
    column = pd.Series([])
    assert is_float(column) == True



def test_is_date_all_dates():
    column = pd.Series(['2020-01-01', '2021-02-15', '2019-12-30'])
    assert is_date(column) == True

def test_is_date_with_nulls():
    column = pd.Series(['2020-01-01', None, '2019-12-30'])
    assert is_date(column) == True

def test_is_date_mixed_date_and_time():
    column = pd.Series(['2020-01-01', '2019-12-30 10:00:00'])
    assert is_date(column) == False

def test_is_date_invalid_dates():
    column = pd.Series(['2020-01-01', 'not a date', '2019-12-30'])
    assert is_date(column) == False

def test_is_date_all_nulls():
    column = pd.Series([None, None, None])
    assert is_date(column) == False

def test_is_date_empty_series():
    column = pd.Series([])
    assert is_date(column) == True



def test_is_datetime_all_datetimes():
    column = pd.Series(['2020-01-01 10:00:00', '2021-02-15 15:30:00', '2019-12-30 08:45:00'])
    assert is_datetime(column) == True

def test_is_datetime_with_nulls():
    column = pd.Series(['2020-01-01 10:00:00', None, '2019-12-30 08:45:00'])
    assert is_datetime(column) == True

def test_is_datetime_mixed_date_and_time():
    column = pd.Series(['2020-01-01', '2019-12-30 10:00:00'])
    assert is_datetime(column) == True

def test_is_datetime_invalid_datetimes():
    column = pd.Series(['2020-01-01 10:00:00', 'not a datetime', '2019-12-30 08:45:00'])
    assert is_datetime(column) == False

def test_is_datetime_all_nulls():
    column = pd.Series([None, None, None])
    assert is_datetime(column) == False

def test_is_datetime_empty_series():
    column = pd.Series([])
    assert is_datetime(column) == True



def test_is_boolean_all_bools():
    column = pd.Series([True, False, True])
    assert is_boolean(column) == True

def test_is_boolean_with_nulls():
    column = pd.Series([True, None, False])
    assert is_boolean(column) == True

def test_is_boolean_mixed_types():
    column = pd.Series([True, 0, False])
    assert is_boolean(column) == False

def test_is_boolean_all_integers_ones_zeros():
    column = pd.Series([1, 0, 1])
    assert is_boolean(column) == False

def test_is_boolean_all_nulls():
    column = pd.Series([None, None, None])
    assert is_boolean(column) == False

def test_is_boolean_empty_series():
    column = pd.Series([])
    assert is_boolean(column) == True



def test_is_categorical_basic():
    column = pd.Series(['cat', 'cat', 'dog', 'dog', 'bird'])
    assert is_categorical(column) == True

def test_is_categorical_with_nulls():
    column = pd.Series(['cat', None, 'dog', 'dog', 'bird'])
    assert is_categorical(column) == True

def test_is_not_categorical_many_unique():
    column = pd.Series(['cat', 'dog', 'bird', 'fish', 'lizard'])
    assert is_categorical(column) == False

def test_is_categorical_specified_threshold():
    column = pd.Series(['cat', 'cat', 'dog', 'dog', 'dog', 'dog', 'bird'])
    assert is_categorical(column, threshold=0.2) == True

def test_is_not_categorical_with_custom_threshold():
    column = pd.Series(['cat', 'dog', 'bird', 'bird', 'bird', 'fish'])
    assert is_categorical(column, threshold=0.1) == False

def test_is_not_categorical_all_unique():
    column = pd.Series([1, 2, 3, 4, 5, 6])
    assert is_categorical(column) == False

def test_is_categorical_all_nulls():
    column = pd.Series([None, None, None])
    assert is_categorical(column) == False

def test_is_categorical_empty_series():
    column = pd.Series([])
    assert is_categorical(column) == False



def test_mean_strategy():
    column = pd.Series([1, 2, None, 4])
    result = handle_missing_values(column, strategy='mean')
    expected = pd.Series([1, 2, 2.333333, 4])
    pd.testing.assert_series_equal(result, expected)

def test_median_strategy():
    column = pd.Series([1, 2, None, 4])
    result = handle_missing_values(column, strategy='median')
    expected = pd.Series([1, 2, 2, 4])
    pd.testing.assert_series_equal(result, expected)

def test_mode_strategy():
    column = pd.Series([1, 1, None, 4])
    result = handle_missing_values(column, strategy='mode')
    expected = pd.Series([1, 1, 1, 4])
    pd.testing.assert_series_equal(result, expected)

def test_drop_strategy():
    column = pd.Series([1, 2, None, 4])
    result = handle_missing_values(column, strategy='drop')
    expected = pd.Series([1, 2, 4]).reset_index(drop=True)
    pd.testing.assert_series_equal(result.reset_index(drop=True), expected)

def test_constant_strategy():
    column = pd.Series([1, 2, None, 4])
    result = handle_missing_values(column, strategy='constant', fill_value=0)
    expected = pd.Series([1, 2, 0, 4])
    pd.testing.assert_series_equal(result, expected)

def test_invalid_strategy():
    column = pd.Series([1, 2, None, 4])
    with pytest.raises(ValueError, match="Unsupported strategy. Choose from 'mean', 'median', 'mode', 'drop', or 'constant'."):
        handle_missing_values(column, strategy='invalid')

def test_missing_fill_value_for_constant():
    column = pd.Series([1, 2, None, 4])
    with pytest.raises(ValueError, match="A fill_value must be provided when using the 'constant' strategy."):
        handle_missing_values(column, strategy='constant')



def test_nan_strategy():
    column = pd.Series([1, np.inf, 3, -np.inf])
    result = handle_infinite_values(column, strategy='nan')
    expected = pd.Series([1, np.nan, 3, np.nan])
    pd.testing.assert_series_equal(result, expected)

def test_replace_strategy():
    column = pd.Series([1, np.inf, 3, -np.inf])
    result = handle_infinite_values(column, strategy='replace', replacement_value=0)
    expected = pd.Series([1, 0, 3, 0])
    pd.testing.assert_series_equal(result, expected)

def test_drop_strategy():
    column = pd.Series([1, np.inf, 3, -np.inf, 4])
    result = handle_infinite_values(column, strategy='drop')
    expected = pd.Series([1, 3, 4]).reset_index(drop=True)
    pd.testing.assert_series_equal(result.reset_index(drop=True), expected)

def test_invalid_strategy():
    column = pd.Series([1, np.inf, 3, -np.inf])
    with pytest.raises(ValueError, match="Unsupported strategy. Choose from 'nan', 'replace', or 'drop'."):
        handle_infinite_values(column, strategy='invalid')

def test_missing_replacement_value():
    column = pd.Series([1, np.inf, 3, -np.inf])
    with pytest.raises(ValueError, match="A replacement_value must be provided when using the 'replace' strategy."):
        handle_infinite_values(column, strategy='replace')



def test_null_column():
    column = pd.Series([None, None, None])
    assert check_null_trivial_columns(column) == True

def test_non_null_non_trivial_column():
    column = pd.Series([1, 2, 3, 4, 5])
    assert check_null_trivial_columns(column) == False

def test_trivial_column_with_identical_values():
    column = pd.Series([5, 5, 5, 5, 5])
    assert check_null_trivial_columns(column) == True

def test_column_with_one_unique_value_and_nulls():
    column = pd.Series([None, 1, None, 1, None])
    assert check_null_trivial_columns(column) == True

def test_column_with_high_uniqueness():
    column = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert check_null_trivial_columns(column) == False

def test_custom_uniqueness_threshold():
    column = pd.Series([1, 1, 1, 2, 2, 2])
    assert check_null_trivial_columns(column, uniqueness_threshold=0.5) == False
