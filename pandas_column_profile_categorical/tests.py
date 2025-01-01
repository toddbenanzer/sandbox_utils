from your_module import CategoricalStatsCalculator
from your_module import display_error_message
from your_module import handle_missing_and_infinite
from your_module import is_categorical
from your_module import validate_input
import numpy as np
import pandas as pd
import pytest


def test_calculate_frequency_distribution():
    data = {'Category': ['A', 'B', 'A', 'C', None, 'B', 'A']}
    df = pd.DataFrame(data)
    calculator = CategoricalStatsCalculator(df, 'Category')
    expected = {'A': 3, 'B': 2, 'C': 1, None: 1}
    result = calculator.calculate_frequency_distribution()
    assert result == expected

def test_find_most_common_values():
    data = {'Category': ['A', 'B', 'A', 'C', 'C', 'B', 'A']}
    df = pd.DataFrame(data)
    calculator = CategoricalStatsCalculator(df, 'Category')
    expected_single = ['A']
    expected_multiple = ['A', 'B']
    result_single = calculator.find_most_common_values()
    result_multiple = calculator.find_most_common_values(n=2)
    assert result_single == expected_single
    assert result_multiple == expected_multiple

def test_count_unique_values():
    data = {'Category': ['A', 'B', 'A', 'C', None, 'B', 'A']}
    df = pd.DataFrame(data)
    calculator = CategoricalStatsCalculator(df, 'Category')
    expected = 3  # 'A', 'B', 'C'
    result = calculator.count_unique_values()
    assert result == expected

def test_count_missing_values():
    data = {'Category': ['A', 'B', 'A', 'C', None, 'B', 'A']}
    df = pd.DataFrame(data)
    calculator = CategoricalStatsCalculator(df, 'Category')
    expected = 1  # One None value
    result = calculator.count_missing_values()
    assert result == expected

def test_identify_trivial_column():
    data_trivial = {'Category': ['A', 'A', 'A', 'A']}
    data_non_trivial = {'Category': ['A', 'B', 'A', 'B']}
    df_trivial = pd.DataFrame(data_trivial)
    df_non_trivial = pd.DataFrame(data_non_trivial)
    calculator_trivial = CategoricalStatsCalculator(df_trivial, 'Category')
    calculator_non_trivial = CategoricalStatsCalculator(df_non_trivial, 'Category')
    assert calculator_trivial.identify_trivial_column() is True
    assert calculator_non_trivial.identify_trivial_column() is False



def test_is_categorical_with_categorical_dtype():
    data = {'Category': pd.Categorical(['A', 'B', 'A', 'C'])}
    df = pd.DataFrame(data)
    assert is_categorical(df, 'Category') is True

def test_is_categorical_with_object_dtype():
    data = {'Category': ['A', 'B', 'A', 'C', 'A', 'B', 'A']}
    df = pd.DataFrame(data)
    assert is_categorical(df, 'Category') is True

def test_is_categorical_with_high_unique_ratio():
    data = {'Category': ['A', 'B', 'C', 'D', 'E', 'F', 'G']}
    df = pd.DataFrame(data)
    assert is_categorical(df, 'Category') is False

def test_is_categorical_with_non_existent_column():
    data = {'Category': ['A', 'B', 'A', 'C', 'A', 'B', 'A']}
    df = pd.DataFrame(data)
    with pytest.raises(ValueError):
        is_categorical(df, 'NonExistent')

def test_is_categorical_with_empty_dataframe():
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        is_categorical(df, 'Category')



def test_ignore_method():
    data = [1, 2, np.nan, np.inf]
    result = handle_missing_and_infinite(data, method='ignore')
    expected = pd.Series([1, 2, np.nan, np.inf], dtype=float)
    pd.testing.assert_series_equal(result, expected)

def test_remove_method():
    data = [1, 2, np.nan, np.inf]
    result = handle_missing_and_infinite(data, method='remove')
    expected = pd.Series([1, 2], dtype=float)
    pd.testing.assert_series_equal(result.reset_index(drop=True), expected)

def test_fill_method():
    data = [1, 2, np.nan, np.inf]
    result = handle_missing_and_infinite(data, method='fill')
    expected = [1, 2, 2, 2]  # median of [1, 2] is 2
    assert result == expected

def test_unsupported_method():
    data = [1, 2, np.nan, np.inf]
    with pytest.raises(ValueError):
        handle_missing_and_infinite(data, method='unknown')

def test_invalid_data_type():
    data = "invalid data type"
    with pytest.raises(TypeError):
        handle_missing_and_infinite(data, method='ignore')



def test_validate_input_with_valid_categorical_column():
    data = {'Category': ['A', 'B', 'C', 'A']}
    df = pd.DataFrame(data)
    assert validate_input(df, 'Category') is True

def test_validate_input_with_non_dataframe():
    data = "not a dataframe"
    with pytest.raises(TypeError):
        validate_input(data, 'Category')

def test_validate_input_with_non_existent_column():
    data = {'Category': ['A', 'B', 'C', 'A']}
    df = pd.DataFrame(data)
    with pytest.raises(ValueError):
        validate_input(df, 'NonExistentColumn')

def test_validate_input_with_non_categorical_column():
    data = {'Numbers': [1, 2, 3, 4]}
    df = pd.DataFrame(data)
    with pytest.raises(ValueError):
        validate_input(df, 'Numbers')



def test_display_error_message_valid_error_code_1():
    assert display_error_message('ERR001') == "The specified column does not exist in the DataFrame."

def test_display_error_message_valid_error_code_2():
    assert display_error_message('ERR002') == "The input is not a valid pandas DataFrame."

def test_display_error_message_valid_error_code_3():
    assert display_error_message('ERR003') == "The input column is not of a categorical data type."

def test_display_error_message_valid_error_code_unknown():
    assert display_error_message('ERR_UNKNOWN') == "An unknown error has occurred."

def test_display_error_message_invalid_error_code():
    assert display_error_message('ERR_INVALID') == "Unknown error code provided."
