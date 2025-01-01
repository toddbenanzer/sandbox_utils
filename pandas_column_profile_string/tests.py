from string_statistics import StringStatistics
from string_statistics import check_trivial_column
from string_statistics import get_descriptive_statistics
from string_statistics import handle_missing_and_infinite
from string_statistics import validate_column
import numpy as np
import pandas as pd
import pytest


def test_calculate_mode():
    data = pd.Series(["apple", "banana", "apple", "orange", "banana", "banana"])
    stats = StringStatistics(data)
    assert stats.calculate_mode() == ["banana"]

def test_calculate_missing_prevalence():
    data = pd.Series(["apple", None, "banana", None, "banana"])
    stats = StringStatistics(data)
    assert stats.calculate_missing_prevalence() == 40.0

def test_calculate_empty_prevalence():
    data = pd.Series(["", "apple", "", "banana", "banana"])
    stats = StringStatistics(data)
    assert stats.calculate_empty_prevalence() == 40.0

def test_calculate_min_length():
    data = pd.Series(["apple", "kiwi", "banana", "pear"])
    stats = StringStatistics(data)
    assert stats.calculate_min_length() == 4

def test_calculate_max_length():
    data = pd.Series(["apple", "kiwi", "banana", "pear"])
    stats = StringStatistics(data)
    assert stats.calculate_max_length() == 6

def test_calculate_avg_length():
    data = pd.Series(["apple", "kiwi", "banana", "pear"])
    stats = StringStatistics(data)
    assert pytest.approx(stats.calculate_avg_length(), 0.1) == 4.75

def test_calculate_mode_multiple_modes():
    data = pd.Series(["apple", "banana", "apple", "orange", "banana"])
    stats = StringStatistics(data)
    assert set(stats.calculate_mode()) == set(["apple", "banana"])

def test_empty_column():
    data = pd.Series(["", "", "", ""])
    stats = StringStatistics(data)
    assert stats.calculate_missing_prevalence() == 0.0
    assert stats.calculate_empty_prevalence() == 100.0
    assert stats.calculate_min_length() == 0
    assert stats.calculate_max_length() == 0
    assert stats.calculate_avg_length() == 0.0



def test_valid_column():
    valid_data = pd.Series(["apple", "banana", "orange"])
    try:
        validate_column(valid_data)
    except ValueError as e:
        pytest.fail(f"Unexpected ValueError: {e}")

def test_non_series_input():
    with pytest.raises(ValueError, match="Input must be a pandas Series."):
        validate_column(["not", "a", "series"])

def test_non_string_series():
    non_string_data = pd.Series([1, 2, 3])
    with pytest.raises(ValueError, match="Series must contain string values."):
        validate_column(non_string_data)

def test_empty_series():
    empty_data = pd.Series([], dtype=object)
    with pytest.raises(ValueError, match="Series is empty or contains only null/empty values."):
        validate_column(empty_data)

def test_null_and_empty_values():
    null_empty_data = pd.Series([None, "", "  ", None])
    with pytest.raises(ValueError, match="Series is empty or contains only null/empty values."):
        validate_column(null_empty_data)



def test_handle_missing_and_infinite_with_nan():
    data = pd.DataFrame({"A": [1, np.nan, 3], "B": [np.nan, 5, np.nan]})
    result = handle_missing_and_infinite(data)
    expected = pd.DataFrame({"A": [1, 1, 3], "B": [5, 5, 5]})
    pd.testing.assert_frame_equal(result, expected)

def test_handle_missing_and_infinite_with_inf():
    data = pd.DataFrame({"A": [1, np.inf, 3], "B": [-np.inf, 5, 6]})
    result = handle_missing_and_infinite(data)
    expected = pd.DataFrame({"A": [1, 1, 3], "B": [5, 5, 6]})
    pd.testing.assert_frame_equal(result, expected)

def test_handle_missing_and_infinite_with_nan_and_inf():
    data = pd.DataFrame({"A": [1, np.inf, np.nan, 3], "B": [np.nan, -np.inf, 5, 6]})
    result = handle_missing_and_infinite(data)
    expected = pd.DataFrame({"A": [1, 1, 3, 3], "B": [5, 5, 5, 6]})
    pd.testing.assert_frame_equal(result, expected)

def test_handle_missing_and_infinite_no_nan_or_inf():
    data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    result = handle_missing_and_infinite(data)
    expected = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    pd.testing.assert_frame_equal(result, expected)

def test_invalid_input():
    with pytest.raises(ValueError, match="Input must be a pandas DataFrame."):
        handle_missing_and_infinite(["not", "a", "dataframe"])



def test_check_trivial_with_single_value():
    data = pd.Series(["same", "same", "same"])
    assert check_trivial_column(data) is True

def test_check_trivial_with_empty_strings():
    data = pd.Series(["", "", ""])
    assert check_trivial_column(data) is True

def test_check_trivial_with_mixed_strings():
    data = pd.Series(["same", "different", "same"])
    assert check_trivial_column(data) is False

def test_check_trivial_with_nan_values():
    data = pd.Series([None, None, None])
    assert check_trivial_column(data) is True

def test_check_trivial_with_stripped_empty_values():
    data = pd.Series([" ", "  ", "\t"])
    assert check_trivial_column(data) is True

def test_check_trivial_with_single_non_empty():
    data = pd.Series(["non-empty", "", ""])
    assert check_trivial_column(data) is False

def test_invalid_input():
    with pytest.raises(ValueError, match="Input must be a pandas Series."):
        check_trivial_column(["not", "a", "series"])



def test_get_descriptive_statistics_basic():
    dataframe = pd.DataFrame({
        "fruits": ["apple", "banana", "apple", "orange", "banana", "banana"]
    })
    stats = get_descriptive_statistics(dataframe, "fruits")
    assert stats["mode"] == ["banana"]
    assert stats["missing_prevalence"] == 0.0
    assert stats["empty_prevalence"] == 0.0
    assert stats["min_length"] == 5
    assert stats["max_length"] == 6
    assert stats["avg_length"] == 5.666666666666667
    assert stats["trivial_column"] is False

def test_get_descriptive_statistics_with_nan():
    dataframe = pd.DataFrame({
        "fruits": ["apple", None, "banana", None, "banana"]
    })
    stats = get_descriptive_statistics(dataframe, "fruits")
    assert stats["missing_prevalence"] == 40.0
    assert stats["trivial_column"] is False

def test_get_descriptive_statistics_trivial_column():
    dataframe = pd.DataFrame({
        "trivial": ["same", "same", "same"]
    })
    stats = get_descriptive_statistics(dataframe, "trivial")
    assert stats["trivial_column"] is True

def test_get_descriptive_statistics_invalid_column():
    dataframe = pd.DataFrame({
        "fruits": ["apple", "banana", "apple"]
    })
    with pytest.raises(ValueError, match="Column 'vegetables' not found in DataFrame."):
        get_descriptive_statistics(dataframe, "vegetables")

def test_get_descriptive_statistics_empty_dataframe():
    dataframe = pd.DataFrame()
    with pytest.raises(ValueError, match="Column 'fruits' not found in DataFrame."):
        get_descriptive_statistics(dataframe, "fruits")
