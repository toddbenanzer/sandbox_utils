from boolean_stats import BooleanDescriptiveStats
from display_function import display_statistics
from io import StringIO
from unittest.mock import patch
import pandas as pd
import pytest


def test_validate_column_valid():
    df = pd.DataFrame({'bool_col': [True, False, True, False]})
    stats = BooleanDescriptiveStats(df, 'bool_col')
    assert stats.validate_column() == True

def test_validate_column_invalid_type():
    df = pd.DataFrame({'bool_col': [1, 0, 1, 0]})
    with pytest.raises(ValueError, match="is not of boolean type"):
        BooleanDescriptiveStats(df, 'bool_col')

def test_validate_column_not_exists():
    df = pd.DataFrame({'bool_col': [True, False, True]})
    with pytest.raises(ValueError, match="not found in DataFrame"):
        BooleanDescriptiveStats(df, 'non_existent_col')

def test_validate_column_trivial():
    df = pd.DataFrame({'bool_col': [True, True, True]})
    with pytest.raises(ValueError, match="is trivial"):
        BooleanDescriptiveStats(df, 'bool_col')

def test_calculate_mean():
    df = pd.DataFrame({'bool_col': [True, False, True, False, True]})
    stats = BooleanDescriptiveStats(df, 'bool_col')
    assert stats.calculate_mean() == 0.6

def test_calculate_true_count():
    df = pd.DataFrame({'bool_col': [True, False, True, False, True]})
    stats = BooleanDescriptiveStats(df, 'bool_col')
    assert stats.calculate_true_count() == 3

def test_calculate_false_count():
    df = pd.DataFrame({'bool_col': [True, False, True, False, True, pd.NA]})
    stats = BooleanDescriptiveStats(df, 'bool_col')
    assert stats.calculate_false_count() == 2

def test_find_most_common_value():
    df = pd.DataFrame({'bool_col': [True, True, False, False, True]})
    stats = BooleanDescriptiveStats(df, 'bool_col')
    assert stats.find_most_common_value() == True

def test_calculate_missing_prevalence():
    df = pd.DataFrame({'bool_col': [True, False, None, True, False, None]})
    stats = BooleanDescriptiveStats(df, 'bool_col')
    assert stats.calculate_missing_prevalence() == 0.3333333333333333

def test_calculate_empty_prevalence():
    df = pd.DataFrame({'bool_col': [True, False, None, True]})
    stats = BooleanDescriptiveStats(df, 'bool_col')
    assert stats.calculate_empty_prevalence() == 0.25

def test_check_trivial_column():
    df = pd.DataFrame({'bool_col': [True, True, True]})
    stats = BooleanDescriptiveStats(df, 'bool_col')
    assert stats.check_trivial_column() == True

def test_handle_missing_infinite_data_warning():
    df = pd.DataFrame({'bool_col': [True, float('inf'), True]})
    stats = BooleanDescriptiveStats(df, 'bool_col')
    with pytest.raises(Warning, match="Infinite values"):
        stats.handle_missing_infinite_data()



def test_display_statistics_dictionary_format():
    stats = {'mean': 0.5, 'true_count': 10, 'false_count': 10, 'mode': True, 'missing_prevalence': 0.1, 'empty_prevalence': 0.1}
    with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
        display_statistics(stats, 'dictionary')
        output = mock_stdout.getvalue().strip()
    assert output == str(stats)

def test_display_statistics_json_format():
    stats = {'mean': 0.5, 'true_count': 10, 'false_count': 10, 'mode': True, 'missing_prevalence': 0.1, 'empty_prevalence': 0.1}
    with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
        display_statistics(stats, 'json')
        output = mock_stdout.getvalue().strip()
    expected_output = json.dumps(stats, indent=4)
    assert output == expected_output

def test_display_statistics_table_format():
    stats = {'mean': 0.5, 'true_count': 10, 'false_count': 10, 'mode': True, 'missing_prevalence': 0.1, 'empty_prevalence': 0.1}
    try:
        from tabulate import tabulate
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            display_statistics(stats, 'table')
            output = mock_stdout.getvalue().strip()
        table = [(key, value) for key, value in stats.items()]
        expected_output = tabulate(table, headers=["Statistic", "Value"], tablefmt="grid")
        assert output == expected_output
    except ImportError:
        pytest.fail("Tabulate library is not installed.")

def test_display_statistics_invalid_format():
    stats = {'mean': 0.5, 'true_count': 10, 'false_count': 10, 'mode': True, 'missing_prevalence': 0.1, 'empty_prevalence': 0.1}
    with pytest.raises(ValueError, match="Unsupported format: xml. Choose 'dictionary', 'json', or 'table'."):
        display_statistics(stats, 'xml')

def test_display_statistics_missing_key():
    stats = {'mean': 0.5, 'true_count': 10}
    with pytest.raises(ValueError, match="Stats dictionary is missing required keys:"):
        display_statistics(stats, 'dictionary')

def test_display_statistics_non_dict():
    stats = ['mean', 0.5]
    with pytest.raises(ValueError, match="Stats must be provided as a dictionary."):
        display_statistics(stats, 'dictionary')
