from date_statistics import DateStatisticsCalculator  # Assume the class is in this module
from datetime import datetime
from mymodule import DataValidator  # Replace `mymodule` with the actual module name
from mymodule import load_dataframe  # Replace `mymodule` with the actual module name
from mymodule import save_descriptive_statistics  # Replace `mymodule` with the actual module name
from mymodule import setup_logging  # Replace `mymodule` with the actual module name
from unittest.mock import patch
import json
import logging
import numpy as np
import os
import pandas as pd
import pytest


def test_init_valid_column():
    data = pd.DataFrame({'date': pd.to_datetime(['2021-01-01', '2021-01-02'])})
    calculator = DateStatisticsCalculator(data, 'date')
    assert calculator.column_name == 'date'

def test_init_invalid_column():
    data = pd.DataFrame({'date': pd.to_datetime(['2021-01-01', '2021-01-02'])})
    with pytest.raises(ValueError):
        DateStatisticsCalculator(data, 'invalid')

def test_calculate_date_range():
    data = pd.DataFrame({'date': pd.to_datetime(['2021-01-01', '2021-01-04', '2021-01-03'])})
    calculator = DateStatisticsCalculator(data, 'date')
    result = calculator.calculate_date_range()
    assert result['min_date'] == pd.Timestamp('2021-01-01')
    assert result['max_date'] == pd.Timestamp('2021-01-04')

def test_count_distinct_dates():
    data = pd.DataFrame({'date': pd.to_datetime(['2021-01-01', '2021-01-01', '2021-01-02'])})
    calculator = DateStatisticsCalculator(data, 'date')
    assert calculator.count_distinct_dates() == 2

def test_find_most_common_dates():
    data = pd.DataFrame({'date': pd.to_datetime(['2021-01-01', '2021-01-01', '2021-01-02'])})
    calculator = DateStatisticsCalculator(data, 'date')
    result = calculator.find_most_common_dates(top_n=1)
    assert result == [(pd.Timestamp('2021-01-01'), 2)]

def test_calculate_missing_and_empty_values():
    data = pd.DataFrame({'date': [pd.NaT, '', pd.Timestamp('2021-01-02')]})
    calculator = DateStatisticsCalculator(data, 'date')
    result = calculator.calculate_missing_and_empty_values()
    assert result['missing_values_count'] == 1
    assert result['empty_values_count'] == 1

def test_check_trivial_column():
    data = pd.DataFrame({'date': pd.to_datetime(['2021-01-01', '2021-01-01', '2021-01-01'])})
    calculator = DateStatisticsCalculator(data, 'date')
    assert calculator.check_trivial_column() is True

def test_analyze_dates():
    data = pd.DataFrame({'date': [pd.NaT, '2021-01-01', pd.Timestamp('2021-01-02'), pd.Timestamp('2021-01-02')]})
    calculator = DateStatisticsCalculator(data, 'date')
    result = calculator.analyze_dates()
    assert result["date_range"] == {'min_date': pd.Timestamp('2021-01-01'), 'max_date': pd.Timestamp('2021-01-02')}
    assert result["distinct_count"] == 2
    assert result["most_common_dates"] == [(pd.Timestamp('2021-01-02'), 2)]
    assert result["missing_and_empty_count"] == {'missing_values_count': 1, 'empty_values_count': 0}
    assert result["is_trivial"] is False



def test_init_valid_column():
    data = pd.DataFrame({'values': [1, 2, 3]})
    validator = DataValidator(data, 'values')
    assert validator.column_name == 'values'

def test_init_invalid_column():
    data = pd.DataFrame({'values': [1, 2, 3]})
    with pytest.raises(ValueError):
        DataValidator(data, 'invalid')

def test_validate_date_column_with_valid_data():
    data = pd.DataFrame({'dates': pd.to_datetime(['2021-01-01', '2021-01-02'])})
    validator = DataValidator(data, 'dates')
    assert validator.validate_date_column() == True

def test_validate_date_column_with_invalid_data():
    data = pd.DataFrame({'dates': ['invalid_date', '2021-01-02']})
    validator = DataValidator(data, 'dates')
    assert validator.validate_date_column() == False

def test_handle_missing_values_drop():
    data = pd.DataFrame({'dates': [pd.NaT, pd.Timestamp('2022-01-01'), pd.NaT]})
    validator = DataValidator(data, 'dates')
    result = validator.handle_missing_values(strategy='drop')
    expected = pd.Series([pd.Timestamp('2022-01-01')])
    pd.testing.assert_series_equal(result.reset_index(drop=True), expected)

def test_handle_missing_values_fill():
    data = pd.DataFrame({'dates': [pd.NaT, pd.Timestamp('2022-01-01'), pd.NaT]})
    validator = DataValidator(data, 'dates')
    fill_date = pd.Timestamp('2022-01-02')
    result = validator.handle_missing_values(strategy='fill', fill_value=fill_date)
    expected = pd.Series([fill_date, pd.Timestamp('2022-01-01'), fill_date])
    pd.testing.assert_series_equal(result.reset_index(drop=True), expected)

def test_handle_infinite_values():
    data = pd.DataFrame({'values': [1, np.inf, -np.inf, 2]})
    validator = DataValidator(data, 'values')
    result = validator.handle_infinite_values()
    expected = pd.Series([1, np.nan, np.nan, 2])
    pd.testing.assert_series_equal(result.reset_index(drop=True), expected)



def test_load_dataframe_success(tmp_path):
    # Create a temporary CSV file
    file_path = tmp_path / "test.csv"
    data = "date,value\n2021-01-01,10\n2021-01-02,20"
    file_path.write_text(data)
    
    df = load_dataframe(str(file_path))
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'date' in df.columns
    assert 'value' in df.columns

def test_load_dataframe_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_dataframe("non_existent_file.csv")

def test_load_dataframe_empty_file(tmp_path):
    # Create an empty temporary CSV file
    file_path = tmp_path / "empty.csv"
    file_path.write_text("")
    
    with pytest.raises(ValueError, match="The provided CSV file is empty."):
        load_dataframe(str(file_path))

def test_load_dataframe_parser_error(tmp_path):
    # Create a temporary CSV file with faulty data
    file_path = tmp_path / "faulty.csv"
    file_path.write_text("date,value\n2021-01,10\n2021-01-02,20")

    # This will simulate a parser error if row length mismatch occurs
    with pytest.raises(ValueError, match="Parsing error: Unable to parse the CSV file."):
        load_dataframe(str(file_path))



def test_save_statistics_as_json(tmp_path):
    statistics = {"date_range": {"min_date": "2021-01-01", "max_date": "2021-12-31"}}
    output_path = tmp_path / "stats.json"
    save_descriptive_statistics(statistics, str(output_path))
    
    assert output_path.exists()
    with open(output_path, 'r') as json_file:
        data = json.load(json_file)
        assert data == statistics

def test_save_statistics_as_csv(tmp_path):
    statistics = {"date_range": {"min_date": "2021-01-01", "max_date": "2021-12-31"}}
    output_path = tmp_path / "stats.csv"
    save_descriptive_statistics(statistics, str(output_path))
    
    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert df['date_range.min_date'][0] == statistics['date_range']['min_date']
    assert df['date_range.max_date'][0] == statistics['date_range']['max_date']

def test_unsupported_file_extension(tmp_path):
    statistics = {"date_range": {"min_date": "2021-01-01", "max_date": "2021-12-31"}}
    output_path = tmp_path / "stats.txt"
    with pytest.raises(ValueError, match="Unsupported file extension. Please use .json or .csv."):
        save_descriptive_statistics(statistics, str(output_path))

def test_invalid_statistics_type():
    output_path = "dummy_path/stats.json"
    with pytest.raises(ValueError, match="Statistics must be provided as a dictionary."):
        save_descriptive_statistics(["Not a dictionary"], output_path)

def test_io_error(tmp_path, monkeypatch):
    statistics = {"date_range": {"min_date": "2021-01-01", "max_date": "2021-12-31"}}
    output_path = tmp_path / "stats.json"
    
    # Simulate IOError by patching open function
    with monkeypatch.context() as m:
        m.setattr("builtins.open", lambda *args, **kwargs: (_ for _ in ()).throw(IOError("Mock IOError")))
        with pytest.raises(IOError, match="Error writing to file"):
            save_descriptive_statistics(statistics, str(output_path))



def test_setup_logging_default_level(caplog):
    # Test the default logging level
    setup_logging()
    with caplog.at_level(logging.INFO):
        logging.info("This is an info message.")
        logging.debug("This is a debug message.")
    
    assert "This is an info message." in caplog.text
    assert "This is a debug message." not in caplog.text

def test_setup_logging_debug_level(caplog):
    # Test logging at DEBUG level
    setup_logging(level=logging.DEBUG)
    with caplog.at_level(logging.DEBUG):
        logging.info("This is an info message.")
        logging.debug("This is a debug message.")
    
    assert "This is an info message." in caplog.text
    assert "This is a debug message." in caplog.text

def test_setup_logging_no_duplicate_handlers():
    # Ensure no duplicate handlers are added
    logger = logging.getLogger()
    initial_handler_count = len(logger.handlers)
    setup_logging()
    setup_logging()
    post_handler_count = len(logger.handlers)
    
    assert initial_handler_count == post_handler_count

def test_setup_logging_custom_format(caplog):
    # Test a custom logging format (not directly set in the function but useful for testing changes)
    setup_logging(level=logging.INFO)
    with patch('logging.basicConfig') as mock_logging:
        setup_logging()
        assert mock_logging.called
        _, kwargs = mock_logging.call_args
        assert kwargs['format'] == '%(asctime)s - %(levelname)s - %(message)s'
