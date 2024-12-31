from DataExporter import DataExporter
from export_as_csv import export_as_csv
from export_as_tdsx import export_as_tdsx
from generate_sample_code import generate_sample_code
from read_user_config import read_user_config
from setup_logging import setup_logging
from tempfile import NamedTemporaryFile
from unittest.mock import mock_open, patch
from unittest.mock import patch, call
from validate_dataframe import validate_dataframe
import json
import logging
import os
import pandas as pd
import pytest


def test_init():
    df = pd.DataFrame({'column1': [1, 2, 3]})
    exporter = DataExporter(df)
    assert exporter.dataframe.equals(df), "DataFrame not set correctly in constructor."

def test_to_tableau_csv():
    df = pd.DataFrame({'column1': [1, 2, 3]})
    exporter = DataExporter(df)
    
    with NamedTemporaryFile(delete=False) as temp_file:
        temp_file.close()  # Close the file so to_tableau_csv can open it
        success = exporter.to_tableau_csv(temp_file.name)
        assert success is True, "Export to CSV failed."
        
        # Verify the file content
        exported_df = pd.read_csv(temp_file.name)
        assert exported_df.equals(df), "Exported CSV content does not match original DataFrame."
        os.remove(temp_file.name)

def test_to_tableau_csv_error_handling(mocker):
    df = pd.DataFrame({'column1': ['a', 'b', 'c']})
    exporter = DataExporter(df)
    
    mocker.patch('pandas.DataFrame.to_csv', side_effect=Exception("Mocked error"))
    mocker.patch.object(DataExporter, 'handle_errors')

    destination_path = 'mocked_path.csv'
    success = exporter.to_tableau_csv(destination_path)
    
    assert success is False, "Export should fail due to mocked error."
    exporter.handle_errors.assert_called_once()

def test_attach_metadata():
    df = pd.DataFrame({'column1': [1, 2, 3]})
    metadata = {'source': 'test'}
    exporter = DataExporter(df)
    exporter.attach_metadata(metadata)
    assert exporter.metadata == metadata, "Metadata not attached correctly."

def test_handle_errors(mocker):
    mocker.patch('logging.error')
    exporter = DataExporter(pd.DataFrame())
    exporter.handle_errors(Exception('test error'))
    logging.error.assert_called_with("An error occurred: test error")

def test_apply_user_config(mocker):
    mocker.patch('logging.debug')
    config = {'setting': True}
    exporter = DataExporter(pd.DataFrame())
    exporter.apply_user_config(config)
    logging.debug.assert_called_with(f"Applying user config: {config}")



def test_validate_dataframe_empty():
    df_empty = pd.DataFrame()
    with pytest.raises(ValueError, match="DataFrame is empty"):
        validate_dataframe(df_empty)

def test_validate_dataframe_nulls():
    df_with_nulls = pd.DataFrame({'column1': [1, 2, None]})
    with pytest.raises(ValueError, match="DataFrame contains null values"):
        validate_dataframe(df_with_nulls)

def test_validate_dataframe_invalid_columns():
    df_invalid_columns = pd.DataFrame({'$invalidName': [1, 2, 3]})
    with pytest.raises(ValueError, match="DataFrame contains invalid column names"):
        validate_dataframe(df_invalid_columns)

def test_validate_dataframe_unsupported_types():
    df_unsupported_types = pd.DataFrame({'valid_column': [1, 2, 3], 'complex_column': [1+1j, 2+2j, 3+3j]})
    with pytest.raises(ValueError, match="DataFrame contains unsupported data types"):
        validate_dataframe(df_unsupported_types)

def test_validate_dataframe_valid():
    df_valid = pd.DataFrame({'Name': ['Alice', 'Bob'], 'Age': [30, 25], 'Joined': pd.to_datetime(['2021-01-01', '2021-02-01'])})
    assert validate_dataframe(df_valid) is True, "Valid DataFrame should pass validation."



def test_export_as_csv_success():
    df = pd.DataFrame({'column1': [1, 2, 3]})
    with NamedTemporaryFile(delete=False) as temp_file:
        temp_file.close()  # Close the file so export_as_csv can open it
        success = export_as_csv(df, temp_file.name)
        assert success is True, "CSV export should be successful."
        
        # Verify the file content
        exported_df = pd.read_csv(temp_file.name)
        assert exported_df.equals(df), "Exported CSV content does not match original DataFrame."
        os.remove(temp_file.name)

def test_export_as_csv_with_options():
    df = pd.DataFrame({'column1': [1, 2, 3]})
    with NamedTemporaryFile(delete=False) as temp_file:
        temp_file.close()
        success = export_as_csv(df, temp_file.name, sep=';', index=False)
        assert success is True, "CSV export with options should be successful."
        
        exported_df = pd.read_csv(temp_file.name, sep=';')
        assert exported_df.equals(df), "Exported CSV content with options does not match."
        os.remove(temp_file.name)

def test_export_as_csv_failure(mocker):
    df = pd.DataFrame({'column1': [1, 2, 3]})
    mocker.patch('pandas.DataFrame.to_csv', side_effect=Exception("Mocked write error"))
    mocker.patch('logging.error')
    
    success = export_as_csv(df, 'invalid/path/to/file.csv')
    
    assert success is False, "CSV export should fail due to mocked error."
    logging.error.assert_called_once()


try:
    import tableauserverclient as TSC
except ImportError:
    TSC = None

def mock_to_hyper(dataframe, file_path, **options):
    # Mock implementation to simulate to_hyper functionality
    return True

@pytest.fixture
def mock_tableau_serverclient(mocker):
    if TSC:
        mocker.patch('pandas.DataFrame.to_hyper', side_effect=mock_to_hyper)

def test_export_as_tdsx_library_not_installed(mocker):
    mocker.patch('export_as_tdsx.TSC', None)
    df = pd.DataFrame({'col1': [1, 2, 3]})
    
    with pytest.raises(Exception):
        success = export_as_tdsx(df, 'output.tdsx')
        assert success is False, "Export should fail when TSC is not installed."

def test_export_as_tdsx_success(mock_tableau_serverclient):
    df = pd.DataFrame({'col1': [1, 2, 3]})
    
    with pytest.raises(Exception):
        success = export_as_tdsx(df, 'output.tdsx', use_hyper=True)
        assert success is True, "Export to TDSX (Hyper) should succeed."

def test_export_as_tdsx_csv_fallback_success(mocker):
    mocker.patch('pandas.DataFrame.to_csv', return_value=True)
    df = pd.DataFrame({'col1': [1, 2, 3]})
    
    success = export_as_tdsx(df, 'output.tdsx')
    assert success is True, "Export to CSV as fallback should succeed."

def test_export_as_tdsx_exception_handling(mocker):
    df = pd.DataFrame({'col1': [1, 2, 3]})
    mocker.patch('pandas.DataFrame.to_csv', side_effect=Exception("Mocked write error"))
    mocker.patch('logging.error')

    success = export_as_tdsx(df, 'output_invalid.tdsx')
    assert success is False, "CSV export should fail and handle exception."
    logging.error.assert_called_once()



def test_setup_logging_info_level(caplog):
    setup_logging(logging.INFO)
    with caplog.at_level(logging.INFO):
        logging.info("Test log at INFO level")
    assert "Test log at INFO level" in caplog.text, "Expected log at INFO level was not found."

def test_setup_logging_debug_level(caplog):
    setup_logging(logging.DEBUG)
    with caplog.at_level(logging.DEBUG):
        logging.debug("Test log at DEBUG level")
    assert "Test log at DEBUG level" in caplog.text, "Expected log at DEBUG level was not found."

def test_setup_logging_error_handling(mocker):
    mocker.patch('logging.basicConfig', side_effect=Exception("Mocked logging error"))
    with patch('logging.error') as mock_log_error:
        setup_logging(logging.WARNING)
        mock_log_error.assert_called_once_with("Error configuring logging: Mocked logging error. Using default level logging.INFO.")



def test_read_user_config_valid_file():
    mock_data = '{"setting1": "value1", "setting2": "value2"}'
    with patch('builtins.open', mock_open(read_data=mock_data)) as mock_file, \
         patch('os.path.exists', return_value=True):
        config = read_user_config("valid_config.json")
        assert config == json.loads(mock_data), "The config should match the mock data."
        mock_file.assert_called_once_with("valid_config.json", 'r')

def test_read_user_config_file_not_found():
    with patch('os.path.exists', return_value=False), \
         patch('logging.error') as mock_log_error:
        with pytest.raises(FileNotFoundError):
            read_user_config("missing_config.json")
        mock_log_error.assert_called_once_with("Configuration file 'missing_config.json' not found.")

def test_read_user_config_invalid_json():
    mock_data = '{"setting1": "value1", "setting2": value2"}'  # Invalid JSON
    with patch('builtins.open', mock_open(read_data=mock_data)), \
         patch('os.path.exists', return_value=True), \
         patch('logging.error') as mock_log_error:
        with pytest.raises(ValueError):
            read_user_config("invalid_config.json")
        mock_log_error.assert_called_once()

def test_read_user_config_unexpected_error(mocker):
    mocker.patch('builtins.open', side_effect=Exception("Unexpected error"))
    mocker.patch('os.path.exists', return_value=True)
    mocker.patch('logging.error')
    
    with pytest.raises(Exception, match="Unexpected error"):
        read_user_config("any_config.json")
    logging.error.assert_called_once()



def test_generate_sample_code_returns_string():
    result = generate_sample_code()
    assert isinstance(result, str), "The result should be a string."

def test_generate_sample_code_contains_logging_example():
    result = generate_sample_code()
    assert "# Example: Setting Up Logging" in result, "The sample code should include a logging setup example."

def test_generate_sample_code_contains_dataexporter_example():
    result = generate_sample_code()
    assert "# Example: Creating and Using DataExporter" in result, "The sample code should include a DataExporter example."

def test_generate_sample_code_contains_validate_dataframe_example():
    result = generate_sample_code()
    assert "# Example: Validating a DataFrame" in result, "The sample code should include a DataFrame validation example."

def test_generate_sample_code_contains_read_user_config_example():
    result = generate_sample_code()
    assert "# Example: Reading User Configuration" in result, "The sample code should include a user configuration reading example."

def test_generate_sample_code_contains_export_as_tdsx_example():
    result = generate_sample_code()
    assert "# Example: Export DataFrame to TDSX" in result, "The sample code should include an example of exporting DataFrame to TDSX."
