from data_analyzer import DataAnalyzer
from data_fetcher import DataFetcher
from data_preprocessor import DataPreprocessor
from data_visualizer import DataVisualizer
from io import StringIO
from load_config_function import load_config
from report_generator import ReportGenerator
from setup_environment_function import setup_environment
from setup_logging_function import setup_logging
from unittest.mock import patch
from unittest.mock import patch, MagicMock
from unittest.mock import patch, call
from validate_data_function import validate_data
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pytest
import requests
import sqlite3
import yaml


# Mock the requests.get function for API testing
@patch('data_fetcher.requests.get')
def test_fetch_data_api_success(mock_get):
    # Configure the mock to return a response with JSON data
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {'key': 'value'}
    
    fetcher = DataFetcher('API', {'url': 'http://fakeapi.com/data'})
    data = fetcher.fetch_data()
    
    assert data == {'key': 'value'}
    mock_get.assert_called_once_with('http://fakeapi.com/data', params={})

@patch('data_fetcher.requests.get')
def test_fetch_data_api_failure(mock_get):
    # Configure the mock to return a response with a status code 404
    mock_get.return_value.status_code = 404
    
    fetcher = DataFetcher('API', {'url': 'http://fakeapi.com/data'})
    
    with pytest.raises(ConnectionError):
        fetcher.fetch_data()

# Mock the sqlite3.connect function for database testing
@patch('data_fetcher.sqlite3.connect')
def test_fetch_data_database(mock_connect):
    # Setup a fake database connection and cursor
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = [(1, 'data')]
    
    fetcher = DataFetcher('database', {'dbname': 'test.db', 'query': 'SELECT * FROM table'})
    data = fetcher.fetch_data()
    
    assert isinstance(data, pd.DataFrame)
    mock_connect.assert_called_once_with('test.db')

# Test for CSV data fetching
@patch('data_fetcher.pd.read_csv')
def test_fetch_data_csv(mock_read_csv):
    # Setup the mock to return a DataFrame
    mock_df = pd.DataFrame({'col1': [1], 'col2': ['data']})
    mock_read_csv.return_value = mock_df
    
    fetcher = DataFetcher('CSV', {'file_path': 'test.csv'})
    data = fetcher.fetch_data()
    
    assert isinstance(data, pd.DataFrame)
    pd.testing.assert_frame_equal(data, mock_df)
    mock_read_csv.assert_called_once_with('test.csv')

def test_fetch_data_invalid_source():
    fetcher = DataFetcher('invalid', {})
    
    with pytest.raises(ValueError):
        fetcher.fetch_data()

@patch('data_fetcher.BackgroundScheduler')
def test_schedule_refresh(mock_scheduler):
    mock_sched = MagicMock()
    mock_scheduler.return_value = mock_sched
    
    fetcher = DataFetcher('CSV', {})
    fetcher.schedule_refresh(10)
    
    mock_sched.add_job.assert_called_once()

@patch('data_fetcher.BackgroundScheduler')
def test_stop_refresh(mock_scheduler):
    mock_sched = MagicMock()
    mock_scheduler.return_value = mock_sched
    
    fetcher = DataFetcher('CSV', {})
    fetcher.stop_refresh()
    
    mock_sched.shutdown.assert_called_once()



def test_handle_missing_values_drop():
    df = pd.DataFrame({"A": [1, np.nan, 3], "B": [4, 5, np.nan]})
    preprocessor = DataPreprocessor(df)
    result = preprocessor.handle_missing_values("drop")
    expected_result = pd.DataFrame({"A": [1], "B": [4]})
    pd.testing.assert_frame_equal(result, expected_result)

def test_handle_missing_values_fill_mean():
    df = pd.DataFrame({"A": [1, np.nan, 3], "B": [4, 5, np.nan]})
    preprocessor = DataPreprocessor(df)
    result = preprocessor.handle_missing_values("fill_mean")
    expected_result = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 4.5]})
    pd.testing.assert_frame_equal(result, expected_result)

def test_normalize_data_min_max():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    preprocessor = DataPreprocessor(df)
    result = preprocessor.normalize_data("min-max")
    expected_result = pd.DataFrame({"A": [0, 0.5, 1], "B": [0, 0.5, 1]})
    pd.testing.assert_frame_equal(result, expected_result)

def test_normalize_data_z_score():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    preprocessor = DataPreprocessor(df)
    result = preprocessor.normalize_data("z-score")
    expected_result = pd.DataFrame({"A": [-1.22474487, 0, 1.22474487], 
                                    "B": [-1.22474487, 0, 1.22474487]})
    pd.testing.assert_frame_equal(result, expected_result)

def test_transform_data_log():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [np.exp(1), np.exp(2), np.exp(3)]})
    preprocessor = DataPreprocessor(df)
    result = preprocessor.transform_data("log")
    expected_result = pd.DataFrame({"A": [0, np.log(2), np.log(3)], 
                                    "B": [1, 2, 3]})
    pd.testing.assert_frame_equal(result, expected_result)

def test_transform_data_sqrt():
    df = pd.DataFrame({"A": [0, 4, 9], "B": [16, 25, 36]})
    preprocessor = DataPreprocessor(df)
    result = preprocessor.transform_data("sqrt")
    expected_result = pd.DataFrame({"A": [0, 2, 3], "B": [4, 5, 6]})
    pd.testing.assert_frame_equal(result, expected_result)

def test_transform_data_encode_categorical():
    df = pd.DataFrame({"A": ["cat", "dog", "cat"]})
    preprocessor = DataPreprocessor(df)
    result = preprocessor.transform_data("encode_categorical")
    expected_result = pd.DataFrame({"A_cat": [1, 0, 1], "A_dog": [0, 1, 0]})
    pd.testing.assert_frame_equal(result, expected_result)

def test_handle_missing_values_unsupported_method():
    df = pd.DataFrame({"A": [1, np.nan, 3]})
    preprocessor = DataPreprocessor(df)
    with pytest.raises(ValueError):
        preprocessor.handle_missing_values("unsupported")

def test_normalize_data_unsupported_method():
    df = pd.DataFrame({"A": [1, 2, 3]})
    preprocessor = DataPreprocessor(df)
    with pytest.raises(ValueError):
        preprocessor.normalize_data("unsupported")

def test_transform_data_unsupported_transformation():
    df = pd.DataFrame({"A": [1, 2, 3]})
    preprocessor = DataPreprocessor(df)
    with pytest.raises(ValueError):
        preprocessor.transform_data("unsupported")



def test_descriptive_statistics():
    data = pd.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [5, 6, 7, 8, 9]
    })
    analyzer = DataAnalyzer(data)
    stats = analyzer.descriptive_statistics()

    expected_stats = {
        'mean': pd.Series({"A": 3.0, "B": 7.0}),
        'median': pd.Series({"A": 3.0, "B": 7.0}),
        'mode': pd.Series({"A": 1.0, "B": 5.0}),
        'std_dev': pd.Series({"A": 1.581139, "B": 1.581139})
    }

    assert stats['mean'].equals(expected_stats['mean'])
    assert stats['median'].equals(expected_stats['median'])
    assert stats['mode'].equals(expected_stats['mode'])
    assert np.allclose(stats['std_dev'], expected_stats['std_dev'], rtol=1e-5)

def test_exploratory_data_analysis():
    data = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [1, 2, 3],
        "C": [3, 2, 1]
    })
    analyzer = DataAnalyzer(data)
    correlation_matrix = analyzer.exploratory_data_analysis()

    expected_matrix = pd.DataFrame({
        "A": [1.0, 1.0, -1.0],
        "B": [1.0, 1.0, -1.0],
        "C": [-1.0, -1.0, 1.0]
    }, index=["A", "B", "C"])

    pd.testing.assert_frame_equal(correlation_matrix, expected_matrix)

def test_linear_regression_analysis():
    data = pd.DataFrame({
        "X": [1, 2, 3, 4, 5],
        "y": [2, 4, 6, 8, 10]
    })
    analyzer = DataAnalyzer(data)
    result = analyzer.regression_analysis("linear")
    
    assert np.allclose(result['coefficients'], [2.0], rtol=1e-2)
    assert np.isclose(result['intercept'], 0.0, atol=1e-5)
    assert np.isclose(result['r2_score'], 1.0, atol=1e-5)

def test_logistic_regression_analysis():
    data = pd.DataFrame({
        "X": [1, 2, 3, 4, 5],
        "y": [0, 0, 1, 1, 1]
    })
    analyzer = DataAnalyzer(data)
    result = analyzer.regression_analysis("logistic")
    
    assert 'coefficients' in result
    assert 'intercept' in result
    assert 'accuracy' in result
    assert 'confusion_matrix' in result

def test_unsupported_regression_model():
    data = pd.DataFrame({
        "X": [1, 2, 3],
        "y": [0, 1, 0]
    })
    analyzer = DataAnalyzer(data)
    
    with pytest.raises(ValueError):
        analyzer.regression_analysis("unsupported")



@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'Category': ['A', 'B', 'C'],
        'Values': [10, 20, 15]
    })

def test_generate_plot_bar(sample_data):
    visualizer = DataVisualizer(sample_data)
    visualizer.generate_plot('bar', x='Category', y='Values')
    assert visualizer.current_plot is not None
    assert isinstance(visualizer.current_plot, plt.Axes)

def test_generate_plot_unsupported_type(sample_data):
    visualizer = DataVisualizer(sample_data)
    with pytest.raises(ValueError):
        visualizer.generate_plot('unsupported_type')

def test_customize_plot(sample_data):
    visualizer = DataVisualizer(sample_data)
    visualizer.generate_plot('bar', x='Category', y='Values')
    visualizer.customize_plot(title='Custom Title', xlabel='X Axis', ylabel='Y Axis', grid=True)
    
    assert visualizer.current_plot.get_title() == 'Custom Title'
    assert visualizer.current_plot.get_xlabel() == 'X Axis'
    assert visualizer.current_plot.get_ylabel() == 'Y Axis'

def test_customize_plot_without_generating():
    visualizer = DataVisualizer(pd.DataFrame())
    with pytest.raises(RuntimeError):
        visualizer.customize_plot(title='Title')

def test_export_visualization(sample_data, tmpdir):
    visualizer = DataVisualizer(sample_data)
    visualizer.generate_plot('bar', x='Category', y='Values')
    file_path = tmpdir.join("test_plot.png")
    visualizer.export_visualization(format='png', file_name=str(file_path))
    
    assert file_path.isfile()
    assert file_path.ext == '.png'

def test_export_without_plot():
    visualizer = DataVisualizer(pd.DataFrame())
    with pytest.raises(RuntimeError):
        visualizer.export_visualization('png')



@pytest.fixture
def sample_analysis_results():
    return {
        "Summary": "This is a summary of the analysis.",
        "Conclusion": "This is the conclusion."
    }

def test_create_report_pdf(sample_analysis_results):
    generator = ReportGenerator(sample_analysis_results)
    file_path = generator.create_report('pdf')
    assert os.path.exists(file_path)
    assert file_path.endswith('.pdf')
    os.remove(file_path)

def test_create_report_html(sample_analysis_results):
    generator = ReportGenerator(sample_analysis_results)
    file_path = generator.create_report('html')
    assert os.path.exists(file_path)
    assert file_path.endswith('.html')
    os.remove(file_path)

def test_create_report_xlsx(sample_analysis_results):
    df = pd.DataFrame(sample_analysis_results, index=[0])
    generator = ReportGenerator(df)
    file_path = generator.create_report('xlsx')
    assert os.path.exists(file_path)
    assert file_path.endswith('.xlsx')
    os.remove(file_path)

def test_create_report_md(sample_analysis_results):
    generator = ReportGenerator(sample_analysis_results)
    file_path = generator.create_report('md')
    assert os.path.exists(file_path)
    assert file_path.endswith('.md')
    os.remove(file_path)

def test_unsupported_format(sample_analysis_results):
    generator = ReportGenerator(sample_analysis_results)
    with pytest.raises(ValueError):
        generator.create_report('unsupported')

def test_schedule_report(sample_analysis_results):
    generator = ReportGenerator(sample_analysis_results)
    generator.schedule_report(1, 'local')

    jobs = generator.scheduler.get_jobs()
    assert len(jobs) > 0
    generator.stop_scheduling()

def test_invalid_distribution_method(sample_analysis_results):
    generator = ReportGenerator(sample_analysis_results)
    with pytest.raises(ValueError):
        generator._generate_and_distribute_report('unsupported')



def test_load_json_config(tmpdir):
    config_data = {"key": "value", "number": 42}
    json_file = tmpdir.join("config.json")
    json_file.write(json.dumps(config_data))
    
    loaded_config = load_config(str(json_file))
    assert loaded_config == config_data

def test_load_yaml_config(tmpdir):
    config_data = {"key": "value", "number": 42}
    yaml_file = tmpdir.join("config.yaml")
    yaml_file.write(yaml.dump(config_data))
    
    loaded_config = load_config(str(yaml_file))
    assert loaded_config == config_data

def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_file.json")

def test_unsupported_file_format(tmpdir):
    text_file = tmpdir.join("config.txt")
    text_file.write("key=value")

    with pytest.raises(ValueError, match="Unsupported configuration file format"):
        load_config(str(text_file))

def test_invalid_json_content(tmpdir):
    invalid_json_file = tmpdir.join("invalid_config.json")
    invalid_json_file.write("{'key': 'value'")  # Invalid JSON

    with pytest.raises(ValueError, match="Error parsing the configuration file"):
        load_config(str(invalid_json_file))

def test_invalid_yaml_content(tmpdir):
    invalid_yaml_file = tmpdir.join("invalid_config.yaml")
    invalid_yaml_file.write("key value")  # Invalid YAML

    with pytest.raises(ValueError, match="Error parsing the configuration file"):
        load_config(str(invalid_yaml_file))



@patch('subprocess.check_call')
def test_setup_environment_all_installed(mock_check_call):
    # Mock check_call to pretend that all packages are already installed
    mock_check_call.side_effect = [None] * 2  # No exception for 'pip show'
    
    dependencies = ['package1', 'package2']
    setup_environment(dependencies)
    
    # Assert that 'pip show' was called for each package
    expected_calls = [call([sys.executable, '-m', 'pip', 'show', 'package1'], stdout=subprocess.DEVNULL),
                      call([sys.executable, '-m', 'pip', 'show', 'package2'], stdout=subprocess.DEVNULL)]
    mock_check_call.assert_has_calls(expected_calls, any_order=True)

@patch('subprocess.check_call')
def test_setup_environment_install_missing(mock_check_call):
    # Mock check_call to pretend 'package1' is not installed, so it will need 'pip install'
    mock_check_call.side_effect = [subprocess.CalledProcessError(1, 'cmd'), None]  # First call fails, second call is for 'pip install'
    
    dependencies = ['package1']
    setup_environment(dependencies)
    
    # Assert that 'pip show' and 'pip install' were called
    expected_calls = [call([sys.executable, '-m', 'pip', 'show', 'package1'], stdout=subprocess.DEVNULL),
                      call([sys.executable, '-m', 'pip', 'install', 'package1'])]
    mock_check_call.assert_has_calls(expected_calls, any_order=True)

@patch('subprocess.check_call')
def test_setup_environment_install_fails(mock_check_call):
    # Mock check_call to always throw an exception to simulate an install failure
    mock_check_call.side_effect = subprocess.CalledProcessError(1, 'cmd')
    
    with pytest.raises(subprocess.CalledProcessError):
        setup_environment(['nonexistent-package'])

    # Check that at least 'pip show' and attempted 'pip install' were called
    expected_calls = [call([sys.executable, '-m', 'pip', 'show', 'nonexistent-package'], stdout=subprocess.DEVNULL),
                      call([sys.executable, '-m', 'pip', 'install', 'nonexistent-package'])]
    mock_check_call.assert_has_calls(expected_calls)



def test_empty_dataframe():
    df = pd.DataFrame()
    is_valid, message = validate_data(df)
    assert not is_valid
    assert message == "The dataset is empty."

def test_missing_values():
    df = pd.DataFrame({
        'column1': [1, 2, None],
        'column2': [0.1, 0.2, 0.3],
        'column3': ['A', 'B', 'C']
    })
    is_valid, message = validate_data(df)
    assert not is_valid
    assert message == "The dataset contains missing values."

def test_missing_columns():
    df = pd.DataFrame({
        'column1': [1, 2, 3],
        'column3': ['A', 'B', 'C']
    })
    is_valid, message = validate_data(df)
    assert not is_valid
    assert message == "Missing required columns: column2"

def test_incorrect_column_type():
    df = pd.DataFrame({
        'column1': [1, 2, 3],
        'column2': [0, 1, 2],  # Incorrect type, should be float64
        'column3': ['A', 'B', 'C']
    })
    is_valid, message = validate_data(df)
    assert not is_valid
    assert message == "Column 'column2' should be of type float64, but found int64."

def test_valid_data():
    df = pd.DataFrame({
        'column1': [1, 2, 3],
        'column2': [0.1, 0.2, 0.3],
        'column3': ['A', 'B', 'C']
    })
    is_valid, message = validate_data(df)
    assert is_valid
    assert message is None



def test_setup_logging_debug_level(caplog):
    setup_logging('DEBUG')
    logger = logging.getLogger()
    logger.debug('This is a debug message')
    assert 'This is a debug message' in caplog.text

def test_setup_logging_info_level(caplog):
    setup_logging('INFO')
    logger = logging.getLogger()
    logger.info('This is an info message')
    logger.debug('This debug message should not appear')
    assert 'This is an info message' in caplog.text
    assert 'This debug message should not appear' not in caplog.text

def test_setup_logging_warning_level(caplog):
    setup_logging('WARNING')
    logger = logging.getLogger()
    logger.warning('This is a warning message')
    logger.info('This info message should not appear')
    assert 'This is a warning message' in caplog.text
    assert 'This info message should not appear' not in caplog.text

def test_setup_logging_invalid_level():
    with pytest.raises(ValueError, match="Invalid logging level: NOTALEVEL"):
        setup_logging('NOTALEVEL')
