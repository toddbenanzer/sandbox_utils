from contextlib import redirect_stdout
from data_analyzer import DataAnalyzer
from data_collector import DataCollector
from experiment_designer import ExperimentDesigner
from io import StringIO
from load_data_module import load_data_from_source
from parse_config import parse_experiment_config
from report_generator import ReportGenerator
from save_results import save_results_to_file
from setup_logging import setup_logging_config
from user_interface import UserInterface
import json
import logging
import os
import pandas as pd
import pytest
import yaml


def test_define_groups():
    designer = ExperimentDesigner()
    group_sizes = designer.define_groups(control_size=100, experiment_size=150)
    assert group_sizes == {'control': 100, 'experiment': 150}

def test_randomize_participants():
    designer = ExperimentDesigner()
    participants = ['A', 'B', 'C', 'D', 'E', 'F']
    allocation = designer.randomize_participants(participants)
    assert set(allocation['control'] + allocation['experiment']) == set(participants)
    assert len(allocation['control']) + len(allocation['experiment']) == len(participants)

def test_set_factors():
    designer = ExperimentDesigner()
    factors = {'color': ['red', 'blue'], 'size': ['small', 'large']}
    designer.set_factors(factors)
    assert designer.factors == factors



def test_capture_metrics():
    collector = DataCollector()
    metrics = ['clicks', 'conversions']
    results = collector.capture_metrics('test_source', metrics)
    assert results == {'clicks': 100, 'conversions': 100}
    assert collector.metrics_data == {'clicks': 100, 'conversions': 100}

def test_integrate_with_platform_success():
    collector = DataCollector()
    platform_details = {'name': 'Google Analytics', 'API_key': 'valid_key'}
    success = collector.integrate_with_platform(platform_details)
    assert success is True
    assert collector.platform == 'Google Analytics'

def test_integrate_with_platform_failure():
    collector = DataCollector()
    platform_details = {'name': 'Google Analytics'}
    success = collector.integrate_with_platform(platform_details)
    assert success is False
    assert collector.platform is None



@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': [5, 6, 7, 8]
    })
    return data

def test_perform_statistical_tests_t_test(sample_data):
    analyzer = DataAnalyzer(sample_data)
    result = analyzer.perform_statistical_tests('t-test')
    assert result['p-value'] == 0.05
    assert result['statistic'] == 2.1
    assert result['test_type'] == 't-test'

def test_perform_statistical_tests_chi_squared(sample_data):
    analyzer = DataAnalyzer(sample_data)
    result = analyzer.perform_statistical_tests('chi-squared')
    assert result['p-value'] == 0.01
    assert result['statistic'] == 5.4
    assert result['test_type'] == 'chi-squared'

def test_perform_statistical_tests_invalid(sample_data):
    analyzer = DataAnalyzer(sample_data)
    result = analyzer.perform_statistical_tests('invalid')
    assert result['error'] == 'Invalid test type'

def test_visualize_data_bar_chart(sample_data, monkeypatch):
    analyzer = DataAnalyzer(sample_data)

    def dummy_show():
        pass
    monkeypatch.setattr(plt, 'show', dummy_show)
    
    analyzer.visualize_data(sample_data, 'bar')  # Should not raise an error

def test_visualize_data_invalid_chart_type(sample_data):
    analyzer = DataAnalyzer(sample_data)
    analyzer.visualize_data(sample_data, 'invalid')  # Testing print output for invalid chart type



@pytest.fixture
def analysis_results_sample():
    return {
        'p-value': 0.03,
        'statistic': 4.56,
        'test_type': 't-test'
    }

@pytest.fixture
def visualization_details_sample():
    return {
        'charts': [
            {'data': {'A': 10, 'B': 15, 'C': 5}, 'type': 'bar', 'title': 'Bar Chart Example'},
            {'data': {'X': 30, 'Y': 25, 'Z': 45}, 'type': 'pie', 'title': 'Pie Chart Example'}
        ]
    }

def test_generate_summary(analysis_results_sample):
    report_gen = ReportGenerator(analysis_results_sample)
    summary = report_gen.generate_summary()
    assert "Summary of Analysis Results:" in summary
    assert "- p-value: 0.03" in summary
    assert "- statistic: 4.56" in summary
    assert "- test_type: t-test" in summary

def test_create_visual_report(monkeypatch, analysis_results_sample, visualization_details_sample):
    report_gen = ReportGenerator(analysis_results_sample)

    def dummy_show():
        pass
    monkeypatch.setattr(plt, 'show', dummy_show)

    report_gen.create_visual_report(visualization_details_sample)  # Should not raise an error



def test_user_interface_initialization(capsys):
    ui = UserInterface()
    captured = capsys.readouterr()
    assert "User Interface initialized" in captured.out

def test_launch_cli(capsys):
    ui = UserInterface()
    ui.launch_cli()
    captured = capsys.readouterr()
    assert "Launching CLI..." in captured.out

def test_launch_gui(capsys):
    ui = UserInterface()
    ui.launch_gui()
    captured = capsys.readouterr()
    assert "Launching GUI..." in captured.out

def test_process_input_parameters(capsys):
    ui = UserInterface()
    params = {'experiment_id': 1, 'group_size': 50}
    processed_params = ui.process_input_parameters(params)
    captured = capsys.readouterr()
    assert "Processing input parameters..." in captured.out
    assert "Processed Parameters: {'experiment_id': 1, 'group_size': 50}" in captured.out
    assert processed_params == params



def test_load_data_from_csv(monkeypatch):
    def mock_read_csv(file_path):
        return pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

    monkeypatch.setattr(pd, 'read_csv', mock_read_csv)
    data = load_data_from_source('test.csv')

    assert isinstance(data, pd.DataFrame)
    assert 'col1' in data.columns
    assert data['col1'].iloc[0] == 1

def test_load_data_from_xlsx(monkeypatch):
    def mock_read_excel(file_path):
        return pd.DataFrame({'col1': [5, 6], 'col2': [7, 8]})

    monkeypatch.setattr(pd, 'read_excel', mock_read_excel)
    data = load_data_from_source('test.xlsx')

    assert isinstance(data, pd.DataFrame)
    assert 'col1' in data.columns
    assert data['col1'].iloc[0] == 5

def test_load_data_from_unsupported_format():
    with pytest.raises(ValueError, match=r"Unsupported source format"):
        load_data_from_source('test.unsupported')

def test_load_data_file_not_found(monkeypatch):
    def mock_read_csv(file_path):
        raise FileNotFoundError("File not found error")

    monkeypatch.setattr(pd, 'read_csv', mock_read_csv)
    with pytest.raises(FileNotFoundError):
        load_data_from_source('nonexistent.csv')



@pytest.fixture
def sample_data():
    return pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})

def test_save_results_to_csv(tmp_path, sample_data):
    file_path = tmp_path / "results.csv"
    save_results_to_file(sample_data, str(file_path))
    assert file_path.exists()
    loaded_data = pd.read_csv(file_path)
    pd.testing.assert_frame_equal(loaded_data, sample_data)

def test_save_results_to_json(tmp_path, sample_data):
    file_path = tmp_path / "results.json"
    save_results_to_file(sample_data, str(file_path))
    assert file_path.exists()
    loaded_data = pd.read_json(file_path, orient='records', lines=True)
    pd.testing.assert_frame_equal(loaded_data, sample_data)

def test_save_results_to_excel(tmp_path, sample_data):
    file_path = tmp_path / "results.xlsx"
    save_results_to_file(sample_data, str(file_path))
    assert file_path.exists()
    loaded_data = pd.read_excel(file_path, engine='openpyxl')
    pd.testing.assert_frame_equal(loaded_data, sample_data)

def test_save_results_to_unsupported_format(tmp_path):
    with pytest.raises(ValueError, match=r"Unsupported file format"):
        save_results_to_file({'key': 'value'}, str(tmp_path / "results.txt"))



def test_logging_level_debug(monkeypatch):
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)  # Reset to default level
    monkeypatch.setattr('sys.stdout', StringIO())
    setup_logging_config('DEBUG')
    assert logger.level == logging.DEBUG

def test_logging_level_info(monkeypatch):
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)  # Reset to default level
    monkeypatch.setattr('sys.stdout', StringIO())
    setup_logging_config('INFO')
    assert logger.level == logging.INFO

def test_logging_level_warning(monkeypatch):
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)  # Reset to default level
    monkeypatch.setattr('sys.stdout', StringIO())
    setup_logging_config('WARNING')
    assert logger.level == logging.WARNING

def test_logging_level_error(monkeypatch):
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)  # Reset to default level
    monkeypatch.setattr('sys.stdout', StringIO())
    setup_logging_config('ERROR')
    assert logger.level == logging.ERROR

def test_logging_level_critical(monkeypatch):
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)  # Reset to default level
    monkeypatch.setattr('sys.stdout', StringIO())
    setup_logging_config('CRITICAL')
    assert logger.level == logging.CRITICAL

def test_invalid_logging_level(monkeypatch):
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)  # Reset to default level
    f = StringIO()
    with redirect_stdout(f):
        setup_logging_config('INVALID_LEVEL')
    output = f.getvalue()
    assert logger.level == logging.WARNING  # Default to WARNING for invalid input
    assert "Invalid logging level 'INVALID_LEVEL' provided. Defaulting to 'WARNING'." in output



def test_parse_json_config(tmp_path):
    config_data = {
        'experiment': 'Test1',
        'groups': ['A', 'B'],
        'metrics': ['clicks', 'views']
    }
    config_file = tmp_path / 'config.json'
    with open(config_file, 'w') as file:
        json.dump(config_data, file)

    parsed_config = parse_experiment_config(str(config_file))
    assert parsed_config == config_data

def test_parse_yaml_config(tmp_path):
    config_data = {
        'experiment': 'Test2',
        'groups': ['C', 'D'],
        'metrics': ['conversions', 'signups']
    }
    config_file = tmp_path / 'config.yaml'
    with open(config_file, 'w') as file:
        yaml.dump(config_data, file)

    parsed_config = parse_experiment_config(str(config_file))
    assert parsed_config == config_data

def test_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        parse_experiment_config('nonexistent_config.json')

def test_invalid_json_format(tmp_path):
    invalid_config = '{"experiment": "Test", "groups": "A", "missing_bracket": "value"'
    config_file = tmp_path / 'invalid_config.json'
    with open(config_file, 'w') as file:
        file.write(invalid_config)

    with pytest.raises(ValueError, match="Invalid configuration file format."):
        parse_experiment_config(str(config_file))

def test_invalid_yaml_format(tmp_path):
    invalid_config = "experiment: Test\nunclosed_dict: {key: value"
    config_file = tmp_path / 'invalid_config.yaml'
    with open(config_file, 'w') as file:
        file.write(invalid_config)

    with pytest.raises(ValueError, match="Invalid configuration file format."):
        parse_experiment_config(str(config_file))

def test_unsupported_file_format(tmp_path):
    config_file = tmp_path / 'config.txt'
    with open(config_file, 'w') as file:
        file.write("sample text content")

    with pytest.raises(ValueError, match="Unsupported file format. Please use JSON or YAML."):
        parse_experiment_config(str(config_file))
