from cli import CLI
from config_setup import setup_config
from data_handler import DataHandler
from export_module import export_to_format
from funnel import Funnel
from io import StringIO
from logging_setup import setup_logging
from metrics_calculator import MetricsCalculator
from unittest.mock import patch, MagicMock
from visualization import Visualization
import json
import logging
import os
import pandas as pd
import pytest


def test_init_valid_stages():
    stages = [{'name': 'Awareness', 'metrics': {'visitors': 1000}}]
    funnel = Funnel(stages)
    assert funnel.get_stages() == stages

def test_init_invalid_stages():
    with pytest.raises(ValueError, match="Stages should be a list of dictionaries."):
        Funnel("invalid_stages")

def test_add_stage_success():
    funnel = Funnel([])
    funnel.add_stage('Awareness', {'visitors': 1000})
    assert funnel.get_stages() == [{'name': 'Awareness', 'metrics': {'visitors': 1000}}]

def test_add_duplicate_stage():
    funnel = Funnel([{'name': 'Awareness', 'metrics': {'visitors': 1000}}])
    with pytest.raises(ValueError, match="Stage 'Awareness' already exists."):
        funnel.add_stage('Awareness', {'visitors': 500})

def test_add_stage_invalid_metrics():
    funnel = Funnel([])
    with pytest.raises(ValueError, match="Metrics should be a dictionary."):
        funnel.add_stage('Consideration', ['invalid_metrics'])

def test_remove_stage_success():
    funnel = Funnel([{'name': 'Awareness', 'metrics': {'visitors': 1000}}])
    funnel.remove_stage('Awareness')
    assert funnel.get_stages() == []

def test_remove_non_existent_stage():
    funnel = Funnel([])
    with pytest.raises(ValueError, match="Stage 'Consideration' does not exist."):
        funnel.remove_stage('Consideration')

def test_get_stages():
    stages = [
        {'name': 'Awareness', 'metrics': {'visitors': 1000}},
        {'name': 'Consideration', 'metrics': {'visitors': 800}}
    ]
    funnel = Funnel(stages)
    assert funnel.get_stages() == stages



@pytest.fixture
def csv_data(tmp_path):
    csv_content = "stage,value\nAwareness,100\nConsideration,200\nAwareness,150\n"
    csv_path = tmp_path / "test_data.csv"
    csv_path.write_text(csv_content)
    return str(csv_path)

@pytest.fixture
def json_data(tmp_path):
    json_content = '[{"stage": "Awareness", "value": 100}, {"stage": "Consideration", "value": 200}, {"stage": "Awareness", "value": 150}]'
    json_path = tmp_path / "test_data.json"
    json_path.write_text(json_content)
    return str(json_path)

def test_load_data_csv(csv_data):
    handler = DataHandler(csv_data)
    data = handler.load_data('csv')
    assert len(data) == 3
    assert data[0] == {'stage': 'Awareness', 'value': 100}

def test_load_data_json(json_data):
    handler = DataHandler(json_data)
    data = handler.load_data('json')
    assert len(data) == 3
    assert data[0] == {'stage': 'Awareness', 'value': 100}

def test_load_data_unsupported_format():
    handler = DataHandler("dummy_path")
    with pytest.raises(ValueError, match="Unsupported format 'xml'. Supported formats are: 'csv', 'json'."):
        handler.load_data('xml')

def test_filter_data_by_stage(csv_data):
    handler = DataHandler(csv_data)
    handler.load_data('csv')
    filtered_data = handler.filter_data_by_stage('Awareness')
    assert len(filtered_data) == 2
    assert all(record['stage'] == 'Awareness' for record in filtered_data)

def test_filter_data_without_loading():
    handler = DataHandler("dummy_path")
    with pytest.raises(ValueError, match="Data is not loaded. Please load data before filtering."):
        handler.filter_data_by_stage('Awareness')



def test_init_valid_data():
    funnel_data = [{'name': 'Awareness', 'metrics': {'user_count': 100, 'conversion_rate': 50}}]
    visualization = Visualization(funnel_data)
    assert visualization.funnel_data == funnel_data

def test_init_invalid_data():
    with pytest.raises(ValueError, match="Funnel data should be a list of dictionaries."):
        Visualization("invalid_data")

def test_create_funnel_chart_valid_data(mocker):
    funnel_data = [{'name': 'Awareness', 'metrics': {'user_count': 100, 'conversion_rate': 50}}]
    visualization = Visualization(funnel_data)
    
    mock_show = mocker.patch.object(plt, 'show', autospec=True)
    visualization.create_funnel_chart(title='Test Funnel Chart')
    assert mock_show.called

def test_create_conversion_chart_valid_data(mocker):
    funnel_data = [{'name': 'Awareness', 'metrics': {'user_count': 100, 'conversion_rate': 50}}]
    visualization = Visualization(funnel_data)
    
    mock_show = mocker.patch.object(plt, 'show', autospec=True)
    visualization.create_conversion_chart(title='Test Conversion Chart')
    assert mock_show.called

def test_export_visualization_not_implemented():
    funnel_data = [{'name': 'Awareness', 'metrics': {'user_count': 100, 'conversion_rate': 50}}]
    visualization = Visualization(funnel_data)
    
    with pytest.raises(NotImplementedError, match="Export logic is not implemented yet."):
        visualization.export_visualization(file_type='png')



@pytest.fixture
def funnel_data():
    return [
        {'name': 'Awareness', 'metrics': {'user_count': 1000, 'conversion_rate': 70}},
        {'name': 'Interest', 'metrics': {'user_count': 700, 'conversion_rate': 60}},
        {'name': 'Consideration', 'metrics': {'user_count': 420, 'conversion_rate': 50}},
    ]

def test_init_valid_data(funnel_data):
    calculator = MetricsCalculator(funnel_data)
    assert calculator.funnel_data == funnel_data

def test_init_invalid_data():
    with pytest.raises(ValueError, match="Funnel data should be a list of dictionaries."):
        MetricsCalculator("invalid_data")

def test_calculate_conversion_rate(funnel_data):
    calculator = MetricsCalculator(funnel_data)
    conversion_rate = calculator.calculate_conversion_rate('Awareness', 'Interest')
    assert conversion_rate == 70.0

def test_calculate_conversion_rate_no_users(funnel_data):
    empty_data = [{'name': 'Stage1', 'metrics': {'user_count': 0}}, {'name': 'Stage2', 'metrics': {'user_count': 0}}]
    calculator = MetricsCalculator(empty_data)
    conversion_rate = calculator.calculate_conversion_rate('Stage1', 'Stage2')
    assert conversion_rate == 0.0

def test_calculate_conversion_rate_stage_not_found(funnel_data):
    calculator = MetricsCalculator(funnel_data)
    with pytest.raises(ValueError, match="Stages 'Unknown' or 'Interest' not found in funnel data."):
        calculator.calculate_conversion_rate('Unknown', 'Interest')

def test_calculate_drop_off(funnel_data):
    calculator = MetricsCalculator(funnel_data)
    drop_off_rate = calculator.calculate_drop_off('Awareness')
    assert drop_off_rate == 30.0

def test_calculate_drop_off_stage_not_found(funnel_data):
    calculator = MetricsCalculator(funnel_data)
    with pytest.raises(ValueError, match="Stage 'Unknown' not found in funnel data."):
        calculator.calculate_drop_off('Unknown')

def test_get_summary_statistics(funnel_data):
    calculator = MetricsCalculator(funnel_data)
    statistics = calculator.get_summary_statistics()
    assert statistics['total_users'] == 2120
    assert statistics['total_conversions'] == pytest.approx(1390)
    assert statistics['average_conversion_rate'] == pytest.approx(60.0)



def test_cli_init():
    cli = CLI()
    assert cli.data_handler is None
    assert cli.funnel is None
    assert cli.visualization is None
    assert cli.metrics_calculator is None

def test_parse_command_help():
    cli = CLI()
    result = cli.parse_command("help")
    assert "Available commands:" in result

def test_parse_command_load_data():
    cli = CLI()
    with patch('cli.DataHandler') as MockDataHandler:
        mock_instance = MockDataHandler.return_value
        mock_instance.load_data.return_value = MagicMock()
        
        result = cli.parse_command("load_data path/to/data csv")
        assert result == "Data loaded from path/to/data in csv format."

def test_parse_command_load_data_invalid_syntax():
    cli = CLI()
    with pytest.raises(ValueError, match="Usage: load_data <source> <format>"):
        cli.parse_command("load_data path/to/data")

def test_parse_command_funnel_chart_not_initialized():
    cli = CLI()
    with pytest.raises(ValueError, match="Visualization object not initialized"):
        cli.parse_command("funnel_chart")

def test_parse_command_calculate_metrics_not_initialized():
    cli = CLI()
    with pytest.raises(ValueError, match="MetricsCalculator not initialized"):
        cli.parse_command("calculate_metrics")

def test_parse_command_unknown():
    cli = CLI()
    with pytest.raises(ValueError, match="Unknown command: unknown_cmd"):
        cli.parse_command("unknown_cmd")



@pytest.fixture
def sample_data():
    return [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame([{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}])

def test_export_to_csv(tmp_path, sample_data):
    file_path = tmp_path / "data.csv"
    message = export_to_format(sample_data, 'csv', file_path)
    assert message == f"Data successfully exported to {file_path}."
    assert file_path.exists()

def test_export_to_json(tmp_path, sample_data):
    file_path = tmp_path / "data.json"
    message = export_to_format(sample_data, 'json', file_path)
    assert message == f"Data successfully exported to {file_path}."
    assert file_path.exists()

def test_export_to_xlsx(tmp_path, sample_data):
    file_path = tmp_path / "data.xlsx"
    message = export_to_format(sample_data, 'xlsx', file_path)
    assert message == f"Data successfully exported to {file_path}."
    assert file_path.exists()

def test_export_invalid_format(sample_data, tmp_path):
    with pytest.raises(ValueError, match="Unsupported format 'xml'. Supported formats are: 'csv', 'json', 'xlsx'."):
        export_to_format(sample_data, 'xml', tmp_path / "data.xml")

def test_export_io_error(sample_dataframe, tmp_path):
    with pytest.raises(IOError, match="An error occurred while writing to file:"):
        os.chmod(tmp_path, 0o400)  # Make the directory read-only
        export_to_format(sample_dataframe, 'csv', tmp_path / "data.csv")
        os.chmod(tmp_path, 0o700)  # Restore permissions



def test_setup_logging_valid_levels(capsys):
    for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        setup_logging(level)
        logger = logging.getLogger()
        assert logger.level == getattr(logging, level)
        with capsys.disabled():
            logging.info("This is a test log message.")

def test_setup_logging_invalid_level():
    with pytest.raises(ValueError, match="Unsupported logging level 'VERBOSE'"):
        setup_logging("VERBOSE")

def test_logging_configuration_exception(mocker):
    mock_basic_config = mocker.patch('logging.basicConfig', side_effect=Exception("Mocked logging failure"))
    with pytest.raises(Exception, match="Failed to configure logging: Mocked logging failure"):
        setup_logging("DEBUG")



def test_setup_config_file_not_found():
    with pytest.raises(FileNotFoundError, match="The configuration file 'non_existent_config.json' does not exist."):
        setup_config('non_existent_config.json')

def test_setup_config_json_parsing_error(tmp_path):
    corrupt_config = tmp_path / "corrupt_config.json"
    corrupt_config.write_text("{invalid_json: }")

    with pytest.raises(ValueError, match="Error parsing configuration file"):
        setup_config(str(corrupt_config))

def test_setup_config_generic_exception(monkeypatch, tmp_path):
    valid_config = tmp_path / "valid_config.json"
    valid_config.write_text("{'key': 'value'}")

    def mock_open(*args, **kwargs):
        raise Exception("Mocked exception")

    monkeypatch.setattr("builtins.open", mock_open)

    with pytest.raises(Exception, match="An error occurred while loading the configuration file: Mocked exception"):
        setup_config(str(valid_config))

def test_setup_config_valid_file(tmp_path, capsys):
    valid_config = tmp_path / "valid_config.json"
    valid_config.write_text(json.dumps({"database": "test_db", "user": "admin"}))

    setup_config(str(valid_config))
    captured = capsys.readouterr()
    assert "database: test_db" in captured.out
    assert "user: admin" in captured.out
