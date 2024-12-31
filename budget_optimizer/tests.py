from cli_interface.cli_interface import display_output
from data_management.data_handler import DataHandler
from integration.analytics_integrator import AnalyticsIntegrator
from io import StringIO
from optimization_algorithms.budget_optimizer import BudgetOptimizer
from reporting.report_generator import ReportGenerator
from unittest.mock import patch
from utils.utility import log, configure_settings
from visualization.visualizer import Visualizer
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pytest
import sys

# tests/test_data_handler.py


@pytest.fixture
def sample_data():
    data = """A,B,C
    1,2,
    4,,6
    7,8,9
    """
    return StringIO(data)

def test_load_data(sample_data):
    handler = DataHandler(sample_data)
    data = handler.load_data()
    assert not data.empty
    assert list(data.columns) == ['A', 'B', 'C']

def test_load_data_invalid_source():
    handler = DataHandler("non_existent_file.csv")
    with pytest.raises(IOError):
        handler.load_data()

def test_preprocess_data(sample_data):
    handler = DataHandler(sample_data)
    handler.load_data()
    filled_data = handler.preprocess_data(method='fill')
    assert filled_data.isnull().sum().sum() == 0

def test_preprocess_data_invalid_method(sample_data):
    handler = DataHandler(sample_data)
    handler.load_data()
    with pytest.raises(ValueError):
        handler.preprocess_data(method='unknown')

def test_validate_data(sample_data):
    handler = DataHandler(sample_data)
    handler.load_data()
    is_valid, errors = handler.validate_data()
    assert not is_valid
    assert "Data contains missing values." in errors

def test_validate_data_after_preprocessing(sample_data):
    handler = DataHandler(sample_data)
    handler.load_data()
    handler.preprocess_data(method='fill')
    is_valid, errors = handler.validate_data()
    assert is_valid
    assert not errors


# tests/test_budget_optimizer.py


@pytest.fixture
def sample_data():
    data = {
        'Channel1': [100, 150, 200],
        'Channel2': [80, 120, 160],
        'Channel3': [60, 90, 120]
    }
    return pd.DataFrame(data)

@pytest.fixture
def constraints():
    return {
        'total_budget': 1,
        'bounds': [(0, 0.5), (0, 0.5), (0, 0.5)]
    }

def test_optimize_with_linear_programming(sample_data, constraints):
    optimizer = BudgetOptimizer(sample_data, constraints)
    allocation = optimizer.optimize_with_linear_programming()
    assert isinstance(allocation, pd.Series)
    assert allocation.sum() == constraints['total_budget']
    assert all(constraints['bounds'][i][0] <= val <= constraints['bounds'][i][1] for i, val in enumerate(allocation))

def test_optimize_with_linear_programming_failure(sample_data):
    constraints = {'total_budget': 0}
    optimizer = BudgetOptimizer(sample_data, constraints)
    with pytest.raises(ValueError):
        optimizer.optimize_with_linear_programming()

def test_optimize_with_genetic_algorithm():
    optimizer = BudgetOptimizer(pd.DataFrame(), {})
    with pytest.raises(NotImplementedError):
        optimizer.optimize_with_genetic_algorithm()

def test_optimize_with_heuristic_method():
    optimizer = BudgetOptimizer(pd.DataFrame(), {})
    with pytest.raises(NotImplementedError):
        optimizer.optimize_with_heuristic_method()


# tests/test_report_generator.py


@pytest.fixture
def sample_optimization_result():
    return {
        'total_budget': 1000,
        'allocations': {
            'Channel1': 500,
            'Channel2': 300,
            'Channel3': 200
        }
    }

def test_generate_summary_report(sample_optimization_result):
    report_generator = ReportGenerator(sample_optimization_result)
    summary_report = report_generator.generate_summary_report()
    assert "Summary Report" in summary_report
    assert "Total Budget Allocated: 1000" in summary_report
    assert "  - Channel1: 500" in summary_report
    assert "  - Channel2: 300" in summary_report
    assert "  - Channel3: 200" in summary_report

def test_generate_detailed_report_pdf(sample_optimization_result):
    report_generator = ReportGenerator(sample_optimization_result)
    detailed_report = report_generator.generate_detailed_report('PDF')
    assert detailed_report == "This is a PDF report."

def test_generate_detailed_report_html(sample_optimization_result):
    report_generator = ReportGenerator(sample_optimization_result)
    detailed_report = report_generator.generate_detailed_report('HTML')
    assert detailed_report == "<html><body>This is an HTML report.</body></html>"

def test_generate_detailed_report_docx(sample_optimization_result):
    report_generator = ReportGenerator(sample_optimization_result)
    detailed_report = report_generator.generate_detailed_report('docx')
    assert detailed_report == "This is a DOCX report."

def test_generate_detailed_report_invalid_format(sample_optimization_result):
    report_generator = ReportGenerator(sample_optimization_result)
    with pytest.raises(ValueError):
        report_generator.generate_detailed_report('TXT')


# tests/test_visualizer.py


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'Allocation': [500, 300, 200],
        'Performance_Before': [1.2, 1.5, 1.1],
        'Performance_After': [1.4, 1.6, 1.3]
    }, index=['Channel1', 'Channel2', 'Channel3'])

def test_plot_budget_distribution(sample_data):
    visualizer = Visualizer(sample_data)
    fig = visualizer.plot_budget_distribution()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_performance_comparison(sample_data):
    visualizer = Visualizer(sample_data)
    fig = visualizer.plot_performance_comparison()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

@pytest.fixture
def sample_data_no_performance_comparison():
    return pd.DataFrame({
        'Channel': ['Channel1', 'Channel2', 'Channel3'],
        'Performance': [1.0, 1.2, 1.3]
    })

def test_plot_performance_comparison_no_before_after(sample_data_no_performance_comparison):
    visualizer = Visualizer(sample_data_no_performance_comparison)
    fig = visualizer.plot_performance_comparison()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# tests/test_cli_interface.py


@pytest.fixture
def sample_results():
    return {
        'total_budget': 1000,
        'allocations': {'Channel1': 500, 'Channel2': 300, 'Channel3': 200},
        'performance_improvement': 0.15,
        'report_path': '/path/to/report.pdf'
    }

def test_display_output(sample_results):
    expected_output = (
        "\nBudget Optimization Results\n"
        "------------------------------\n"
        "Total Budget Allocated: 1000\n\n"
        "Channel Allocations:\n"
        "  - Channel1: 500\n"
        "  - Channel2: 300\n"
        "  - Channel3: 200\n\n"
        "Performance Improvement: 0.15\n\n"
        "Detailed Report: /path/to/report.pdf\n"
        "------------------------------\n"
    )
    
    with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
        display_output(sample_results)
        assert mock_stdout.getvalue() == expected_output


# tests/test_analytics_integrator.py


@pytest.fixture
def sample_tools_config():
    return {
        'Tableau': {'auth_token': 'abc123', 'api_endpoint': 'https://api.tableau.com'},
        'PowerBI': {'auth_token': 'xyz789', 'workspace_url': 'https://api.powerbi.com'}
    }

def test_integration_successful(sample_tools_config):
    integrator = AnalyticsIntegrator(sample_tools_config)
    status, details = integrator.integrate_with_tool('Tableau')
    assert status is True
    assert details == "Successfully integrated with Tableau."

def test_integration_tool_not_found(sample_tools_config):
    integrator = AnalyticsIntegrator(sample_tools_config)
    status, details = integrator.integrate_with_tool('Looker')
    assert status is False
    assert details == "Configuration for tool not found."

def test_integration_with_exception_handling(mocker, sample_tools_config):
    integrator = AnalyticsIntegrator(sample_tools_config)
    
    # Mock the json.dumps to throw an exception to simulate an integration failure scenario
    mocker.patch('json.dumps', side_effect=Exception('Mocked exception'))
    
    status, details = integrator.integrate_with_tool('Tableau')
    assert status is False
    assert 'Integration with Tableau failed' in details
    assert 'Mocked exception' in details


# tests/test_utility.py


def test_log_info_level(caplog):
    message = "This is an info log message."
    with caplog.at_level(logging.INFO):
        log(message, level='INFO')
    assert message in caplog.text

def test_log_error_level(caplog):
    message = "This is an error log message."
    with caplog.at_level(logging.ERROR):
        log(message, level='ERROR')
    assert message in caplog.text

def test_log_invalid_level(caplog):
    message = "This is a log message with invalid level."
    with caplog.at_level(logging.INFO):
        log(message, level='INVALID')
    assert message in caplog.text

def test_configure_settings_json(tmpdir):
    json_file = tmpdir.join("settings.json")
    json_file.write('{"setting1": "value1", "setting2": "value2"}')
    configurations = configure_settings(str(json_file))
    assert configurations['setting1'] == "value1"
    assert configurations['setting2'] == "value2"

def test_configure_settings_yaml(tmpdir):
    yaml_file = tmpdir.join("settings.yaml")
    yaml_file.write('setting1: value1\nsetting2: value2')
    configurations = configure_settings(str(yaml_file))
    assert configurations['setting1'] == "value1"
    assert configurations['setting2'] == "value2"

def test_configure_settings_ini(tmpdir):
    ini_file = tmpdir.join("settings.ini")
    ini_file.write("[DEFAULT]\nsetting1 = value1\nsetting2 = value2")
    configurations = configure_settings(str(ini_file))
    assert configurations['DEFAULT']['setting1'] == "value1"
    assert configurations['DEFAULT']['setting2'] == "value2"

def test_configure_settings_unsupported_format(caplog, tmpdir):
    unsupported_file = tmpdir.join("settings.unsupported")
    unsupported_file.write('setting1: value1\nsetting2: value2')
    configurations = configure_settings(str(unsupported_file))
    assert "Unsupported file format" in caplog.text
