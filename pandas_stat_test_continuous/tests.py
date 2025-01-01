from data_analyzer import DataAnalyzer  # Adjust the import as per your module name and structure
from dependencies_installer import install_dependencies  # Adjust the import as per your module structure
from statistical_tests import StatisticalTests  # Adjust the import as per your module structure
from summary_report import generate_summary_report  # Adjust the import as per your module structure
from unittest import mock
from visualize import visualize_statistics  # Adjust the import as per your module structure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


class TestDataAnalyzer:

    def test_calculate_descriptive_stats(self):
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1]
        })
        analyzer = DataAnalyzer(df, ['A', 'B'])
        stats = analyzer.calculate_descriptive_stats()
        
        assert stats['A']['mean'] == 3.0
        assert stats['B']['mean'] == 3.0
        assert stats['A']['std_dev'] == pytest.approx(1.5811, 0.0001)
        assert stats['B']['std_dev'] == pytest.approx(1.5811, 0.0001)
        
    def test_detect_and_handle_missing_data(self):
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [5, np.nan, 3, 2, 1]
        })
        analyzer = DataAnalyzer(df, ['A', 'B'])
        analyzer.detect_and_handle_missing_data(method='fill_mean')
        
        assert df['A'].isnull().sum() == 0
        assert df['B'].isnull().sum() == 0
        assert df['A'][2] == df['A'].mean()
        assert df['B'][1] == df['B'].mean()

    def test_detect_and_handle_infinite_data(self):
        df = pd.DataFrame({
            'A': [1, 2, np.inf, 4, 5],
            'B': [5, -np.inf, 3, 2, 1]
        })
        analyzer = DataAnalyzer(df, ['A', 'B'])
        analyzer.detect_and_handle_infinite_data(method='replace_nan')
        
        assert np.isinf(df['A']).sum() == 0
        assert np.isinf(df['B']).sum() == 0
        assert np.isnan(df['A'][2])
        assert np.isnan(df['B'][1])
        
    def test_exclude_null_and_trivial_columns(self):
        df = pd.DataFrame({
            'A': [1, 1, 1, 1, 1],
            'B': [np.nan, np.nan, np.nan, np.nan, np.nan],
            'C': [1, 2, 3, 4, 5]
        })
        analyzer = DataAnalyzer(df, ['A', 'B', 'C'])
        excluded = analyzer.exclude_null_and_trivial_columns()
        
        assert 'A' in excluded
        assert 'B' in excluded
        assert analyzer.columns == ['C']



class TestStatisticalTests:

    def test_perform_one_sample_t_test(self):
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [2, 3, 4, 5, 6]
        })
        tests = StatisticalTests(df, ['A', 'B'])
        results = tests.perform_t_tests(test_type='one-sample', popmean=3)

        assert 'A' in results
        assert 'B' in results
        assert isinstance(results['A']['t-statistic'], float)
        assert isinstance(results['A']['p-value'], float)

    def test_perform_independent_t_test(self):
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 6],
            'Group': ['X', 'X', 'Y', 'Y', 'Y', 'X']
        })
        tests = StatisticalTests(df, ['A'])
        results = tests.perform_t_tests(test_type='independent', group_column='Group')

        assert 'independent' in results
        assert isinstance(results['independent']['t-statistic'], float)
        assert isinstance(results['independent']['p-value'], float)

    def test_perform_paired_t_test(self):
        df = pd.DataFrame({
            'Pre': [89, 67, 78, 91, 85],
            'Post': [90, 68, 80, 92, 88]
        })
        tests = StatisticalTests(df, ['Pre', 'Post'])
        results = tests.perform_t_tests(test_type='paired')

        assert 'paired' in results
        assert isinstance(results['paired']['t-statistic'], float)
        assert isinstance(results['paired']['p-value'], float)

    def test_perform_anova(self):
        df = pd.DataFrame({
            'A': [5, 6, 5, 7, 5, 6],
            'Group': ['G1', 'G1', 'G2', 'G2', 'G3', 'G3']
        })
        tests = StatisticalTests(df, ['A'])
        results = tests.perform_anova(group_column='Group')

        assert 'F-statistic' in results
        assert 'p-value' in results
        assert isinstance(results['F-statistic'], float)
        assert isinstance(results['p-value'], float)
        
    def test_perform_chi_squared_test(self):
        df = pd.DataFrame({
            'C1': ['A', 'A', 'B', 'B', 'C', 'C'],
            'C2': ['X', 'X', 'Y', 'Y', 'X', 'Y']
        })
        tests = StatisticalTests(df, ['C1', 'C2'])
        results = tests.perform_chi_squared_test()

        assert 'chi2-statistic' in results
        assert 'p-value' in results
        assert isinstance(results['chi2-statistic'], float)
        assert isinstance(results['p-value'], float)



def test_generate_summary_report_basic():
    data = {
        'Metric A': {
            'Mean': 3.1415,
            'Std Dev': 1.234
        },
        'Metric B': {
            'Minimum': 1,
            'Maximum': 10
        }
    }
    expected_in_report = [
        "Statistical Summary Report",
        "Metric A:",
        "Mean: 3.1415",
        "Std Dev: 1.2340",
        "Metric B:",
        "Minimum: 1",
        "Maximum: 10",
        "=" * 50
    ]

    report = generate_summary_report(data)
    for line in expected_in_report:
        assert line in report

def test_generate_summary_report_single_value():
    data = {'Single Metric': 42}
    expected_in_report = [
        "Statistical Summary Report",
        "Single Metric:",
        "Value: 42",
        "=" * 50
    ]

    report = generate_summary_report(data)
    for line in expected_in_report:
        assert line in report

def test_generate_summary_report_empty_dict():
    empty_data = {}
    expected_in_report = [
        "Statistical Summary Report",
        "=" * 50,
        "=" * 50
    ]

    report = generate_summary_report(empty_data)
    for line in expected_in_report:
        assert line in report

def test_generate_summary_report_complex_values():
    data = {
        'Complex Stat': {
            'Complex Value': 2 + 3j
        }
    }
    expected_in_report = [
        "Statistical Summary Report",
        "Complex Stat:",
        "Complex Value: (2+3j)",
        "=" * 50
    ]
    
    report = generate_summary_report(data)
    for line in expected_in_report:
        assert line in report



def test_visualize_statistics_bar_plot(mocker):
    statistics_data = {
        'Category1': {'A': 10, 'B': 20},
        'Category2': {'A': 15, 'B': 25}
    }
    mocker.patch('matplotlib.pyplot.show')  # Mock plt.show to avoid displaying the plot
    visualize_statistics(statistics_data, plot_type='bar', title='Bar Plot Test', xlabel='Categories', ylabel='Values')

def test_visualize_statistics_scatter_plot(mocker):
    statistics_data = {
        'X_values': [1, 2, 3],
        'Y_values': [4, 5, 6]
    }
    mocker.patch('matplotlib.pyplot.show')  # Mock plt.show to avoid displaying the plot
    visualize_statistics(statistics_data, plot_type='scatter', title='Scatter Plot Test', xlabel='X-axis', ylabel='Y-axis')

def test_visualize_statistics_histogram(mocker):
    statistics_data = {
        'DataSet1': [1, 2, 2, 3, 3, 3],
        'DataSet2': [2, 3, 3, 4, 5, 5]
    }
    mocker.patch('matplotlib.pyplot.show')  # Mock plt.show to avoid displaying the plot
    visualize_statistics(statistics_data, plot_type='histogram', title='Histogram Test', xlabel='Values', ylabel='Frequency')

def test_visualize_statistics_invalid_plot_type():
    statistics_data = {'Category': [1, 2, 3]}
    with pytest.raises(ValueError, match="Unsupported plot type"):
        visualize_statistics(statistics_data, plot_type='invalid')

def test_visualize_statistics_scatter_missing_data():
    statistics_data = {
        'X_values': [1, 2, 3]
    }
    with pytest.raises(ValueError, match="Scatter plot requires exactly two sets of data"):
        visualize_statistics(statistics_data, plot_type='scatter')



def test_install_dependencies_file_not_found():
    with pytest.raises(FileNotFoundError):
        install_dependencies('non_existent_file.txt')

@mock.patch('subprocess.check_call')
def test_install_dependencies_success(mock_check_call):
    mock_check_call.return_value = 0
    try:
        install_dependencies('requirements.txt')
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")

@mock.patch('subprocess.check_call', side_effect=subprocess.CalledProcessError(1, 'cmd'))
def test_install_dependencies_installation_failure(mock_check_call):
    with pytest.raises(Exception, match="There was an error installing the dependencies."):
        install_dependencies('requirements.txt')
