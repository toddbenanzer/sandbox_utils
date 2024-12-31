from configure_integration import configure_integration
from data_handler import DataHandler
from io import StringIO
from logging import getLogger, StreamHandler
from results_interpreter import ResultsInterpreter
from scipy import stats
from setup_logging import setup_logging
from typing import List, Dict, Any
from unittest.mock import patch, MagicMock
from visualizer import Visualizer
import importlib
import logging
import numpy as np
import pandas as pd
import pytest


class TestStatisticalTests:

    def setup(self):
        """Setup any state specific to the execution of the given module."""
        self.stat_tests = StatisticalTests()

    def test_perform_t_test_independent(self):
        data_a = np.array([1.1, 2.2, 3.1, 4.8, 5.5])
        data_b = np.array([2.3, 3.3, 3.8, 4.4, 5.0])
        result = self.stat_tests.perform_t_test(data_a, data_b, paired=False)
        assert isinstance(result, dict)
        assert 't_stat' in result
        assert 'p_value' in result
        assert 'degrees_of_freedom' in result

    def test_perform_t_test_paired(self):
        data_a = np.array([1.1, 2.5, 3.1, 4.2, 5.3])
        data_b = np.array([1.2, 2.6, 3.0, 4.0, 5.1])
        result = self.stat_tests.perform_t_test(data_a, data_b, paired=True)
        assert isinstance(result, dict)
        assert 't_stat' in result
        assert 'p_value' in result
        assert 'degrees_of_freedom' in result

    def test_perform_anova(self):
        group1 = np.array([1.1, 2.2, 3.1])
        group2 = np.array([2.3, 3.3, 4.4])
        group3 = np.array([1.5, 2.5, 3.5])
        result = self.stat_tests.perform_anova(group1, group2, group3)
        assert isinstance(result, dict)
        assert 'f_stat' in result
        assert 'p_value' in result

    def test_perform_chi_square(self):
        observed = np.array([10, 20, 30])
        expected = np.array([15, 15, 30])
        result = self.stat_tests.perform_chi_square(observed, expected)
        assert isinstance(result, dict)
        assert 'chi_stat' in result
        assert 'p_value' in result

    def test_perform_regression_analysis(self):
        data = pd.DataFrame({
            'independent_var1': [1, 2, 3, 4, 5],
            'independent_var2': [2, 3, 4, 5, 6],
            'dependent_var': [2.3, 3.1, 4.0, 5.1, 6.0]
        })
        result = self.stat_tests.perform_regression_analysis(data, ['independent_var1', 'independent_var2'], 'dependent_var')
        assert isinstance(result, dict)
        assert 'coefficients' in result
        assert 'r_squared' in result
        assert 'standard_error' in result
        assert 'p_values' in result



class TestDataHandler:

    def setup(self):
        """Setup that creates a DataHandler instance."""
        self.data_handler = DataHandler()

    def test_load_data_csv(self):
        csv_data = """col1,col2,col3
                      1,2,3
                      4,5,6"""
        file_path = "test_data.csv"
        data_frame = pd.read_csv(StringIO(csv_data))
        data_frame.to_csv(file_path, index=False)

        loaded_data = self.data_handler.load_data(file_path)
        assert isinstance(loaded_data, pd.DataFrame)
        assert not loaded_data.empty
        assert list(loaded_data.columns) == ["col1", "col2", "col3"]

    def test_cleanse_data(self):
        raw_data = pd.DataFrame({
            'A': [1, None, 2, 2, 4],
            'B': [5, 6, 7, 8, 5],
            'C': ['cat', 'dog', 'cat', 'dog', 'dog']
        })
        cleansed_data = self.data_handler.cleanse_data(raw_data)
        assert len(cleansed_data) == 4
        assert cleansed_data.isnull().sum().sum() == 0

    def test_preprocess_data(self):
        raw_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': ['cat', 'dog', 'cat', 'dog', 'cat']
        })
        preprocessed_data = self.data_handler.preprocess_data(raw_data)
        assert 'B_dog' in preprocessed_data.columns
        assert preprocessed_data['A'].mean() == pytest.approx(0, abs=1e-9)
        assert preprocessed_data['A'].std() == pytest.approx(1, abs=1e-9)



class TestResultsInterpreter:

    def setup(self):
        """Setup any state specific to the execution of the given class."""
        self.interpreter = ResultsInterpreter()

    def test_interpret_t_test_significant(self):
        result = {'t_stat': 2.5, 'p_value': 0.01, 'degrees_of_freedom': 10}
        interpretation = self.interpreter.interpret_t_test(result)
        assert "statistically significant" in interpretation

    def test_interpret_t_test_not_significant(self):
        result = {'t_stat': 1.0, 'p_value': 0.3, 'degrees_of_freedom': 10}
        interpretation = self.interpreter.interpret_t_test(result)
        assert "not statistically significant" in interpretation

    def test_interpret_anova_significant(self):
        result = {'f_stat': 5.5, 'p_value': 0.01, 'df_between': 2, 'df_within': 27}
        interpretation = self.interpreter.interpret_anova(result)
        assert "significant differences" in interpretation

    def test_interpret_anova_not_significant(self):
        result = {'f_stat': 2.0, 'p_value': 0.2, 'df_between': 2, 'df_within': 27}
        interpretation = self.interpreter.interpret_anova(result)
        assert "no significant differences" in interpretation

    def test_calculate_effect_size(self):
        test_result = {'t_stat': 2.5, 'degrees_of_freedom': 10}
        effect_size = self.interpreter.calculate_effect_size(test_result)
        assert effect_size == pytest.approx(0.735, abs=0.001)

    def test_compute_confidence_interval(self):
        test_result = {'t_stat': 2.0, 'degrees_of_freedom': 10, 'standard_error': 0.5}
        confidence_interval = self.interpreter.compute_confidence_interval(test_result)
        assert len(confidence_interval) == 2
        assert confidence_interval[0] < confidence_interval[1]



class TestVisualizer:

    def setup(self):
        """Setup that creates a Visualizer instance."""
        self.visualizer = Visualizer()

    def test_plot_distribution(self):
        data = np.random.normal(loc=0, scale=1, size=100)
        try:
            self.visualizer.plot_distribution(data, "Normal Distribution")
            assert True  # If no exception, test passes
        except Exception:
            assert False  # Test fails if any exception is raised

    def test_create_result_summary_plot(self):
        results = {
            'categories': ['A', 'B', 'C'],
            'means': [1.5, 2.0, 1.8],
            'conf_int_lower': [0.1, 0.2, 0.1],
            'conf_int_upper': [0.2, 0.3, 0.2]
        }
        try:
            self.visualizer.create_result_summary_plot(results)
            assert True  # If no exception, test passes
        except Exception:
            assert False  # Test fails if any exception is raised

    def test_generate_correlation_matrix(self):
        data = pd.DataFrame({
            'var1': np.random.rand(100),
            'var2': np.random.rand(100),
            'var3': np.random.rand(100)
        })
        try:
            self.visualizer.generate_correlation_matrix(data)
            assert True  # If no exception, test passes
        except Exception:
            assert False  # Test fails if any exception is raised



def test_setup_logging_valid_levels(caplog):
    # Test if setting up all valid logging levels works without errors
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    for level in valid_levels:
        setup_logging(level)
        logger = getLogger()
        assert logger.level == getattr(logging, level)

def test_setup_logging_invalid_level():
    # Test if an invalid logging level raises a ValueError
    with pytest.raises(ValueError, match='Invalid log level: INVALID'):
        setup_logging("INVALID")

def test_logging_output(caplog):
    # Test if logging outputs messages as expected
    setup_logging("INFO")
    logger = getLogger()
    with caplog.at_level(logging.INFO):
        logger.info("Test info message")
    assert "Test info message" in caplog.text



def test_configure_integration_success(caplog):
    # Mock import module success
    with patch('importlib.import_module', return_value=MagicMock()) as mocked_import:
        configure_integration(['os', 'sys'])
        
        assert "Successfully integrated with os." in caplog.text
        assert "Successfully integrated with sys." in caplog.text
        mocked_import.assert_any_call('os')
        mocked_import.assert_any_call('sys')

def test_configure_integration_import_error(caplog):
    # Mock import module to raise ImportError
    with patch('importlib.import_module', side_effect=ImportError("No module named 'fake_lib'")):
        configure_integration(['fake_lib'])
        
        assert "Library fake_lib is not available. Please install it." in caplog.text

def test_configure_integration_generic_error(caplog):
    # Mock import module to raise a generic exception
    with patch('importlib.import_module', side_effect=Exception("Some error")):
        configure_integration(['os'])
        
        assert "An error occurred while configuring os: Some error" in caplog.text
