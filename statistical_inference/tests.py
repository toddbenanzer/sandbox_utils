from your_module_name import BooleanTest
from your_module_name import CategoricalTest
from your_module_name import NumericTest
from your_module_name import StatisticalTest
from your_module_name import compute_descriptive_statistics
from your_module_name import impute_missing_values
from your_module_name import validate_input_data
from your_module_name import visualize_results
import matplotlib.pyplot as plt
import numpy as np
import pytest


def test_initialization():
    data1 = np.array([1, 2, 3])
    data2 = np.array([4, 5, 6])
    params = {}
    test = StatisticalTest(data1, data2, params)
    assert np.array_equal(test.data1, data1)
    assert np.array_equal(test.data2, data2)
    assert test.test_params == params

def test_check_zero_variance():
    data1 = np.array([1, 1, 1])
    data2 = np.array([2, 3, 4])
    params = {}
    test = StatisticalTest(data1, data2, params)
    assert test.check_zero_variance() is True

    data1 = np.array([1, 2, 3])
    data2 = np.array([4, 4, 4])
    test = StatisticalTest(data1, data2, params)
    assert test.check_zero_variance() is True

    data1 = np.array([1, 2, 3])
    data2 = np.array([4, 5, 6])
    test = StatisticalTest(data1, data2, params)
    assert test.check_zero_variance() is False

def test_handle_missing_values_remove():
    data1 = np.array([1, np.nan, 3])
    data2 = np.array([4, 5, np.nan])
    params = {'missing_values_method': 'remove'}
    test = StatisticalTest(data1, data2, params)
    test.handle_missing_values()
    assert np.array_equal(test.data1, np.array([1, 3]))
    assert np.array_equal(test.data2, np.array([4, 5]))

def test_handle_missing_values_impute():
    data1 = np.array([1, np.nan, 3])
    data2 = np.array([4, 5, np.nan])
    params = {'missing_values_method': 'impute'}
    test = StatisticalTest(data1, data2, params)
    test.handle_missing_values()
    assert np.allclose(test.data1, np.array([1, 2, 3]), equal_nan=True)
    assert np.allclose(test.data2, np.array([4, 5, 4.5]), equal_nan=True)

def test_handle_constant_values():
    data1 = np.array([1, 1, 1])
    data2 = np.array([2, 3, 4])
    params = {}
    test = StatisticalTest(data1, data2, params)
    test.handle_constant_values()
    # Outputs should be printed, here we are simply ensuring that the function does not raise errors
    assert True

    data1 = np.array([1, 2, 3])
    data2 = np.array([5, 5, 5])
    test = StatisticalTest(data1, data2, params)
    test.handle_constant_values()
    # Outputs should be printed, here we are simply ensuring that the function does not raise errors
    assert True



def test_perform_t_test():
    data1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    test_params = {}
    numeric_test = NumericTest(data1, data2, test_params)
    
    t_statistic, p_value = numeric_test.perform_t_test()
    
    assert isinstance(t_statistic, float), "Expected t-statistic to be a float."
    assert isinstance(p_value, float), "Expected p-value to be a float."
    assert p_value < 1.0, "Expected a valid p-value less than 1."

def test_perform_anova():
    data1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    test_params = {}
    numeric_test = NumericTest(data1, data2, test_params)
    
    f_statistic, p_value = numeric_test.perform_anova()
    
    assert isinstance(f_statistic, float), "Expected F-statistic to be a float."
    assert isinstance(p_value, float), "Expected p-value to be a float."
    assert p_value < 1.0, "Expected a valid p-value less than 1."

def test_t_test_with_missing_values():
    data1 = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
    data2 = np.array([2.0, 3.0, np.nan, 5.0, 6.0])
    test_params = {'missing_values_method': 'remove'}
    numeric_test = NumericTest(data1, data2, test_params)
    
    t_statistic, p_value = numeric_test.perform_t_test()
    
    assert isinstance(t_statistic, float), "Expected t-statistic to be a float."
    assert isinstance(p_value, float), "Expected p-value to be a float."
    assert p_value < 1.0, "Expected a valid p-value less than 1."

def test_anova_with_constant_values():
    data1 = np.array([4.0, 4.0, 4.0, 4.0, 4.0])
    data2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    test_params = {}
    numeric_test = NumericTest(data1, data2, test_params)
    
    f_statistic, p_value = numeric_test.perform_anova()
    
    assert isinstance(f_statistic, float), "Expected F-statistic to be a float."
    assert isinstance(p_value, float), "Expected p-value to be a float."
    assert p_value >= 0.0, "Expected a valid p-value greater than or equal to 0."



def test_perform_chi_squared_test():
    data1 = np.array([10, 20, 30])
    data2 = np.array([20, 30, 10])
    test_params = {}
    categorical_test = CategoricalTest(data1, data2, test_params)
    
    chi2_statistic, p_value, degrees_of_freedom, expected_frequencies = categorical_test.perform_chi_squared_test()
    
    assert isinstance(chi2_statistic, float), "Expected chi2_statistic to be a float."
    assert isinstance(p_value, float), "Expected p_value to be a float."
    assert isinstance(degrees_of_freedom, int), "Expected degrees_of_freedom to be an int."
    assert isinstance(expected_frequencies, np.ndarray), "Expected expected_frequencies to be a numpy array."
    assert p_value < 1.0, "Expected a valid p-value less than 1."

def test_chi_squared_with_missing_values():
    data1 = np.array([10, np.nan, 30])
    data2 = np.array([20, 30, np.nan])
    test_params = {'missing_values_method': 'remove'}
    categorical_test = CategoricalTest(data1, data2, test_params)
    
    chi2_statistic, p_value, degrees_of_freedom, expected_frequencies = categorical_test.perform_chi_squared_test()
    
    assert isinstance(chi2_statistic, float), "Expected chi2_statistic to be a float."
    assert isinstance(p_value, float), "Expected p_value to be a float."
    assert isinstance(degrees_of_freedom, int), "Expected degrees_of_freedom to be an int."
    assert isinstance(expected_frequencies, np.ndarray), "Expected expected_frequencies to be a numpy array."
    assert p_value < 1.0, "Expected a valid p-value less than 1."

def test_chi_squared_constant_values():
    data1 = np.array([5, 5, 5])
    data2 = np.array([5, 5, 5])
    test_params = {}
    categorical_test = CategoricalTest(data1, data2, test_params)
    
    chi2_statistic, p_value, degrees_of_freedom, expected_frequencies = categorical_test.perform_chi_squared_test()
    
    assert isinstance(chi2_statistic, float), "Expected chi2_statistic to be a float."
    assert isinstance(p_value, float), "Expected p_value to be a float."
    assert isinstance(degrees_of_freedom, int), "Expected degrees_of_freedom to be an int."
    assert isinstance(expected_frequencies, np.ndarray), "Expected expected_frequencies to be a numpy array."
    assert p_value >= 0.0, "Expected a valid p-value greater than or equal to 0."



def test_perform_proportions_test():
    data1 = np.array([True, True, False, True, False])
    data2 = np.array([False, True, False, False, True])
    test_params = {}
    boolean_test = BooleanTest(data1, data2, test_params)
    
    z_statistic, p_value = boolean_test.perform_proportions_test()
    
    assert isinstance(z_statistic, float), "Expected z_statistic to be a float."
    assert isinstance(p_value, float), "Expected p_value to be a float."
    assert p_value < 1.0, "Expected a valid p-value less than 1."

def test_proportions_test_with_missing_values():
    data1 = np.array([True, np.nan, True, False, np.nan])
    data2 = np.array([False, True, False, np.nan, True])
    test_params = {'missing_values_method': 'remove'}
    boolean_test = BooleanTest(data1, data2, test_params)
    
    z_statistic, p_value = boolean_test.perform_proportions_test()
    
    assert isinstance(z_statistic, float), "Expected z_statistic to be a float."
    assert isinstance(p_value, float), "Expected p_value to be a float."
    assert p_value < 1.0, "Expected a valid p-value less than 1."

def test_proportions_test_constant_values():
    data1 = np.array([True, True, True, True, True])
    data2 = np.array([False, False, False, False, False])
    test_params = {}
    boolean_test = BooleanTest(data1, data2, test_params)
    
    z_statistic, p_value = boolean_test.perform_proportions_test()
    
    assert isinstance(z_statistic, float), "Expected z_statistic to be a float."
    assert isinstance(p_value, float), "Expected p_value to be a float."
    assert p_value >= 0.0, "Expected a valid p-value greater than or equal to 0."



def test_compute_descriptive_statistics():
    data = np.array([1, 2, 3, 4, 5, 5, 6, 7, 8, 9])
    expected_stats = {
        'mean': 5.0,
        'median': 5.0,
        'mode': 5,
        'variance': 7.0,
        'standard_deviation': np.sqrt(7),
        'min': 1,
        'max': 9,
        'count': 10
    }
    statistics = compute_descriptive_statistics(data)
    
    for key in expected_stats:
        assert np.isclose(statistics[key], expected_stats[key]), f"Expected {key} to be {expected_stats[key]}, got {statistics[key]}"

def test_with_nan_and_inf_values():
    data = np.array([1, 2, np.nan, 4, np.inf, 6, -np.inf, 8, 9])
    expected_count = 6  # Only finite numbers
    statistics = compute_descriptive_statistics(data)
    
    assert statistics['count'] == expected_count, f"Expected count to be {expected_count}, got {statistics['count']}"

def test_with_single_value():
    data = np.array([10])
    expected_stats = {
        'mean': 10.0,
        'median': 10.0,
        'mode': 10,
        'variance': np.nan,  # Variance is undefined for a single value with ddof=1 in numpy
        'standard_deviation': np.nan,  # Standard deviation is undefined for a single value with ddof=1 in numpy
        'min': 10,
        'max': 10,
        'count': 1
    }
    statistics = compute_descriptive_statistics(data)
    
    for key in ['mean', 'median', 'mode', 'min', 'max', 'count']:
        assert np.isclose(statistics[key], expected_stats[key]), f"Expected {key} to be {expected_stats[key]}, got {statistics[key]}"
    
    for key in ['variance', 'standard_deviation']:
        assert np.isnan(statistics[key]), f"Expected {key} to be NaN, got {statistics[key]}"



@pytest.fixture
def sample_results():
    return {
        'data': np.random.normal(size=100),
        't_statistic': 1.96,
        'p_value': 0.05,
        'box_data': [np.random.normal(loc=0, scale=1, size=100), np.random.normal(loc=1, scale=1.5, size=100)],
        'category_proportions': {'A': 50, 'B': 30, 'C': 20}
    }

def test_visualize_histogram(sample_results):
    plt.ioff()  # Turn off interactive mode to prevent figures from displaying
    visualize_results({'data': sample_results['data']})
    plt.ion()  # Turn it back on
    # No assertions are needed for plotting, instead we ensure no errors during execution

def test_visualize_bar_chart(sample_results):
    plt.ioff()
    visualize_results({'t_statistic': sample_results['t_statistic'], 'p_value': sample_results['p_value']})
    plt.ion()

def test_visualize_box_plot(sample_results):
    plt.ioff()
    visualize_results({'box_data': sample_results['box_data']})
    plt.ion()

def test_visualize_pie_chart(sample_results):
    plt.ioff()
    visualize_results({'category_proportions': sample_results['category_proportions']})
    plt.ion()



def test_validate_correct_data():
    data1 = np.array([1, 2, 3, 4])
    data2 = np.array([5, 6, 7, 8])
    assert validate_input_data(data1, data2) is True

def test_invalid_data_type():
    with pytest.raises(ValueError, match="data1 must be a numpy array or a list."):
        validate_input_data("invalid_data_type", [1, 2, 3, 4])

    with pytest.raises(ValueError, match="data2 must be a numpy array or a list."):
        validate_input_data([1, 2, 3, 4], "invalid_data_type")

def test_non_empty_check():
    with pytest.raises(ValueError, match="data1 is empty."):
        validate_input_data([], [1, 2, 3])

    with pytest.raises(ValueError, match="data2 is empty."):
        validate_input_data([1, 2, 3], [])

def test_dimension_check():
    data1 = np.array([1, 2, 3])
    data2 = np.array([[4, 5], [6, 7]])
    with pytest.raises(ValueError, match="data1 and data2 must have the same number of dimensions."):
        validate_input_data(data1, data2)

def test_length_check():
    data1 = np.array([1, 2, 3, 4])
    data2 = np.array([5, 6, 7])
    with pytest.raises(ValueError, match="data1 and data2 must be of the same length for paired analysis."):
        validate_input_data(data1, data2)

def test_missing_values_check():
    data1 = np.array([np.nan, np.nan, np.nan, 4])
    data2 = np.array([5, 6, 7, 8])
    with pytest.raises(ValueError, match="data1 contains too many missing values."):
        validate_input_data(data1, data2)

    data1 = np.array([1, 2, 3, 4])
    data2 = np.array([np.nan, np.nan, np.nan, np.nan])
    with pytest.raises(ValueError, match="data2 contains too many missing values."):
        validate_input_data(data1, data2)



def test_mean_imputation():
    data = np.array([1, np.nan, 3, 4, np.nan])
    imputed_data = impute_missing_values(data, method='mean')
    expected_data = np.array([1, 2.66666667, 3, 4, 2.66666667])
    assert np.allclose(imputed_data, expected_data, equal_nan=True)

def test_median_imputation():
    data = np.array([1, np.nan, 3, 4, 5])
    imputed_data = impute_missing_values(data, method='median')
    expected_data = np.array([1, 3, 3, 4, 5])
    assert np.allclose(imputed_data, expected_data, equal_nan=True)

def test_mode_imputation():
    data = np.array([1, np.nan, 3, 3, 5, np.nan])
    imputed_data = impute_missing_values(data, method='mode')
    expected_data = np.array([1, 3, 3, 3, 5, 3])
    assert np.array_equal(imputed_data, expected_data)

def test_zero_imputation():
    data = np.array([1, np.nan, 3, 4, np.nan])
    imputed_data = impute_missing_values(data, method='zero')
    expected_data = np.array([1, 0, 3, 4, 0])
    assert np.array_equal(imputed_data, expected_data)

def test_invalid_method():
    data = np.array([1, 2, np.nan])
    with pytest.raises(ValueError, match="Unsupported imputation method"):
        impute_missing_values(data, method='unknown')
