from calculate_statistics import calculate_statistics
from data_handler import DataHandler
from descriptive_statistics import DescriptiveStatistics
from detect_outliers import detect_outliers
from estimate_distribution import estimate_likely_distribution
from visualize_distribution import visualize_distribution
import numpy as np
import pandas as pd
import pytest


def test_compute_central_tendency():
    data = pd.Series([1, 2, 2, 3, 4, 5, np.nan])
    stats = DescriptiveStatistics(data)
    result = stats.compute_central_tendency()
    assert result['mean'] == pytest.approx(2.83, 0.01)
    assert result['median'] == 2.5
    assert result['mode'] == 2

def test_compute_dispersion():
    data = pd.Series([1, 2, 2, 3, 4, 5, np.nan])
    stats = DescriptiveStatistics(data)
    result = stats.compute_dispersion()
    assert result['variance'] == pytest.approx(2.16, 0.01)
    assert result['std_dev'] == pytest.approx(1.47, 0.01)
    assert result['range'] == 4
    assert result['IQR'] == 2

def test_detect_outliers_z_score():
    data = pd.Series([1, 2, 2, 3, 4, 5, 100])
    stats = DescriptiveStatistics(data)
    outliers = stats.detect_outliers(method='z-score')
    assert outliers == [100]

def test_detect_outliers_iqr():
    data = pd.Series([1, 2, 2, 3, 4, 5, 100])
    stats = DescriptiveStatistics(data)
    outliers = stats.detect_outliers(method='IQR')
    assert outliers == [100]

def test_detect_outliers_invalid_method():
    data = pd.Series([1, 2, 3])
    stats = DescriptiveStatistics(data)
    with pytest.raises(ValueError, match="Unsupported method provided for outlier detection."):
        stats.detect_outliers(method='invalid')

def test_estimate_distribution():
    data = pd.Series(np.random.normal(0, 1, 1000))
    stats = DescriptiveStatistics(data)
    distribution = stats.estimate_distribution()
    assert distribution == 'normal'



def test_handle_missing_values_mean():
    data = pd.Series([1, 2, np.nan, 4])
    handler = DataHandler(data)
    result = handler.handle_missing_values(strategy='mean')
    expected = pd.Series([1, 2, 2.333, 4])
    pd.testing.assert_series_equal(result, expected, atol=0.001)

def test_handle_missing_values_median():
    data = pd.Series([1, 2, np.nan, 4])
    handler = DataHandler(data)
    result = handler.handle_missing_values(strategy='median')
    expected = pd.Series([1, 2, 2, 4])
    pd.testing.assert_series_equal(result, expected)

def test_handle_missing_values_mode():
    data = pd.Series([1, 2, 2, np.nan, 4])
    handler = DataHandler(data)
    result = handler.handle_missing_values(strategy='mode')
    expected = pd.Series([1, 2, 2, 2, 4])
    pd.testing.assert_series_equal(result, expected)

def test_handle_missing_values_drop():
    data = pd.Series([1, 2, np.nan, 4])
    handler = DataHandler(data)
    result = handler.handle_missing_values(strategy='drop')
    expected = pd.Series([1, 2, 4])
    pd.testing.assert_series_equal(result.reset_index(drop=True), expected)

def test_handle_infinite_values():
    data = pd.Series([1, np.inf, 3, -np.inf])
    handler = DataHandler(data)
    result = handler.handle_infinite_values(strategy='mean')
    expected = pd.Series([1, 2, 3, 2])
    pd.testing.assert_series_equal(result, expected)

def test_check_null_trivial_all_null():
    data = pd.Series([np.nan, np.nan])
    handler = DataHandler(data)
    assert handler.check_null_trivial() is True

def test_check_null_trivial_single_value():
    data = pd.Series([5, 5, np.nan, 5])
    handler = DataHandler(data)
    assert handler.check_null_trivial() is True

def test_check_null_trivial_regular_values():
    data = pd.Series([1, 2, 3, 4])
    handler = DataHandler(data)
    assert handler.check_null_trivial() is False



def test_calculate_statistics_non_trivial_data():
    data = pd.Series([1, 2, 2, 3, 4, 5])
    results = calculate_statistics(data)
    
    assert 'central_tendency' in results
    assert 'dispersion' in results
    assert 'outliers' in results
    assert 'estimated_distribution' in results
    assert results['central_tendency']['mean'] == pytest.approx(2.833, 0.001)
    assert results['dispersion']['variance'] == pytest.approx(2.167, 0.001)

def test_calculate_statistics_trivial_data():
    data = pd.Series([5, 5, 5, 5])
    results = calculate_statistics(data)
    
    assert results['message'] == "The data is null or trivial and cannot be processed for statistics."
    assert results['handled_data'].equals(data)

def test_calculate_statistics_with_missing_infinite_values():
    data = pd.Series([1, 2, np.nan, 4, np.inf, -np.inf])
    results = calculate_statistics(data)
    
    assert 'central_tendency' in results
    assert 'dispersion' in results
    assert results['central_tendency']['mean'] == pytest.approx(2.25, 0.001)
    assert results['handled_data'].isnull().sum() == 0

def test_calculate_statistics_empty_data():
    data = pd.Series([])
    results = calculate_statistics(data)
    
    assert results['message'] == "The data is null or trivial and cannot be processed for statistics."
    assert results['handled_data'].equals(data)



def test_visualize_distribution_empty_data(capfd):
    data = pd.Series([])
    visualize_distribution(data)
    captured = capfd.readouterr()
    assert "The provided data series is empty." in captured.out

def test_visualize_distribution_non_numeric_data(capfd):
    data = pd.Series(['a', 'b', 'c'])
    visualize_distribution(data)
    captured = capfd.readouterr()
    assert "The provided data series is not numeric." in captured.out

def test_visualize_distribution_numeric_data():
    data = pd.Series([1, 2, 3, 4, 5, 6])
    # Visual test - No assertion possible, mock `plt.show()` if needed
    visualize_distribution(data)



def test_detect_outliers_empty_data(capfd):
    data = pd.Series([])
    outliers = detect_outliers(data)
    assert outliers == []
    captured = capfd.readouterr()
    assert "The provided data series is empty." in captured.out

def test_detect_outliers_non_numeric_data(capfd):
    data = pd.Series(['a', 'b', 'c'])
    outliers = detect_outliers(data)
    assert outliers == []
    captured = capfd.readouterr()
    assert "The provided data series is not numeric." in captured.out

def test_detect_outliers_no_outliers():
    data = pd.Series([1, 2, 3, 4, 5, 6])
    outliers = detect_outliers(data)
    assert outliers == []

def test_detect_outliers_with_outliers():
    data = pd.Series([1, 2, 3, 4, 100])
    outliers = detect_outliers(data)
    assert outliers == [100]

def test_detect_outliers_unsupported_method():
    data = pd.Series([1, 2, 3])
    with pytest.raises(ValueError, match="Unsupported method provided for outlier detection."):
        detect_outliers(data, method='unsupported')



def test_estimate_likely_distribution_empty_data(capfd):
    data = pd.Series([])
    distribution = estimate_likely_distribution(data)
    assert distribution == "No Data"
    captured = capfd.readouterr()
    assert "The provided data series is empty." in captured.out

def test_estimate_likely_distribution_non_numeric_data(capfd):
    data = pd.Series(['a', 'b', 'c'])
    distribution = estimate_likely_distribution(data)
    assert distribution == "Non-numeric Data"
    captured = capfd.readouterr()
    assert "The provided data series is not numeric." in captured.out

def test_estimate_likely_distribution_normal():
    data = pd.Series(np.random.normal(loc=0, scale=1, size=1000))
    distribution = estimate_likely_distribution(data)
    assert distribution == "normal"

def test_estimate_likely_distribution_uniform():
    data = pd.Series(np.random.uniform(low=0, high=1, size=1000))
    distribution = estimate_likely_distribution(data)
    assert distribution == "uniform"

def test_estimate_likely_distribution_exponential():
    data = pd.Series(np.random.exponential(scale=1, size=1000))
    distribution = estimate_likely_distribution(data)
    assert distribution == "exponential"
