from your_module_name import DataAggregator  # Replace with the actual module name
from your_module_name import DataBinner  # Replace with the actual module name
from your_module_name import DataNormalizer  # Make sure to replace with your actual module name
from your_module_name import DataValidator  # Replace with the actual module name
import numpy as np
import pandas as pd
import pytest


def test_min_max_normalize_list():
    data = [1, 2, 3, 4, 5]
    normalizer = DataNormalizer(data)
    normalized_data = normalizer.normalize_data('min-max')
    expected = [0.0, 0.25, 0.5, 0.75, 1.0]
    assert np.allclose(normalized_data, expected)

def test_z_score_normalize_list():
    data = [1, 2, 3, 4, 5]
    normalizer = DataNormalizer(data)
    normalized_data = normalizer.normalize_data('z-score')
    expected = [-1.26491, -0.632455, 0.0, 0.632455, 1.26491]
    assert np.allclose(normalized_data, expected, atol=1e-5)

def test_min_max_normalize_dataframe():
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    normalizer = DataNormalizer(data)
    normalized_data = normalizer.normalize_data('min-max')
    expected = pd.DataFrame({'A': [0.0, 0.5, 1.0], 'B': [0.0, 0.5, 1.0]})
    pd.testing.assert_frame_equal(normalized_data, expected)

def test_z_score_normalize_dataframe():
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    normalizer = DataNormalizer(data)
    normalized_data = normalizer.normalize_data('z-score')
    expected = pd.DataFrame({'A': [-1.0, 0.0, 1.0], 'B': [-1.0, 0.0, 1.0]})
    pd.testing.assert_frame_equal(normalized_data, expected)

def test_min_max_normalize_ndarray():
    data = np.array([[1, 2], [2, 3], [3, 4]])
    normalizer = DataNormalizer(data)
    normalized_data = normalizer.normalize_data('min-max')
    expected = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    assert np.allclose(normalized_data, expected)

def test_z_score_normalize_ndarray():
    data = np.array([[1, 2], [2, 3], [3, 4]])
    normalizer = DataNormalizer(data)
    normalized_data = normalizer.normalize_data('z-score')
    expected = np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
    assert np.allclose(normalized_data, expected)

def test_unsupported_method():
    data = [1, 2, 3, 4, 5]
    normalizer = DataNormalizer(data)
    with pytest.raises(ValueError, match="Unsupported normalization method 'unsupported'. Choose 'min-max' or 'z-score'."):
        normalizer.normalize_data('unsupported')

def test_invalid_data_initialization():
    with pytest.raises(ValueError, match="Data must be a list, pandas DataFrame, or numpy ndarray."):
        DataNormalizer(123)  # Invalid data type



def test_bin_data_list_with_bin_count():
    data = [1, 2, 3, 4, 5]
    binner = DataBinner(data)
    binned_data = binner.bin_data(2)
    expected = [0, 0, 0, 1, 1]
    assert binned_data == expected

def test_bin_data_list_with_bin_edges():
    data = [1, 2, 3, 4, 5]
    binner = DataBinner(data)
    binned_data = binner.bin_data([1, 3, 5])
    expected = [0, 0, 1, 1, 2]
    assert binned_data == expected

def test_bin_data_dataframe_with_bin_count():
    data = pd.DataFrame({'A': [10, 20, 30], 'B': [40, 50, 60]})
    binner = DataBinner(data)
    binned_data = binner.bin_data(3)
    expected = pd.DataFrame({'A': [0, 1, 2], 'B': [0, 1, 2]})
    pd.testing.assert_frame_equal(binned_data, expected)

def test_bin_data_dataframe_with_bin_edges():
    data = pd.DataFrame({'A': [10, 20, 30], 'B': [40, 50, 60]})
    binner = DataBinner(data)
    binned_data = binner.bin_data([10, 25, 60])
    expected = pd.DataFrame({'A': [0, 0, 1], 'B': [0, 0, 1]})
    pd.testing.assert_frame_equal(binned_data, expected)

def test_bin_data_ndarray_with_bin_count():
    data = np.array([[5, 10], [15, 20], [25, 30]])
    binner = DataBinner(data)
    binned_data = binner.bin_data(3)
    expected = np.array([[0, 0], [1, 1], [2, 2]])
    assert np.array_equal(binned_data, expected)

def test_bin_data_ndarray_with_bin_edges():
    data = np.array([[5, 10], [15, 20], [25, 30]])
    binner = DataBinner(data)
    binned_data = binner.bin_data([5, 15, 30])
    expected = np.array([[0, 0], [1, 1], [2, 2]])
    assert np.array_equal(binned_data, expected)

def test_invalid_bin_specifications():
    data = [1, 2, 3, 4, 5]
    binner = DataBinner(data)
    with pytest.raises(ValueError, match="Number of bins must be a positive integer."):
        binner.bin_data(0)

    with pytest.raises(ValueError, match="Bin edges must be a list of scalars."):
        binner.bin_data(['a', 'b', 'c'])

def test_invalid_data_initialization():
    with pytest.raises(ValueError, match="Data must be a list, pandas DataFrame, or numpy ndarray."):
        DataBinner("invalid data type")  # Invalid data type



def test_aggregate_single_column():
    data = pd.DataFrame({
        'Category': ['A', 'A', 'B', 'B'],
        'Value': [10, 20, 30, 40]
    })
    aggregator = DataAggregator(data)
    result = aggregator.aggregate_data('Category', 'sum')
    expected = pd.DataFrame({'Value': [30, 70]}, index=['A', 'B'])
    pd.testing.assert_frame_equal(result, expected)

def test_aggregate_multiple_columns():
    data = pd.DataFrame({
        'Category': ['A', 'A', 'B', 'B'],
        'Value1': [10, 20, 30, 40],
        'Value2': [5, 15, 25, 35]
    })
    aggregator = DataAggregator(data)
    result = aggregator.aggregate_data('Category', {'Value1': 'sum', 'Value2': 'mean'})
    expected = pd.DataFrame({'Value1': [30, 70], 'Value2': [10, 30]}, index=['A', 'B'])
    pd.testing.assert_frame_equal(result, expected)

def test_aggregate_by_multiple_columns():
    data = pd.DataFrame({
        'Category': ['A', 'A', 'B', 'B'],
        'Type': ['X', 'Y', 'X', 'Y'],
        'Value': [10, 20, 30, 40]
    })
    aggregator = DataAggregator(data)
    result = aggregator.aggregate_data(['Category', 'Type'], {'Value': 'sum'})
    expected = pd.DataFrame({'Value': [10, 20, 30, 40]}, index=pd.MultiIndex.from_tuples([('A', 'X'), ('A', 'Y'), ('B', 'X'), ('B', 'Y')], names=['Category', 'Type']))
    pd.testing.assert_frame_equal(result, expected)

def test_invalid_groupby_column():
    data = pd.DataFrame({
        'Category': ['A', 'A', 'B', 'B'],
        'Value': [10, 20, 30, 40]
    })
    aggregator = DataAggregator(data)
    with pytest.raises(ValueError, match="Group by column 'Invalid' does not exist in the DataFrame."):
        aggregator.aggregate_data('Invalid', 'sum')

def test_invalid_aggregation_function():
    data = pd.DataFrame({
        'Category': ['A', 'A', 'B', 'B'],
        'Value': [10, 20, 30, 40]
    })
    aggregator = DataAggregator(data)
    with pytest.raises(ValueError, match="Aggregation function failed with error:"):
        aggregator.aggregate_data('Category', {'Value': 'nonexistent_func'})

def test_invalid_data_initialization():
    with pytest.raises(ValueError, match="Data must be a pandas DataFrame."):
        DataAggregator([1, 2, 3, 4])  # Invalid data type



def test_validate_dataframe():
    data = pd.DataFrame({
        'col1': [1, 2, np.nan],
        'col2': [np.nan, 2, 3],
    })
    validator = DataValidator(data)
    validation_report = validator.validate_data()
    
    expected_missing_values = {'col1': 1, 'col2': 1}
    expected_data_types = {'col1': 'float64', 'col2': 'float64'}
    
    assert validation_report['missing_values'] == expected_missing_values
    assert validation_report['data_type_inconsistencies'] == expected_data_types

def test_validate_ndarray():
    data = np.array([1, 2, np.nan])
    validator = DataValidator(data)
    validation_report = validator.validate_data()
    
    expected_missing_values = 1
    expected_data_type = 'float64'
    
    assert validation_report['missing_values'] == expected_missing_values
    assert str(validation_report['data_type_inconsistencies']) == expected_data_type

def test_validate_list():
    data = [1, 2, np.nan]
    validator = DataValidator(data)
    validation_report = validator.validate_data()
    
    expected_missing_values = 1
    expected_data_type = 'float64'
    
    assert validation_report['missing_values'] == expected_missing_values
    assert str(validation_report['data_type_inconsistencies']) == expected_data_type

def test_invalid_data_initialization():
    with pytest.raises(ValueError, match="Data must be a list, pandas DataFrame, or numpy ndarray."):
        DataValidator("invalid data type")  # Invalid data type
