andas as pd
import numpy as np
import pytest

@pytest.fixture
def sample_dataframe():
    data = {'Column1': [1, 2, 3, None, 'apple', True],
            'Column2': ['2021-01-01', '2021-02-02', '2021-03-03', '2021-04-04', '2021-05-05', '2021-06-06'],
            'Column3': [np.nan, np.nan, np.nan, np.nan, np.inf, -np.inf]}
    return pd.DataFrame(data)

def test_handle_mixed_data_trivial_column(sample_dataframe):
    column_name = 'Column1'
    expected_result = pd.DataFrame({'Column1': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                    'Column2': ['2021-01-01', '2021-02-02', '2021-03-03', '2021-04-04', '2021-05-05', '2021-06-06'],
                                    'Column3': [np.nan, np.nan, np.nan, np.nan, np.inf, -np.inf]})
    
    result = handle_mixed_data(sample_dataframe.copy(), column_name)
    
    assert result.equals(expected_result)

def test_handle_mixed_data_null_column(sample_dataframe):
    column_name = 'Column3'
    expected_result = pd.DataFrame({'Column1': [1, 2, 3, None, 'apple', True],
                                    'Column2': ['2021-01-01', '2021-02-02', '2021-03-03', '2021-04-04', '2021-05-05', '2021-06-06'],
                                    'Column3': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]})
    
    result = handle_mixed_data(sample_dataframe.copy(), column_name)
    
    assert result.equals(expected_result)

def test_handle_mixed_data_boolean_column(sample_dataframe):
    column_name = 'Column1'
    expected_result = pd.DataFrame({'Column1': ['True', 'True', 'True', 'nan', 'apple', 'True'],
                                    'Column2': ['2021-01-01', '2021-02-02', '2021-03-03', '2021-04-04', '2021-05-05', '2021-06-06'],
                                    'Column3': [np.nan, np.nan, np.nan, np.nan, np.inf, -np.inf]})
    
    result = handle_mixed_data(sample_dataframe.copy(), column_name)
    
    assert result.equals(expected_result)

def test_handle_mixed_data_categorical_column(sample_dataframe):
    column_name = 'Column2'
    expected_result = pd.DataFrame({'Column1': [1, 2, 3, None, 'apple', True],
                                    'Column2': ['2021-01-01', '2021-02-02', '2021-03-03', '2021-04-04', '2021-05-05', '2021-06-06'],
                                    'Column3': [np.nan, np.nan, np.nan, np.nan, np.inf, -np.inf]})
    expected_result['Column2'] = expected_result['Column2'].astype(str)
    
    result = handle_mixed_data(sample_dataframe.copy(), column_name)
    
    assert result.equals(expected_result)

def test_handle_mixed_data_datetime_column(sample_dataframe):
    column_name = 'Column2'
    expected_result = pd.DataFrame({'Column1': [1, 2, 3, None, 'apple', True],
                                    'Column2': pd.to_datetime(['2021-01-01', '2021-02-02', '2021-03-03', '2021-04-04', '2021-05-05', '2021-06-06']),
                                    'Column3': [np.nan, np.nan, np.nan, np.nan, np.inf, -np.inf]})
    
    result = handle_mixed_data(sample_dataframe.copy(), column_name)
    
    assert result.equals(expected_result)

def test_handle_mixed_data_infinite_float_column(sample_dataframe):
    column_name = 'Column3'
    expected_result = pd.DataFrame({'Column1': [1, 2, 3, None, 'apple', True],
                                    'Column2': ['2021-01-01', '2021-02-02', '2021-03-03', '2021-04-04', '2021-05-05', '2021-06-06'],
                                    'Column3': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]})
    
    result = handle_mixed_data(sample_dataframe.copy(), column_name)
    
    assert result.equals(expected_result)

def test_handle_mixed_data_string_column(sample_dataframe):
    column_name = 'Column4'
    expected_result = sample_dataframe.copy()
    
    result = handle_mixed_data(sample_dataframe.copy(), column_name)
    
    assert result.equals(expected_result