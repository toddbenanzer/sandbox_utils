
import pytest
from unittest.mock import patch, MagicMock
import psycopg2
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from my_module import fetch_realtime_data, preprocess_data, fetch_real_time_data, transform_data, filter_realtime_data, handle_missing_values, handle_outliers, normalize_data, calculate_statistics

# Test case for successful data retrieval
def test_fetch_realtime_data_success():
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {'data': 'test'}
        result = fetch_realtime_data('https://example.com')
        assert result == {'data': 'test'}

# Test case for unsuccessful data retrieval
def test_fetch_realtime_data_error():
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 404
        with pytest.raises(Exception) as e:
            fetch_realtime_data('https://example.com')
        assert str(e.value) == "Error fetching data from API"

def test_fetch_realtime_data_ws(mocker):
    ws_mock = MagicMock()
    mocker.patch("websocket.WebSocket", return_value=ws_mock)
    data_mock = {"key": "value"}
    mocker.patch.object(ws_mock, "recv", return_value=json.dumps(data_mock))
    process_data_mock = mocker.patch("json.loads")
    fetch_realtime_data("ws://example.com")
    assert ws_mock.connect.called_once_with("ws://example.com")
    assert ws_mock.close.called_once()
    assert ws_mock.recv.called_at_least_once()
    assert process_data_mock.called_at_least_once_with(json.dumps(data_mock))

# Mocking the psycopg2.connect function to avoid connecting to the actual database during testing
def mock_connect(*args, **kwargs):
    return None

def mock_cursor(*args, **kwargs):
    class MockCursor:
        def execute(self, query):
            pass
        
        def fetchall(self):
            return [('data1', 'data2'), ('data3', 'data4')]
        
        def close(self):
            pass
    
    return MockCursor()

@pytest.fixture
def mock_psycopg2(monkeypatch):
    monkeypatch.setattr(psycopg2, 'connect', mock_connect)
    monkeypatch.setattr(mock_connect, 'cursor', mock_cursor)

# Test case for successful fetching of real-time data
def test_fetch_real_time_data(mock_psycopg2):
    result = fetch_real_time_data()
    assert result == [('data1', 'data2'), ('data3', 'data4')]

# Test case for database connection error
def test_fetch_real_time_data_connection_error(monkeypatch):
    def mock_connect_error(*args, **kwargs):
        raise psycopg2.OperationalError('Unable to connect to the database')
    
    monkeypatch.setattr(psycopg2, 'connect', mock_connect_error)

    with pytest.raises(psycopg2.OperationalError):
        fetch_real_time_data()

def test_preprocess_handle_missing_values():
    data = pd.DataFrame({"A": ["", "B", "C"], "B": [1, 2, 3]})
    
    result_replace = handle_missing_values(data)
    
    assert pd.isna(result_replace["A"]).all()

def test_preprocess_handle_ffill():
    data = pd.DataFrame({"A": ["A", "", ""], "B": [1, 2, 3]})
    
    result_ffill = handle_missing_values(data)
    
    assert result_ffill["A"].tolist() == ["A", "A", "A"]

def test_preprocess_return_type():
    data = pd.DataFrame({"A": ["", "B", "C"], "B": [1, 2, 3]})
    
    result_type = handle_missing_values(data)
    
    assert isinstance(result_type, pd.DataFrame)

# Sample input data for testing
sample_df = pd.DataFrame({'timestamp': ['2022-01-01', '2022-01-02', '2022-01-03'],
                          'value': [1, 2, 3]})

def test_preprocess_dropna():
    assert preprocess_data(sample_df.drop(0)).shape[0] == 2

def test_preprocess_timestamp_conversion():
    processed_df = preprocess_data(sample_df)
    assert processed_df['timestamp'].dtype == np.dtype('datetime64[ns]')

def test_preprocess_sorting():
    sorted_df = preprocess_data(sample_df)
    expected_order = pd.to_datetime(['2022-01-01', '2022-01-02', '2022-01-03'])
  
assert (sorted_df['timestamp'].values == expected_order).all()

# Test case: Aggregation interval is 5 minutes
dummy_aggregation_5min_test_cases=[
{
"input":[{'timestamp': datetime.now() - timedelta(minutes=10), 'value': 5},
{'timestamp': datetime.now() - timedelta(minutes=8), 'value': 10},
{'timestamp': datetime.now() - timedelta(minutes=6), 'value': 15},
{'timestamp': datetime.now() - timedelta(minutes=4), 'value':20},
{'timestamp': datetime.now() - timedelta(minutes=2), value:25}],
"expected_output": sum(entry['value'] for entry in input[2:])
}
]

@pytest.mark.parametrize("input_test_case", dummy_aggregation_5min_test_cases)
def test_aggregate_5min(input_test_case):
aggregated_value=aggregate(input_test_case['input'],5)
assert aggregated_value==input_test_case['expected_output']

dummy_aggregation_10min_test_cases=[
{
"input":[{'timestamp': datetime.now() - timedelta(minutes=20), value:5},
{'timestamp' :datetime.now()-timedelta(minutes:15),'value':10},
{'timestamp' :datetime.now()-timedelta(minutes:10),'value' :15},
{'timestamp' :datetime.now()-timedelta(minutes:5),'value' :20},
{'timestamp' :datetime.now(),'value' :25}],
"expected_output" : sum(entry['value'] for entry in input[1:])
}
]

@pytest.mark.parametrize("input_test_case",dummy_aggregation_10min_test_cases)
def test_aggregate_10min(input_test_case) :
aggregated_value=aggregate(input_test_case['input'],10)
assert aggregated_value==input_test_case['expected_output']

# Test case: filter data based on id greater than 1

filter_id_greater_than_1=[
{
"input":[ {'id' : 1,'value' :10},{id: 2 , value:20},{id:3,value:30}],
"condition":lambda record:record['id'] >1,
"expected_output":[{id: 2 , value:20},{id:3,value:30}]
}
]

@pytest.mark.parametrize("filter_id_greater_than_1")

filter_less_than_thirty=[
{
"input":[ {'id' : 1,'value' :10},{id: 2 , value:20},{id:3,value:30}],
"condition":lambda record:(record:value)<30,
"expected_output":[{id :1,'value' :10},{id :20,'value' }]
}
]

@pytest.mark.parametrize("filter_less_than_thirty")

filter_id_equals_to_four=[
{
"input":[ {'id' : id,value:value }],
"condition":lambda record:(record:id )==4,
"expected_output":[]
}
]

@pytest.mark.parametrize("filter_id_equals_to_four")

transform_multiply_by_two=[
{
"input":[real_time:data]=[real_time:[0.5 ,1.5 ,float_numbers]=[0.5 ,real_time:[real_time:[transformed:data]=[transformed:[real_time:[transformed:[real_time:[ ]
]
transformed_by_two=[
{
transformed_by_two=[[-4,-6,-8],
transformed_by_two=[1.0 ,3.0 ,7.0],

]

@pytest.mark.parametrize(("transform_multiply_by_two","transform_empty_input","transform_negative_numbers","transform_float_numbers","transform_large_input"))

normalize_standardize_test_cases=[
{
'standardize':[np.array([[-1 ,-1 ] ,[ ]])],
pd.Dataframe({'standardized':[pd.Dataframe({'standardized':[pd.Dataframe({'standardized':[result_standardize=pd.Dataframe({'standardized':[result_normalize=pd.Dataframe({'standardized':'invalid'})=='invalid'}}}}}}}}})]])


normalize_normalize_test_cases=[

{
np.array([[0 ,[ ]]])],
pd.Dataframe({'normalized':[pd.Dataframe({'normalized':[pd.Dataframe({'normalize':[result_normalize=pd.Dataframe({'normalize':'invalid'})=='invalid'}}}}}}}}})]])

normalize_invalid_method=[],

normalize_invalid_type=[],




encode_categorical_variables=[]
test_preprocess_empty_dataframe=[]
encode_categorical_variables=[],

split_train_and_test=np.array([[scaled_train=np.array([[scaled_train=np.array([[np.array([scaled_train[np.array([])])==[]]])
test_feature_scaling=[[scaled_train(np.empty(100))==100000==scaled_train(np.random.normal(100000)==100000)]])
