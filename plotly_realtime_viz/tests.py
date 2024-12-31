from data_preprocessing_module import DataCleaner, DataTransformer  # Assuming the classes are in a module named data_preprocessing_module
from realtime_plot_module import RealTimePlot  # Assume the class is in a module named realtime_plot_module
from tempfile import NamedTemporaryFile
from unittest.mock import patch, Mock
from user_interaction_module import enable_zoom, enable_pan, hover_info, update_plot_params  # Assume the functions are in a module named user_interaction_module
from utils_module import configure_logging, read_config  # Assume the functions are in a module named utils_module
import json
import logging
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
import pytest
import requests


# Assume that DataConnector is already imported

def test_init():
    connector = DataConnector("http://example.com", "HTTP")
    assert connector.source == "http://example.com"
    assert connector.protocol == "http"
    assert connector.connection is None

@patch('requests.Session')
def test_connect_http(mock_session):
    connector = DataConnector("http://example.com", "HTTP")
    connection = connector.connect()
    
    mock_session.assert_called_once()
    assert connection is not None
    assert connector.connection == connection

@patch('websocket.create_connection')
def test_connect_websocket(mock_create_connection):
    mock_create_connection.return_value = Mock()
    connector = DataConnector("ws://example.com", "WebSocket")
    connection = connector.connect()
    
    mock_create_connection.assert_called_once_with("ws://example.com")
    assert connection is not None
    assert connector.connection == connection

def test_connect_unsupported_protocol():
    connector = DataConnector("http://example.com", "FTP")
    with pytest.raises(ValueError) as exc_info:
        connector.connect()
    assert str(exc_info.value) == "Unsupported protocol: ftp"

@patch('requests.get')
def test_fetch_data_http(mock_get):
    mock_response = Mock()
    mock_response.ok = True
    mock_response.json.return_value = {"key": "value"}
    mock_get.return_value = mock_response

    connector = DataConnector("http://example.com", "HTTP")
    connector.connect()
    data = connector.fetch_data()

    mock_get.assert_called_once_with("http://example.com")
    assert data == {"key": "value"}

@patch.object(websocket, 'create_connection')
def test_fetch_data_websocket(mock_create_connection):
    mock_conn = Mock()
    mock_conn.recv.return_value = '{"key": "value"}'
    mock_create_connection.return_value = mock_conn

    connector = DataConnector("ws://example.com", "WebSocket")
    connector.connect()
    data = connector.fetch_data()

    assert data == '{"key": "value"}'
    mock_conn.recv.assert_called_once()

@patch.object(DataConnector, 'connect')
def test_reconnect_success(mock_connect):
    connector = DataConnector("http://example.com", "HTTP")
    result = connector.reconnect()
    
    mock_connect.assert_called_once()
    assert result is True

@patch.object(DataConnector, 'connect', side_effect=Exception("Failed"))
def test_reconnect_failure(mock_connect):
    connector = DataConnector("http://example.com", "HTTP")
    result = connector.reconnect()
    
    mock_connect.assert_called_once()
    assert result is False



def test_handle_missing_values_drop():
    data = pd.DataFrame({'A': [1, 2, np.nan], 'B': [np.nan, 5, 6]})
    cleaner = DataCleaner(data)
    cleaned_data = cleaner.handle_missing_values(method='drop')
    expected_data = pd.DataFrame({'A': [2.0], 'B': [5.0]})
    pd.testing.assert_frame_equal(cleaned_data.reset_index(drop=True), expected_data)

def test_handle_missing_values_fill_mean():
    data = pd.DataFrame({'A': [1, 2, np.nan], 'B': [np.nan, 5, 6]})
    cleaner = DataCleaner(data)
    cleaned_data = cleaner.handle_missing_values(method='fill_mean')
    expected_data = pd.DataFrame({'A': [1, 2, 1.5], 'B': [5.5, 5, 6]})
    pd.testing.assert_frame_equal(cleaned_data, expected_data)

def test_remove_outliers_z_score():
    data = pd.DataFrame({'A': [1, 2, 100], 'B': [1, 5, 6]})
    cleaner = DataCleaner(data)
    cleaned_data = cleaner.remove_outliers(method='z-score')
    expected_data = pd.DataFrame({'A': [1, 2], 'B': [1, 5]})
    pd.testing.assert_frame_equal(cleaned_data.reset_index(drop=True), expected_data)

def test_remove_outliers_iqr():
    data = pd.DataFrame({'A': [1, 2, 100, 2], 'B': [1, 5, 6, 1.5]})
    cleaner = DataCleaner(data)
    cleaned_data = cleaner.remove_outliers(method='iqr')
    expected_data = pd.DataFrame({'A': [1, 2, 2], 'B': [1, 5, 1.5]})
    pd.testing.assert_frame_equal(cleaned_data.reset_index(drop=True), expected_data)

def test_normalize_data_min_max():
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    cleaner = DataCleaner(data)
    normalized_data = cleaner.normalize_data(strategy='min-max')
    expected_data = pd.DataFrame({'A': [0.0, 0.5, 1.0], 'B': [0.0, 0.5, 1.0]})
    pd.testing.assert_frame_equal(normalized_data, expected_data)

def test_normalize_data_z_score():
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    cleaner = DataCleaner(data)
    normalized_data = cleaner.normalize_data(strategy='z-score')
    expected_data = pd.DataFrame({'A': [-1.224744871391589, 0.0, 1.224744871391589], 
                                  'B': [-1.224744871391589, 0.0, 1.224744871391589]})
    pd.testing.assert_frame_equal(normalized_data, expected_data)

def test_aggregate_data_sum():
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    transformer = DataTransformer(data)
    aggregated_data = transformer.aggregate_data(method='sum')
    expected_data = pd.Series({'A': 6, 'B': 15})
    pd.testing.assert_series_equal(aggregated_data, expected_data)

def test_aggregate_data_mean():
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    transformer = DataTransformer(data)
    aggregated_data = transformer.aggregate_data(method='mean')
    expected_data = pd.Series({'A': 2.0, 'B': 5.0})
    pd.testing.assert_series_equal(aggregated_data, expected_data)



def test_init_properties():
    plot = RealTimePlot('line', title='Test Plot', xaxis_title='x', yaxis_title='y')
    assert isinstance(plot.plot, go.Figure)
    assert plot.plot_type == 'line'
    assert plot.plot.layout.title.text == 'Test Plot'
    assert plot.plot.layout.xaxis.title.text == 'x'
    assert plot.plot.layout.yaxis.title.text == 'y'

def test_update_plot_dataframe():
    data = pd.DataFrame({'value': [1, 2, 3]}, index=[0, 1, 2])
    plot = RealTimePlot('line')
    plot.update_plot(data)
    assert len(plot.plot.data) == 1
    assert plot.plot.data[0].x.tolist() == [0, 1, 2]
    assert plot.plot.data[0].y.tolist() == [1, 2, 3]

def test_update_plot_dict():
    data = {'x': [0, 1, 2], 'y': [3, 4, 5]}
    plot = RealTimePlot('scatter')
    plot.update_plot(data)
    assert len(plot.plot.data) == 1
    assert plot.plot.data[0].x.tolist() == [0, 1, 2]
    assert plot.plot.data[0].y.tolist() == [3, 4, 5]

def test_update_plot_invalid_data():
    plot = RealTimePlot('bar')
    with pytest.raises(ValueError) as exc_info:
        plot.update_plot([1, 2, 3])  # Invalid data type
    
    assert "new_data must be a pandas DataFrame or a dictionary with 'x' and 'y' keys." in str(exc_info.value)

def test_update_plot_invalid_plot_type():
    data = {'x': [0, 1, 2], 'y': [3, 4, 5]}
    plot = RealTimePlot('invalid_type')  # Invalid plot type
    with pytest.raises(ValueError) as exc_info:
        plot.update_plot(data)
    
    assert "Unsupported plot_type" in str(exc_info.value)

def test_customize_plot():
    plot = RealTimePlot('line')
    plot.customize_plot(bgcolor='lightgrey', title_font_size=18)
    assert plot.plot.layout.plot_bgcolor == 'lightgrey'
    assert plot.plot.layout.title.font.size == 18



def test_enable_zoom():
    plot = go.Figure()
    enable_zoom(plot)
    assert plot.layout.dragmode == 'zoom'

def test_enable_pan():
    plot = go.Figure()
    enable_pan(plot)
    assert plot.layout.dragmode == 'pan'

def test_hover_info():
    plot = go.Figure(data=[go.Scatter(x=[1, 2], y=[3, 4], text=["Point 1", "Point 2"])])
    hover_info(plot)
    assert all(trace.hoverinfo == 'text+x+y' for trace in plot.data)
    assert all(trace.hovertemplate == '%{text}<br>X: %{x}<br>Y: %{y}' for trace in plot.data)

def test_update_plot_params():
    plot = go.Figure()
    update_plot_params(plot, title='Test Title', xaxis_range=[0, 10])
    assert plot.layout.title.text == 'Test Title'
    assert plot.layout.xaxis.range == [0, 10]



def test_configure_logging_valid_level(caplog):
    with caplog.at_level(logging.INFO):
        configure_logging('info')
        assert 'Logging configured at INFO level.' in caplog.text

def test_configure_logging_invalid_level():
    with pytest.raises(ValueError) as exc_info:
        configure_logging('invalid_level')
    assert 'Invalid log level' in str(exc_info.value)

def test_read_config_success():
    with NamedTemporaryFile('w', delete=False) as temp_file:
        json.dump({"key": "value"}, temp_file)
        temp_file_path = temp_file.name

    try:
        config = read_config(temp_file_path)
        assert config == {"key": "value"}
    finally:
        os.remove(temp_file_path)

def test_read_config_file_not_found(caplog):
    fake_file_path = 'non_existent_file.json'
    with pytest.raises(Exception):
        read_config(fake_file_path)
        assert f"Failed to read configuration from {fake_file_path}" in caplog.text

def test_read_config_invalid_json_format(caplog):
    with NamedTemporaryFile('w', delete=False) as temp_file:
        temp_file.write("{invalid_json}")
        temp_file_path = temp_file.name

    try:
        with pytest.raises(Exception):
            read_config(temp_file_path)
            assert f"Failed to read configuration from {temp_file_path}" in caplog.text
    finally:
        os.remove(temp_file_path)
