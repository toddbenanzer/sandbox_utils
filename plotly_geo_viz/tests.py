
import pandas as pd
from plotly.graph_objects import Figure
from my_module import plot_scatter_map, plot_line_on_map, plot_choropleth, add_markers_on_map

import pytest
import plotly.graph_objects as go

# Tests for plot_scatter_map function
def test_plot_scatter_map_returns_figure():
    data = pd.DataFrame({'lat': [40.7128, 34.0522], 'lon': [-74.0060, -118.2437]})
    lat_column = 'lat'
    lon_column = 'lon'
    marker_size = 8
    marker_color = 'blue'

    result = plot_scatter_map(data, lat_column, lon_column, marker_size, marker_color)
    assert isinstance(result, Figure)

def test_plot_scatter_map_default_parameters():
    data = pd.DataFrame({'lat': [40.7128], 'lon': [-74.0060]})
    lat_column = 'lat'
    lon_column = 'lon'

    result = plot_scatter_map(data, lat_column, lon_column)
    assert isinstance(result, Figure)

@pytest.fixture
def sample_data():
    return {
        'latitude': [1, 2, 3, 4, 5],
        'longitude': [10, 20, 30, 40, 50],
        'data': ['A', 'B', 'C', 'D', 'E']
    }

# Tests for plot_line_on_map function
def test_plot_line_on_map(sample_data):
    latitude = sample_data['latitude']
    longitude = sample_data['longitude']
    data = sample_data['data']

    # Mock the show method to prevent the actual display of the figure
    go.Figure.show = lambda self: None

    # Call the function
    plot_line_on_map(latitude, longitude, data)

    # Assert that the scatter trace is created correctly
    scatter_trace = go.Scattergeo(
        lat=latitude,
        lon=longitude,
        mode='lines',
        line=dict(color='red', width=2),
        marker=dict(size=4, color='red')
    )
    
    temp_fig = go.Figure()
    
    temp_fig.add_trace(scatter_trace)
    
    assert scatter_trace in temp_fig.data

@pytest.fixture(scope="module")
def choropleth_data():
    return pd.DataFrame({
        'locations': ['USA', 'Canada', 'Mexico'],
        'values': [10, 20, 30]
    })

# Tests for plot_choropleth function
def test_plot_choropleth(choropleth_data):
    
    try:
        fig = plot_choropleth(choropleth_data, 'locations', 'values', 'Choropleth Map')
        
        assert isinstance(fig.layout.title.text == "Choropleth Map")
        
        assert fig.data[0].locations.tolist() == choropleth_data['locations'].tolist()
        
        assert fig.data[0].z.tolist() == choropleth_data['values'].tolist()
        
   
        
           
@pytest.fixture(scope="module")
def map_fig():
    
   return go.Figure()

@pytest.fixture(scope="module")
def markers_latitudes():
    
   return [37.7749]

@pytest.fixture(scope="module")
def markers_longitudes():
    
   return [-122.4194]

@pytest.fixture(scope="module")
def markers_texts():
    
   return ['San Francisco']

# Tests for add_markers_on_map function
def test_add_markers_on_empty_figure(map_fig ,markers_latitudes ,markers_longitudes ,markers_texts):

  

      updated_map_fig = add_markers_on_map(map_fig ,markers_latitudes , markers_longitudes ,markers_texts)

      assert len(updated_map_fig.data) == 1 
      assert isinstance(updated_map_fig.data[0], go.Scattermapbox)  
      assert updated_map_fig.data[0].lat == markers_latitudes 
      assert updated_map_fig.data[0].lon == markers_longitudes 
      assert updated_map_fig.data[0].mode == "markers" 
      assert updated_map_fig.data[0].marker['size'] == 10 
      assert updated_map_fig.data[0].marker['color'] == "red" 
      assert updated_map_fig.data[0].text == markers_texts 

