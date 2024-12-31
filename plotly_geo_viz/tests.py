from DataIntegrator import DataIntegrator  # assuming the class is in DataIntegrator.py
from DrillDownInteractor import DrillDownInteractor  # assuming the class is in DrillDownInteractor.py
from MapPlotter import MapPlotter  # assuming the class is in MapPlotter.py
from RegionHighlighter import RegionHighlighter  # assuming the class is in RegionHighlighter.py
from customize_aesthetics import customize_aesthetics  # assuming the function is in customize_aesthetics.py
from io import StringIO
from my_module import GeoPlotlyCore  # assuming the class is in my_module.py
from plotly.graph_objects import Figure
from setup_config import setup_config  # assuming the function is in setup_config.py
from setup_logging import setup_logging  # assuming the function is in setup_logging.py
import geopandas as gpd
import json
import logging
import os
import pandas as pd
import plotly.graph_objects as go
import pytest


@pytest.fixture
def setup_core():
    return GeoPlotlyCore()

def test_load_data_csv(setup_core):
    csv_data = """latitude,longitude
    34.05,-118.25
    40.71,-74.00
    """
    df = pd.read_csv(StringIO(csv_data))
    setup_core.raw_data = df
    loaded_data = setup_core.load_data(StringIO(csv_data), 'csv')
    assert isinstance(loaded_data, pd.DataFrame)
    assert loaded_data.equals(df)

def test_load_data_geojson(setup_core):
    geojson_data = '{"type": "FeatureCollection", "features": []}'
    gdf = gpd.read_file(StringIO(geojson_data))
    setup_core.raw_data = gdf
    loaded_data = setup_core.load_data(StringIO(geojson_data), 'geojson')
    assert isinstance(loaded_data, gpd.GeoDataFrame)
    assert loaded_data.equals(gdf)

def test_load_data_invalid_format(setup_core):
    with pytest.raises(ValueError):
        setup_core.load_data("fake_path", "xml")

def test_preprocess_data_dataframe(setup_core):
    df = pd.DataFrame({'latitude': [34.05, None, 40.71], 'longitude': [-118.25, -118.25, None]})
    setup_core.raw_data = df
    processed_data = setup_core.preprocess_data()
    expected_df = df.dropna().reset_index(drop=True)
    assert processed_data.equals(expected_df)

def test_preprocess_data_geodataframe(setup_core):
    # Empty GeoDataFrame example for simplicity
    gdf = gpd.GeoDataFrame({'geometry': []})
    setup_core.raw_data = gdf
    processed_data = setup_core.preprocess_data()
    expected_gdf = gdf.dropna().reset_index(drop=True)
    assert processed_data.equals(expected_gdf)

def test_preprocess_data_no_data(setup_core):
    with pytest.raises(RuntimeError):
        setup_core.preprocess_data()



def test_initialization_with_dataframe():
    df = pd.DataFrame({'lat': [34, 40], 'lon': [-118, -74]})
    plotter = MapPlotter(df)
    assert plotter.data.equals(df)

def test_initialization_with_geodataframe():
    gdf = gpd.GeoDataFrame({'geometry': []})
    plotter = MapPlotter(gdf)
    assert plotter.data.equals(gdf)

def test_initialization_with_invalid_data():
    with pytest.raises(TypeError):
        MapPlotter("invalid_data")

def test_plot_choropleth_map():
    df = pd.DataFrame({'location': ['USA', 'CAN'], 'value': [100, 200]})
    plotter = MapPlotter(df)
    fig = plotter.plot_choropleth_map(locations='location', color='value')
    assert isinstance(fig, Figure)
    assert 'layout' in fig.to_dict()

def test_plot_scatter_geo_map():
    df = pd.DataFrame({'lat': [34, 40], 'lon': [-118, -74]})
    plotter = MapPlotter(df)
    fig = plotter.plot_scatter_geo_map(lat='lat', lon='lon')
    assert isinstance(fig, Figure)
    assert 'layout' in fig.to_dict()

def test_plot_choropleth_map_missing_columns():
    df = pd.DataFrame({'lat': [34, 40], 'lon': [-118, -74]})
    plotter = MapPlotter(df)
    with pytest.raises(ValueError):
        plotter.plot_choropleth_map(locations='location', color='value')

def test_plot_scatter_geo_map_missing_columns():
    df = pd.DataFrame({'location': ['USA', 'CAN']})
    plotter = MapPlotter(df)
    with pytest.raises(ValueError):
        plotter.plot_scatter_geo_map(lat='lat', lon='lon')



def test_initialization_with_valid_map_object():
    fig = go.Figure()
    highlighter = RegionHighlighter(fig)
    assert highlighter.map_object == fig

def test_initialization_with_invalid_map_object():
    with pytest.raises(TypeError):
        RegionHighlighter("not_a_figure")

def test_highlight_regions_valid_criteria():
    fig = go.Figure(go.Choropleth(locations=['USA', 'CAN'], z=[1, 2]))
    highlighter = RegionHighlighter(fig)
    new_fig = highlighter.highlight_regions({'USA': 1}, highlight_color='blue', border_thickness=3)
    trace = new_fig.data[0]
    assert trace.marker.line.color == 'blue'
    assert trace.marker.line.width == 3
    assert trace.locations == ['USA']

def test_highlight_regions_no_matching_criteria():
    fig = go.Figure(go.Choropleth(locations=['USA', 'CAN'], z=[1, 2]))
    highlighter = RegionHighlighter(fig)
    new_fig = highlighter.highlight_regions({'MEX': 1}, highlight_color='red')
    trace = new_fig.data[0]
    assert trace.locations == []

def test_highlight_regions_invalid_criteria():
    fig = go.Figure(go.Choropleth(locations=['USA', 'CAN'], z=[1, 2]))
    highlighter = RegionHighlighter(fig)
    with pytest.raises(ValueError):
        highlighter.highlight_regions({'USA': 'invalid_value'})



def dummy_callback(data):
    """A simple callback function for testing."""
    return f"Clicked on data: {data}"

def test_initialization_with_valid_map_object():
    fig = go.Figure(go.Scattergeo())
    interactor = DrillDownInteractor(fig)
    assert interactor.map_object == fig

def test_initialization_with_invalid_map_object():
    with pytest.raises(TypeError):
        DrillDownInteractor("invalid_object")

def test_add_interactivity_with_valid_callback():
    fig = go.Figure(go.Scattergeo())
    interactor = DrillDownInteractor(fig)
    updated_fig = interactor.add_interactivity(dummy_callback)
    for trace in updated_fig.data:
        assert hasattr(trace, "on_click")
        assert trace.on_click is dummy_callback

def test_add_interactivity_with_invalid_callback():
    fig = go.Figure(go.Scattergeo())
    interactor = DrillDownInteractor(fig)
    with pytest.raises(ValueError):
        interactor.add_interactivity("not_callable")

def test_add_interactivity_raises_error():
    fig = go.Figure(go.Scattergeo())
    interactor = DrillDownInteractor(fig)
    def faulty_callback(data):
        raise Exception("Simulated error")
    with pytest.raises(ValueError):
        interactor.add_interactivity(faulty_callback)



def test_customize_title():
    fig = go.Figure(go.Scattergeo())
    updated_fig = customize_aesthetics(fig, title="New Title")
    assert updated_fig.layout.title.text == "New Title"

def test_customize_background_color():
    fig = go.Figure(go.Scattergeo())
    updated_fig = customize_aesthetics(fig, background_color='lightgrey')
    assert updated_fig.layout.paper_bgcolor == 'lightgrey'
    assert updated_fig.layout.plot_bgcolor == 'lightgrey'

def test_customize_legend_position():
    fig = go.Figure(go.Scattergeo())
    updated_fig = customize_aesthetics(fig, legend_x=0.5, legend_y=0.5)
    assert updated_fig.layout.legend.x == 0.5
    assert updated_fig.layout.legend.y == 0.5

def test_customize_marker_colors():
    fig = go.Figure(go.Scattergeo(marker=dict(color='red')))
    updated_fig = customize_aesthetics(fig, marker_colors='blue')
    for trace in updated_fig.data:
        if 'marker' in trace:
            assert trace.marker.color == 'blue'

def test_invalid_customization():
    fig = go.Figure(go.Scattergeo())
    with pytest.raises(ValueError):
        customize_aesthetics(fig, some_invalid_param='invalid')



def test_merge_datasets_basic():
    di = DataIntegrator()
    df1 = pd.DataFrame({'id': [1, 2, 3], 'value1': ['a', 'b', 'c']})
    df2 = pd.DataFrame({'id': [3, 4, 5], 'value2': ['x', 'y', 'z']})
    expected = pd.DataFrame({'id': [1, 2, 3, 4, 5], 'value1': ['a', 'b', 'c', None, None], 'value2': [None, None, 'x', 'y', 'z']})
    result = di.merge_datasets(df1, df2, 'id')
    pd.testing.assert_frame_equal(result, expected)

def test_merge_datasets_key_error():
    di = DataIntegrator()
    df1 = pd.DataFrame({'id1': [1, 2], 'value1': ['a', 'b']})
    df2 = pd.DataFrame({'id2': [1, 2], 'value2': ['c', 'd']})
    with pytest.raises(KeyError):
        di.merge_datasets(df1, df2, 'id')

def test_merge_datasets_multiple_keys():
    di = DataIntegrator()
    df1 = pd.DataFrame({'id': [1, 1, 2], 'key2': ['x', 'y', 'z'], 'value1': ['a', 'b', 'c']})
    df2 = pd.DataFrame({'id': [1, 1, 2], 'key2': ['x', 'z', 'z'], 'value2': ['u', 'v', 'w']})
    expected = pd.DataFrame({'id': [1, 1, 1, 2, 2], 'key2': ['x', 'y', 'z', 'z', 'z'], 'value1': ['a', 'b', 'c', None, 'c'], 'value2': ['u', None, 'v', 'w', 'w']})
    result = di.merge_datasets(df1, df2, ['id', 'key2'])
    pd.testing.assert_frame_equal(result.sort_values(by=['id', 'key2']).reset_index(drop=True), expected)

def test_merge_datasets_value_error():
    di = DataIntegrator()
    df1 = pd.DataFrame({'id': [1, 2], 'value1': ['a', 'b']})
    df2 = pd.DataFrame({'id': [1, 2], 'value2': ['c', 'd']})
    with pytest.raises(ValueError):
        di.merge_datasets(df1, 'not_a_dataframe', 'id')



def test_setup_logging_valid_level(caplog):
    setup_logging(logging.DEBUG)
    logging.getLogger().debug("Debug message")
    assert "Debug message" in caplog.text

def test_setup_logging_invalid_level():
    with pytest.raises(ValueError):
        setup_logging("INVALID_LEVEL")

def test_setup_logging_info_level(caplog):
    setup_logging(logging.INFO)
    logging.getLogger().info("Info message")
    assert "Info message" in caplog.text

def test_setup_logging_no_lower_levels_recorded(caplog):
    setup_logging(logging.WARNING)
    logging.getLogger().info("This should not appear")
    assert "This should not appear" not in caplog.text

def test_setup_logging_warning_level(caplog):
    setup_logging(logging.WARNING)
    logging.getLogger().warning("Warning message")
    assert "Warning message" in caplog.text



def test_setup_config_valid_file(tmpdir):
    config_file = tmpdir.join("config.json")
    config_data = {"setting1": "value1", "setting2": "value2"}
    config_file.write(json.dumps(config_data))
    
    result = setup_config(str(config_file))
    assert result == config_data

def test_setup_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        setup_config("non_existent_file.json")

def test_setup_config_invalid_json(tmpdir):
    config_file = tmpdir.join("invalid_config.json")
    config_file.write("Invalid JSON content")
    
    with pytest.raises(ValueError):
        setup_config(str(config_file))

def test_setup_config_empty_file(tmpdir):
    config_file = tmpdir.join("empty_config.json")
    config_file.write("")
    
    with pytest.raises(ValueError):
        setup_config(str(config_file))

def test_setup_config_malformed_json(tmpdir):
    config_file = tmpdir.join("malformed_config.json")
    config_file.write('{"setting1": "value1", "setting2": value2')  # missing quotes around value2
    
    with pytest.raises(ValueError):
        setup_config(str(config_file))
