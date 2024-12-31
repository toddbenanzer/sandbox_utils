from DataIntegrator import DataIntegrator  # assuming the class is in DataIntegrator.py
from DrillDownInteractor import DrillDownInteractor  # assuming the class is in DrillDownInteractor.py
from GeoPlotlyCore import GeoPlotlyCore  # assuming the class is in GeoPlotlyCore.py
from MapPlotter import MapPlotter  # assuming the class is in MapPlotter.py
from RegionHighlighter import RegionHighlighter  # assuming the class is in RegionHighlighter.py
from customize_aesthetics import customize_aesthetics  # assuming the function is in customize_aesthetics.py
from io import StringIO
from setup_config import setup_config  # assuming the function is in setup_config.py
from setup_logging import setup_logging  # assuming the function is in setup_logging.py
import geopandas as gpd
import json
import logging
import pandas as pd
import plotly.graph_objects as go


# Example 1: Load and preprocess CSV data
csv_data = """latitude,longitude
34.05,-118.25
40.71,-74.00
"""
core = GeoPlotlyCore()
# Load the CSV data
core.load_data(StringIO(csv_data), 'csv')
# Preprocess the CSV data
csv_processed = core.preprocess_data()
print("Processed CSV Data:\n", csv_processed)

# Example 2: Load and preprocess GeoJSON data
geojson_data = '{"type": "FeatureCollection", "features": []}'
# Load the GeoJSON data
core.load_data(StringIO(geojson_data), 'geojson')
# Preprocess the GeoJSON data
geojson_processed = core.preprocess_data()
print("Processed GeoJSON Data:\n", geojson_processed)

# Example 3: Handling invalid format
try:
    core.load_data("invalid_path", "xml")
except ValueError as e:
    print("Caught an expected exception:", e)

# Example 4: Attempting to preprocess without loading data
try:
    core_no_data = GeoPlotlyCore()
    core_no_data.preprocess_data()
except RuntimeError as e:
    print("Caught an expected exception:", e)



# Example 1: Create a simple choropleth map
data_choropleth = pd.DataFrame({
    'location': ['USA', 'CAN'],
    'value': [100, 200]
})
plotter_choropleth = MapPlotter(data_choropleth)
fig_choropleth = plotter_choropleth.plot_choropleth_map(locations='location', color='value', title='Choropleth Example')
fig_choropleth.show()

# Example 2: Create a simple scatter geo map
data_scatter_geo = pd.DataFrame({
    'lat': [34.05, 40.71],
    'lon': [-118.25, -74.00],
    'city': ['Los Angeles', 'New York']
})
plotter_scatter_geo = MapPlotter(data_scatter_geo)
fig_scatter_geo = plotter_scatter_geo.plot_scatter_geo_map(lat='lat', lon='lon', text='city', title='Scatter Geo Example', marker=dict(size=10))
fig_scatter_geo.show()

# Example 3: Customize a choropleth map with a color scale
data_custom_choropleth = pd.DataFrame({
    'location': ['USA', 'CAN'],
    'value': [300, 150]
})
plotter_custom_choropleth = MapPlotter(data_custom_choropleth)
fig_custom_choropleth = plotter_custom_choropleth.plot_choropleth_map(
    locations='location', color='value', color_continuous_scale='Viridis', title='Customized Choropleth')
fig_custom_choropleth.show()

# Example 4: Customize a scatter geo map with hover text
data_custom_scatter_geo = pd.DataFrame({
    'lat': [51.5074, 48.8566],
    'lon': [-0.1278, 2.3522],
    'city': ['London', 'Paris']
})
plotter_custom_scatter_geo = MapPlotter(data_custom_scatter_geo)
fig_custom_scatter_geo = plotter_custom_scatter_geo.plot_scatter_geo_map(
    lat='lat', lon='lon', text='city', title='Customized Scatter Geo', hover_name='city', marker=dict(size=12))
fig_custom_scatter_geo.show()



# Example 1: Highlighting regions with specific values
fig = go.Figure(go.Choropleth(locations=['USA', 'CAN', 'MEX'], z=[100, 200, 100]))
highlighter = RegionHighlighter(fig)
fig_with_highlights = highlighter.highlight_regions({'USA': 100, 'MEX': 100}, highlight_color='blue')
fig_with_highlights.show()

# Example 2: Highlighting with custom border thickness
fig = go.Figure(go.Choropleth(locations=['FRA', 'DEU', 'ESP'], z=[5, 10, 15]))
highlighter = RegionHighlighter(fig)
fig_with_custom_highlight = highlighter.highlight_regions({'DEU': 10}, border_thickness=5)
fig_with_custom_highlight.show()

# Example 3: Handle no matching criteria
fig = go.Figure(go.Choropleth(locations=['BRA', 'ARG'], z=[50, 75]))
highlighter = RegionHighlighter(fig)
fig_no_match = highlighter.highlight_regions({'CHL': 50}, highlight_color='red')
print("Regions with no matching criteria will not be highlighted.")
fig_no_match.show()



# Define a callback function to be triggered on interaction
def example_callback(data):
    print(f"Drill-down triggered on data: {data}")

# Example 1: Add interactivity to a simple scattergeo map
fig = go.Figure(go.Scattergeo(lon=[10, 20, 30], lat=[0, 10, 20], mode='markers'))
interactor = DrillDownInteractor(fig)
interactive_map = interactor.add_interactivity(example_callback)
interactive_map.show()

# Example 2: Add interactivity to a line geo map
fig_line_geo = go.Figure(go.Scattergeo(lon=[-50, -40, -30], lat=[30, 20, 10], mode='lines'))
interactor_line = DrillDownInteractor(fig_line_geo)
interactive_line_map = interactor_line.add_interactivity(example_callback)
interactive_line_map.show()

# Example 3: Interactivity added to a choropleth map
fig_choropleth = go.Figure(go.Choropleth(locations=['USA', 'CAN'], z=[1, 2]))
interactor_choropleth = DrillDownInteractor(fig_choropleth)
interactive_choropleth_map = interactor_choropleth.add_interactivity(example_callback)
interactive_choropleth_map.show()



# Example 1: Customize the title and background color of a map
fig = go.Figure(go.Scattergeo(lon=[30, 40], lat=[10, 20], mode='markers'))
customized_fig = customize_aesthetics(fig, title="Customized Map", background_color='lightblue')
customized_fig.show()

# Example 2: Adjust legend position and marker colors
fig_leg = go.Figure(go.Scattergeo(lon=[50, 60], lat=[20, 30], mode='markers', marker=dict(color='red')))
customized_leg_fig = customize_aesthetics(fig_leg, legend_x=0.5, legend_y=0.5, marker_colors='green')
customized_leg_fig.show()

# Example 3: Apply multiple customizations at once
fig_multi = go.Figure(go.Scattergeo(lon=[-80, -70], lat=[25, 35], mode='markers'))
multi_customized_fig = customize_aesthetics(fig_multi, title="Multi Customized Map", 
                                            background_color='black', marker_colors='orange')
multi_customized_fig.show()



# Example 1: Basic merge with single key
primary_df = pd.DataFrame({'id': [1, 2, 3], 'value1': ['A', 'B', 'C']})
secondary_df = pd.DataFrame({'id': [3, 4, 5], 'value2': ['X', 'Y', 'Z']})
integrator = DataIntegrator()
merged_df = integrator.merge_datasets(primary_df, secondary_df, 'id')
print("Merged DataFrame with single key:\n", merged_df)

# Example 2: Merge with multiple keys
primary_df_multi = pd.DataFrame({'id': [1, 1, 2], 'key2': ['x', 'y', 'z'], 'value1': ['A', 'B', 'C']})
secondary_df_multi = pd.DataFrame({'id': [1, 2, 2], 'key2': ['x', 'z', 'z'], 'value2': ['U', 'V', 'W']})
merged_df_multi = integrator.merge_datasets(primary_df_multi, secondary_df_multi, ['id', 'key2'])
print("Merged DataFrame with multiple keys:\n", merged_df_multi)

# Example 3: Handling missing keys
try:
    primary_df_missing = pd.DataFrame({'id1': [1, 2], 'value1': ['A', 'B']})
    secondary_df_missing = pd.DataFrame({'id2': [1, 2], 'value2': ['C', 'D']})
    merged_df_missing = integrator.merge_datasets(primary_df_missing, secondary_df_missing, 'id')
except KeyError as e:
    print("Caught an expected KeyError:", e)

# Example 4: Merge resulting in empty intersection
primary_df_empty = pd.DataFrame({'id': [1, 2], 'value1': ['A', 'B']})
secondary_df_empty = pd.DataFrame({'id': [3, 4], 'value2': ['C', 'D']})
merged_df_empty = integrator.merge_datasets(primary_df_empty, secondary_df_empty, 'id')
print("Merged DataFrame with no common keys:\n", merged_df_empty)



# Example 1: Setup logging at DEBUG level and log a debug message
setup_logging(logging.DEBUG)
logging.debug("This is a debug message.")

# Example 2: Setup logging at INFO level and log an info message
setup_logging(logging.INFO)
logging.info("This is an info message.")

# Example 3: Switching to WARNING level and logging messages
setup_logging(logging.WARNING)
logging.info("This info message will not be shown.")
logging.warning("This is a warning message.")

# Example 4: Setup logging at ERROR level and log an error message
setup_logging(logging.ERROR)
logging.error("This is an error message.")
logging.critical("This is a critical message.")



# Example 1: Load a valid configuration file
config_path_valid = 'path/to/valid_config.json'
try:
    config_settings = setup_config(config_path_valid)
    print("Loaded config settings:", config_settings)
except (FileNotFoundError, ValueError) as e:
    logging.error(f"Failed to load configuration: {e}")

# Example 2: Attempt to load a non-existent configuration file
config_path_non_existent = 'path/to/non_existent_config.json'
try:
    setup_config(config_path_non_existent)
except FileNotFoundError as e:
    logging.error(f"Expected error for non-existent config file: {e}")

# Example 3: Attempt to load a configuration file with invalid JSON
config_path_invalid_json = 'path/to/invalid_json_config.json'
try:
    setup_config(config_path_invalid_json)
except ValueError as e:
    logging.error(f"Expected error for invalid JSON config file: {e}")
