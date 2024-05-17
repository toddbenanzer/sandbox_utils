## Overview

This Python script provides several functions to create interactive and visually appealing maps using the Plotly library. The script includes functions for plotting scatter plots on a map, drawing lines on a map, creating bar charts on a map, plotting choropleth maps, adding markers to an existing map, customizing color scales and marker sizes, highlighting specific regions on a map, creating drill-down visualizations, zooming and panning maps, displaying multiple layers of maps, creating heat maps, creating contour maps, creating bubble plots on geographical maps, and creating three-dimensional geospatial visualizations.

## Usage

To use this script, you will need to have the following dependencies installed:
- plotly
- pandas
- geopy

You can install these dependencies using pip:
```
pip install plotly pandas geopy
```

Once you have the dependencies installed, you can import the necessary functions from the script for your specific use case.

## Examples

### Plot Scatter Map
```python
import pandas as pd
import plotly.graph_objects as go

# Create a sample DataFrame with latitude and longitude columns
data = pd.DataFrame({
    'lat': [40.7128, 34.0522],
    'lon': [-74.0060, -118.2437]
})

# Plot the scatter map
fig = plot_scatter_map(data, 'lat', 'lon')

# Show the figure
fig.show()
```

### Plot Choropleth Map
```python
import pandas as pd

# Create a sample DataFrame with country names and values
data = pd.DataFrame({
    'country': ['USA', 'Canada'],
    'value': [10, 20]
})

# Plot the choropleth map
plot_choropleth(data, 'country', 'value', 'Choropleth Map')
```

### Add Markers to Map
```python
import plotly.graph_objects as go

# Create a sample map figure
fig = go.Figure()

# Add markers to the map
fig = add_markers_on_map(fig, [40.7128, 34.0522], [-74.0060, -118.2437], ['New York', 'Los Angeles'])

# Show the figure
fig.show()
```

These are just a few examples of how you can use the functions in this script to create different types of geospatial visualizations. You can explore more functionalities and customize the visualizations according to your needs.