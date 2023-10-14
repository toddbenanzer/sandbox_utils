# Plotly Highlighted Regions Customization

## Overview
This package provides a function called `customize_highlighted_regions` that allows you to customize the color and opacity of highlighted regions in a Plotly figure. It takes as input a Plotly figure object, the name of the trace to be customized, and the desired color and opacity values for the highlighted regions. The function then updates the specified trace with the new color and opacity values.

## Usage
To use this package, you need to have Plotly installed. You can install it using pip:

```
pip install plotly
```

Once installed, you can import the necessary modules:

```python
import plotly.graph_objects as go
```

Then, you can call the `customize_highlighted_regions` function to customize the highlighted regions in your Plotly figure:

```python
fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 5, 6], name='trace')])
customized_fig = customize_highlighted_regions(fig, 'trace', 'red', 0.5)
```

The `fig` variable is a Plotly figure object that represents your chart. In this example, we create a scatter plot with x-values [1, 2, 3] and y-values [4, 5, 6], and name it 'trace'. We then pass this figure object along with the trace name ('trace'), desired color ('red'), and desired opacity (0.5) to the `customize_highlighted_regions` function. The function returns an updated Plotly figure object (`customized_fig`) with the customized highlighted regions.

## Examples
Here are some examples that demonstrate how to use the `customize_highlighted_regions` function:

### Example 1: Customize Highlighted Regions in a Line Chart

```python
fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 5, 6], name='trace')])
customized_fig = customize_highlighted_regions(fig, 'trace', 'blue', 0.7)
```

In this example, we have a line chart with x-values [1, 2, 3] and y-values [4, 5, 6]. We want to customize the highlighted regions of the trace named 'trace' with the color 'blue' and opacity of 0.7. The function `customize_highlighted_regions` is called with the figure object (`fig`), trace name ('trace'), color ('blue'), and opacity (0.7). The function returns an updated Plotly figure object (`customized_fig`) with the customized highlighted regions.

### Example 2: Customize Highlighted Regions in a Bar Chart

```python
fig = go.Figure(data=[go.Bar(x=['A', 'B', 'C'], y=[10, 20, 30], name='trace')])
customized_fig = customize_highlighted_regions(fig, 'trace', 'green', 0.8)
```

In this example, we have a bar chart with x-values ['A', 'B', 'C'] and y-values [10, 20, 30]. We want to customize the highlighted regions of the trace named 'trace' with the color 'green' and opacity of 0.8. The function `customize_highlighted_regions` is called with the figure object (`fig`), trace name ('trace'), color ('green'), and opacity (0.8). The function returns an updated Plotly figure object (`customized_fig`) with the customized highlighted regions.

## Conclusion
The `customize_highlighted_regions` function provides a convenient way to customize the color and opacity of highlighted regions in your Plotly figures. It allows you to create visually appealing and informative charts by highlighting specific areas of interest.