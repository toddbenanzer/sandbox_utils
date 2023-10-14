# Animated Plotly Charts

This package provides a set of functions to create animated charts using Plotly, a powerful and flexible data visualization library for Python. With these functions, you can easily create animated line charts, bar charts, scatter plots, bubble charts, area charts, heatmaps, pie charts, histograms, and more.

## Installation

To use this package, you need to have Plotly installed. You can install it via pip:

```shell
pip install plotly
```

## Usage

The package consists of several functions that allow you to create different types of animated charts. Here is an overview of the available functions:

- `create_animated_line_chart`: Create an animated line chart.
- `create_animated_bar_chart`: Create an animated bar chart.
- `create_animated_scatter_plot`: Create an animated scatter plot.
- `create_animated_bubble_chart`: Create an animated bubble chart.
- `create_animated_area_chart`: Create an animated area chart.
- `create_animated_heatmap`: Create an animated heatmap.
- `create_animated_pie_chart`: Create an animated pie chart.
- `create_animated_histogram`: Create an animated histogram.

In addition to these functions, there are also some utility functions:

- `set_animation_speed`: Customize the animation speed.
- `add_labels_and_titles`: Add labels and titles to an animation.
- `control_animation_playback`: Control the playback of an animation.
- `export_animation_as_video`: Export an animation as a video file.
- `save_animation_as_html`: Save an animation as an HTML file for web embedding.

To use the functions in your code, you need to import them from the package:

```python
from plotly_animations import create_animated_line_chart
```

## Examples

### Animated Line Chart

Here is an example that demonstrates how to create an animated line chart:

```python
import numpy as np
from plotly_animations import create_animated_line_chart

# Generate some data
x = np.linspace(0, 2 * np.pi, 100)
y = [np.sin(x + i / 10) for i in range(len(x))]

# Create the animated line chart
fig = create_animated_line_chart(x, y)

# Display the chart
fig.show()
```

### Animated Bar Chart

Here is an example that demonstrates how to create an animated bar chart:

```python
import numpy as np
from plotly_animations import create_animated_bar_chart

# Generate some data
data = [[np.random.randint(1, 10) for _ in range(5)] for _ in range(10)]
labels = ['A', 'B', 'C', 'D', 'E']

# Create the animated bar chart
fig = create_animated_bar_chart(data, labels, title='Animated Bar Chart')

# Display the chart
fig.show()
```

### Animated Scatter Plot

Here is an example that demonstrates how to create an animated scatter plot:

```python
import pandas as pd
from plotly_animations import create_animated_scatter_plot

# Generate some data
data = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6], 'frame': [0, 1, 2]})

# Create the animated scatter plot
fig = create_animated_scatter_plot(data, x='x', y='y', animation_frame='frame')

# Display the chart
fig.show()
```

### Animated Bubble Chart

Here is an example that demonstrates how to create an animated bubble chart:

```python
import numpy as np
from plotly_animations import create_animated_bubble_chart

# Generate some data
x_data = [[np.random.normal(0, 1) for _ in range(100)] for _ in range(10)]
y_data = [[np.random.normal(0, 1) for _ in range(100)] for _ in range(10)]
size_data = [[np.random.randint(1, 10) for _ in range(100)] for _ in range(10)]
text_data = [['Bubble {}'.format(i) for i in range(100)] for _ in range(10)]
animation_frame = [i for i in range(10)]

# Create the animated bubble chart
fig = create_animated_bubble_chart(x_data, y_data, size_data, text_data, animation_frame)

# Display the chart
fig.show()
```

### Animated Area Chart

Here is an example that demonstrates how to create an animated area chart:

```python
import pandas as pd
from plotly_animations import create_animated_area_chart

# Generate some data
data = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6], 'frame': [0, 1, 2]})

# Create the animated area chart
fig = create_animated_area_chart(data, x='x', y='y', animation_frame='frame', title='Animated Area Chart')

# Display the chart
fig.show()
```

### Animated Heatmap

Here is an example that demonstrates how to create an animated heatmap:

```python
import numpy as np
from plotly_animations import create_animated_heatmap

# Generate some data
data = np.random.rand(3, 5, 5)
labels = ['Frame {}'.format(i) for i in range(data.shape[0])]

# Create the animated heatmap
fig = create_animated_heatmap(data, labels)

# Display the chart
fig.show()
```

### Animated Pie Chart

Here is an example that demonstrates how to create an animated pie chart:

```python
from plotly_animations import create_animated_pie_chart

# Generate some data
labels = ['A', 'B', 'C']
values = [10, 20, 30]

# Create the animated pie chart
fig = create_animated_pie_chart(labels, values)

# Display the chart
fig.show()
```

### Animated Histogram

Here is an example that demonstrates how to create an animated histogram:

```python
import numpy as np
from plotly_animations import create_animated_histogram

# Generate some data
data = [[np.random.normal(0, 1) for _ in range(100)] for _ in range(10)]

# Create the animated histogram
fig = create_animated_histogram(data, frames=10, title='Animated Histogram')

# Display the chart
fig.show()
```

## Contributing

If you find any bugs or have suggestions for improvement, please feel free to open an issue or submit a pull request on GitHub.

## License

This package is licensed under the MIT License. See the LICENSE file for details.