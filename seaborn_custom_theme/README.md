# Overview

This Python script provides a collection of functions for creating customized charts using the seaborn and matplotlib libraries. The functions in this script allow users to customize the theme, style, and appearance of their charts. Additionally, there are helper functions for validation, calculation, label and legend customization, tick label formatting, reference lines, dataset scaling and series detection, and additional customization options.

# Usage

To use this script, you need to have seaborn and matplotlib installed in your Python environment. You can install these libraries using pip:

```python
pip install seaborn matplotlib
```

Once you have the required libraries installed, you can import the functions from the script into your own Python code by including the following line at the top of your script:

```python
from <script_name> import *
```

Replace `<script_name>` with the actual name of the script file.

# Examples

Here are a few examples showcasing how to utilize some of the key functions in this script:

## Example 1: Creating a Stacked Bar Chart

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from <script_name> import create_stacked_bar_chart

data = pd.DataFrame({'x': [0.5, 1.5, 2.5], 'y': [4, 5, 6], 'hue': ['A', 'B', 'C']})
colors = ['#FF0000', '#00FF00', '#0000FF']

create_stacked_bar_chart(data[['x', 'y']], colors)
```

## Example 2: Creating a Custom Line Chart

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from <script_name> import create_custom_line_chart

data = pd.DataFrame({'x': [0.5, 1.5, 2.5], 'y': [4, 5, 6]})
colors = ['#FF0000', '#00FF00', '#0000FF']

create_custom_line_chart(data, 'x', 'y', colors)
```

## Example 3: Customizing Tick Labels

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from <script_name> import format_xaxis_tick_labels, format_y_axis_tick_labels

data = pd.DataFrame({'x': [0.5, 1.5, 2.5], 'y': [4, 5, 6]})
tick_labels = ['label1', 'label2', 'label3']

formatted_labels = format_xaxis_tick_labels(data, tick_labels)
format_y_axis_tick_labels(data['y'])
```

These are just a few examples of how to use the functions in this script. For more detailed examples and information on each function, refer to the function descriptions and associated comments within the script itself.