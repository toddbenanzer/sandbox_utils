# Overview

This package provides functions to convert dataframes into various types of charts in a PowerPoint presentation. The supported chart types are: bar chart, line chart, scatter plot, pie chart, area chart, stacked bar chart, stacked area chart, and stacked line chart. The package utilizes the pandas library for handling dataframes and the pptx library for creating PowerPoint presentations.

# Usage

To use this package, you need to have Python 3 installed along with the required dependencies: pandas and pptx. You can install these dependencies using pip:

```bash
pip install pandas pptx
```

Once the dependencies are installed, you can import the necessary functions from the package:

```python
from dataframe_to_pptcharts import dataframe_to_bar_chart, dataframe_to_line_chart, dataframe_to_scatter_plot,
                                convert_dataframe_to_pie_chart, customize_x_axis_label, customize_y_axis_label
```

You can then use these functions to convert your dataframes into PowerPoint charts.

# Examples

Here are some examples of how to use the functions provided by this package:

## Bar Chart

```python
import pandas as pd
from dataframe_to_pptcharts import dataframe_to_bar_chart

df = pd.DataFrame({'Category': ['A', 'B', 'C'],
                   'Value': [10, 20, 30]})

presentation = dataframe_to_bar_chart(df, "Bar Chart Example")
presentation.save("bar_chart.pptx")
```

This will create a PowerPoint presentation with a single slide containing a bar chart representing the data in the dataframe.

## Line Chart

```python
import pandas as pd
from dataframe_to_pptcharts import dataframe_to_line_chart

df = pd.DataFrame({'Year': [2019, 2020, 2021],
                   'Sales': [1000, 2000, 1500],
                   'Profit': [500, 1000, 800]})

presentation = dataframe_to_line_chart(df, "Line Chart Example")
presentation.save("line_chart.pptx")
```

This will create a PowerPoint presentation with a single slide containing a line chart representing the data in the dataframe.

## Scatter Plot

```python
import pandas as pd
from dataframe_to_pptcharts import dataframe_to_scatter_plot

df = pd.DataFrame({'x': [1, 2, 3],
                   'y': [4, 5, 6]})

presentation = dataframe_to_scatter_plot(df, "Scatter Plot Example")
presentation.save("scatter_plot.pptx")
```

This will create a PowerPoint presentation with a single slide containing a scatter plot representing the data in the dataframe.

## Pie Chart

```python
import pandas as pd
from dataframe_to_pptcharts import convert_dataframe_to_pie_chart

df = pd.DataFrame({'Category': ['A', 'B', 'C'],
                   'Value': [10, 20, 30]})

presentation = convert_dataframe_to_pie_chart(df, "Pie Chart Example")
presentation.save("pie_chart.pptx")
```

This will create a PowerPoint presentation with a single slide containing a pie chart representing the data in the dataframe.

## Customizing Axis Labels

You can also customize the axis labels of the charts using the provided functions `customize_x_axis_label` and `customize_y_axis_label`.

```python
from dataframe_to_pptcharts import customize_x_axis_label, customize_y_axis_label

# Customize x-axis label
customize_x_axis_label(chart, "Year", font_size=12, font_color=(0x00, 0x00, 0x00))

# Customize y-axis label
customize_y_axis_label(chart, "Sales", font_size=14)
```

These functions allow you to set the text, font size, and font color of the axis labels of a given PowerPoint chart object.