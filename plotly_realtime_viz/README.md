# DataConnector Class Documentation

## Overview
The `DataConnector` class provides a means to handle connections to various data sources and fetch real-time data using specified protocols such as HTTP and WebSocket.

## Methods

### `__init__(self, source: str, protocol: str)`
Initializes the DataConnector with a given data source and protocol.
  
**Args:**
- `source` (str): The URI or connection string of the data source.
- `protocol` (str): The protocol to be used for connecting to the data source (e.g., 'HTTP', 'WebSocket').

### `connect(self) -> Optional[object]`
Establishes a connection to the data source using the specified protocol.

**Returns:**
- `object`: The established connection object if the connection succeeds, or `None` if the protocol is unsupported.

**Raises:**
- `ValueError`: If the provided protocol is not supported.

### `fetch_data(self) -> Optional[dict]`
Fetches real-time data from the connected data source.

**Returns:**
- `dict`: The real-time data fetched from the source if successful, or `None` if the fetch fails.

### `reconnect(self) -> bool`
Attempts to re-establish the connection to the data source if the connection has been lost.

**Returns:**
- `bool`: Returns `True` if the reconnection is successful, or `False` otherwise.


# DataCleaner Class Documentation

## Overview
The `DataCleaner` class provides functionality to clean datasets by handling missing values, removing outliers, and normalizing data.

## Methods

### `__init__(self, data: pd.DataFrame)`
Initializes the DataCleaner with the provided dataset.

**Args:**
- `data` (pd.DataFrame): The dataset to be cleaned.

### `handle_missing_values(self, method: str = 'drop') -> pd.DataFrame`
Handles missing values in the dataset using the specified method.

**Args:**
- `method` (str): The method to handle missing values. Options include:
  - `'drop'`: Remove rows with missing values.
  - `'fill_mean'`: Fill missing values with the mean of each column.
  - `'fill_median'`: Fill missing values with the median of each column.

**Returns:**
- `pd.DataFrame`: The cleaned dataset.

**Raises:**
- `ValueError`: If the specified method is unsupported.

### `remove_outliers(self, method: str = 'z-score') -> pd.DataFrame`
Removes outliers from the dataset using the specified method.

**Args:**
- `method` (str): The method to remove outliers. Options include:
  - `'z-score'`: Remove rows that have a z-score greater than 3.
  - `'iqr'`: Use the interquartile range (IQR) method to determine outliers.

**Returns:**
- `pd.DataFrame`: The dataset with outliers removed.

**Raises:**
- `ValueError`: If the specified method is unsupported.

### `normalize_data(self, strategy: str = 'min-max') -> pd.DataFrame`
Normalizes the data using the specified normalization strategy.

**Args:**
- `strategy` (str): The normalization strategy. Options include:
  - `'min-max'`: Scale the data to a range between 0 and 1.
  - `'z-score'`: Standardize the data using z-scores.

**Returns:**
- `pd.DataFrame`: The normalized dataset.

**Raises:**
- `ValueError`: If the specified strategy is unsupported.

---

# DataTransformer Class Documentation

## Overview
The `DataTransformer` class provides functionality to transform datasets by aggregating data.

## Methods

### `__init__(self, data: pd.DataFrame)`
Initializes the DataTransformer with the provided dataset.

**Args:**
- `data` (pd.DataFrame): The dataset to be transformed.

### `aggregate_data(self, method: str = 'sum') -> pd.DataFrame`
Aggregates data in the dataset using the specified method.

**Args:**
- `method` (str): The aggregation method. Options include:
  - `'sum'`: Return the sum of each column.
  - `'mean'`: Return the mean of each column.
  - `'count'`: Return the count of non-null values in each column.

**Returns:**
- `pd.DataFrame`: The aggregated data.

**Raises:**
- `ValueError`: If the specified method is unsupported.


# RealTimePlot Class Documentation

## Overview
The `RealTimePlot` class is designed to create and manage real-time visualizations using Plotly. It allows for the seamless update of plots with new data and customization of visual elements to enhance the presentation of the data.

## Methods

### `__init__(self, plot_type: str, **kwargs)`
Initializes the `RealTimePlot` with the specified plot type and customizable properties.

**Args:**
- `plot_type` (str): The type of plot to create. Supported types include:
  - `'line'`: A line plot.
  - `'scatter'`: A scatter plot.
  - `'bar'`: A bar plot.
- `**kwargs`: Additional keyword arguments for setting initial plot properties, such as:
  - `title`: Title of the plot.
  - `xaxis_title`: Title of the x-axis.
  - `yaxis_title`: Title of the y-axis.
  - Other layout settings supported by Plotly.

### `update_plot(self, new_data: Any)`
Updates the existing plot with new data for real-time visualization.

**Args:**
- `new_data` (Any): The new dataset to integrate into the current plot. Can be provided as either:
  - A `pandas.DataFrame`: The DataFrame's index will be used for the x-axis and the values for the y-axis.
  - A dictionary containing keys `'x'` and `'y'`: These will serve as the x and y data for the plot.

**Raises:**
- `ValueError`: If `new_data` is neither a `pandas.DataFrame` nor a dictionary with required keys.

### `customize_plot(self, **kwargs)`
Customizes the plot's visual aspects like layout, colors, labels, and other aesthetic properties.

**Args:**
- `**kwargs`: Customization parameters for the Plotly figure layout and appearance, such as:
  - `bgcolor`: Background color of the plot area.
  - `title_font_size`: Font size for the title.
  - `legend`: A dictionary specifying the placement and appearance of the legend.
  - Any other layout properties supported by Plotly.

## Example Usage


# User Interaction Functions Documentation

## Overview
This module provides functions to enhance user interaction with Plotly visualizations. It allows dynamic adjustments such as enabling zoom and pan functionality, configuring hover information, and updating plot parameters.

## Functions

### `enable_zoom(plot: go.Figure)`
Enables zoom functionality on the provided Plotly plot.

**Args:**
- `plot` (go.Figure): The Plotly figure object on which zooming will be enabled.

**Returns:** 
None

**Usage:**


# Utilities Module Documentation

## Overview
This module provides utility functions for configuring logging within an application and reading configuration settings from a file.

## Functions

### `configure_logging(level: str)`
Sets up logging configuration for the application.

**Args:**
- `level` (str): The logging level to be set. Acceptable values include:
  - `'DEBUG'`: Detailed information, typically of interest only when diagnosing problems.
  - `'INFO'`: Confirmation that things are working as expected.
  - `'WARNING'`: An indication that something unexpected happened, or indicative of some problem in the near future (e.g., ‘disk space low’).
  - `'ERROR'`: Due to a more serious problem, the software has not been able to perform some function.
  - `'CRITICAL'`: A very serious error, indicating that the program itself may be unable to continue running.

**Returns:** 
None

**Raises:**
- `ValueError`: If an invalid log level is provided.

**Example:**
