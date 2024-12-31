from typing import Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Example 1: Using DataConnector with HTTP protocol
http_connector = DataConnector("https://api.example.com/data", "HTTP")
http_connection = http_connector.connect()

if http_connection:
    data = http_connector.fetch_data()
    print("Fetched Data over HTTP:", data)
else:
    print("Failed to establish HTTP connection.")

# Example 2: Using DataConnector with WebSocket protocol
ws_connector = DataConnector("ws://live.example.com/stream", "WebSocket")
ws_connection = ws_connector.connect()

if ws_connection:
    data = ws_connector.fetch_data()
    print("Fetched Data over WebSocket:", data)
else:
    print("Failed to establish WebSocket connection.")

# Example 3: Handling reconnection
http_connector = DataConnector("https://api.example.com/data", "HTTP")

if http_connector.connect():
    data = http_connector.fetch_data()
    print("Initial fetch over HTTP:", data)
else:
    print("Initial HTTP connection failed.")

# Simulate a lost connection and attempt to reconnect
if not http_connector.reconnect():
    print("Reconnection failed.")
else:
    data = http_connector.fetch_data()
    print("Data after reconnection over HTTP:", data)



# Example dataset
data = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 100],
    'B': [5, np.nan, np.nan, 8, 10],
    'C': [1, 2, 3, 4, 5]
})

# Initialize DataCleaner
cleaner = DataCleaner(data)

# Example 1: Handling missing values by filling with mean
cleaned_data_fill_mean = cleaner.handle_missing_values(method='fill_mean')
print("Data after filling missing values with mean:\n", cleaned_data_fill_mean)

# Example 2: Removing outliers using z-score method
cleaned_data_remove_outliers = cleaner.remove_outliers(method='z-score')
print("Data after removing outliers:\n", cleaned_data_remove_outliers)

# Example 3: Normalizing data using min-max
normalized_data = cleaner.normalize_data(strategy='min-max')
print("Normalized data using min-max:\n", normalized_data)

# Example 4: Aggregating data by summing all columns
transformer = DataTransformer(data)
aggregated_data_sum = transformer.aggregate_data(method='sum')
print("Aggregated data (sum):\n", aggregated_data_sum)



# Example 1: Creating a line plot with initial data
data_line = pd.DataFrame({'value': [1, 2, 3, 4]}, index=[0, 1, 2, 3])
line_plot = RealTimePlot(plot_type='line', title='Line Plot Example', xaxis_title='X Axis', yaxis_title='Y Axis')
line_plot.update_plot(data_line)
line_plot.customize_plot(bgcolor='lightblue', title_font_size=16)
line_plot.plot.show()

# Example 2: Creating a scatter plot with dictionary data
data_scatter = {'x': [0, 1, 2, 3], 'y': [10, 15, 13, 17]}
scatter_plot = RealTimePlot(plot_type='scatter', title='Scatter Plot Example')
scatter_plot.update_plot(data_scatter)
scatter_plot.customize_plot(title_font_color='red')
scatter_plot.plot.show()

# Example 3: Creating a bar plot and updating with new DataFrame
data_bar = pd.DataFrame({'value': [5, 6, 7, 8]}, index=['A', 'B', 'C', 'D'])
bar_plot = RealTimePlot(plot_type='bar', title='Bar Plot Example', yaxis_title='Values')
bar_plot.update_plot(data_bar)
bar_plot.customize_plot(xaxis_title='Categories', plot_bgcolor='whitesmoke')
bar_plot.plot.show()



# Example 1: Enabling zoom on a scatter plot
scatter_plot = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 5, 6], mode='markers')])
enable_zoom(scatter_plot)
scatter_plot.show()

# Example 2: Enabling pan on a line plot
line_plot = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 5, 6], mode='lines')])
enable_pan(line_plot)
line_plot.show()

# Example 3: Display hover information on a bar plot
bar_plot = go.Figure(data=[go.Bar(x=['A', 'B', 'C'], y=[4, 5, 6], text=['Apple', 'Banana', 'Cherry'])])
hover_info(bar_plot)
bar_plot.show()

# Example 4: Updating plot parameters to set title and axis ranges
plot = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 5, 6], mode='lines+markers')])
update_plot_params(plot, title='Updated Plot', xaxis_range=[0, 4], yaxis_range=[0, 7])
plot.show()


# Example 1: Configuring logging with INFO level
configure_logging('INFO')
logging.info("This is an info log message.")
logging.debug("This debug message will not be shown because the level is set to INFO.")

# Example 2: Reading a configuration from a JSON file
sample_config_path = 'config.json'
with open(sample_config_path, 'w') as config_file:
    json.dump({"database": "data.db", "timeout": 30}, config_file)

try:
    config = read_config(sample_config_path)
    print("Database:", config["database"])
    print("Timeout:", config["timeout"])
finally:
    import os
    os.remove(sample_config_path)

# Example 3: Handling invalid log level
try:
    configure_logging('VERBOSE')  # Invalid level
except ValueError as e:
    print(f"Caught an exception: {e}")

# Example 4: Handling a non-existent configuration file
try:
    config = read_config('non_existent_config.json')
except Exception as e:
    print(f"Caught an exception while reading config: {e}")
