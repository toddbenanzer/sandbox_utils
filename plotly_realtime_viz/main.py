from typing import Any
from typing import Any, Optional
from typing import Dict
from typing import Optional
import json
import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import websocket


class DataConnector:
    """
    A class to handle connections to data sources and fetch real-time data.
    """

    def __init__(self, source: str, protocol: str):
        """
        Initializes the DataConnector with a data source and protocol.

        Args:
            source (str): The URI or connection string of the data source.
            protocol (str): The protocol for connecting to the data source (e.g., 'HTTP', 'WebSocket').
        """
        self.source = source
        self.protocol = protocol.lower()
        self.connection = None

    def connect(self) -> Optional[object]:
        """
        Establishes a connection to the data source using the specified protocol.

        Returns:
            object: The established connection object, or None if the protocol is unsupported.

        Raises:
            ValueError: If the protocol is not supported.
        """
        if self.protocol == 'http':
            self.connection = requests.Session()
            return self.connection
        elif self.protocol == 'websocket':
            self.connection = websocket.create_connection(self.source)
            return self.connection
        else:
            raise ValueError(f"Unsupported protocol: {self.protocol}")

    def fetch_data(self) -> Optional[dict]:
        """
        Fetches real-time data from the connected data source.

        Returns:
            dict: The real-time data fetched from the source, or None if fetch failed.
        """
        if self.protocol == 'http' and self.connection:
            response = self.connection.get(self.source)
            if response.ok:
                return response.json()
            else:
                response.raise_for_status()

        elif self.protocol == 'websocket' and self.connection:
            data = self.connection.recv()
            return data

        else:
            print("Connection not established or unsupported protocol.")
            return None

    def reconnect(self) -> bool:
        """
        Attempts to re-establish connection to the data source if the connection is lost.

        Returns:
            bool: True if the reconnection is successful, False otherwise.
        """
        try:
            self.connect()
            return True
        except Exception as e:
            print(f"Reconnection failed: {e}")
            return False



class DataCleaner:
    """
    A class for performing data cleaning operations on datasets.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initializes the DataCleaner with the given dataset.

        Args:
            data (pd.DataFrame): The dataset to be cleaned.
        """
        self.data = data

    def handle_missing_values(self, method: str = 'drop') -> pd.DataFrame:
        """
        Handles missing values in the dataset according to the specified method.

        Args:
            method (str): The method to handle missing values ('drop', 'fill_mean', 'fill_median').

        Returns:
            pd.DataFrame: The cleaned dataset.
        """
        if method == 'drop':
            return self.data.dropna()
        elif method == 'fill_mean':
            return self.data.fillna(self.data.mean())
        elif method == 'fill_median':
            return self.data.fillna(self.data.median())
        else:
            raise ValueError(f"Unsupported method: {method}")

    def remove_outliers(self, method: str = 'z-score') -> pd.DataFrame:
        """
        Removes outliers from the dataset using the specified method.

        Args:
            method (str): The method to use for removing outliers ('z-score', 'iqr').

        Returns:
            pd.DataFrame: The dataset with outliers removed.
        """
        if method == 'z-score':
            z_scores = np.abs((self.data - self.data.mean()) / self.data.std())
            return self.data[(z_scores < 3).all(axis=1)]
        elif method == 'iqr':
            Q1 = self.data.quantile(0.25)
            Q3 = self.data.quantile(0.75)
            IQR = Q3 - Q1
            return self.data[~((self.data < (Q1 - 1.5 * IQR)) | (self.data > (Q3 + 1.5 * IQR))).any(axis=1)]
        else:
            raise ValueError(f"Unsupported method: {method}")

    def normalize_data(self, strategy: str = 'min-max') -> pd.DataFrame:
        """
        Normalizes the data using the specified normalization strategy.

        Args:
            strategy (str): The normalization strategy ('min-max', 'z-score').

        Returns:
            pd.DataFrame: The normalized dataset.
        """
        if strategy == 'min-max':
            return (self.data - self.data.min()) / (self.data.max() - self.data.min())
        elif strategy == 'z-score':
            return (self.data - self.data.mean()) / self.data.std()
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

class DataTransformer:
    """
    A class to perform data transformation operations on datasets.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initializes the DataTransformer with the given dataset.

        Args:
            data (pd.DataFrame): The dataset to be transformed.
        """
        self.data = data

    def aggregate_data(self, method: str = 'sum') -> pd.DataFrame:
        """
        Aggregates data in the dataset using the specified method.

        Args:
            method (str): The aggregation method ('sum', 'mean', 'count').

        Returns:
            pd.DataFrame: The aggregated data.
        """
        if method == 'sum':
            return self.data.sum()
        elif method == 'mean':
            return self.data.mean()
        elif method == 'count':
            return self.data.count()
        else:
            raise ValueError(f"Unsupported method: {method}")



class RealTimePlot:
    """
    A class to create and manage real-time visualizations using Plotly.
    """

    def __init__(self, plot_type: str, **kwargs):
        """
        Initializes the RealTimePlot with the specified plot type and customizable properties.

        Args:
            plot_type (str): The type of plot to create (e.g., 'line', 'scatter', 'bar').
            **kwargs: Additional keyword arguments for setting initial plot properties (e.g., title, xaxis_title, yaxis_title).
        """
        self.plot_type = plot_type
        self.plot = go.Figure()

        # Apply initial settings from kwargs
        self.plot.update_layout(**kwargs)

    def update_plot(self, new_data: Any):
        """
        Updates the existing plot with new data for real-time visualization.

        Args:
            new_data (Any): The new dataset to integrate into the current plot. Can be a dictionary or DataFrame.
        """
        if isinstance(new_data, pd.DataFrame):
            x_data = new_data.index
            y_data = new_data.values.T
        elif isinstance(new_data, dict):
            x_data = new_data.get('x')
            y_data = new_data.get('y')
        else:
            raise ValueError("new_data must be a pandas DataFrame or a dictionary with 'x' and 'y' keys.")
        
        if self.plot_type == 'line':
            self.plot.add_trace(go.Scatter(x=x_data, y=y_data[0], mode='lines'))
        elif self.plot_type == 'scatter':
            self.plot.add_trace(go.Scatter(x=x_data, y=y_data[0], mode='markers'))
        elif self.plot_type == 'bar':
            self.plot.add_trace(go.Bar(x=x_data, y=y_data[0]))
        else:
            raise ValueError(f"Unsupported plot_type: {self.plot_type}")

    def customize_plot(self, **kwargs):
        """
        Customizes the plot's visual aspects like layout, colors, labels, and other aesthetic properties.

        Args:
            **kwargs: Customization parameters for Plotly figure layout and appearance (e.g., bgcolor, font_size, legend_position).
        """
        self.plot.update_layout(**kwargs)

# Example usage
if __name__ == "__main__":
    data = pd.DataFrame({'value': [1, 2, 3, 4]}, index=[0, 1, 2, 3])
    plot = RealTimePlot(plot_type='line', title='Real-Time Line Plot', xaxis_title='Index', yaxis_title='Value')
    plot.update_plot(data)
    plot.customize_plot(title_font_size=20, legend=dict(x=0.5, y=0.9))
    plot.plot.show()



def enable_zoom(plot: go.Figure):
    """
    Enables zoom functionality on the provided Plotly plot.

    Args:
        plot (go.Figure): The Plotly figure object on which zooming will be enabled.
    """
    plot.update_layout(dragmode='zoom')

def enable_pan(plot: go.Figure):
    """
    Enables pan functionality on the provided Plotly plot.

    Args:
        plot (go.Figure): The Plotly figure object on which panning will be enabled.
    """
    plot.update_layout(dragmode='pan')

def hover_info(plot: go.Figure):
    """
    Configures hover functionality on the provided Plotly plot to display additional
    information when hovering over data points.

    Args:
        plot (go.Figure): The Plotly figure object on which hover functionality will be enabled.
    """
    for trace in plot.data:
        trace.hoverinfo = 'text+x+y'
        trace.hovertemplate = '%{text}<br>X: %{x}<br>Y: %{y}'

def update_plot_params(plot: go.Figure, **kwargs):
    """
    Updates the Plotly plot's parameters, allowing dynamic adjustments of aesthetics and config.

    Args:
        plot (go.Figure): The Plotly figure object whose parameters will be updated.
        **kwargs: A dictionary of plot parameters to update (e.g., title, xaxis_range, yaxis_range).
    """
    plot.update_layout(**kwargs)



def configure_logging(level: str):
    """
    Set up logging configuration for the application.

    Args:
        level (str): The logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging configured at {level} level.")

def read_config(file_path: str) -> Dict:
    """
    Reads configuration settings from a specified file.

    Args:
        file_path (str): The path to the configuration file to be read.

    Returns:
        Dict: A dictionary containing configuration parameters as key-value pairs.
    """
    try:
        with open(file_path, 'r') as file:
            config = json.load(file)
        logging.info(f"Configuration successfully read from {file_path}.")
        return config
    except Exception as e:
        logging.error(f"Failed to read configuration from {file_path}: {e}")
        raise
