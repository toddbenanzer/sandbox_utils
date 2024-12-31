from typing import Any, Dict
from typing import Callable
from typing import Optional, Union
from typing import Union
import geopandas as gpd
import json
import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class GeoPlotlyCore:
    """
    Core class for handling data operations in the GeoPlotly package,
    including loading and preprocessing geospatial data.
    """
    
    def __init__(self):
        """
        Initializes the GeoPlotlyCore object.
        Sets up necessary attributes and configurations for data operations.
        """
        self.raw_data = None
        self.processed_data = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

    def load_data(self, source: Union[str], format: str) -> Optional[Union[pd.DataFrame, gpd.GeoDataFrame]]:
        """
        Loads geospatial data from a specified source in the given format.

        Args:
            source (str): Path or URL to the data source.
            format (str): Format of the data, e.g., 'GeoJSON', 'CSV'.

        Returns:
            Union[pd.DataFrame, gpd.GeoDataFrame]: Raw data object loaded from the given source.

        Raises:
            ValueError: If the format is not supported.
            IOError: If the data cannot be read from the source.
        """
        try:
            if format.lower() == 'geojson':
                self.raw_data = gpd.read_file(source)
            elif format.lower() == 'csv':
                self.raw_data = pd.read_csv(source)
            else:
                self.logger.error(f"Unsupported format: {format}")
                raise ValueError(f"Unsupported format '{format}'. Supported formats: 'GeoJSON', 'CSV'.")
            self.logger.info(f"Data loaded successfully from {source}")
            return self.raw_data
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def preprocess_data(self) -> Optional[Union[pd.DataFrame, gpd.GeoDataFrame]]:
        """
        Preprocesses raw geospatial data to prepare it for visualization.

        Returns:
            Union[pd.DataFrame, gpd.GeoDataFrame]: Structured data ready for plotting.

        Raises:
            RuntimeError: If preprocessing fails due to invalid data.
        """
        try:
            if self.raw_data is None:
                self.logger.error("No data to preprocess.")
                raise RuntimeError("No data loaded. Please load data before preprocessing.")
            
            # Example preprocessing steps
            if isinstance(self.raw_data, gpd.GeoDataFrame):
                self.processed_data = self.raw_data.dropna().reset_index(drop=True)
                self.logger.debug("GeoDataFrame preprocessed successfully.")
            elif isinstance(self.raw_data, pd.DataFrame):
                self.processed_data = self.raw_data.dropna().reset_index(drop=True)
                self.logger.debug("DataFrame preprocessed successfully.")
            else:
                self.logger.error("Unsupported data type for preprocessing.")
                raise TypeError("Unsupported data type for preprocessing.")
            
            self.logger.info("Data preprocessed successfully.")
            return self.processed_data
        except Exception as e:
            self.logger.error(f"Error during data preprocessing: {e}")
            raise



class MapPlotter:
    """
    Class for generating various types of map plots using Plotly
    including choropleth and scatter geo maps.
    """

    def __init__(self, data: Union[pd.DataFrame, gpd.GeoDataFrame]):
        """
        Initializes the MapPlotter object with data.

        Args:
            data (Union[pd.DataFrame, gpd.GeoDataFrame]): Geospatial data to be used for plotting.
    
        Raises:
            TypeError: If input data is not a DataFrame or GeoDataFrame.
        """
        if not isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
            raise TypeError("Data must be a pandas DataFrame or geopandas GeoDataFrame.")
        
        self.data = data

    def plot_choropleth_map(self, **kwargs) -> go.Figure:
        """
        Generates a choropleth map using the provided dataset.

        Args:
            **kwargs: Additional parameters for customizing the plot.
        
        Returns:
            plotly.graph_objects.Figure: Interactive choropleth map object.
        
        Raises:
            ValueError: If necessary data columns are missing.
        """
        try:
            fig = px.choropleth(self.data, **kwargs)
            fig.update_geos(fitbounds="locations")
            fig.update_layout(title=kwargs.get('title', 'Choropleth Map'))
            return fig
        except Exception as e:
            raise ValueError(f"Error generating choropleth map: {e}")

    def plot_scatter_geo_map(self, **kwargs) -> go.Figure:
        """
        Generates a scatter geo map using the provided dataset.

        Args:
            **kwargs: Additional parameters for customizing the plot.
        
        Returns:
            plotly.graph_objects.Figure: Interactive scatter geo map object.
        
        Raises:
            ValueError: If necessary data columns are missing.
        """
        try:
            fig = px.scatter_geo(self.data, **kwargs)
            fig.update_geos(fitbounds="locations")
            fig.update_layout(title=kwargs.get('title', 'Scatter Geo Map'))
            return fig
        except Exception as e:
            raise ValueError(f"Error generating scatter geo map: {e}")



class RegionHighlighter:
    """
    Class for highlighting specific regions on a Plotly map based on given criteria.
    """

    def __init__(self, map_object: go.Figure):
        """
        Initializes the RegionHighlighter with a Plotly map object.

        Args:
            map_object (plotly.graph_objects.Figure): The map object on which regions will be highlighted.
        
        Raises:
            TypeError: If map_object is not an instance of plotly.graph_objects.Figure.
        """
        if not isinstance(map_object, go.Figure):
            raise TypeError("map_object must be a Plotly Figure.")
        
        self.map_object = map_object

    def highlight_regions(self, criteria: Dict[str, Any], **kwargs) -> go.Figure:
        """
        Highlights regions on the map based on specified criteria.

        Args:
            criteria (dict): Criteria for which regions to highlight (e.g., {'region': 'value'}).
            **kwargs: Additional customization options for highlighting.
        
        Returns:
            plotly.graph_objects.Figure: The updated map object with highlighted regions.
        
        Raises:
            ValueError: If criteria is not valid or highlighting operation fails.
        """
        try:
            highlight_color = kwargs.get('highlight_color', 'yellow')
            border_thickness = kwargs.get('border_thickness', 2)

            for trace in self.map_object.data:
                if 'locations' in trace:
                    trace.update(
                        marker=dict(
                            line=dict(
                                color=highlight_color,
                                width=border_thickness
                            ),
                            opacity=0.5
                        )
                    )
                    filtered_data = [loc for loc, val in zip(trace['locations'], trace.get('z', []))
                                     if criteria.get(loc) == val]
                    trace.update(locations=filtered_data)

            return self.map_object
        except Exception as e:
            raise ValueError(f"Error during region highlighting: {e}")




class DrillDownInteractor:
    """
    Class for adding drill-down interactivity to a Plotly map, allowing users to interact with map elements.
    """

    def __init__(self, map_object: go.Figure):
        """
        Initializes the DrillDownInteractor with a Plotly map object.

        Args:
            map_object (plotly.graph_objects.Figure): The map object to which interactive drill-down features will be added.
        
        Raises:
            TypeError: If map_object is not an instance of plotly.graph_objects.Figure.
        """
        if not isinstance(map_object, go.Figure):
            raise TypeError("map_object must be a Plotly Figure.")
        
        self.map_object = map_object

    def add_interactivity(self, callback_function: Callable[[dict], None]) -> go.Figure:
        """
        Adds interactivity to the map, enabling users to perform drill-down actions using a callback function.

        Args:
            callback_function (Callable): A function triggered upon interaction with map elements, used to modify details or load additional data.
        
        Returns:
            plotly.graph_objects.Figure: The updated map object with added drill-down capabilities.
        
        Raises:
            ValueError: If callback_function is not callable or if applying interactivity fails.
        """
        if not callable(callback_function):
            raise ValueError("callback_function must be callable.")
        
        try:
            # Simulate adding interactivity by utilizing Plotly's JavaScript API concept
            # In reality, this requires client-side enabling (e.g., Dash app)
            for trace in self.map_object.data:
                trace.on_click = callback_function  # Hypothetical usage
                
            return self.map_object
        except Exception as e:
            raise ValueError(f"Error adding interactivity: {e}")




def customize_aesthetics(map_object: go.Figure, **kwargs) -> go.Figure:
    """
    Adjusts visual elements of a Plotly map based on the provided keyword arguments.

    Args:
        map_object (plotly.graph_objects.Figure): The base map object whose aesthetics will be customized.
        **kwargs: A collection of key-value pairs for aesthetic customizations 
                  (e.g., title, background_color, legend_x, legend_y, marker_colors).

    Returns:
        plotly.graph_objects.Figure: The map object with applied aesthetic customizations.
    
    Raises:
        ValueError: If customization arguments are incompatible or misconfigured.
    """
    try:
        # Customize title
        if 'title' in kwargs:
            map_object.update_layout(title=kwargs['title'])
        
        # Customize background color
        if 'background_color' in kwargs:
            map_object.update_layout(paper_bgcolor=kwargs['background_color'],
                                     plot_bgcolor=kwargs['background_color'])
        
        # Customize legend position
        if 'legend_x' in kwargs or 'legend_y' in kwargs:
            map_object.update_layout(
                legend=dict(
                    x=kwargs.get('legend_x', 1),
                    y=kwargs.get('legend_y', 1)
                )
            )
        
        # Customize marker colors (assuming marker is a part of the traces)
        if 'marker_colors' in kwargs:
            for trace in map_object.data:
                if 'marker' in trace:
                    trace.update(marker=dict(color=kwargs['marker_colors']))
        
        return map_object
    except Exception as e:
        raise ValueError(f"Error customizing aesthetics: {e}")




class DataIntegrator:
    """
    Class for merging datasets based on a common key for comprehensive geospatial analysis.
    """

    def __init__(self):
        """
        Initializes the DataIntegrator, preparing it for dataset operations.
        """
        pass

    def merge_datasets(self, primary_data: pd.DataFrame, secondary_data: pd.DataFrame, key: str) -> pd.DataFrame:
        """
        Merges two datasets based on a specified key, creating a unified dataset.

        Args:
            primary_data (pd.DataFrame): The primary dataset to be merged.
            secondary_data (pd.DataFrame): The secondary dataset that supplies additional information.
            key (str or list of str): The key(s) to use for merging both datasets.

        Returns:
            pd.DataFrame: The merged dataset containing combined information from both sources.

        Raises:
            KeyError: If the key is not present in both datasets.
            ValueError: If there is a mismatch or inconsistency in merging.
        """
        try:
            # Verify presence of key(s) in both datasets
            if isinstance(key, str):
                keys = [key]
            else:
                keys = key

            missing_keys_primary = [k for k in keys if k not in primary_data.columns]
            missing_keys_secondary = [k for k in keys if k not in secondary_data.columns]

            if missing_keys_primary or missing_keys_secondary:
                raise KeyError(f"Missing keys in datasets: primary({missing_keys_primary}), secondary({missing_keys_secondary}).")

            # Perform the merge
            merged_data = pd.merge(primary_data, secondary_data, on=key, how='outer')  # or 'inner', 'left', 'right'
            return merged_data

        except Exception as e:
            raise ValueError(f"Error merging datasets: {e}")



def setup_logging(level: int) -> None:
    """
    Configures the logging settings for the application.

    Args:
        level (int): The logging level to configure, using constants from the logging module (e.g., logging.DEBUG, logging.INFO).
    
    Returns:
        None
    
    Raises:
        ValueError: If the provided logging level is not within the predefined logging module constants.
    """
    try:
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        # Optionally add a file handler
        # file_handler = logging.FileHandler('app.log')
        # file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        # logging.getLogger().addHandler(file_handler)
        
    except Exception as e:
        raise ValueError(f"Error setting up logging: {e}")



def setup_config(config_file: str) -> dict:
    """
    Loads and applies configuration settings from a specified file.

    Args:
        config_file (str): The path to the configuration file containing key-value pairs for application settings.

    Returns:
        dict: A dictionary representing the configuration settings loaded from the file.

    Raises:
        FileNotFoundError: If the configuration file cannot be found.
        ValueError: If the configuration file is improperly formatted or contains invalid data.
    """
    try:
        with open(config_file, 'r') as file:
            config_data = json.load(file)
            
            # Example application of configuration settings
            # for key, value in config_data.items():
            #     apply_setting_globally(key, value)
            
            return config_data

    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {e}")
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing the configuration file: {e}")
        raise ValueError(f"Error parsing configuration file: {config_file}")
    except Exception as e:
        logging.error(f"Unhandled exception during configuration setup: {e}")
        raise ValueError(f"Unhandled exception while setting up configuration: {e}")
