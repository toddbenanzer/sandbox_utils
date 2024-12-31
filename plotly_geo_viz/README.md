# GeoPlotlyCore Documentation

## Class: GeoPlotlyCore

### Overview
The `GeoPlotlyCore` class is designed for handling data operations in the GeoPlotly package, specifically for loading and preprocessing geospatial data in various formats, including GeoJSON and CSV.

### Attributes
- `raw_data`: Optional[Union[pd.DataFrame, gpd.GeoDataFrame]]
  - Holds the raw geospatial data loaded from the specified source.
  
- `processed_data`: Optional[Union[pd.DataFrame, gpd.GeoDataFrame]]
  - Stores the geospatial data after preprocessing.

- `logger`: logging.Logger
  - Logger instance for recording events and errors during data operations.

### Methods

#### __init__(self)
- **Description**: Initializes a new instance of the GeoPlotlyCore class. Sets up necessary attributes and logging configurations.

#### load_data(self, source: Union[str], format: str) -> Optional[Union[pd.DataFrame, gpd.GeoDataFrame]]
- **Description**: Loads geospatial data from a specified source in the given format.
- **Parameters**:
  - `source` (str): Path or URL to the data source.
  - `format` (str): Format of the data, can be 'GeoJSON' or 'CSV'.
- **Returns**: 
  - Union[pd.DataFrame, gpd.GeoDataFrame]: The raw data object loaded from the given source.
- **Raises**:
  - `ValueError`: If the provided format is not supported.
  - `IOError`: If there is an issue accessing the data source.

#### preprocess_data(self) -> Optional[Union[pd.DataFrame, gpd.GeoDataFrame]]
- **Description**: Preprocesses the loaded geospatial data to prepare it for visualization.
- **Returns**:
  - Union[pd.DataFrame, gpd.GeoDataFrame]: Structured data that is ready for plotting.
- **Raises**:
  - `RuntimeError`: If preprocessing is attempted without loaded data.
  - `TypeError`: If the raw data type is unsupported for preprocessing.


# MapPlotter Documentation

## Class: MapPlotter

### Overview
The `MapPlotter` class provides functionalities for generating various types of map plots using Plotly, including choropleth maps and scatter geo maps. 

### Attributes
- `data`: Union[pd.DataFrame, gpd.GeoDataFrame]
  - The geospatial data to be used for plotting. Must be in either a pandas DataFrame or a geopandas GeoDataFrame.

### Methods

#### __init__(self, data: Union[pd.DataFrame, gpd.GeoDataFrame])
- **Description**: Initializes the MapPlotter object with the provided geospatial data.
- **Parameters**:
  - `data`: Union[pd.DataFrame, gpd.GeoDataFrame]. The geospatial data to be used for plotting.
- **Returns**: None
- **Raises**:
  - `TypeError`: If the provided data is neither a pandas DataFrame nor a geopandas GeoDataFrame.

#### plot_choropleth_map(self, **kwargs) -> go.Figure
- **Description**: Generates a choropleth map using the provided dataset.
- **Parameters**:
  - `**kwargs`: Additional parameters for customizing the plot (e.g., locations, color).
- **Returns**:
  - `plotly.graph_objects.Figure`: An interactive choropleth map object.
- **Raises**:
  - `ValueError`: If necessary data columns for plotting are missing or if there's an error during plot creation.

#### plot_scatter_geo_map(self, **kwargs) -> go.Figure
- **Description**: Generates a scatter geo map using the provided dataset.
- **Parameters**:
  - `**kwargs`: Additional parameters for customizing the plot (e.g., lat, lon, text).
- **Returns**:
  - `plotly.graph_objects.Figure`: An interactive scatter geo map object.
- **Raises**:
  - `ValueError`: If necessary data columns for plotting are missing or if there's an error during plot creation.


# RegionHighlighter Documentation

## Class: RegionHighlighter

### Overview
The `RegionHighlighter` class is designed for highlighting specific regions on a Plotly map based on given criteria. This class allows for easy customization of highlight colors and border properties.

### Attributes
- `map_object`: plotly.graph_objects.Figure
  - The Plotly figure object on which regions will be highlighted. It must be a valid instance of a Plotly figure.

### Methods

#### __init__(self, map_object: go.Figure)
- **Description**: Initializes the RegionHighlighter with a provided Plotly map object.
- **Parameters**:
  - `map_object`: plotly.graph_objects.Figure
    - The map object containing the geographical data where regions will be highlighted.
- **Returns**: None
- **Raises**:
  - `TypeError`: If `map_object` is not an instance of `plotly.graph_objects.Figure`.

#### highlight_regions(self, criteria: Dict[str, Any], **kwargs) -> go.Figure
- **Description**: Highlights regions on the map based on specified criteria.
- **Parameters**:
  - `criteria` (dict): A dictionary specifying the regions to highlight, formatted as `{location: value}` (e.g., `{'region': 'value'}`).
  - `**kwargs`: Additional customization options for highlighting, including:
    - `highlight_color` (str): The color to use for highlighting regions (default is 'yellow').
    - `border_thickness` (int): The width of the border around highlighted regions (default is 2).
- **Returns**:
  - plotly.graph_objects.Figure: The updated Plotly map object with highlighted regions.
- **Raises**:
  - `ValueError`: If the criteria is invalid or if the highlighting operation fails.


# DrillDownInteractor Documentation

## Class: DrillDownInteractor

### Overview
The `DrillDownInteractor` class is designed to add drill-down interactivity to Plotly maps, allowing users to interact with map elements and execute specific actions upon those interactions.

### Attributes
- `map_object`: plotly.graph_objects.Figure
  - The Plotly figure object on which drill-down interactivity will be added.

### Methods

#### __init__(self, map_object: go.Figure)
- **Description**: Initializes the DrillDownInteractor with a specific Plotly map object.
- **Parameters**:
  - `map_object`: plotly.graph_objects.Figure
    - The map object that will gain interactive drill-down features.
- **Returns**: None
- **Raises**:
  - `TypeError`: If `map_object` is not an instance of `plotly.graph_objects.Figure`.

#### add_interactivity(self, callback_function: Callable[[dict], None]) -> go.Figure
- **Description**: Adds interactivity to the map, enabling drill-down actions triggered by user interactions with map elements.
- **Parameters**:
  - `callback_function`: Callable
    - A function that is called upon interaction with map elements, used to modify details or load additional data.
- **Returns**:
  - plotly.graph_objects.Figure: The updated Plotly map object with added drill-down capabilities.
- **Raises**:
  - `ValueError`: If `callback_function` is not callable or if an error occurs while adding interactivity.


# customize_aesthetics Documentation

## Function: customize_aesthetics

### Overview
The `customize_aesthetics` function is used to adjust the visual elements of a Plotly map based on customization options provided by the user. This allows for improved aesthetics, making the map visually appealing and easier to interpret.

### Parameters

- **map_object** (plotly.graph_objects.Figure):  
  The base map object whose aesthetics will be customized.

- **kwargs**:  
  A collection of key-value pairs for aesthetic customizations, which can include:
  - `title` (str): The title of the map.
  - `background_color` (str): The color for the paper and plot backgrounds.
  - `legend_x` (float): The x-coordinate position of the legend.
  - `legend_y` (float): The y-coordinate position of the legend.
  - `marker_colors` (str or list): The color(s) to use for markers in the traces.

### Returns
- **plotly.graph_objects.Figure**:  
  The map object with applied aesthetic customizations.

### Raises
- **ValueError**:  
  If customization arguments are incompatible or misconfigured, an error is raised indicating the reason for the failure.

### Example Usage


# DataIntegrator Documentation

## Class: DataIntegrator

### Overview
The `DataIntegrator` class is designed for merging datasets based on a common key, enabling comprehensive geospatial analysis by combining information from multiple sources.

### Methods

#### __init__(self)
- **Description**: Initializes the DataIntegrator object, preparing it for dataset operations.
- **Returns**: None

#### merge_datasets(self, primary_data: pd.DataFrame, secondary_data: pd.DataFrame, key: str) -> pd.DataFrame
- **Description**: Merges two datasets based on a specified key, creating a unified dataset that incorporates data from both sources.
- **Parameters**:
  - `primary_data` (pd.DataFrame): The primary dataset to be merged.
  - `secondary_data` (pd.DataFrame): The secondary dataset that supplies additional information.
  - `key` (str or list of str): The key(s) to use for merging both datasets.
- **Returns**:
  - pd.DataFrame: The merged dataset containing combined information from both sources.
- **Raises**:
  - `KeyError`: If the specified key is not present in either dataset.
  - `ValueError`: If there is a mismatch or inconsistency during the merging process.
  
### Example Usage


# setup_logging Documentation

## Function: setup_logging

### Overview
The `setup_logging` function configures the logging settings for the application, allowing for consistent logging behavior across different components. It enables capturing and recording significant events and errors at the specified logging level.

### Parameters

- **level** (int):  
  The logging level to configure, which should be a constant from Python's `logging` module (e.g., `logging.DEBUG`, `logging.INFO`, `logging.WARNING`, `logging.ERROR`, etc.).

### Returns
- **None**:  
  This function does not return a value but configures the logging settings globally.

### Raises
- **ValueError**:  
  If the provided logging level is not a valid integer constant from the `logging` module, this exception is raised to indicate that logging configuration has failed.

### Example Usage


# setup_config Documentation

## Function: setup_config

### Overview
The `setup_config` function loads and applies configuration settings from a specified file, facilitating the initialization of application settings.

### Parameters

- **config_file** (str):  
  The path to the configuration file that contains key-value pairs for application settings, typically formatted as JSON.

### Returns
- **dict**:  
  A dictionary representing the configuration settings that have been loaded from the specified file.

### Raises
- **FileNotFoundError**:  
  Raised if the configuration file cannot be located at the specified path.

- **ValueError**:  
  Raised if the configuration file is improperly formatted or contains invalid data that cannot be parsed (such as malformed JSON).

### Example Usage
