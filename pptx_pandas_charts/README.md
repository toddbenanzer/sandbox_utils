# DataFrameToPPT Documentation

## Overview
The `DataFrameToPPT` class provides functionality to convert pandas DataFrames into various types of charts in PowerPoint presentations. It supports creating bar, line, and pie charts, and allows customization of chart appearance.

## Class: DataFrameToPPT

### Constructor

#### `__init__(self, dataframe: pd.DataFrame)`
Initializes the `DataFrameToPPT` instance with the specified DataFrame.

- **Args:**
  - `dataframe` (pd.DataFrame): The source data to be converted into charts.

### Methods

#### `convert_to_chart(self, slide, chart_type: XL_CHART_TYPE, **kwargs)`
General function to convert DataFrame data into a specified chart type on a PowerPoint slide.

- **Args:**
  - `slide` (pptx.slide.Slide): The slide to which the chart will be added.
  - `chart_type` (XL_CHART_TYPE): The type of chart to be created (e.g., bar, line, pie).
  - `**kwargs`: Additional parameters for chart customization.

- **Returns:** 
  - `pptx.chart.Chart`: The created chart object.

#### `convert_to_bar_chart(self, slide, **kwargs)`
Converts DataFrame data into a bar chart on the specified slide.

- **Args:**
  - `slide` (pptx.slide.Slide): The slide to which the bar chart will be added.
  - `**kwargs`: Additional parameters for chart customization.

- **Returns:** 
  - `pptx.chart.Chart`: The created bar chart object.

#### `convert_to_line_chart(self, slide, **kwargs)`
Converts DataFrame data into a line chart on the specified slide.

- **Args:**
  - `slide` (pptx.slide.Slide): The slide to which the line chart will be added.
  - `**kwargs`: Additional parameters for chart customization.

- **Returns:** 
  - `pptx.chart.Chart`: The created line chart object.

#### `convert_to_pie_chart(self, slide, **kwargs)`
Converts DataFrame data into a pie chart on the specified slide.

- **Args:**
  - `slide` (pptx.slide.Slide): The slide to which the pie chart will be added.
  - `**kwargs`: Additional parameters for chart customization.

- **Returns:** 
  - `pptx.chart.Chart`: The created pie chart object.

#### `update_chart(self, chart, **kwargs)`
Updates an existing PowerPoint chart with new data or customization.

- **Args:**
  - `chart` (pptx.chart.Chart): The PowerPoint chart to be updated.
  - `**kwargs`: Various update parameters (new data, title, colors, etc.).

- **Returns:** 
  - `None`

### Example Usage



# PPTCustomizer Documentation

## Overview
The `PPTCustomizer` class provides functionality to customize PowerPoint chart objects, allowing users to modify chart titles, axes labels, legends, and colors for better presentation quality.

## Class: PPTCustomizer

### Constructor

#### `__init__(self, chart)`
Initializes the `PPTCustomizer` with the specified PowerPoint chart object.

- **Args:**
  - `chart` (pptx.chart.Chart): The PowerPoint chart to be customized.

### Methods

#### `set_title(self, title, **kwargs)`
Sets or updates the title of the specified chart.

- **Args:**
  - `title` (str): The title text to be set on the chart.
  - `**kwargs`: Additional customization parameters for the title (e.g., font size, boldness).
  
- **Returns:** 
  - None

#### `set_axes_labels(self, x_label, y_label, **kwargs)`
Sets the labels for the x-axis and y-axis on the specified chart.

- **Args:**
  - `x_label` (str): Label for the x-axis.
  - `y_label` (str): Label for the y-axis.
  - `**kwargs`: Additional customization parameters (e.g., font size, rotation).
  
- **Returns:** 
  - None

#### `set_legend(self, display, **kwargs)`
Configures the legend display settings for the specified chart.

- **Args:**
  - `display` (bool): Flag to show (`True`) or hide (`False`) the legend.
  - `**kwargs`: Additional customization parameters for the legend (e.g., position, font).
  
- **Returns:** 
  - None

#### `set_colors(self, color_scheme)`
Applies a color scheme to the series in the specified chart.

- **Args:**
  - `color_scheme` (list of RGB tuples): e.g., [(255, 0, 0), (0, 255, 0)] specifying colors for the series.
  
- **Returns:** 
  - None

### Example Usage



# PPTManager Documentation

## Overview
The `PPTManager` class provides functionality to manage PowerPoint presentations, allowing users to create new presentations, add slides, and save their work to specified file paths.

## Class: PPTManager

### Constructor

#### `__init__(self, presentation_path: str)`
Initializes the `PPTManager` with the specified path for a new or existing PowerPoint presentation.

- **Args:**
  - `presentation_path` (str): Path to the PowerPoint file to open or create.

### Methods

#### `add_slide(self, layout_index: int)`
Adds a new slide with the specified layout to the PowerPoint presentation.

- **Args:**
  - `layout_index` (int): Index of the layout in the presentation's slide layout collection.
  
- **Returns:**
  - `pptx.slide.Slide`: The newly created slide object.

#### `save_presentation(self, path: str = None)`
Saves the current PowerPoint presentation to the specified file path.

- **Args:**
  - `path` (str, optional): Path to save the PowerPoint presentation. If None, saves to the initial presentation path.

- **Returns:**
  - None

### Example Usage



# validate_dataframe Documentation

## Overview
The `validate_dataframe` function checks the validity of a pandas DataFrame to ensure it is suitable for conversion into charts. It verifies various aspects of the DataFrame, including data types and content.

## Function: validate_dataframe

### Parameters

#### `dataframe`
- **Type:** `pd.DataFrame`
- **Description:** The DataFrame to be validated.

### Returns
- **Type:** `bool`
- **Description:** Returns `True` if the DataFrame is valid for chart conversion; otherwise, it raises an exception.

### Raises
- **TypeError:** If the input is not a pandas DataFrame.
- **ValueError:** If the following conditions are met:
  - The DataFrame is empty.
  - The DataFrame contains non-numeric data.
  - The DataFrame contains NaN or missing values.
  - The DataFrame has zero dimensions (either zero rows or zero columns).

### Example Usage



# setup_logging Documentation

## Overview
The `setup_logging` function configures the logging utility for the application. It sets the appropriate logging level, formats the log messages, and defines the output destinations for the logs, enabling effective monitoring and debugging.

## Function: setup_logging

### Parameters

#### `level`
- **Type:** `int`
- **Description:** The logging level to set. It determines the severity of messages to capture. Valid levels include:
  - `logging.DEBUG`: Detailed information, typically of interest only when diagnosing problems.
  - `logging.INFO`: Confirmation that things are working as expected.
  - `logging.WARNING`: An indication that something unexpected happened or indicative of some problem in the near future.
  - `logging.ERROR`: Due to a more serious problem, the software has not been able to perform some function.
  - `logging.CRITICAL`: A very serious error, indicating that the program itself may be unable to continue running.

#### `log_file`
- **Type:** `str`, optional
- **Description:** The path to the log file. If `None`, logs are written only to the console. If a file path is provided, the logging output will also be recorded in that file.

#### `**kwargs`
- **Type:** Additional keyword arguments
- **Description:** Accepts further customization options for the logging configuration, such as specifying additional handlers or formatting options.

### Returns
- **Type:** `None`
- **Description:** The function does not return any values; it configures the logging system in place.

### Raises
- **ValueError:** If an invalid logging level is provided (i.e., a level that is not one of DEBUG, INFO, WARNING, ERROR, CRITICAL).

### Example Usage

