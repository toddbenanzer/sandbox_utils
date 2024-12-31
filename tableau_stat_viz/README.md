# DistributionVisualizer Documentation

## Overview
The `DistributionVisualizer` class enables the creation of distribution visualizations, such as histograms and box plots, using a provided dataset in the form of a Pandas DataFrame.

## Installation
To use this class, ensure you have the necessary packages installed:


# CorrelationVisualizer Documentation

## Overview
The `CorrelationVisualizer` class provides tools to create visualizations that explore correlations between variables in a dataset. It can generate a correlation matrix heatmap and scatter plots, allowing users to visualize relationships and trends within their data.

## Installation
To use this class, ensure you have the necessary packages installed:


# DataHandler Documentation

## Overview
The `DataHandler` class is designed to manage the import and preprocessing of datasets from various formats, allowing for easy manipulation and preparation of data for analysis or visualization.

## Installation
To use this class, ensure you have the necessary packages installed:


# StatisticalAnalyzer Documentation

## Overview
The `StatisticalAnalyzer` class is designed to perform statistical analysis on datasets, including the computation of basic statistics and the execution of hypothesis tests.

## Installation
To use this class, ensure you have the necessary packages installed:


# TableauExporter Documentation

## Overview
The `TableauExporter` class is designed to facilitate the export of visualization objects created with Matplotlib to a format that is compatible with Tableau. This enables users to integrate their visualizations into Tableau for further analysis and presentation.

## Installation
To use this class, ensure you have the necessary package installed:


# setup_visualization_style Documentation

## Overview
The `setup_visualization_style` function is used to configure the visual aesthetics and style properties for Matplotlib visualizations. It allows users to customize various aspects of the plots to ensure consistency and meet specific design requirements.

## Parameters

- **`style_options`** (`dict`): A dictionary containing style parameters and their corresponding values. The following keys are recognized:
  - `'color_palette'`: `str` - Name of the color palette to use (e.g., 'viridis', 'plasma', 'ggplot').
  - `'font_size'`: `int` - Default font size for text elements in all plots.
  - `'line_style'`: `str` - Style of lines in the plot (options include '-', '--', '-.', ':').
  - `'figure_size'`: `tuple` - A tuple specifying the width and height of the figure in inches (e.g., `(10, 5)`).
  - `'background_color'`: `str` - Background color for the visualization axes (e.g., 'white', 'black', 'lightgrey').

## Returns
- **None**

## Raises
- **`ValueError`**: If an unknown style option is provided or if there is an issue while setting the style.

## Example Usage



# setup_logging Documentation

## Overview
The `setup_logging` function configures the Python logging system to manage and display log messages at a specified severity level. This function allows developers to control the verbosity of log output, which is crucial for debugging and application monitoring.

## Parameters

- **`level`** (`str`): The desired logging level. Accepted values include:
  - `'DEBUG'`: Detailed information, typically for diagnosing issues (default verbosity).
  - `'INFO'`: Confirmation that things are working as expected.
  - `'WARNING'`: An indication that something unexpected happened, but the application is still functioning.
  - `'ERROR'`: A more serious problem that affects functionality.
  - `'CRITICAL'`: A severe problem, potentially leading to application termination.

## Returns

- **None**

## Raises

- **`ValueError`**: If an unsupported or invalid logging level is provided.

## Example Usage

