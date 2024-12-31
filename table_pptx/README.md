# PresentationGenerator Class

## Overview
The `PresentationGenerator` class is designed to simplify the creation of PowerPoint presentations with customizable slides. It allows users to add title slides, as well as text and table slides with various content specifications.

## Methods

### __init__(self, title, **kwargs)
Initializes a new instance of the `PresentationGenerator` class.

#### Parameters:
- `title` (str): The title of the presentation.
- `**kwargs`: Optional additional settings for the presentation (e.g., template configurations).

#### Raises:
- `ValueError`: If the title is not provided as a string.

### add_slide(self, title, content_type, **kwargs)
Adds a new slide to the presentation with the specified content type.

#### Parameters:
- `title` (str): The title for the slide.
- `content_type` (str): The type of content for the slide. Acceptable values are:
  - 'text': For adding a text box to the slide.
  - 'table': For a future implementation to add tables.
- `**kwargs`: Additional options for customizing the slide content.

#### Raises:
- `ValueError`: If the `content_type` is not valid.

### save_presentation(self, file_path)
Saves the current presentation to a specified file path.

#### Parameters:
- `file_path` (str): The path (including filename) where the presentation should be saved, which must end with the `.pptx` extension.

#### Raises:
- `ValueError`: If the provided file path is not valid.
- `IOError`: If there is an error while trying to save the presentation.

## Usage Example



# DataFrameToSlide Class

## Overview
The `DataFrameToSlide` class is designed to convert a pandas DataFrame into a styled table within a PowerPoint slide, utilizing the `python-pptx` library. This allows users to present data in a visually appealing format for presentations.

## Methods

### __init__(self, dataframe, style_config)
Initializes a new instance of the `DataFrameToSlide` class.

#### Parameters:
- `dataframe` (pd.DataFrame): The data to be displayed in the PowerPoint slide.
- `style_config` (StyleConfig): An instance of `StyleConfig` that provides styling options for the table display.

#### Raises:
- `ValueError`: If the provided dataframe is not a valid pandas DataFrame.

### create_table_slide(self, slide)
Creates a table slide in the specified PowerPoint slide.

#### Parameters:
- `slide` (Slide): The PowerPoint slide object to which the table will be added.

#### Raises:
- `ValueError`: If the provided slide is not a valid slide object.

### apply_styles(self, table, style_config)
Applies the specified styles from the `StyleConfig` to the created table.

#### Parameters:
- `table`: The table object within the slide to which styles will be applied.
- `style_config` (StyleConfig): The styling attributes to be applied to the table.

## Usage Example



# SlideManager Class

## Overview
The `SlideManager` class is designed to manage and rearrange slides within a PowerPoint presentation. This class provides functionality to modify the order of slides, allowing for better organization and presentation flow.

## Methods

### __init__(self, presentation)
Initializes a new instance of the `SlideManager` class with the specified PowerPoint presentation.

#### Parameters:
- `presentation` (Presentation): An existing PowerPoint presentation to manage.

#### Raises:
- `ValueError`: If the provided presentation is not a valid `Presentation` object.

### arrange_slides(self, order)
Arranges the slides in the presentation according to the specified order.

#### Parameters:
- `order` (list of int): A list of indices representing the desired order of slides. The list should contain all slide indices in a permutation.

#### Raises:
- `ValueError`: If the provided order list does not cover all slides or if any indices are out of range.

## Usage Example



# StyleConfig Class

## Overview
The `StyleConfig` class is designed to configure and validate the styles applied to tables or text in PowerPoint slides. It ensures that all styling attributes are set correctly and conform to the expected formats.

## Methods

### __init__(self, font_style, font_size, font_color, background_color)
Initializes a new instance of the `StyleConfig` class with specified styling attributes.

#### Parameters:
- `font_style` (str): The font style to apply (e.g., 'Arial').
- `font_size` (int or float): The font size in points.
- `font_color` (tuple): An RGB color tuple for the font (e.g., (0, 0, 0) for black).
- `background_color` (tuple): An RGB color tuple for the background (e.g., (255, 255, 255) for white).

#### Raises:
- `ValueError`: If any of the styling attributes are invalid.

### validate_styles(self)
Validates the provided styling attributes to ensure correctness.

#### Raises:
- `ValueError`: If the font size is non-positive, colors are not valid RGB tuples, or the font style is not a string.

## Usage Example



# load_data Function

## Overview
The `load_data` function is designed to read data from a specified file path and load it into a pandas DataFrame. It supports various file formats and provides a convenient way to import data for processing and analysis.

## Parameters

### file_path (str)
The path to the file containing the data. This could refer to local files or remote resources. The function currently supports CSV, Excel, and JSON file formats.

### **kwargs
Optional keyword arguments that allow for additional configurations for the specific read functions used (e.g., `delimiter`, `sheet_name`).

## Returns
- **DataFrame**
  - A pandas DataFrame containing the data loaded from the specified file.

## Raises
- **ValueError**
  - If the file format is unsupported (not CSV, Excel, or JSON) or if the file cannot be read.
  - If the `file_path` is not a valid non-empty string.

- **FileNotFoundError**
  - If the specified file path does not exist.

## Usage Example



# style_table Function

## Overview
The `style_table` function is designed to apply various styling attributes to a PowerPoint table, enhancing its appearance according to user-defined styles. This function modifies font properties, cell background colors, and text alignment within the table.

## Parameters

### table
- **Type:** Table
- **Description:** The table object within a PowerPoint slide to which styles are to be applied.

### **style_args
- **Type:** Keyword arguments
- **Description:** A variable number of keyword arguments representing different styling attributes, such as:
  - `font_style` (str): The font style to use (e.g., 'Calibri').
  - `font_size` (int or float): The size of the font in points.
  - `font_color` (tuple): An RGB tuple representing the font color (e.g., (0, 0, 0) for black).
  - `background_color` (tuple): An RGB tuple representing the cell background color (e.g., (255, 255, 255) for white).
  - `alignment` (str): The text alignment within the cells ('left', 'center', or 'right').

## Returns
- **None**
  - The function modifies the table object in place and does not return any value.

## Raises
- **ValueError**
  - If any style argument is invalid, such as an unsupported color format or incorrect alignment string.

## Usage Example



# customize_slide Function

## Overview
The `customize_slide` function is designed to enhance a PowerPoint slide by setting its title, adding main text content, and applying various formatting styles. This function allows for extensive customization of slide presentation to fit specific needs.

## Parameters

### slide
- **Type:** Slide
- **Description:** The PowerPoint slide object that will be customized.

### title
- **Type:** str
- **Description:** The title text to be set in the slide's title placeholder.

### text_content
- **Type:** str
- **Description:** The main content text to be added to the body of the slide.

### **style_args
- **Type:** Keyword arguments
- **Description:** A set of optional keyword arguments that allow for customization of the slide's text appearance, including:
  - `font_style` (str): The font style to apply (e.g., 'Arial').
  - `font_size` (int or float): The size of the font in points.
  - `font_color` (tuple): An RGB tuple representing the font color (e.g., (255, 0, 0)).
  - `alignment` (str): Text alignment within the text box ('left', 'center', 'right').
  - `placeholder_bg_color` (tuple): An RGB tuple for the background color of the text box (e.g., (200, 200, 200)).

## Returns
- **None**
  - The function modifies the provided slide object in-place and does not return any value.

## Raises
- **ValueError**
  - Raised if any provided style argument is invalid (e.g., out of RGB range, incorrect alignment).

## Usage Example



# setup_logging Function

## Overview
The `setup_logging` function configures the logging system for an application to capture and format log messages based on a specified logging level. This helps in monitoring application behavior, debugging issues, and maintaining logs in a structured format.

## Parameters

### level
- **Type:** str or int
- **Description:** The logging level to set for the application. Acceptable values include:
  - `'DEBUG'`: Detailed information, typically used for diagnosing problems.
  - `'INFO'`: General information that highlights the progress of the application.
  - `'WARNING'`: An indication that something unexpected happened but the application is still functioning.
  - `'ERROR'`: A serious problem that prevented the application from performing a function.
  - `'CRITICAL'`: A very serious error that may prevent the program from continuing to run.
  - Alternatively, the logging levels can be specified using their corresponding integer values.

## Raises
- **ValueError**
  - Raised if the provided logging level is invalid, either as a string or integer.

## Usage Example

