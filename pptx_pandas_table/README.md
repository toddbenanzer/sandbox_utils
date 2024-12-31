# PowerPointIntegration Documentation

## Class: PowerPointIntegration

The `PowerPointIntegration` class handles the integration of pandas DataFrames into PowerPoint presentations, allowing for easy addition of data tables to specified slides.

### Initialization

#### `__init__(self, presentation_file: str)`

Initializes the `PowerPointIntegration` class with a specific PowerPoint presentation.

- **Parameters:**
  - `presentation_file` (str): The path to the PowerPoint file to be edited or created.

- **Raises:**
  - `FileNotFoundError`: If the presentation file cannot be found or opened.

### Methods

#### `add_dataframe_as_table(self, dataframe: pd.DataFrame, slide_number: int, **kwargs)`

Adds a pandas DataFrame as a formatted table to a specified slide in the PowerPoint presentation.

- **Parameters:**
  - `dataframe` (pd.DataFrame): The pandas DataFrame to be inserted as a table.
  - `slide_number` (int): The index of the slide where the table should be added.
  - `**kwargs`: Additional options for customization. Example options include:
    - `table_style` (str): A string specifying the style to apply to the table.
    - `number_format` (dict): A dictionary specifying the formatting options for columns.

- **Raises:**
  - `IndexError`: If the slide number is out of range.
  - `ValueError`: If the DataFrame is empty or invalid.

### Usage Example



# TableFormatter Documentation

## Class: TableFormatter

The `TableFormatter` class handles the formatting of PowerPoint tables. It provides methods for applying number formats and visual styles to enhance table presentation.

### Initialization

#### `__init__(self, table: Table, dataframe: pd.DataFrame)`

Initializes the `TableFormatter` class with a PowerPoint table along with a corresponding pandas DataFrame.

- **Parameters:**
  - `table` (pptx.table.Table): 
    - The PowerPoint table object that needs to be formatted.
  - `dataframe` (pandas.DataFrame): 
    - The pandas DataFrame that corresponds to the data within the table.

### Methods

#### `apply_number_formatting(self, column_formats: dict)`

Applies specified number formatting to columns within the table based on the DataFrame.

- **Parameters:**
  - `column_formats` (dict): 
    - A dictionary that specifies the desired number format for each column. 
    - Example: `{'Sales': '${:,.2f}'}` will format the values in the 'Sales' column as currency.

- **Raises:**
  - `KeyError`: 
    - Raised if a specified column in `column_formats` does not exist in the DataFrame.

#### `apply_style(self, style_name: str)`

Applies a visual style to the table.

- **Parameters:**
  - `style_name` (str): 
    - The name of the style to be applied to the table. 
    - Example styles include 'Light Style 1 - Accent 1' and 'Dark Style 1 - Accent 2'.

### Usage Example



# UserInputHandler Documentation

## Class: UserInputHandler

The `UserInputHandler` class is designed to facilitate user input for obtaining formatting preferences for DataFrame columns, specifically when creating tables in PowerPoint presentations.

### Initialization

#### `__init__(self)`

Initializes the `UserInputHandler` class.

- **Parameters:** None

### Methods

#### `get_column_formatting_preferences(self)`

Obtains user-specified formatting preferences for each column.

- **Returns:**
  - `dict`: A dictionary where each key represents the column name, and each value is the user's preferred formatting string. 
    - Example: `{'Revenue': '${:,.2f}'}` indicates that the 'Revenue' column should be formatted as currency with two decimal places.

- **Functionality:**
  - Prompts the user to input column names and their respective format specifications.
  - Validates the correct formatting of user inputs using a sample test value.
  - Allows users to enter 'done' to finish the input process.
  - If an invalid formatting string is entered, an error message is displayed, and the user is prompted to re-enter the value.

### Usage Example



# format_numbers Documentation

## Function: format_numbers

The `format_numbers` function formats numerical data according to a specified format string, enabling customization of how numbers are displayed in string format.

### Parameters

#### `data`
- **Type:** `list`
- **Description:** A collection of numerical values (e.g., integers, floats) to be formatted.

#### `format_spec`
- **Type:** `str`
- **Description:** A format string outlining how each numerical value within `data` should be formatted. 
  - Examples include:
    - `"{:,.2f}"` for formatting to two decimal places with commas as thousand separators.
    - `"{:.1%}"` for displaying as a percentage with one decimal place.

### Returns
- **Type:** `list`
- **Description:** A list of formatted strings where each string corresponds to a formatted value from `data` as specified by `format_spec`.

### Raises
- **ValueError:** 
  - Raised if any element in `data` is non-numeric and cannot be formatted with the given `format_spec`.
- **TypeError:** 
  - Raised if `format_spec` is not a valid format string.

### Example Usage



# get_table_styles Documentation

## Function: get_table_styles

The `get_table_styles` function retrieves a list of available styles that can be applied to PowerPoint tables, allowing users to enhance the visual presentation of their data.

### Returns

- **Type:** `list of dict`
- **Description:** A list of dictionaries, where each dictionary represents a table style.
  - Each dictionary contains:
    - `name` (str): The name of the table style (e.g., "Light Style 1 - Accent 1").
    - `description` (str): A brief description of what the style entails (e.g., "Light style with a subtle accent on the first column.").

### Example Usage



# validate_dataframe Documentation

## Function: validate_dataframe

The `validate_dataframe` function checks the provided pandas DataFrame to ensure it meets the necessary criteria for further processing, such as exporting to PowerPoint tables.

### Parameters

#### `dataframe`
- **Type:** `pandas.DataFrame`
- **Description:** The DataFrame to be validated to ensure it adheres to required structural and content rules.

### Returns

- **Type:** `bool`
- **Description:** Returns `True` if the DataFrame passes all validation checks; otherwise, an appropriate exception is raised.

### Raises

- **ValueError:** 
  - Raised if:
    - The DataFrame is empty or does not contain any rows.
    - The DataFrame has zero columns.
    - Any column names are empty.
    - There are duplicate column names present in the DataFrame.
  
- **TypeError:** 
  - Raised if the DataFrame includes unsupported data types.

### Example Usage



# export_presentation Documentation

## Function: export_presentation

The `export_presentation` function saves a PowerPoint presentation object to a specified file path, allowing for easy export of presentation files.

### Parameters

#### `presentation`
- **Type:** `pptx.Presentation`
- **Description:** The PowerPoint presentation object that you wish to save. This object should be a valid instance of `Presentation` from the `python-pptx` library.

#### `file_path`
- **Type:** `str`
- **Description:** The file path where the presentation will be saved. This path must include the `.pptx` extension.

### Returns

- **Type:** `None`
- **Description:** The function does not return a value. Its success is confirmed by the saved file at the specified location.

### Raises

- **FileNotFoundError:** 
  - Raised if the specified directory in `file_path` cannot be found and cannot be created.

- **PermissionError:** 
  - Raised if there are insufficient permissions to write to the specified `file_path`.

- **ValueError:** 
  - Raised if `file_path` does not end with the `.pptx` extension.

### Example Usage



# setup_logger Documentation

## Function: setup_logger

The `setup_logger` function configures the logging mechanism for the application, allowing for effective tracking of events and errors through the use of a logger.

### Parameters

#### `log_level`
- **Type:** `int`
- **Description:** The logging level that determines the severity of messages to be captured by the logger. Common values include:
  - `logging.DEBUG`: Detailed debugging messages.
  - `logging.INFO`: General information messages.
  - `logging.WARNING`: Warnings that may require attention.
  - `logging.ERROR`: Error messages indicating actual problems.
  - `logging.CRITICAL`: Serious errors that may prevent the program from continuing.

### Returns

- **Type:** `logging.Logger`
- **Description:** Returns a configured logger instance that can be used throughout the application for logging messages.

### Example Usage

