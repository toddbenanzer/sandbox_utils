# DataExporter Class Documentation

## Overview
The `DataExporter` class is responsible for exporting data to Tableau-friendly formats, including CSV and TDS. It provides methods for exporting the DataFrame, attaching metadata, handling errors, and applying user-defined configurations.

## Attributes
- `dataframe (pd.DataFrame)`: The DataFrame that is to be exported.

## Methods

### `__init__(self, dataframe: pd.DataFrame)`
Initializes a new instance of the `DataExporter` class.

#### Parameters
- `dataframe (pd.DataFrame)`: The DataFrame to be exported.

### `to_tableau_csv(self, destination_path: str, **kwargs) -> bool`
Exports the DataFrame to a CSV file optimized for Tableau.

#### Parameters
- `destination_path (str)`: The path where the CSV file will be saved.
- `**kwargs`: Additional keyword arguments for the pandas `DataFrame.to_csv` method.

#### Returns
- `bool`: Returns `True` if the export is successful, `False` otherwise.

### `to_tableau_tds(self, destination_path: str, **kwargs) -> bool`
Exports the DataFrame to a TDS file optimized for Tableau.

#### Parameters
- `destination_path (str)`: The path where the TDS file will be saved.
- `**kwargs`: Additional arguments related to TDS export settings.

#### Returns
- `bool`: Returns `True` if the export is successful, `False` otherwise.

### `attach_metadata(self, metadata: dict)`
Attaches metadata to the DataFrame for the purpose of Tableau export.

#### Parameters
- `metadata (dict)`: Metadata to be attached, containing information like field names and data types.

### `handle_errors(self, error: Exception)`
Handles errors during the export process by logging them.

#### Parameters
- `error (Exception)`: The exception that was raised.

### `apply_user_config(self, config: dict)`
Applies user-defined configuration settings to the export process.

#### Parameters
- `config (dict)`: User configuration settings.


# validate_dataframe Function Documentation

## Overview
The `validate_dataframe` function validates a pandas DataFrame to ensure it meets the necessary requirements for export to Tableau-friendly formats.

## Parameters
- `dataframe (pd.DataFrame)`: The DataFrame to be validated.

## Returns
- `bool`: Returns `True` if the DataFrame is valid for export. 
  - If the DataFrame is invalid, it raises a `ValueError` with specific details about the issue.

## Raises
- `ValueError`: 
  - If the DataFrame is empty, it raises an error indicating the necessity of a non-empty DataFrame.
  - If the DataFrame contains null values, it raises an error advising to handle or remove these values.
  - If the DataFrame includes invalid column names (which may contain special characters or do not adhere to identifier rules), it raises an error detailing which column names are invalid.
  - If there are unsupported data types present in the DataFrame, it raises an error indicating which data types are problematic and notes that only certain types are suitable for Tableau export.

## Usage
This function should be used prior to attempting to export a DataFrame to Tableau to ensure compatibility and avoid errors during the export process.


# export_as_csv Function Documentation

## Overview
The `export_as_csv` function is designed to export a pandas DataFrame to a CSV file. It allows users to apply various formatting options to customize the output.

## Parameters

- `dataframe (pd.DataFrame)`: 
  - The DataFrame that will be exported to a CSV file.

- `file_path (str)`: 
  - The destination path where the CSV file will be saved. This should include the desired filename and extension.

- `**options`: 
  - Additional keyword arguments to customize the output of the `to_csv` method from pandas. This includes options for delimiter, header, index, encoding, and other CSV formatting parameters.

## Returns
- `bool`: 
  - Returns `True` if the export is successful, and `False` otherwise.

## Error Handling
- The function will log a success message if the export completes without issues.
- If an error occurs during the export process (for example, due to invalid file paths or write permissions), it logs an error message containing the details of the failure.

## Usage
This function should be used when there is a need to export DataFrame objects to CSV format, especially when specific formatting options are required for further processing or analysis.


# export_as_tdsx Function Documentation

## Overview
The `export_as_tdsx` function is designed to export a pandas DataFrame to a TDSX (Tableau Data Source) file. It allows for additional customization options that can be utilized during the export process to ensure compatibility and functionality within Tableau.

## Parameters

- `dataframe (pd.DataFrame)`: 
  - The pandas DataFrame containing the data that needs to be exported.

- `file_path (str)`:
  - The destination path where the TDSX file will be saved. This should include the desired filename and the `.tdsx` extension.

- `**options`: 
  - Additional keyword arguments for customization during the export process. This can include parameters specific to the Tableau export, such as settings for hyper file processing or other export configurations.

## Returns
- `bool`: 
  - Returns `True` if the export is successful and `False` if it fails.

## Error Handling
- If the Tableau Server Client (TSC) library is not installed, the function logs an error message and returns `False`.
- In the event of an exception during the export process, the function logs specific error information, helping diagnose what went wrong.

## Logging
- Upon successful export, a confirmation message is logged stating that the DataFrame was successfully exported to the specified file path.
- If an error occurs, an error message detailing the failure is logged.

## Usage
This function is useful for exporting DataFrame objects into a format that is compatible with Tableau, making it easier for data analysts and scientists to share and visualize data within Tableau platforms.


# setup_logging Function Documentation

## Overview
The `setup_logging` function configures the logging settings for the module, allowing for customizable output of log messages based on the specified logging level.

## Parameters

- `level (int)`: 
  - The logging level to set. This can be one of the standard logging levels provided by the `logging` module, such as:
    - `logging.DEBUG`: Detailed information, typically of interest only when diagnosing problems.
    - `logging.INFO`: Confirmation that things are working as expected.
    - `logging.WARNING`: An indication that something unexpected happened, or indicative of some problem in the near future (e.g., ‘disk space low’).
    - `logging.ERROR`: Due to a more serious problem, the software has not been able to perform some function.
    - `logging.CRITICAL`: A very serious error, indicating that the program itself may be unable to continue running.

## Returns
- `None`: This function does not return any value.

## Error Handling
- If an error occurs while configuring logging (e.g., invalid configurations), the function logs an error message detailing the failure and defaults the logging level to `INFO`.

## Logging Configuration
- The function uses `logging.basicConfig` to set the logging level, format, and handlers.
- The default logging format includes timestamps, log levels, and the log message.
- By default, log messages are output to the console, but file logging can be enabled by uncommenting the associated line.

## Usage
This function is utilized to establish the logging configuration at the beginning of a script or application to ensure that all log messages are handled consistently throughout the program's execution. This setup is particularly useful for debugging and monitoring the application.


# read_user_config Function Documentation

## Overview
The `read_user_config` function is designed to read and parse user-specific configuration settings from a specified file, returning those settings in a structured format. The function is particularly useful for loading configurations needed for the operation of a program.

## Parameters

- `config_file (str)`: 
  - The path to the configuration file, which should be in a supported format (currently assumed to be JSON).

## Returns
- `dict`: 
  - Returns a dictionary containing the configuration settings loaded from the specified file.

## Raises
- `FileNotFoundError`: 
  - Raised if the specified configuration file does not exist.
  
- `ValueError`: 
  - Raised if the contents of the configuration file are not in a valid JSON format, indicating that the file cannot be decoded properly.
  
- `Exception`: 
  - Any other unexpected errors when reading or parsing the file will also raise an exception, and details will be logged.

## Logging
- Errors encountered will be logged with appropriate messages to help diagnose issues, including cases where the file cannot be found, where there are issues decoding the JSON content, or other unexpected errors.

## Usage
This function can be used at the beginning of a script or application to load necessary configurations before proceeding with further processing or data handling. It ensures that all required settings are read and available, allowing the program to behave according to user-defined parameters.


