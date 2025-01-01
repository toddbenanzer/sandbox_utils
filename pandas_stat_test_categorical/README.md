# DataFrameHandler Class Documentation

## Overview
The `DataFrameHandler` class is designed to manage and preprocess a pandas DataFrame specifically for categorical data analysis. It validates the DataFrame's structure, ensures the applicable columns are present and of the right type, and manages missing and infinite values effectively.

## Initialization

### `__init__(self, dataframe: pd.DataFrame, columns: List[str])`
Initializes the `DataFrameHandler` with the specified DataFrame and the columns to be analyzed.

#### Parameters:
- `dataframe` (`pd.DataFrame`): The DataFrame to be processed.
- `columns` (`List[str]`): A list of column names from the DataFrame to include in the analysis.

#### Raises:
- `DataFrameValidationError`: If validation of the DataFrame fails due to missing columns, an empty DataFrame, or unsuitable column types.

## Methods

### `validate_dataframe(self) -> bool`
Validates the structure and content of the DataFrame.

#### Returns:
- `bool`: Returns `True` if validation is successful.

#### Raises:
- `DataFrameValidationError`: If validation checks fail, including:
  - Missing columns.
  - Empty DataFrame.
  - Non-categorical columns.

### `handle_missing_infinite_values(self) -> pd.DataFrame`
Handles missing and infinite values in the specified columns of the DataFrame.

#### Returns:
- `pd.DataFrame`: The preprocessed DataFrame, ready for further analysis, with missing values filled and infinite values managed.

## Example Usage


# DescriptiveStatistics Class Documentation

## Overview
The `DescriptiveStatistics` class is designed to compute descriptive statistics for categorical data within a specified pandas DataFrame. The class includes methods for calculating frequency counts, modes, and generating contingency tables for categorical variables.

## Initialization

### `__init__(self, dataframe: pd.DataFrame, columns: List[str])`
Initializes the `DescriptiveStatistics` object with the specified DataFrame and the columns for analysis.

#### Parameters:
- `dataframe` (`pd.DataFrame`): The DataFrame containing the categorical data to be analyzed.
- `columns` (`List[str]`): A list of column names for which to compute descriptive statistics.

#### Raises:
- `ValueError`: If any of the specified columns do not exist in the DataFrame or are not of categorical type.

## Methods

### `_validate_columns(self) -> None`
Validates that all specified columns exist in the DataFrame and are of the categorical data type.

#### Raises:
- `ValueError`: If any of the specified columns are not found or are not categorical types.

### `calculate_frequencies(self) -> Dict[str, pd.Series]`
Calculates the frequency of each category in the specified columns.

#### Returns:
- `Dict[str, pd.Series]`: A dictionary where keys are column names and values are pandas Series with frequency counts for each category.

### `compute_mode(self) -> Dict[str, pd.Series]`
Computes the mode (most frequent category) for each specified column.

#### Returns:
- `Dict[str, pd.Series]`: A dictionary where keys are column names and values are pandas Series containing the mode(s) for each column.

### `generate_contingency_table(self, column1: str, column2: str) -> pd.DataFrame`
Generates a contingency table representing the cross-tabulation of two specified columns.

#### Parameters:
- `column1` (`str`): The first column for cross-tabulation.
- `column2` (`str`): The second column for cross-tabulation.

#### Returns:
- `pd.DataFrame`: A contingency table with counts for each combination of categories from the specified columns.

#### Raises:
- `ValueError`: If either column does not exist or is not of categorical type.

## Example Usage


# StatisticalTests Class Documentation

## Overview
The `StatisticalTests` class is designed for performing statistical tests on categorical data contained within a pandas DataFrame. It provides methods to conduct the Chi-squared test and Fisher's exact test for assessing the independence of categorical variables.

## Initialization

### `__init__(self, dataframe: pd.DataFrame, columns: List[str])`
Initializes the `StatisticalTests` object with the provided DataFrame and specified columns for analysis.

#### Parameters:
- `dataframe` (`pd.DataFrame`): The DataFrame containing the categorical data to be analyzed.
- `columns` (`List[str]`): A list of column names within the DataFrame for which statistical tests will be performed.

#### Raises:
- `ValueError`: If any specified columns do not exist in the DataFrame or are not of categorical type.

## Methods

### `_validate_columns(self) -> None`
Validates that the specified columns exist in the DataFrame and that they are of categorical data type.

#### Raises:
- `ValueError`: If any of the specified columns are not found in the DataFrame or are not categorical.

### `perform_chi_squared_test(self, column1: str, column2: str) -> Tuple[float, float, int, pd.DataFrame]`
Performs the Chi-squared test for independence between two categorical columns.

#### Parameters:
- `column1` (`str`): The name of the first column to be tested.
- `column2` (`str`): The name of the second column to be tested.

#### Returns:
- `Tuple[float, float, int, pd.DataFrame]`: Returns a tuple containing the Chi-squared statistic, p-value, degrees of freedom, and the expected frequency table as a DataFrame.

#### Raises:
- `ValueError`: If either column is not found in the DataFrame or is not of categorical type.

### `perform_fishers_exact_test(self, table: pd.DataFrame) -> Tuple[float, float]`
Performs Fisher's exact test on a provided 2x2 contingency table.

#### Parameters:
- `table` (`pd.DataFrame`): A 2x2 contingency table containing the categorical data.

#### Returns:
- `Tuple[float, float]`: Returns a tuple containing the odds ratio and the p-value resulting from Fisher's exact test.

#### Raises:
- `ValueError`: If the provided table is not a valid 2x2 contingency table.

## Example Usage


# OutputManager Class Documentation

## Overview
The `OutputManager` class is responsible for managing the output of statistical analysis results. This includes functionality to export the results to various file formats (CSV and Excel) and to generate visualizations (bar and pie charts) using matplotlib.

## Initialization

### `__init__(self, results: Any)`
Initializes the `OutputManager` with the provided analysis results.

#### Parameters:
- `results` (`Any`): The results of the analysis to be managed, which can include DataFrames or other data structures.

## Methods

### `export_to_csv(self, file_path: str) -> None`
Exports the results to a CSV file.

#### Parameters:
- `file_path` (`str`): The file path where the CSV file will be saved.

#### Raises:
- `ValueError`: If the results cannot be converted to a CSV-compatible format (i.e., if they are not a DataFrame).

### `export_to_excel(self, file_path: str) -> None`
Exports the results to an Excel file.

#### Parameters:
- `file_path` (`str`): The file path where the Excel file will be saved.

#### Raises:
- `ValueError`: If the results cannot be converted to an Excel-compatible format (i.e., if they are not a DataFrame).

### `generate_visualization(self, chart_type: str) -> None`
Generates a visualization of the results.

#### Parameters:
- `chart_type` (`str`): A string specifying the type of chart to generate (e.g., 'bar', 'pie').

#### Raises:
- `ValueError`: If the chart type is unsupported or if visualization generation fails due to invalid result format.

## Example Usage


# load_dataframe Function Documentation

## Overview
The `load_dataframe` function is designed to load data from a specified file path into a pandas DataFrame. It supports both CSV and Excel file formats.

## Parameters

### `file_path` (str)
- The path to the file that contains the data to be loaded. The function determines the file format based on the file extension.

## Returns
- `pd.DataFrame`: A pandas DataFrame containing the data loaded from the specified file.

## Raises
- `FileNotFoundError`: Raised if the specified file is not found at the provided path.
- `ValueError`: Raised if the file format is unsupported or if an error occurs while reading the file.

## Example Usage


# save_summary Function Documentation

## Overview
The `save_summary` function is designed to save analytical results to a specified file path in a selectable format, which includes CSV and Excel formats. It ensures the results are suitable for the chosen format and provides appropriate error handling for unsupported formats and incompatible result types.

## Parameters

### `results` (Any)
- The analysis results to be saved. This parameter is expected to be a structure that can be converted to a file format, such as a pandas DataFrame.

### `file_path` (str)
- The file path where the results will be saved. The path should include the desired file name and an appropriate file extension.

### `format` (str)
- A string indicating the format in which to save the results. Supported formats include:
  - `'csv'`: For saving as a CSV file.
  - `'excel'`: For saving as an Excel file.

## Returns
- `None`

## Raises
- `ValueError`: Raised under the following conditions:
  - If the specified format is unsupported (not 'csv' or 'excel').
  - If the `results` cannot be saved in the specified format (e.g., if not a DataFrame).
  - If an error occurs during the file writing process, providing additional details about the error.

## Example Usage


# configure_logging Function Documentation

## Overview
The `configure_logging` function sets up the logging configuration for the application. It allows users to specify the desired log level, which determines the severity of log messages that will be captured and displayed. The function supports handling log levels in a case-insensitive manner and provides meaningful error messages for invalid input.

## Parameters

### `level` (str)
- A string that specifies the log level to set. This can be one of the following:
  - `'DEBUG'`: Detailed information, typically of interest only when diagnosing problems.
  - `'INFO'`: Confirming that things are working as expected.
  - `'WARNING'`: An indication that something unexpected happened or indicative of some problem in the near future (e.g., 'disk space low').
  - `'ERROR'`: Due to a more serious problem, the software has not been able to perform some function.
  - `'CRITICAL'`: A very serious error, indicating that the program itself may be unable to continue running.

## Returns
- `None`

## Raises
- `ValueError`: Raised if the provided log level is not valid. The error message specifies the invalid log level and lists the accepted values.

## Example Usage
