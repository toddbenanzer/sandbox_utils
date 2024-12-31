# DataValidator Class Documentation

## Overview
The `DataValidator` class provides a mechanism to validate data within a Pandas DataFrame by checking for missing values, detecting outliers, and identifying inconsistencies according to user-defined rules.

## Initialization

### `__init__(self, dataframe: pd.DataFrame)`
Initializes the `DataValidator` object with the provided DataFrame.

- **Args:**
  - `dataframe` (`pd.DataFrame`): The dataset on which validations will be performed.

## Methods

### `check_missing_values(self) -> pd.Series`
Identifies and reports missing values within the DataFrame.

- **Returns:**
  - `pd.Series`: A series indicating the count of missing values per column.

### `detect_outliers(self, method: str = "z-score", threshold: float = 3.0) -> Dict[str, pd.DataFrame]`
Detects outliers in the numerical columns of the DataFrame based on a specified method and threshold.

- **Args:**
  - `method` (`str`): The method to use for detecting outliers. Options include `'z-score'` or `'IQR'`.
  - `threshold` (`float`): The threshold beyond which a data point is considered an outlier (default is `3.0`).

- **Returns:**
  - `Dict[str, pd.DataFrame]`: A dictionary with column names as keys and DataFrames of detected outliers as values.

### `find_inconsistencies(self, column_rules: Dict[str, str]) -> pd.DataFrame`
Identifies data inconsistencies within the DataFrame based on user-defined rules for specific columns.

- **Args:**
  - `column_rules` (`Dict[str, str]`): A dictionary where keys are column names and values are rules/conditions in string format.

- **Returns:**
  - `pd.DataFrame`: A DataFrame highlighting inconsistent records found according to the provided rules.


# DataCorrector Class Documentation

## Overview
The `DataCorrector` class provides functionality to correct inaccuracies in a Pandas DataFrame by imputing missing values and managing outliers. This class can be useful for preprocessing data before analysis.

## Initialization

### `__init__(self, dataframe: pd.DataFrame)`
Initializes the `DataCorrector` object with the provided DataFrame.

- **Args:**
  - `dataframe` (`pd.DataFrame`): The dataset to perform corrections on.

## Methods

### `impute_missing_values(self, method: Union[str, Callable[[pd.Series], Union[int, float]]] = "mean") -> pd.DataFrame`
Fills in missing values in the DataFrame using a specified imputation method.

- **Args:**
  - `method` (`str` or `callable`): The method to use for imputing missing values. Options include:
    - `'mean'`: Use the mean of the column to replace missing values.
    - `'median'`: Use the median of the column to replace missing values.
    - `'mode'`: Use the mode of the column to replace missing values.
    - A user-defined function that takes a Pandas Series and returns a scalar.
  
- **Returns:**
  - `pd.DataFrame`: A DataFrame with missing values imputed based on the specified method.

### `handle_outliers(self, method: str = "remove") -> pd.DataFrame`
Manages outliers in the DataFrame using a specified method.

- **Args:**
  - `method` (`str`): The method to use for handling outliers. Options include:
    - `'remove'`: Remove rows with outliers based on the Interquartile Range (IQR).
    - `'clip'`: Clip outliers to the upper and lower bounds defined by the IQR.
    - A user-defined function to process outliers.
  
- **Returns:**
  - `pd.DataFrame`: A DataFrame with outliers handled according to the specified method.

### Errors
- Raises a `ValueError` if an unsupported method is specified for either imputation or outlier handling.


# DataStandardizer Class Documentation

## Overview
The `DataStandardizer` class is designed to standardize the format of specified columns in a Pandas DataFrame. This is useful for ensuring consistent data representation, facilitating data analysis, and improving data quality.

## Initialization

### `__init__(self, dataframe: pd.DataFrame)`
Initializes the `DataStandardizer` object with the provided DataFrame.

- **Args:**
  - `dataframe` (`pd.DataFrame`): The dataset on which standardization will be performed.

## Methods

### `standardize_column_format(self, column_name: str, format_type: str) -> pd.DataFrame`
Standardizes the format of a specified column in the DataFrame.

- **Args:**
  - `column_name` (`str`): The name of the column to standardize.
  - `format_type` (`str`): The format type to standardize to. Options include:
    - `'date'`: Converts the column to a datetime format.
    - `'currency'`: Converts the column to a float representation by removing currency symbols.
    - `'percentage'`: Converts the column to a float representation where percentages are converted to decimal form (e.g., '10%' becomes 0.10).

- **Returns:**
  - `pd.DataFrame`: A DataFrame with the specified column standardized to the desired format.

### Errors
- Raises a `ValueError` if the specified column does not exist in the DataFrame.
- Raises a `ValueError` if the provided `format_type` is unsupported. Supported formats include `'date'`, `'currency'`, and `'percentage'`.


# ValidationRuleManager Class Documentation

## Overview
The `ValidationRuleManager` class manages validation rules for data processing, allowing users to define, save, and load rules to facilitate consistent validation operations.

## Initialization

### `__init__(self)`
Initializes the `ValidationRuleManager` object with an empty dictionary to store rules.

- **Returns:** None

## Methods

### `load_rules(self, rules_file: str) -> None`
Loads validation rules from a specified file into the rule manager.

- **Args:**
  - `rules_file` (`str`): Path to the file containing validation rules, expected in JSON format.

- **Returns:** None

- **Raises:**
  - `FileNotFoundError`: If the specified rules file does not exist.
  - `json.JSONDecodeError`: If an error occurs while decoding JSON from the file.

### `save_rules(self, rules_file: str) -> None`
Saves the current set of validation rules to a specified file.

- **Args:**
  - `rules_file` (`str`): Path to the file where the validation rules will be saved in JSON format.

- **Returns:** None

- **Raises:**
  - `IOError`: If an error occurs while writing to the file.

### `define_rule(self, rule_name: str, rule_definition: str) -> None`
Defines a new validation rule by specifying its name and definition.

- **Args:**
  - `rule_name` (`str`): The name of the rule to be added.
  - `rule_definition` (`str`): The definition or condition of the rule.

- **Returns:** None

- **Notes:**
  - If the specified rule name already exists, the existing rule will be updated with the new definition.


# ValidationReport Class Documentation

## Overview
The `ValidationReport` class generates summaries and reports based on validation results for datasets. It supports various output formats for reporting and exporting the validation results.

## Initialization

### `__init__(self, validation_results: Any)`
Initializes the `ValidationReport` object with the given validation results.

- **Args:**
  - `validation_results` (`Any`): The results from a data validation process. This can be any data structure containing validation information.

## Methods

### `generate_summary(self, output_format: str = "text") -> Union[str, pd.DataFrame]`
Generates a summary of the validation results in the specified format.

- **Args:**
  - `output_format` (`str`): The format of the summary output; options include:
    - `'text'`: Returns a textual representation of the validation results.
    - `'html'`: Returns an HTML representation of the validation results in a table format.

- **Returns:**
  - `Union[str, pd.DataFrame]`: A summary in the specified format.

- **Raises:**
  - `ValueError`: If an unsupported output format is provided.

### `export_report(self, file_path: str, format: str = "csv") -> None`
Exports the validation results to a specified file in the desired format.

- **Args:**
  - `file_path` (`str`): Path to the file where the report will be saved.
  - `format` (`str`): The file format to use for exporting the report; options include:
    - `'csv'`: Exports the validation results to a CSV file.
    - `'json'`: Exports the validation results to a JSON file.
    - `'xlsx'`: Exports the validation results to an Excel file.

- **Returns:** None

- **Raises:**
  - `ValueError`: If an unsupported export format is provided.


# integrate_with_pandas Function Documentation

## Overview
The `integrate_with_pandas` function integrates custom validation, correction, and standardization functionalities directly into the Pandas DataFrame class. This allows users to apply data validation, corrections, standardization, and generate reports seamlessly using DataFrame methods.

## Functionality

The function adds the following methods to the Pandas DataFrame:

### `validate(self, column_rules: Dict[str, str] = None) -> Dict[str, Any]`
Validates the DataFrame to check for missing values, outliers, and inconsistencies based on optional column rules.

- **Args:**
  - `column_rules` (`Dict[str, str]`, optional): A dictionary of column names and their corresponding validation rules.

- **Returns:**
  - `Dict[str, Any]`: A dictionary containing:
    - `missing_values`: Summary of missing values in the DataFrame.
    - `outliers`: Summary of outliers found.
    - `inconsistencies`: Summary of any inconsistencies based on column rules (if provided).

### `correct(self) -> pd.DataFrame`
Corrects the DataFrame by imputing missing values and handling outliers.

- **Returns:**
  - `pd.DataFrame`: A corrected DataFrame with missing values imputed and outliers handled.

### `standardize(self, column_name: str, format_type: str) -> pd.DataFrame`
Standardizes the format of a specified column in the DataFrame.

- **Args:**
  - `column_name` (`str`): The name of the column to standardize.
  - `format_type` (`str`): The desired format type (e.g., 'date', 'currency', 'percentage').

- **Returns:**
  - `pd.DataFrame`: A DataFrame with the specified column standardized to the desired format.

### `apply_rules(self, rule_manager: ValidationRuleManager) -> pd.DataFrame`
Applies validation rules from the specified `ValidationRuleManager` to the DataFrame.

- **Args:**
  - `rule_manager` (`ValidationRuleManager`): An instance of the ValidationRuleManager containing defined rules.

- **Returns:**
  - `pd.DataFrame`: The DataFrame after applying the specified validation rules.

### `generate_report(self, format: str = "text") -> str`
Generates a report summarizing the validation results in the specified format.

- **Args:**
  - `format` (`str`): The format of the report summary; options include 'text' and 'html'.

- **Returns:**
  - `str`: A summary of the validation results in the specified format.

### `export_report(self, file_path: str, format: str = "csv") -> None`
Exports the validation results to a specified file in the desired format.

- **Args:**
  - `file_path` (`str`): Path to the file where the report will be saved.
  - `format` (`str`): The file format to use for exporting the report; options include 'csv', 'json', or 'xlsx'.

- **Returns:** None

## Integration
After calling the `integrate_with_pandas` function, the above methods will be available as part of the Pandas DataFrame instance, enabling enhanced data processing capabilities directly on DataFrames.


# validate_and_correct Function Documentation

## Overview
The `validate_and_correct` function is designed to perform validation and correction of a given Pandas DataFrame based on specified validation rules. It identifies issues within the dataset, applies corrections, and returns the updated DataFrame along with a summary report of the actions taken.

## Parameters

### `dataframe`
- **Type:** `pd.DataFrame`
- **Description:** The dataset to be validated and corrected.

### `rules`
- **Type:** `Dict[str, str]`
- **Description:** A dictionary of validation rules that define conditions to check against the data. Each key represents a column name, and each value is a condition that should hold true for the data in that column.

## Returns
- **Type:** `Tuple[pd.DataFrame, Dict[str, Any]]`
- **Description:** A tuple containing:
  - A corrected DataFrame with missing values imputed and outliers handled.
  - A report of the validation and correction processes, which includes:
    - `missing_values`: Summary of missing values in the DataFrame.
    - `outliers`: Summary of detected outliers.
    - `inconsistencies`: Summary of any inconsistencies found based on the specified rules.
    - `corrections`: A description of the corrections applied to the dataset.

## Usage
The function is typically used in data preprocessing workflows where data quality needs to be ensured before analysis. It is useful for automating checks and corrections, enhancing the reliability of the data used for further analysis.

## Example
