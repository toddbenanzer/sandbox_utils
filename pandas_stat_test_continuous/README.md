# DataAnalyzer Class Documentation

## Overview
The `DataAnalyzer` class provides methods to analyze a pandas DataFrame specifically for continuous data. It calculates a variety of descriptive statistics and is equipped to handle missing values, infinite values, and trivial (constant) columns.

## Initialization

### `__init__(self, dataframe: pd.DataFrame, columns: List[str])`
Initializes the `DataAnalyzer` with a DataFrame and a list of columns to analyze.

#### Parameters:
- `dataframe` (`pd.DataFrame`): The DataFrame containing data to be analyzed.
- `columns` (`List[str]`): A list of column names within the DataFrame to include in the analysis.

## Methods

### `calculate_descriptive_stats(self) -> Dict[str, Dict[str, float]]`
Calculates descriptive statistics for the specified columns.

#### Returns:
- `Dict[str, Dict[str, float]]`: A dictionary where each key is a column name, and each value is another dictionary containing the calculated statistics (mean, median, mode, variance, standard deviation, range, interquartile range, skewness, and kurtosis).

### `detect_and_handle_missing_data(self, method: str = 'drop') -> None`
Detects and handles missing data in the specified columns.

#### Parameters:
- `method` (`str`, optional): The method used to handle missing data. Choose from:
  - `'drop'`: Drops rows with missing values in specified columns.
  - `'fill_mean'`: Fills missing values with the mean of the respective column.
  - `'fill_median'`: Fills missing values with the median of the respective column.
  
  Defaults to `'drop'`.

#### Raises:
- `ValueError`: If the method provided is not supported.

### `detect_and_handle_infinite_data(self, method: str = 'replace_nan') -> None`
Detects and handles infinite data in the specified columns.

#### Parameters:
- `method` (`str`, optional): The method used to handle infinite data. Choose from:
  - `'drop'`: Replaces infinite values with NaN and then drops rows with NaN in specified columns.
  - `'replace_nan'`: Replaces infinite values with NaN without dropping rows.

  Defaults to `'replace_nan'`.

#### Raises:
- `ValueError`: If the method provided is not supported.

### `exclude_null_and_trivial_columns(self) -> List[str]`
Excludes columns that are entirely null or trivial (containing constant values).

#### Returns:
- `List[str]`: A list of column names that were excluded from analysis.


# StatisticalTests Class Documentation

## Overview
The `StatisticalTests` class provides methods to perform various statistical tests on a pandas DataFrame. It includes functionalities for t-tests, ANOVA tests, and chi-squared tests.

## Initialization

### `__init__(self, dataframe: pd.DataFrame, columns: List[str])`
Initializes the `StatisticalTests` with a DataFrame and a list of columns to analyze.

#### Parameters:
- `dataframe` (`pd.DataFrame`): The DataFrame containing data to be analyzed.
- `columns` (`List[str]`): A list of column names within the DataFrame to include in the analysis.

## Methods

### `perform_t_tests(self, test_type: str, **kwargs) -> Dict[str, float]`
Performs t-tests on the specified columns or between specified columns.

#### Parameters:
- `test_type` (`str`): The type of t-test to perform, which can be one of the following:
  - `'one-sample'`: Test against a population mean.
  - `'independent'`: Compare means between two independent groups.
  - `'paired'`: Compare means from the same group at different times.
- `**kwargs`: Additional parameters such as:
  - `popmean`: float (Population mean for one-sample t-test.)
  - `group_column`: str (The name of the column defining the groups for independent t-tests.)

#### Returns:
- `dict`: A dictionary containing test statistics (`t-statistic`) and `p-value` for the performed t-tests.

#### Raises:
- `ValueError`: If an unsupported test type is specified or if the number of groups for an independent t-test is not exactly two.

### `perform_anova(self, **kwargs) -> Dict[str, float]`
Conducts ANOVA tests on the specified columns or among groups defined in a column.

#### Parameters:
- `**kwargs`: Additional parameters such as:
  - `group_column`: str (The name of the column that defines the groups for ANOVA.)

#### Returns:
- `dict`: A dictionary containing ANOVA statistics (`F-statistic`) and `p-value`.

#### Raises:
- `ValueError`: If no `group_column` is specified.

### `perform_chi_squared_test(self, **kwargs) -> Dict[str, float]`
Performs chi-squared tests on the data.

#### Parameters:
- `**kwargs`: Additional parameters such as:
  - `columns`: list (Specific columns to be used for constructing contingency tables if different from `self.columns`.)

#### Returns:
- `dict`: A dictionary containing chi-squared test statistics (`chi2-statistic`) and `p-value`.

#### Raises:
- `ValueError`: If the number of specified columns is not exactly two.


# generate_summary_report Function Documentation

## Overview
The `generate_summary_report` function generates a formatted summary report from statistical data provided in the form of a dictionary. It summarizes key metrics and results in a clear, human-readable format.

## Parameters

### `data: dict`
- A dictionary containing statistical data to be summarized.
- **Keys:** Should represent metric names (e.g., 'Mean', 't-statistic', etc.).
- **Values:** Correspond to the numerical results associated with those metrics. Values can be:
  - Numerical values (e.g., floats, integers)
  - Nested dictionaries for grouped or multi-part metrics.

## Returns

### `str`
- A formatted string representing the summary report. The string contains:
  - A header indicating that it is a statistical summary report.
  - A breakdown of each metric and its associated value, formatted for clarity.
  
## Example Usage


# visualize_statistics Function Documentation

## Overview
The `visualize_statistics` function creates visual representations of statistical data, allowing users to interpret and analyze the information effectively. It supports multiple types of plots, including bar charts, scatter plots, and histograms.

## Parameters

### `statistics_data: Dict[str, Any]`
- A dictionary containing the statistical data to be visualized.
- **Keys:** Should represent metric names or categories.
- **Values:** Can be:
  - Single values for bar plots.
  - Lists of values for scatter and histogram plots.
  - Dictionaries for grouped bar plots where keys are categories and values are counts.

### `**kwargs`
- Additional parameters to customize the visualizations. Possible options include:
  - `plot_type` (`str`): The type of plot to create; default is `'bar'`. Options include `'bar'`, `'scatter'`, and `'histogram'`.
  - `title` (`str`): Title of the plot; default is `'Statistical Visualization'`.
  - `xlabel` (`str`): Label for the x-axis; default is an empty string.
  - `ylabel` (`str`): Label for the y-axis; default is an empty string.
  - `figsize` (`tuple`): Size of the figure; default is `(10, 6)`.

## Returns
- `None`: The function displays the generated plot directly and does not return a value.

## Example Usage


# install_dependencies Function Documentation

## Overview
The `install_dependencies` function installs Python package dependencies defined in a requirements file. This function automates the installation process, ensuring that all necessary packages are available for a project.

## Parameters

### `file_path: str`
- The path to the requirements file that contains package specifications.
- The file should be formatted according to Python's package manager (pip) requirements format (e.g., `packagename==version`).

## Raises
- `FileNotFoundError`: 
  - This exception is raised if the specified requirements file does not exist at the provided path.
- `Exception`: 
  - This exception is raised if there is an error during the installation process, such as an invalid package name in the requirements file.

## Returns
- `None`: 
  - The function does not return a value. Instead, it performs the installation and prints a success message upon successful installation of all dependencies.

## Example Usage
