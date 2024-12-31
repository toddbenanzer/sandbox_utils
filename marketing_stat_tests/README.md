# StatisticalTests Class Documentation

## Overview
The `StatisticalTests` class provides methods for performing common statistical tests used in marketing analytics. This class can conduct t-tests, ANOVA, chi-square tests, and regression analysis, enabling users to evaluate the effectiveness of marketing strategies and campaigns quantitatively.

## Methods

### 1. `perform_t_test(data_a: np.ndarray, data_b: np.ndarray, paired: bool = False) -> Dict[str, Any]`
Performs a t-test to compare the means of two samples.

#### Parameters:
- `data_a`: `np.ndarray`
  - First set of data for the t-test.
- `data_b`: `np.ndarray`
  - Second set of data for the t-test.
- `paired`: `bool`
  - If `True`, performs a paired t-test. Default is `False`.

#### Returns:
- `Dict[str, Any]`
  - A dictionary containing:
    - `t_stat`: The calculated t-value.
    - `p_value`: The p-value of the test.
    - `degrees_of_freedom`: The degrees of freedom used in the test.

---

### 2. `perform_anova(*groups: np.ndarray) -> Dict[str, Any]`
Performs an ANOVA test to determine if there are significant differences between group means.

#### Parameters:
- `groups`: `np.ndarray`
  - Multiple arrays, each corresponding to a group to be tested.

#### Returns:
- `Dict[str, Any]`
  - A dictionary containing:
    - `f_stat`: The calculated F-statistic.
    - `p_value`: The p-value of the ANOVA test.
    - `df_between`: The degrees of freedom between groups.
    - `df_within`: The degrees of freedom within groups.

---

### 3. `perform_chi_square(observed: np.ndarray, expected: np.ndarray) -> Dict[str, Any]`
Performs a chi-square test for goodness of fit.

#### Parameters:
- `observed`: `np.ndarray`
  - Observed frequency counts.
- `expected`: `np.ndarray`
  - Expected frequency counts.

#### Returns:
- `Dict[str, Any]`
  - A dictionary containing:
    - `chi_stat`: The chi-square statistic.
    - `p_value`: The p-value of the chi-square test.

---

### 4. `perform_regression_analysis(data: pd.DataFrame, independent_vars: List[str], dependent_var: str) -> Dict[str, Any]`
Performs a regression analysis to model the relationship between independent and dependent variables.

#### Parameters:
- `data`: `pd.DataFrame`
  - Dataset containing independent and dependent variables.
- `independent_vars`: `List[str]`
  - Column names in data representing independent variables.
- `dependent_var`: `str`
  - Column name in data representing the dependent variable.

#### Returns:
- `Dict[str, Any]`
  - A dictionary containing:
    - `coefficients`: The regression coefficients.
    - `r_squared`: The R-squared value of the regression model.
    - `standard_error`: The standard error of the regression.
    - `p_values`: The p-values for the coefficients.


# DataHandler Class Documentation

## Overview
The `DataHandler` class provides functionalities for managing data handling tasks, such as loading, cleansing, and preprocessing data. This class is designed to facilitate the preparation of data for analysis in a user-friendly manner.

## Methods

### 1. `load_data(file_path: str) -> pd.DataFrame`
Loads data from the specified file path into a pandas DataFrame.

#### Parameters:
- `file_path`: `str`
  - The path to the data file. Acceptable formats are CSV and Excel.

#### Returns:
- `pd.DataFrame`
  - The loaded data as a pandas DataFrame.

#### Raises:
- `ValueError`
  - If the file format is unsupported or if an error occurs while loading the data.

---

### 2. `cleanse_data(data: pd.DataFrame) -> pd.DataFrame`
Cleanses the data by handling missing values and removing duplicates.

#### Parameters:
- `data`: `pd.DataFrame`
  - The input data to cleanse.

#### Returns:
- `pd.DataFrame`
  - The cleansed data with missing values handled and duplicates removed.

---

### 3. `preprocess_data(data: pd.DataFrame) -> pd.DataFrame`
Preprocesses the data by normalizing numeric values and encoding categorical features.

#### Parameters:
- `data`: `pd.DataFrame`
  - Input data for preprocessing.

#### Returns:
- `pd.DataFrame`
  - The preprocessed data, with normalized numeric features and encoded categorical variables.



# ResultsInterpreter Class Documentation

## Overview
The `ResultsInterpreter` class provides methods for interpreting the results of statistical tests, calculating effect sizes, and computing confidence intervals. It is designed to facilitate the understanding of statistical outcomes in hypothesis testing.

## Methods

### 1. `interpret_t_test(result: Dict[str, float]) -> str`
Interprets the results of a t-test.

#### Parameters:
- `result`: `Dict[str, float]`
  - A dictionary containing:
    - `t_stat`: The calculated t-statistic.
    - `p_value`: The p-value of the test.
    - `degrees_of_freedom`: The degrees of freedom used in the test.

#### Returns:
- `str`
  - A human-readable interpretation of the t-test results, indicating whether the result is statistically significant.

---

### 2. `interpret_anova(result: Dict[str, float]) -> str`
Interprets the results of an ANOVA test.

#### Parameters:
- `result`: `Dict[str, float]`
  - A dictionary containing:
    - `f_stat`: The calculated F-statistic.
    - `p_value`: The p-value of the ANOVA test.
    - `df_between`: The degrees of freedom between groups.
    - `df_within`: The degrees of freedom within groups.

#### Returns:
- `str`
  - A human-readable interpretation of the ANOVA results, indicating whether there are significant differences between groups.

---

### 3. `calculate_effect_size(test_result: Dict[str, float]) -> float`
Calculates the effect size (Cohen's d) for a t-test.

#### Parameters:
- `test_result`: `Dict[str, float]`
  - A t-test result dictionary containing:
    - `t_stat`: The calculated t-statistic.
    - `degrees_of_freedom`: The degrees of freedom used in the test.

#### Returns:
- `float`
  - The calculated effect size (Cohen's d).

---

### 4. `compute_confidence_interval(test_result: Dict[str, float], confidence_level: float = 0.95) -> Tuple[float, float]`
Computes the confidence interval for the mean difference.

#### Parameters:
- `test_result`: `Dict[str, float]`
  - A t-test result dictionary containing:
    - `t_stat`: The calculated t-statistic.
    - `degrees_of_freedom`: The degrees of freedom used in the test.
- `confidence_level`: `float`
  - The confidence level for the interval (default is 0.95).

#### Returns:
- `Tuple[float, float]`
  - The lower and upper bounds of the confidence interval for the mean difference.


# Visualizer Class Documentation

## Overview
The `Visualizer` class provides methods for visualizing data distributions, summarizing statistical results, and generating correlation matrices. It utilizes the `matplotlib` and `seaborn` libraries to produce informative and aesthetically pleasing plots.

## Methods

### 1. `plot_distribution(data: Union[np.ndarray, pd.Series], test_type: str) -> None`
Plots the distribution of the given data.

#### Parameters:
- `data`: `Union[np.ndarray, pd.Series]`
  - The dataset for which to plot the distribution.
- `test_type`: `str`
  - Specifies the type of distribution to plot (e.g., 't-test', 'ANOVA').

#### Returns:
- `None`
  - This method displays a histogram with a kernel density estimate (KDE) overlay.

---

### 2. `create_result_summary_plot(results: dict) -> None`
Creates a plot summarizing statistical test results.

#### Parameters:
- `results`: `dict`
  - Contains statistical test outcomes and relevant metrics to visualize, such as:
    - `categories`: List of categories to display.
    - `means`: List of means for each category.
    - `conf_int_lower`: List of lower confidence intervals.
    - `conf_int_upper`: List of upper confidence intervals.

#### Returns:
- `None`
  - This method displays an error bar plot summarizing the results.

---

### 3. `generate_correlation_matrix(data: pd.DataFrame) -> None`
Generates and visualizes a correlation matrix of the dataset.

#### Parameters:
- `data`: `pd.DataFrame`
  - The dataset for which to compute and visualize correlations.

#### Returns:
- `None`
  - This method displays a heatmap representing the correlation coefficients among the variables in the dataset.


# setup_logging Function Documentation

## Overview
The `setup_logging` function configures the logging settings for the module, allowing for standardized logging throughout the application. It sets the desired logging level and the format of log messages.

## Parameters

### 1. `level`
- **Type:** `str`
- **Description:** The desired logging level. Acceptable values are `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, and `"CRITICAL"`.
  
## Returns
- **Type:** `None`
- **Description:** This function does not return a value. It configures the logging system for the module.

## Exceptions
- **ValueError:** 
  - Raised if an invalid logging level is provided. The error message will indicate the invalid level.

## Usage
To use the `setup_logging` function, call it with the desired log level string to configure the logging for your module:



# configure_integration Function Documentation

## Overview
The `configure_integration` function is responsible for setting up integration with specified external libraries in a Python application. It checks for the availability of the libraries and performs any necessary configurations.

## Parameters

### 1. `libraries`
- **Type:** `list of str`
- **Description:** A list of library names (as strings) that the function will attempt to integrate and configure.

## Returns
- **Type:** `None`
- **Description:** This function does not return a value; it logs the outcome of the integration process.

## Exceptions
- **ImportError:** 
  - Raised if a library specified in the `libraries` list is not available for import. The function logs an error message indicating which library is missing.
- **Exception:** 
  - Catches any other exceptions that may occur during the import or configuration process and logs an appropriate error message.

## Usage
To use the `configure_integration` function, pass a list of libraries you want to integrate:

