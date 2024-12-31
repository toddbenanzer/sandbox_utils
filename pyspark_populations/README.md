# PopulationManager Documentation

## Overview
The `PopulationManager` class is designed to manage and manipulate defined populations using PySpark. It allows users to define populations based on SQL queries and retrieve the corresponding DataFrames.

## Class Definition

### `PopulationManager`

#### Description
A class to manage and manipulate defined populations using PySpark.

#### Parameters
- **spark_session** (`SparkSession`): An active Spark session to execute SQL queries and manage data operations.

### Methods

#### `__init__(self, spark_session: SparkSession)`

Initializes the PopulationManager with a Spark session.

**Args:**
- `spark_session` (`SparkSession`): An active Spark session.

---

#### `define_population(self, population_name: str, sql_query: str) -> None`

Defines a population based on the provided SQL query and stores it with the specified name.

**Args:**
- `population_name` (`str`): The name for the defined population.
- `sql_query` (`str`): SQL query to define the population.

**Raises:**
- `ValueError`: If the population name already exists or if the SQL query is invalid.

---

#### `get_population(self, population_name: str) -> DataFrame`

Retrieves the DataFrame corresponding to the specified population name.

**Args:**
- `population_name` (`str`): The name of the population to retrieve.

**Returns:**
- `DataFrame`: The DataFrame of the specified population.

**Raises:**
- `ValueError`: If the population name is not found.


# OverlapAnalyzer Documentation

## Overview
The `OverlapAnalyzer` class is designed to analyze overlaps among defined populations using PySpark. It allows users to determine intersections between multiple populations based on a common identifier.

## Class Definition

### `OverlapAnalyzer`

#### Description
A class to analyze overlap among defined populations using PySpark.

#### Parameters
- **population_manager** (`PopulationManager`): An instance of the PopulationManager class used to manage and access defined populations.

### Methods

#### `__init__(self, population_manager)`

Initializes the OverlapAnalyzer with an instance of PopulationManager.

**Args:**
- `population_manager` (`PopulationManager`): The instance to manage and access the defined populations.

---

#### `analyze_overlap(self, population_names: List[str]) -> DataFrame`

Analyzes the overlap among the specified populations and produces a DataFrame detailing the intersections.

**Args:**
- `population_names` (`List[str]`): A list containing the names of the populations to be analyzed for overlap.

**Returns:**
- `DataFrame`: A DataFrame containing the overlap information between specified populations, including details such as size of intersections.

**Raises:**
- `ValueError`: If any population name is invalid or not defined, or if fewer than two populations are specified for analysis.


# ProfileCalculator Documentation

## Overview
The `ProfileCalculator` class is designed to calculate profiles for defined populations based on specified metrics using PySpark. It enables users to define metrics and subsequently compute profiles by combining those metrics with population data.

## Class Definition

### `ProfileCalculator`

#### Description
A class to calculate profiles for defined populations based on specified metrics using PySpark.

#### Parameters
- **spark_session** (`SparkSession`): An active Spark session for executing SQL queries and managing data operations.

### Methods

#### `__init__(self, spark_session: SparkSession)`

Initializes the ProfileCalculator with a Spark session.

**Args:**
- `spark_session` (`SparkSession`): An active Spark session.

---

#### `define_metrics(self, metric_name: str, sql_query: str) -> None`

Defines a metric by executing a SQL query and stores the resultant DataFrame under the given metric name.

**Args:**
- `metric_name` (`str`): The name to assign to the metric.
- `sql_query` (`str`): SQL query used to calculate the metric.

**Raises:**
- `ValueError`: If the metric name already exists or if the SQL query is invalid.

---

#### `calculate_profiles(self, population_name: str, metric_names: List[str]) -> DataFrame`

Calculates profiles for a given population based on specified metrics and returns a summary DataFrame.

**Args:**
- `population_name` (`str`): The name of the population.
- `metric_names` (`List[str]`): A list of metric names to include in the profile.

**Returns:**
- `DataFrame`: A DataFrame summarizing the profiles for the specified population against the selected metrics.

**Raises:**
- `ValueError`: If the population or any metric is not found.


# load_data Documentation

## Overview
The `load_data` function loads data from a specified file path into a Spark DataFrame, using the designated file format (CSV, JSON, or Parquet). It facilitates data integration from external sources into a Spark environment for further processing and analysis.

## Function Definition

### `load_data`

#### Description
Loads data from a specified file path into a Spark DataFrame, using the specified file format.

#### Parameters
- **spark_session** (`SparkSession`): 
  An active Spark session used to facilitate the data loading process.

- **file_path** (`str`): 
  The path to the file containing the data to be loaded.

- **file_format** (`str`): 
  The format of the file to be loaded. Common formats include:
  - `'csv'`: Comma-separated values.
  - `'json'`: JSON formatted data.
  - `'parquet'`: Parquet formatted data.

#### Returns
- **DataFrame**: 
  A Spark DataFrame containing the data loaded from the specified file.

#### Raises
- **IOError**: 
  If the file cannot be accessed or read (e.g., file not found or permission issues).
  
- **ValueError**: 
  If the file format is unsupported or invalid (e.g., if a format other than 'csv', 'json', or 'parquet' is provided).

#### Examples


# save_results Documentation

## Overview
The `save_results` function saves a given Spark DataFrame to a specified file path in the designated file format. It supports common formats such as CSV, JSON, and Parquet.

## Function Definition

### `save_results`

#### Description
Saves a given Spark DataFrame to a specified file path in the designated file format.

#### Parameters
- **dataframe** (`DataFrame`): 
  The Spark DataFrame to be saved.

- **file_path** (`str`): 
  The destination file path where the DataFrame will be saved.

- **file_format** (`str`): 
  The format in which to save the DataFrame. Common formats include:
  - `'csv'`: Comma-separated values.
  - `'json'`: JSON formatted data.
  - `'parquet'`: Parquet formatted data.

#### Returns
- **None**

#### Raises
- **IOError**: 
  If the file path is invalid or the DataFrame cannot be saved (e.g., due to permission issues or nonexistent directories).
  
- **ValueError**: 
  If the file format is unsupported or invalid (e.g., if a format other than 'csv', 'json', or 'parquet' is specified).

#### Examples


# validate_sql_query Documentation

## Overview
The `validate_sql_query` function checks the syntax of a given SQL query to ensure it is valid and ready for execution in a Spark environment. This function is useful for identifying common SQL errors before attempting to execute the query.

## Function Definition

### `validate_sql_query`

#### Description
Validates a given SQL query to ensure it is syntactically correct and ready for execution.

#### Parameters
- **spark_session** (`SparkSession`): 
  An active Spark session used for validation purposes.

- **sql_query** (`str`): 
  The SQL query string to be validated for correctness.

#### Returns
- **bool**: 
  A boolean value indicating whether the SQL query is valid (`True`) or not (`False`).

#### Raises
- **ValueError**: 
  If the input SQL query is empty or null.

#### Examples


# setup_logging Documentation

## Overview
The `setup_logging` function configures the logging settings for the application by setting the specified logging level globally. This function is essential for controlling the verbosity of log output and ensuring that relevant logging information is captured during application runtime.

## Function Definition

### `setup_logging`

#### Description
Configures the logging settings for the application, setting the specified logging level globally.

#### Parameters
- **level** (`str`): 
  The logging level to be set. Common levels include:
  - `'DEBUG'`: Detailed information for diagnosing problems.
  - `'INFO'`: General information about application progress.
  - `'WARNING'`: An indication that something unexpected happened, or indicative of some problem in the near future.
  - `'ERROR'`: A more serious problem that prevented the program from performing a function.
  - `'CRITICAL'`: A very serious error indicating that the program itself may be unable to continue running.

#### Raises
- **ValueError**: 
  If the provided logging level is invalid or not recognized by the logging module.

#### Examples
