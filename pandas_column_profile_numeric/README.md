# DescriptiveStatistics Class Documentation

## Overview
The `DescriptiveStatistics` class provides methods for calculating various descriptive statistics on a numeric column represented as a pandas Series. It helps in understanding the central tendency, dispersion, outlier detection, and statistical distribution of the data.

## Initialization
### `__init__(self, data: pd.Series)`
Initializes the `DescriptiveStatistics` object with the provided numeric data.

- **Parameters:**
  - `data (pd.Series)`: The column of numeric data for analysis. Missing values will be dropped during initialization.

## Methods

### `compute_central_tendency(self) -> Dict[str, float]`
Calculates and returns measures of central tendency.

- **Returns:**
  - `dict`: A dictionary containing the following keys:
    - `mean`: The mean of the data.
    - `median`: The median of the data.
    - `mode`: The mode of the data.

### `compute_dispersion(self) -> Dict[str, float]`
Calculates and returns measures of dispersion.

- **Returns:**
  - `dict`: A dictionary containing:
    - `variance`: The variance of the data.
    - `std_dev`: The standard deviation of the data.
    - `range`: The range (max - min) of the data.
    - `IQR`: The interquartile range of the data.

### `detect_outliers(self, method: str = 'z-score') -> List[float]`
Detects outliers in the data using the specified method.

- **Parameters:**
  - `method (str)`: The method to use for outlier detection. Options are:
    - `'z-score'`: Uses Z-score method for outlier detection.
    - `'IQR'`: Uses the Interquartile Range method for outlier detection. 
    - Default is `'z-score'`.

- **Returns:**
  - `list`: A list of detected outliers.

- **Raises:**
  - `ValueError`: If an unsupported method is provided for outlier detection.

### `estimate_distribution(self) -> str`
Estimates and displays the statistical distribution of the data.

- **Returns:**
  - `str`: The estimated distribution type (e.g., `'normal'`, `'binomial'`).
  
- **Visualization:**
  Displays a histogram with a kernel density estimate (KDE) overlay for visual analysis of the data distribution.

## Usage Example


# DataHandler Class Documentation

## Overview
The `DataHandler` class provides methods for preprocessing and validating a numeric dataset in the form of a pandas Series. It supports handling missing and infinite values, as well as checking for null or trivial data.

## Initialization
### `__init__(self, data: pd.Series)`
Initializes a `DataHandler` object with the provided numeric data.

- **Parameters:**
  - `data (pd.Series)`: The column of numeric data to be preprocessed and validated.

## Methods

### `handle_missing_values(self, strategy: str = 'mean') -> pd.Series`
Identifies and handles missing values in the dataset using a specified strategy.

- **Parameters:**
  - `strategy (str)`: The strategy to handle missing values. Options include:
    - `'mean'`: Replace missing values with the mean of the data.
    - `'median'`: Replace missing values with the median of the data.
    - `'mode'`: Replace missing values with the mode of the data.
    - `'drop'`: Remove all rows with missing values.
  - Default is `'mean'`.

- **Returns:**
  - `pd.Series`: A Series with missing values handled according to the specified strategy.

- **Raises:**
  - `ValueError`: If an unsupported strategy is provided.

### `handle_infinite_values(self, strategy: str = 'mean') -> pd.Series`
Identifies and processes infinite values in the dataset, applying the same strategy as defined for handling missing values.

- **Parameters:**
  - `strategy (str)`: The strategy to handle infinite values. Options are the same as for `handle_missing_values`. Default is `'mean'`.

- **Returns:**
  - `pd.Series`: A Series with infinite values handled according to the specified strategy.

### `check_null_trivial(self) -> bool`
Checks if the data is null or trivial (contains only a single unique value).

- **Returns:**
  - `bool`: Returns True if the data is entirely null or trivial; otherwise returns False.

- **Output:**
  - Prints a message if the series is entirely null or contains only one unique value.

## Usage Example


# calculate_statistics Function Documentation

## Overview
The `calculate_statistics` function computes comprehensive descriptive statistics for a given pandas Series containing numeric data. It handles preprocessing for missing and infinite values before applying statistical methods, making it a robust tool for data analysis.

## Parameters

### `data`
- **Type:** `pd.Series`
- **Description:** A pandas Series that contains the numeric column of data for analysis. The function performs statistical calculations based on this input.

## Returns
- **Type:** `dict`
- **Description:** A dictionary containing the computed descriptive statistics, including:
  - `central_tendency`: A dictionary with mean, median, and mode.
  - `dispersion`: A dictionary with variance, standard deviation, range, and interquartile range (IQR).
  - `outliers`: A list of detected outliers based on the Z-score method.
  - `estimated_distribution`: A string representing the estimated statistical distribution of the data.
  - `handled_data`: A pandas Series that contains the cleaned data after handling missing and infinite values.

## Behavior
1. **Data Preprocessing:**
   - The function checks if the input data is null or trivial (contains a single unique value).
   - Handles any infinite values by replacing them according to the specified strategy (default is mean).

2. **Statistical Calculations:**
   - Creates an instance of `DescriptiveStatistics` to compute central tendency and dispersion measures.
   - Detects outliers using the Z-score method.
   - Estimates the likely statistical distribution of the cleaned data.

3. **Output Compilation:**
   - Gathers the computed statistics into a returnable dictionary format.

## Example Usage


# visualize_distribution Function Documentation

## Overview
The `visualize_distribution` function visualizes the distribution of numeric data within a pandas Series. It employs a histogram combined with a kernel density estimate (KDE) to provide an intuitive graphical representation of the data's distribution, allowing for easy identification of patterns and characteristics.

## Parameters

### `data`
- **Type:** `pd.Series`
- **Description:** A pandas Series containing the numeric column of data to be visualized.

## Returns
- **Type:** `None`
- **Description:** This function does not return any value. Instead, it directly displays the histogram and KDE plot, alongside an optional box plot for further insight.

## Behavior
1. **Data Validation:**
   - The function first checks if the input Series is empty. If it is, it prints a message and exits without displaying a plot.
   - It also verifies whether the data is numeric. If the data type is non-numeric, it prints a corresponding message and exits.

2. **Visualization:**
   - Creates a histogram displaying the frequency distribution of the data with a KDE overlay for smooth approximation of the distribution.
   - Includes a box plot to provide additional visual insight into the spread and potential outliers in the data.

3. **Plot Customization:**
   - Sets up the figure size, colors, titles, and labels for the axes to enhance the clarity and presentation of the visualized data.

## Example Usage


# detect_outliers Function Documentation

## Overview
The `detect_outliers` function identifies and returns outliers in a given pandas Series based on specified statistical methods. This function is designed to help in recognizing anomalies or extreme values in the dataset that may affect statistical analyses and modeling.

## Parameters

### `data`
- **Type:** `pd.Series`
- **Description:** A pandas Series containing the numeric column of data for outlier detection. The function works with numeric types only.

### `method`
- **Type:** `str`
- **Optional:** Yes, default value is `'z-score'`.
- **Description:** The method used for outlier detection. Currently supported:
  - `'z-score'`: Identifies outliers as points that fall outside a threshold defined by Z-scores, specifically those with an absolute value greater than 3.

## Returns
- **Type:** `List[float]`
- **Description:** A list of detected outlier values based on the specified method. If no outliers are found, the list will be empty.

## Behavior
1. **Data Validation:**
   - Checks if the provided Series is empty. If so, a message is printed and an empty list is returned.
   - Verifies whether the data is numeric. If the data type is non-numeric, a message is printed and an empty list is returned.

2. **Outlier Detection Using Z-score:**
   - Calculates the mean and standard deviation of the dataset.
   - Computes Z-scores for the data points.
   - Identifies outliers as those with a |Z-score| greater than 3.

3. **Error Handling:**
   - Raises a `ValueError` if an unsupported method is provided for outlier detection.

## Example Usage


# estimate_likely_distribution Function Documentation

## Overview
The `estimate_likely_distribution` function analyzes a given pandas Series containing numeric data and estimates the most likely statistical distribution that fits the data. This function aids in statistical analysis by providing insights into the data's underlying distribution.

## Parameters

### `data`
- **Type:** `pd.Series`
- **Description:** A pandas Series that contains the numeric column of data for analysis. The function performs statistical tests to evaluate which distribution the data is likely to follow.

## Returns
- **Type:** `str`
- **Description:** The function returns a string that indicates the estimated statistical distribution of the data. Possible values include:
  - `'normal'`: Indicates that the data follows a normal distribution.
  - `'exponential'`: Indicates that the data follows an exponential distribution.
  - `'uniform'`: Indicates that the data follows a uniform distribution.
  - If no distribution is likely or if the input data is empty/non-numeric, it returns "No Data" or "Non-numeric Data" accordingly.

## Behavior
1. **Data Validation:**
   - If the input Series is empty, a message is printed, and the function returns "No Data."
   - If the Series contains non-numeric data, a message is printed, and the function returns "Non-numeric Data."

2. **Distribution Fitting:**
   - The function cleans the data by removing NaN values.
   - It performs goodness-of-fit tests for three distributions: normal, exponential, and uniform.
   - For each distribution, a p-value is calculated, and if the p-value indicates a good fit, the distribution type is added to a list of potential fits.

3. **Visualization:**
   - A Q-Q plot is generated for visual analysis of the normality of the data.

4. **Conclusion:**
   - The function returns the first successfully fitted distribution or indicates that no likely distribution was found.

## Example Usage
