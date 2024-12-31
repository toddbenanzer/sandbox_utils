# StatisticalTest Class Documentation

## Overview
The `StatisticalTest` class provides a framework for performing statistical tests on two datasets. It includes methods to handle zero variance, missing values, and constant values. This class is useful for data preprocessing before conducting statistical analyses.

## Initialization


# NumericTest Class Documentation

## Overview
The `NumericTest` class is designed for performing statistical tests on numeric datasets. It inherits from the `StatisticalTest` class, utilizing its data preprocessing capabilities such as handling missing values and constant values. This class includes methods for conducting t-tests and ANOVA tests.

## Inheritance
Inherits from:
- `StatisticalTest`: This class provides foundational functionality for managing and preprocessing datasets prior to performing statistical analyses.

## Methods

### perform_t_test


# CategoricalTest Class Documentation

## Overview
The `CategoricalTest` class is designed for performing statistical tests on categorical datasets. It inherits from the `StatisticalTest` class, allowing it to utilize its data preprocessing capabilities such as handling missing values and constant values. This class focuses on executing the Chi-squared test to evaluate relationships between categorical variables.

## Inheritance
Inherits from:
- `StatisticalTest`: This base class provides foundational functionality for processing datasets prior to statistical analysis.

## Methods

### perform_chi_squared_test


# BooleanTest Class Documentation

## Overview
The `BooleanTest` class is designed for performing statistical tests on boolean datasets. It inherits from the `StatisticalTest` class, allowing it to utilize its data preprocessing functionalities such as handling missing values and constant values. This class focuses on executing a z-test for comparing two proportions from boolean data.

## Inheritance
Inherits from:
- `StatisticalTest`: This base class provides foundational functionality for data management and preprocessing before statistical testing.

## Methods

### perform_proportions_test


# compute_descriptive_statistics Function Documentation

## Overview
The `compute_descriptive_statistics` function calculates a set of key descriptive statistics for a given numeric dataset. It provides insight into the dataset's distribution, central tendency, and variability.

## Parameters
- `data` (`np.ndarray`): A NumPy array containing numeric data. The function will handle NaN and infinite values by excluding them from the calculations.

## Returns
- `dict`: A dictionary containing the following descriptive statistics:
  - `mean` (`float`): The arithmetic average of the data.
  - `median` (`float`): The middle value of the dataset when ordered.
  - `mode` (`array` or `value`): The most frequently occurring value(s) in the dataset.
  - `variance` (`float`): The measure of how much the data points vary from the mean.
  - `standard_deviation` (`float`): The square root of the variance, indicating the average distance of the data points from the mean.
  - `min` (`float`): The smallest value in the dataset.
  - `max` (`float`): The largest value in the dataset.
  - `count` (`int`): The total number of non-NaN and finite values in the dataset.

## Behavior
- The function first removes NaN and infinite values from the input data to ensure accurate calculations.
- It then computes the desired statistics using NumPy and SciPy libraries, returning a summarized statistical profile of the dataset.


# visualize_results Function Documentation

## Overview
The `visualize_results` function generates visual representations of statistical test results or descriptive statistics. It utilizes Matplotlib and Seaborn to create various types of plots that facilitate the interpretation of data insights.

## Parameters
- `results` (`dict`): A dictionary containing information to be visualized. This may include:
  - `'data'`: A numeric dataset for which a histogram will be generated.
  - `'t_statistic'`: The calculated T-statistic, which, along with the `p_value`, is used to create a bar chart.
  - `'p_value'`: The p-value corresponding to the statistical test, displayed alongside the T-statistic in the bar chart.
  - `'box_data'`: A collection of datasets used to generate box plots for visualizing data spread and outliers.
  - `'category_proportions'`: A dictionary of categories and their corresponding proportions, used to generate a pie chart.

## Behavior
- If the `results` dictionary contains the key `'data'`, a histogram with density estimation (KDE) will be displayed to visualize the data distribution.
- If the keys `'t_statistic'` and `'p_value'` are present, a bar chart will illustrate these statistical test results.
- If `'box_data'` is provided, a box plot will be generated to show individual data distributions and to highlight any outliers.
- Lastly, if `'category_proportions'` is present, a pie chart will depict the proportions of different categories.

## Visualization Details
- Histograms and bar charts are customized with titles and labeled axes.
- Box plots are labeled appropriately to indicate the variable being analyzed.
- Pie charts feature percentages displayed on the slices and are created with equal aspect ratios for proper circular representation.


# validate_input_data Function Documentation

## Overview
The `validate_input_data` function checks the input datasets to determine if they are suitable for statistical analysis. It ensures the integrity and compatibility of the datasets before performing any statistical operations.

## Parameters
- `data1`: The first dataset, which can be a NumPy array or a list.
- `data2`: The second dataset, which can also be a NumPy array or a list.

## Returns
- `bool`: Returns `True` if both datasets are valid for analysis.

## Raises
- `ValueError`: This exception is raised if the datasets are found to have any of the following issues:
  - The data types of `data1` or `data2` are not NumPy arrays or lists.
  - Either dataset is empty.
  - The datasets have different numbers of dimensions.
  - The lengths of `data1` and `data2` do not match when a paired analysis is required.
  - Either dataset contains more missing values than the predefined threshold (default set to 50%).

## Behavior
- The function first checks that both `data1` and `data2` are of appropriate types.
- It then converts any lists into NumPy arrays to maintain consistency.
- Following this, it verifies that the datasets are not empty, that they have the same number of dimensions, and that their lengths are compatible for paired analyses.
- Lastly, it checks the fraction of missing values in the datasets and raises an error if the number exceeds the specified threshold.


# impute_missing_values Function Documentation

## Overview
The `impute_missing_values` function handles missing values in a dataset by applying a specified imputation method. This ensures a complete dataset that can be used for further statistical analysis, addressing potential biases caused by missing data.

## Parameters
- `data` (`np.ndarray` or `list`): A NumPy array or list containing numeric data with potential missing values (e.g., NaN).
- `method` (`str`): A string indicating the imputation method to be used. The options include:
  - `'mean'`: Replace missing values with the mean of the non-missing values.
  - `'median'`: Replace missing values with the median of the non-missing values.
  - `'mode'`: Replace missing values with the mode (most frequent value) of the non-missing values.
  - `'zero'`: Replace missing values with zero.

## Returns
- `np.ndarray`: An array with the missing values imputed based on the chosen method.

## Raises
- `ValueError`: Raised if the specified imputation method is unsupported or if the mode is requested for a dataset containing all NaN values, making mode calculation undefined.

## Behavior
- The function converts the input to a NumPy array for consistency before performing imputation.
- It identifies and replaces missing values according to the specified method, utilizing appropriate NumPy and SciPy functions to compute necessary statistics.
- A message indicating the completion of imputation, along with the method used, is printed to the console for user feedback.
