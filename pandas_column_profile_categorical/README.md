# Python Data Analysis Package

This package provides various functions for analyzing categorical data in Python. It includes functionality for calculating statistics, handling missing values, encoding categorical columns, performing statistical tests, and visualizing data.

## Installation

To use this package, you need to have Python installed. You can install the package using pip:

```
pip install python-data-analysis
```

## Usage

Once the package is installed, you can import the functions and use them in your code. Here's an example of how to import and use the `calculate_unique_categories` function:

```python
import pandas as pd
from python_data_analysis import calculate_unique_categories

# Create a pandas Series with categorical data
data = pd.Series(['A', 'B', 'C', 'A', 'B', 'B'])

# Calculate the number of unique categories
unique_categories = calculate_unique_categories(data)
print(unique_categories)  # Output: 3
```

## Examples

Here are some examples of how to use the functions provided by this package:

### Calculate Category Count

The `calculate_category_count` function calculates the count of each category in a categorical column. Here's an example:

```python
import pandas as pd
from python_data_analysis import calculate_category_count

# Create a pandas Series with categorical data
data = pd.Series(['A', 'B', 'C', 'A', 'B', 'B'])

# Calculate the count of each category
category_count = calculate_category_count(data)
print(category_count)
```

Output:
```
B    3
A    2
C    1
dtype: int64
```

### Handle Missing Values

The `handle_missing_values` function replaces missing values in a categorical column with a specified value. Here's an example:

```python
import pandas as pd
from python_data_analysis import handle_missing_values

# Create a pandas DataFrame with a categorical column
data = pd.DataFrame({'Category': ['A', 'B', '', 'A', 'B', 'B']})

# Replace missing values with 'Unknown'
data = handle_missing_values(data, 'Category', 'Unknown')
print(data)
```

Output:
```
  Category
0        A
1        B
2  Unknown
3        A
4        B
5        B
```

### Visualize Categories

The `visualize_categories` function can be used to visualize the distribution of categories in a categorical column. It supports both bar and pie charts. Here's an example:

```python
import pandas as pd
from python_data_analysis import visualize_categories

# Create a pandas Series with categorical data
data = pd.Series(['A', 'B', 'C', 'A', 'B', 'B'])

# Visualize the categories using a bar chart
visualize_categories(data, chart_type='bar')

# Visualize the categories using a pie chart
visualize_categories(data, chart_type='pie')
```

### Perform Statistical Tests

This package also provides functions for performing statistical tests on categorical data. For example, you can use the `chi_square_test` function to perform a chi-square test of independence between two categorical variables. Here's an example:

```python
import pandas as pd
from python_data_analysis import chi_square_test

# Create a pandas DataFrame with two categorical columns
data = pd.DataFrame({'Category1': ['A', 'B', 'C'], 'Category2': ['X', 'Y', 'Z']})

# Perform the chi-square test
result = chi_square_test(data, 'Category1', 'Category2')
print(result)
```

Output:
```
(2.0, 0.36787944117144233, 2,
 array([[0.66666667, 0.33333333],
        [0.66666667, 0.33333333],
        [0.66666667, 0.33333333]]))
```

## Documentation

For more information on the functions and their parameters, please refer to the function docstrings or the package documentation.