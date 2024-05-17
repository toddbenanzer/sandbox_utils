# Package Name

## Overview

This package provides a collection of utility functions for working with Boolean columns in pandas DataFrames. It includes functions for handling missing data, calculating statistics, performing hypothesis testing, visualizing distributions, and more.

## Usage

To use this package, you will need to have pandas and numpy installed in your Python environment. You can install them using pip:

```
pip install pandas numpy
```

After installing the required dependencies, you can import the package and start using the functions. Here is an example:

```python
import boolean_utils

# Create a DataFrame
data = pd.DataFrame({'A': [True, False, True, True],
                     'B': [False, False, True, False]})

# Calculate the number of true values in column A
count = boolean_utils.count_true_values(data, 'A')
print(count)
```

## Examples

### Checking Column Existence

```python
import boolean_utils

data = pd.DataFrame({'A': [True, False]})
column_name = 'B'

try:
    boolean_utils.check_column_exists(data, column_name)
except ValueError as e:
    print(str(e))
```

Output:
```
Column 'B' does not exist in the dataframe.
```

### Handling Missing Data

```python
import boolean_utils

data = pd.DataFrame({'A': [True, False]})
column_name = 'A'
replace_value = True

data = boolean_utils.handle_missing_data(data, column_name, replace_value)
print(data)
```

Output:
```
       A
0   True
1  False
```

### Calculating Statistics

```python
import boolean_utils

data = pd.DataFrame({'A': [True, False]})
column_name = 'A'

stats = boolean_utils.boolean_stats(data, column_name)
print(stats)
```

Output:
```
{'mean': 0.5, 'median': 0.5, 'mode': [False, True]}
```

### Visualizing Distribution

```python
import boolean_utils

data = pd.DataFrame({'A': [True, False]})
column_name = 'A'

boolean_utils.visualize_boolean_distribution(data[column_name])
```

Output:
![Boolean Column Distribution](path_to_image)

These are just a few examples of the functionality provided by this package. For more details and additional examples, please refer to the documentation of each individual function.