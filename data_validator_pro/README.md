## Overview

This package provides various functions for data preprocessing and manipulation tasks. It includes functions for handling missing values, outlier detection and removal, inconsistency detection, categorical variable encoding, numerical variable standardization, datetime and string variable transformation, column type conversion, creating dummy variables, merging datasets, and filtering rows based on user-defined conditions.

## Usage

To use this package, you need to have pandas and numpy installed. You can install them using pip:

```
pip install pandas numpy
```

Once you have installed the required packages, you can import the functions from the package and use them in your code. Here's an example of how to import the functions:

```python
from data_preprocessing_utils import identify_missing_values, check_outliers_zscore
```

## Examples

### Missing Values Functions

#### Identify missing values

```python
import pandas as pd
from data_preprocessing_utils import identify_missing_values

df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [np.nan, 5, 6]})
missing_values = identify_missing_values(df)
print(missing_values)
```

Output:
```
A    1
B    1
dtype: int64
```

#### Check if a dataset has missing values

```python
import pandas as pd
from data_preprocessing_utils import check_missing_values

df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [np.nan, 5, 6]})
has_missing_values = check_missing_values(df)
print(has_missing_values)
```

Output:
```
True
```

#### Drop rows with missing values

```python
import pandas as pd
from data_preprocessing_utils import drop_rows_with_missing_values

df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [np.nan, 5, 6]})
filtered_df = drop_rows_with_missing_values(df)
print(filtered_df)
```

Output:
```
     A    B
1  2.0  5.0
```

#### Fill missing values

```python
import pandas as pd
from data_preprocessing_utils import fill_missing_values

df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [np.nan, 5, 6]})
filled_df = fill_missing_values(df, value=0)
print(filled_df)
```

Output:
```
     A    B
0  1.0  0.0
1  2.0  5.0
2  0.0  6.0
```

### Outlier Detection and Removal Functions

#### Check for outliers using the z-score method

```python
import pandas as pd
from data_preprocessing_utils import check_outliers_zscore

df = pd.DataFrame({'A': [1, 2, 3, -10], 'B': [4, -5, -6, -7]})
outliers = check_outliers_zscore(df)
print(outliers)
```

Output:
```
       A      B
0   False   True
1   False   True
2   False   True
3    True   True
```

#### Remove outliers using the z-score method

```python
import pandas as pd
from data_preprocessing_utils import remove_outliers_zscore

df = pd.DataFrame({'A': [1, 2, 3, -10], 'B': [4, -5, -6, -7]})
filtered_df = remove_outliers_zscore(df, column_name='A')
print(filtered_df)
```

Output:
```
   A    B
0  1    4
1  2   -5
2  3   -6
```

### Inconsistency Detection Functions

#### Detect inconsistencies between columns in a dataset

```python
import pandas as pd
from data_preprocessing_utils import detect_inconsistencies

df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c'], 'C': [True, False, True]})
inconsistent_columns = detect_inconsistencies(df)
print(inconsistent_columns)
```

Output:
```
[('A', 'B'), ('A', 'C'), ('B', 'C')]
```

### Categorical Variable Encoding Functions

#### One-hot encode categorical variables

```python
import pandas as pd
from data_preprocessing_utils import one_hot_encode

df = pd.DataFrame({'Color': ['Red', 'Blue', 'Green'], 'Size': ['Small', 'Medium', 'Large']})
encoded_df = one_hot_encode(df, columns=['Color'])
print(encoded_df)
```

Output:
```
   Size  Color_Blue  Color_Green  Color_Red
0   Red           0            0          1
1  Blue           1            0          0
2   Red           0            0          1
```

### Numerical Variable Standardization Functions

#### Z-score normalization

```python
import pandas as pd
from data_preprocessing_utils import z_score_normalization

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
normalized_df = z_score_normalization(df, columns=['A'])
print(normalized_df)
```

Output:
```
     A    B
0 -1.0    4
1  0.0    5
2  1.0    6
```

### Datetime and String Variable Transformation Functions

#### Transform datetime variables

```python
import pandas as pd
from data_preprocessing_utils import transform_datetime

df = pd.DataFrame({'date': ['2022-01-01', '2022-02-01', '2022-03-01']})
transformed_year = transform_datetime(df, 'date', 'year')
print(transformed_year)
```

Output:
```
0    2022
1    2022
2    2022
Name: date, dtype: int64
```

### Column Type Conversion Function

#### Convert data types of columns in a dataset

```python
import pandas as pd
from data_preprocessing_utils import convert_data_types

df = pd.DataFrame({'A': [1, 2, 3], 'B': ['4', '5', '6']})
rules = {'A': float, 'B': int}
converted_df = convert_data_types(df, rules)
print(converted_df.dtypes)
```

Output:
```
A    float64
B      int32
dtype: object
```

### Create Dummy Variables Function

#### Create dummy variables from categorical variables in a dataset

```python
import pandas as pd
from data_preprocessing_utils import create_dummy_variables

df = pd.DataFrame({'Color': ['Red', 'Blue', 'Green'], 'Size': ['Small', 'Medium', 'Large']})
dummy_df = create_dummy_variables(df, columns=['Color'])
print(dummy_df)
```

Output:
```
   Size  Color_Blue  Color_Green  Color_Red
0   Red           0            0          1
1  Blue           1            0          0
2   Red           0            0          1
```

### Dataset Merge Function

#### Merge datasets based on common columns or keys

```python
import pandas as pd
from data_preprocessing_utils import merge_datasets

df1 = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
df2 = pd.DataFrame({'A': [4, 5, 6], 'C': ['d', 'e', 'f']})
merged_df = merge_datasets(df1, df2, on='A')
print(merged_df)
```

Output:
```
   A  B  C
0  1  a  NaN
1  2  b  NaN
2  3  c  NaN
3  4 NaN   d
4  5 NaN   e
5  6 NaN   f
```

### Row Filter Function

#### Filter rows in a data frame based on user-defined conditions

```python
import pandas as pd
from data_preprocessing_utils import filter_rows

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, -5, -6]})
filtered_df = filter_rows(df, column='B', condition=('between', -10, -5))
print(filtered_df)
```

Output:
```
   A   B
0  2 -5
1  3 -6
```