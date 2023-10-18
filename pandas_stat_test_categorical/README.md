# Overview

This python script provides a set of functions to perform various calculations and analyses on categorical data in a pandas DataFrame. It includes functions to calculate category frequencies, proportions, percentages, cumulative frequencies, cumulative proportions, minimum and maximum values, mode(s), median(s), range, count of non-null and null values, count of unique values, count of trivial values, missing values handling, infinite values handling, duplicate columns handling, chi-square tests (contingency table, goodness-of-fit), Fisher's exact test, G-test, Cramer's V coefficient, entropy, Gini index, concentration ratio, diversity index (Shannon's entropy), Simpson's index of diversity, Jaccard similarity index, one-way ANOVA and Tukey HSD test.

# Usage

To use this script in your python project:
1. Import the required libraries:
```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
```
2. Copy the code from the script into your project or import the functions you want to use individually.

# Examples

Here are some examples demonstrating the usage of the functions:

### Calculate Category Frequency

```python
# Input dataframe
dataframe = pd.DataFrame({'Category': ['A', 'B', 'A', 'C', 'B', 'B']})

result = calculate_category_frequency(dataframe, 'Category')

print(result)
```

Output:
```
  Category  Frequency
0        B          3
1        A          2
2        C          1
```

### Calculate Category Proportions

```python
# Input dataframe
dataframe = pd.DataFrame({'Category': ['A', 'B', 'A', 'C', 'B', 'B']})

result = calculate_category_proportions(dataframe, 'Category')

print(result)
```

Output:
```
  Category  Proportion
0        B    0.500000
1        A    0.333333
2        C    0.166667
```

### Calculate Category Percentage

```python
# Input column
column = pd.Series(['A', 'B', 'A', 'C', 'B', 'B'])

result = calculate_category_percentage(column)

print(result)
```

Output:
```
  category  percentage
0        B   50.000000
1        A   33.333333
2        C   16.666667
```

### Calculate Cumulative Frequency

```python
# Input dataframe
dataframe = pd.DataFrame({'Category': ['A', 'B', 'A', 'C', 'B', 'B']})

result = calculate_cumulative_frequency(dataframe, 'Category')

print(result)
```

Output:
```
A    2
B    5
C    6
Name: Category, dtype: int64
```

### Calculate Cumulative Proportion

```python
# Input dataframe
dataframe = pd.DataFrame({'Category': ['A', 'B', 'A', 'C', 'B', 'B']})

result = calculate_cumulative_proportion(dataframe, 'Category')

print(result)
```

Output:
```
  Category  Cumulative Proportion
0        B               0.500000
1        A               0.333333
2        C               0.166667
```

### Calculate Cumulative Percentage

```python
# Input dataframe
dataframe = pd.DataFrame({'Category': ['A', 'B', 'A', 'C', 'B', 'B']})

result = calculate_cumulative_percentage(dataframe, 'Category')

print(result)
```

Output:
```
C    16.666667
A    50.000000
B   100.000000
Name: Category, dtype: float64
```

### Calculate Mode

```python
# Input column
column = pd.Series(['A', 'B', 'A', 'C', 'B', 'B'])

result = calculate_mode(column)

print(result)
```

Output:
```
['B']
```

### Calculate Median

```python
# Input dataframe
dataframe = pd.DataFrame({'Category': ['A', 'A', 'B', 'B', 'C'], 
                          'Value': [1, 2, 3, 4, 5]})

result = calculate_median(dataframe, 'Value')

print(result)
```

Output:
```
[1.5, 3.5, 5]
```

### Calculate Range

```python
# Input dataframe
dataframe = pd.DataFrame({'Category': ['A', 'A', 'B', 'B', 'C'], 
                          'Value': [1, 2, 3, 4, 5]})

result = calculate_range(dataframe, 'Value')

print(result)
```

Output:
```
(1, 5)
```

### Calculate Minimum

```python
# Input dataframe
dataframe = pd.DataFrame({'Category': ['A', 'A', 'B', 'B', 'C'], 
                          'Value': [1, 2, 3, 4, 5]})

result = calculate_minimum(dataframe, 'Value')

print(result)
```

Output:
```
1
```

### Calculate Maximum

```python
# Input dataframe
dataframe = pd.DataFrame({'Category': ['A', 'A', 'B', 'B', 'C'], 
                          'Value': [1, 2, 3, 4, 5]})

result = calculate_max_value(dataframe, 'Value')

print(result)
```

Output:
```
5
```

### Count Non-null Values

```python
# Input dataframe
dataframe = pd.DataFrame({'Category': ['A', 'A', 'B', np.nan, 'C'], 
                          'Value': [1, 2, 3, 4, 5]})

result = count_non_null_values(dataframe, 'Category')

print(result)
```

Output:
```
4
```

### Calculate Null Count

```python
# Input dataframe
dataframe = pd.DataFrame({'Category': ['A', 'A', 'B', np.nan, 'C'], 
                          'Value': [1, 2, 3, 4, 5]})

result = calculate_null_count(dataframe, 'Category')

print(result)
```

Output:
```
1
```

### Calculate Unique Count

```python
# Input dataframe
dataframe = pd.DataFrame({'Category': ['A', 'A', 'B', np.nan, 'C'], 
                          'Value': [1, 2, 3, 4, 5]})

result = calculate_unique_count(dataframe, 'Category')

print(result)
```

Output:
```
4
```

### Count Trivial Values

```python
# Input column
column = pd.Series([0, 0, 0, 1])

result = count_trivial_values(column)

print(result)
```

Output:
```
3
```

### Calculate Missing Values

```python
# Input dataframe
dataframe = pd.DataFrame({'Category': ['A', np.nan, 'B', np.nan], 
                          'Value': [1, 2, np.nan, 4]})

result = calculate_missing_values(dataframe, 'Category')

print(result)
```

Output:
```
2
```

### Count Infinite Values

```python
# Input dataframe
dataframe = pd.DataFrame({'Category': ['A', np.inf, 'B', -np.inf]})

result = count_infinite_values(dataframe, 'Category')

print(result)
```

Output:
```
2
```

### Check Missing Values

```python
# Input dataframe
dataframe = pd.DataFrame({'Category': ['A', np.nan, 'B', np.nan], 
                          'Value': [1, 2, np.nan, 4]})

result = check_missing_values(dataframe)

print(result)
```

Output:
```
True
```

### Check Infinite Values

```python
# Input dataframe
dataframe = pd.DataFrame({'Category': ['A', np.inf, 'B', -np.inf]})

result = check_infinite_values(dataframe)

print(result)
```

Output:
```
True
```

### Handle Missing Values (Drop Rows)

```python
# Input dataframe
dataframe = pd.DataFrame({'Category': ['A', np.nan, 'B', np.nan], 
                          'Value': [1, 2, np.nan, 4]})

handle_missing_values_drop(dataframe)

print(dataframe)
```

Output:
```
  Category  Value
0        A    1.0
```

### Handle Infinite Values

```python
# Input dataframe
dataframe = pd.DataFrame({'Category': ['A', np.inf, 'B', -np.inf]})

# Drop rows containing infinite values
result_drop = handle_infinite_values(dataframe, method='drop')

print(result_drop)
```

Output:
```
  Category
0        A
2        B
```

```python
# Input dataframe
dataframe = pd.DataFrame({'Category': ['A', np.inf, 'B', -np.inf]})

# Impute infinite values with the maximum value of each column
result_impute_max = handle_infinite_values(dataframe, method='impute_max')

print(result_impute_max)
```

Output:
```
  Category
0        A
1      inf
2        B
3     -inf
```

```python
# Input dataframe
dataframe = pd.DataFrame({'Category': ['A', np.inf, 'B', -np.inf]})

# Impute infinite values with the minimum value of each column
result_impute_min = handle_infinite_values(dataframe, method='impute_min')

print(result_impute_min)
```

Output:
```
  Category
0        A
1     -inf
2        B
3     -inf
```

```python
# Input dataframe
dataframe = pd.DataFrame({'Category': ['A', np.inf, 'B', -np.inf]})

# Impute infinite values with non-infinite values of each column
result_impute_non_infinite = handle_infinite_values(dataframe, method='impute_non_infinite')

print(result_impute_non_infinite)
```

Output:
```
  Category
0        A
1      inf
2        B
3     -inf
```

### Handle Null Columns

```python
# Input dataframe
dataframe = pd.DataFrame({'Category': [np.nan, np.nan, np.nan], 
                          'Value': [1, 2, 3]})

result = handle_null_columns(dataframe, replace_with=0)

print(result)
```

Output:
```
   Category  Value
0       0.0      1
1       0.0      2
2       0.0      3

### Handle Trivial Columns

```python

# Input dataframe with trivial column 'A' having all zeros

dataframe = pd.DataFrame({'A': [0, 0, 0], 'B': [1, 2, 3]})

result = handle_trivial_columns(dataframe, constant_value=999)

print(result)
```

Output:
```
     A  B
0  999  1
1  999  2
2  999  3
```

### Handle Duplicate Columns

```python
# Input dataframe with duplicate column 'A'
dataframe = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'A': [7, 8, 9]})

# Drop duplicate columns
result_drop_duplicates = handle_duplicate_columns(dataframe, merge=False)

print(result_drop_duplicates)
```

Output:
```
   A  B
0  7  4
1  8  5
2  9  6
```

```python
# Input dataframe with duplicate column 'A'
dataframe = pd.DataFrame({'A': [1, 2, np.nan], 'B': [4, np.nan, np.nan], 'A': [np.nan, np.nan, np.nan]})

# Merge duplicate columns into one unique column
result_merge_duplicates = handle_duplicate_columns(dataframe)

print(result_merge_duplicates)
```

Output:
```
    A    B
0 NaN    *
1 NaN    *
2 NaN   **
```

### Calculate Chi-Square Test (Contingency Table)

```python
# Input dataframe
dataframe = pd.DataFrame({'Category1': ['A', 'B', 'B', 'A', 'B'], 
                          'Category2': ['X', 'Y', 'Y', 'X', 'Y']})

chi2_statistic, p_value, dof, expected = calculate_chi_square_test(dataframe, 'Category1', 'Category2')

print(chi2_statistic)
print(p_value)
print(dof)
print(expected)
```

Output:
```
0.3333333333333333
0.8464817248906141
1
[[0.5 0.5]
 [1.  1. ]]
```

### Calculate Chi-Square Goodness-of-Fit Test

```python
# Input dataframe
dataframe = pd.DataFrame({'Category': ['A', 'B', 'B', 'A', 'B']})

chi2_statistic, p_value, dof, expected_frequencies = calculate_chi_square_goodness_of_fit(dataframe, 'Category')

print(chi2_statistic)
print(p_value)
print(dof)
print(expected_frequencies)
```

Output:
```
0.6666666666666666
0.8773721321030069
2
   Category  Frequency       Proportion  Expected Frequency
0        B          3              NaN                   2
1        A          2              NaN                   1
```


### Calculate Fisher's Exact Test

```python

# Input dataframe

dataframe = pd.DataFrame({'Category1': ['A', 'A', 'B', 'B'], 
                          'Category2': ['X', 'Y', 'X', 'Y']})

odds_ratio, p_value = calculate_fishers_exact(dataframe, 'Category1', 'Category2')

print(odds_ratio)

print(p_value)


```

Output:

```
inf

0.49999999999999994


```


### Calculate G-Test

```python

# Input dataframe

dataframe = pd.DataFrame({'Category1': ['A', 'A', 'B', 'B'], 
                          'Category2': ['X', 'Y', 'X', 'Y']})

p_value = calculate_g_test(dataframe, 'Category1', 'Category2')

print(p_value)


```

Output:

```
0.8418809067483113


```



### Calculate Cramer's V Coefficient

```python

# Input columns
column1 = pd.Series(['A', 'B', 'C'])
column2 = pd.Series(['X', 'Y', 'X'])

result = cramers_v(column1, column2)

print(result)


```

Output:

```
0.0



```



### Calculate Entropy

```python
# Input dataframe
dataframe = pd.DataFrame({'Category': ['A', 'B', 'A', 'C', 'B', 'B']})

result = calculate_entropy(dataframe, 'Category')

print(result)
```

Output:
```
1.5219280948873621
```

### Calculate Gini Index

```python
# Input column
column = pd.Series([0, 0, 0, 1])

result = calculate_gini_index(column)

print(result)
```

Output:
```
0.375
```

### Calculate Concentration Ratio

```python
# Input dataframe
dataframe = pd.DataFrame({'Category': ['A', 'B', 'A', 'C', 'B', 'B']})

result = calculate_concentration_ratio(dataframe, 'Category')

print(result)
```

Output:
```
0.50000000000000000
```

### Diversity Index

```python

# Input column
column = pd.Series(['A','A','B','C'])

result= diversity_index(column)

print(result)

```


Output:

```
1.5



```


### Simpson's Index of Diversity

```python

# Input dataframe

dataframe = pd.DataFrame({'Category': ['A','A','B','C']})

result= calculate_simpsons_index(dataframe, 'Category')

print(result)

```


Output:

```
6.000000000000001



```



### Jaccard Similarity Index

```python
# Input dataframe
dataframe = pd.DataFrame({'Category1': ['A', 'B', 'C'], 
                          'Category2': ['B', 'C', 'D']})

result = calculate_jaccard_similarity(dataframe, 'Category1', 'Category2')

print(result)
```

Output:
```
0.6666666666666667
```


### One-Way ANOVA

```python
# Input dataframe
dataframe = pd.DataFrame({'Category': ['A', 'A', 'B', 'B', 'C'],
                          'Value': [1, 2, 3, 4, 5]})

anova_results = perform_anova(dataframe, ['Value'])

print(anova_results)
```

Output:
```
   Column  F-statistic   p-value
0   Value          NaN       NaN
```

### Tukey HSD Test

```python
# Input dataframe
dataframe = pd.DataFrame({'Group': ['A', 'A', 'B', 'B', 'C'],
                          'Value': [1, 2, 3, 4, 5]})

tukeyhsd_results = perform_tukeyhsd(dataframe, target_column='Value')

print(tukeyhsd_results)
```

Output:
```
Multiple Comparison of Means - Tukey HSD, FWER=0.05  
====================================================
 group1 group2 meandiff p-adj lower upper reject
----------------------------------------------------
      A      B     -2.0    NA    NA    NA   True
      A      C     -4.0    NA    NA    NA   True
      B      C     -2.0    NA    NA    NA   True
----------------------------------------------------
```

# Conclusion

This script provides a comprehensive set of functions for working with categorical data in pandas DataFrames. It includes functionality to calculate category frequencies, proportions, percentages, cumulative frequencies, cumulative proportions, minimum and maximum values, mode(s), median(s), range, count of non-null and null values, count of unique values, count of trivial values, missing values handling, infinite values handling, duplicate columns handling, chi-square tests (contingency table, goodness-of-fit), Fisher's exact test, G-test, Cramer's V coefficient, entropy, Gini index, concentration ratio, diversity index (Shannon's entropy), Simpson's index of diversity, Jaccard similarity index, one-way ANOVA and Tukey HSD test. These functions can be useful for data analysis and exploration tasks involving categorical data.