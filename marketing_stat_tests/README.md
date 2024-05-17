## Overview

This python script provides a collection of statistical calculations and tests commonly used in data analysis. It includes functions for calculating mean, standard deviation, t-statistic, degrees of freedom, confidence interval, effect size, and Cohen's d. Additionally, it provides various t-tests (paired t-test, one-sample t-test, two-sample t-test, independent t-test with equal variances assumption, independent t-test with unequal variances assumption) for comparing means between groups. Furthermore, it includes functions for performing ANOVA (analysis of variance) to compare means across multiple groups and conducting post-hoc tests using the Tukey method. It also offers chi-square tests for goodness-of-fit and expected frequencies calculation. Lastly, it includes functions for visualizing distributions, conducting correlation analysis, performing logistic regression analysis, and regression analysis.

## Usage

To utilize this script, you will need to have the following packages installed:

- numpy
- scipy
- matplotlib
- sklearn
- pandas
- seaborn
- statsmodels

To use a particular function or test from this script, import the necessary package(s) and call the desired function(s) with the appropriate arguments.

## Examples

Here are some examples demonstrating how to use the functions provided in this script:

1. Calculating mean:

```python
import numpy as np

group = [1, 2, 3, 4, 5]
mean = np.mean(group)
print(mean)
```

2. Calculating standard deviation:

```python
import numpy as np

group = [1, 2, 3, 4, 5]
std_dev = np.std(group)
print(std_dev)
```

3. Conducting a paired t-test:

```python
from scipy.stats import ttest_rel

before = [1, 2, 3]
after = [4, 5, 6]
t_statistic, p_value = ttest_rel(before, after)
print(t_statistic, p_value)
```

4. Performing ANOVA:

```python
import pandas as pd
import statsmodels.api as sm

data = pd.DataFrame({
    'group1': [1, 2, 3],
    'group2': [4, 5, 6],
    'group3': [7, 8, 9]
})

f_stats, p_value = sm.stats.anova.anova_lm(data)
print(f_stats, p_value)
```

5. Conducting logistic regression analysis:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

df = pd.DataFrame({
    'feature1': [1, 2, 3],
    'feature2': [4, 5, 6],
    'target': [0, 1, 0]
})

X = df[['feature1', 'feature2']]
y = df['target']

model = LogisticRegression()
model.fit(X, y)

scores = cross_val_score(model, X, y)
print(scores)
```

These are just a few examples to demonstrate the usage of the functions provided in this script. You can explore the various functions and tests available and use them according to your specific analysis requirements.