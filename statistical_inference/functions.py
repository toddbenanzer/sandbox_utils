umpy as np
from scipy.stats import ttest_ind
from scipy.stats import f_oneway
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact
from scipy.stats import pearsonr
from scipy.stats import cov
from statistics import mode
from scipy.stats import mannwhitneyu
from scipy.stats import chi2
from scipy.stats import wilcoxon

def t_test(array1, array2):
    if np.var(array1) == 0 or np.var(array2) == 0:
        raise ValueError("Arrays have zero variance")
    if np.unique(array1).shape[0] == 1 or np.unique(array2).shape[0] == 1:
        raise ValueError("Arrays have constant values")

    array1 = array1[np.logical_and(~np.isnan(array1), array1 != 0)]
    array2 = array2[np.logical_and(~np.isnan(array2), array2 != 0)]

    t_statistic, p_value = ttest_ind(array1, array2)

    return {
        "t_statistic": t_statistic,
        "p_value": p_value,
        "array1_mean": np.mean(array1),
        "array2_mean": np.mean(array2),
        "array1_std": np.std(array1),
        "array2_std": np.std(array2)
    }

def perform_anova_test(*args):
    anova_result = f_oneway(*args)
    return anova_result

def perform_chi_squared_test(data1, data2):
    observed_freq = np.histogram2d(data1, data2)[0]
    
    chi2, p_value, _, _ = chi2_contingency(observed_freq)
    
    return chi2, p_value

def fishers_exact_test(array1, array2):
    contingency_table = np.array([array1, array2])
    odds_ratio, p_value = fisher_exact(contingency_table)
    return odds_ratio, p_value

def calculate_correlation_coefficient(x, y):
    if len(x) != len(y):
        raise ValueError("Input arrays must have the same length")

    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    if len(x) == 0 or len(y) == 0:
        raise ValueError("Arrays must have non-missing values")

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    covariance = np.sum((x - x_mean) * (y - y_mean))

    x_std = np.std(x)
    y_std = np.std(y)

    correlation_coefficient = covariance / (x_std * y_std)

    return correlation_coefficient

def calculate_covariance(array1, array2):
    return cov(array1, array2)[0][1]

def calculate_mean(data):
    return np.mean(data)

def calculate_median(data):
    median = np.median(data)
    return median

def calculate_mode(data):
    data_list = data.tolist()
    
    mode_value = mode(data_list)
    
    return mode_value

def calculate_std_dev(numeric_array):
    return np.std(numeric_array)

def calculate_variance(data):
    return np.var(data)

def check_zero_variance(data):
    return np.var(data) == 0

def check_constant_values(arr):
   unique_values = np.unique(arr)
   return len(unique_values) == 1

def handle_missing_values(array, method='impute'):
    
   if method == 'impute':
        mean = np.nanmean(array)
        array[np.isnan(array)] = mean
   elif method == 'remove':
        array = array[~np.isnan(array).any(axis=1)]
    
   return array

def handle_zeroes(array, method='remove'):
    if method == 'remove':
        return array[np.all(array != 0, axis=1)]
    elif method == 'impute':
        non_zero_values = array[array != 0]
        mean = np.mean(non_zero_values)
        return np.where(array == 0, mean, array)
    else:
        raise ValueError("Invalid method. Please choose either 'remove' or 'impute'.")

def calculate_descriptive_stats(arr):
    
    if np.var(arr) == 0:
        return 'Zero variance'
    
    if len(np.unique(arr)) == 1:
        return 'Constant values'
    
    mean = np.mean(arr)
    median = np.median(arr)
    std_dev = np.std(arr)
    min_val = np.min(arr)
    max_val = np.max(arr)
    
    return mean, median, std_dev, min_val, max_val

def calculate_categorical_stats(data1, data2):
    
    unique_categories = np.unique(np.concatenate((data1, data2)))
    
    counts_data1 = np.array([np.count_nonzero(data1 == category) for category in unique_categories])
    counts_data2 = np.array([np.count_nonzero(data2 == category) for category in unique_categories])
    
    total_counts = counts_data1 + counts_data2
    
    proportions_data1 = counts_data1 / total_counts
    proportions_data2 = counts_data2 / total_counts
    
    return unique_categories, counts_data1, proportions_data1, counts_data2, proportions_data

def descriptive_stats_boolean(data):
    
    count = data.size
    true_count = np.count_nonzero(data)
    false_count = count - true_count
    true_percentage = (true_count / count) * 100
    false_percentage = (false_count / count) * 100

    return {
        "count": count,
        "true_count": true_count,
        "false_count": false_count,
        "true_percentage": true_percentage,
        "false_percentage": false_percentage
    }

def mann_whitney_test(data1, data2):
    
    if len(data1) != len(data2):
        raise ValueError("Input arrays must have the same length.")

    statistic, p_value = mannwhitneyu(data1, data2)

    n1 = len(data1)
    n2 = len(data2)
    u = statistic
    r1 = n1 * n2 + (n1 * (n1 + 1)) / 2 - u
    r2 = n1 * n2 - r1
    mean_rank_1 = r1 / n1
    mean_rank_2 = r2 / n2

    results = {
        "statistic": statistic,
        "p-value": p_value,
        "mean rank 1": mean_rank_1,
        "mean rank 2": mean_rank_2,
        "size of group 1": n1,
        "size of group 2": n2,
    }

    return results

def chi_square_test(array1, array2):
    
    contingency_table = np.array([array1, array2])
    
    statistic, p_value, _, _ = chi2_contingency(contingency_table)
    
    return statistic, p_value


def mcnemar_test(array1, array2):
    
    if len(array1) != len(array2):
        raise ValueError("Input arrays must be of the same length.")
    
    if not np.issubdtype(array1.dtype, np.bool) or not np.issubdtype(array2.dtype, np.bool):
        raise ValueError("Input arrays must be boolean arrays.")
    
   
    table = [[0, 0], [0, 0]]
    
    for i in range(len(array1)):
        if array1[i] and array2[i]:
            table[0][0] += 1
        elif array1[i] and not array2[i]:
            table[0][1] += 1
        elif not array1[i] and array2[i]:
            table[1][0] += 1
        else:
            table[1][1] += 1
    
    n = table[0][1] + table[1][0]
    statistic = (abs(table[0][1] - table[1][0]) - 1) ** 2 / float(n)
    p_value = chi2.sf(statistic, df=1)
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'df': 1
    }


def wilcoxon_signed_rank_test(data1, data2, zero_method='wilcox', correction=False):
    
    statistic, p_value = wilcoxon(data1, data2, zero_method=zero_method, correction=correction)
    
    return statistic, p_valu