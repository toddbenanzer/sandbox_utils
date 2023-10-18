
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def calculate_phi_coefficient(dataframe, variable1, variable2):
    contingency_table = pd.crosstab(dataframe[variable1], dataframe[variable2])
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    phi_coefficient = np.sqrt(chi2 / n)
    return phi_coefficient

def calculate_chi_square(dataframe, variable1, variable2):
    contingency_table = pd.crosstab(dataframe[variable1], dataframe[variable2])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    return chi2, p_value

def calculate_cramers_v(variable1, variable2):
    confusion_matrix = pd.crosstab(variable1, variable2)
    chi2, _, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    v_cramer = np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1)))
    return v_cramer

def perform_one_way_anova(dataframe, categorical_column, continuous_column):
    groups = dataframe.groupby(categorical_column)[continuous_column].apply(list).values
    f_statistic, p_value = stats.f_oneway(*groups)
    return f_statistic, p_value

def perform_factor_analysis(dataframe):
    result = dataframe.isnull().sum().rename('Count')
    result['Percentage'] = result / len(dataframe)
    return result

def handle_missing_data(dataframe):
    return dataframe.dropna()

def generate_bar_plot(dataframe, column_name):
    dataframe[column_name].value_counts().plot(kind='bar')

def generate_count_plot(dataframe, column_name):
    sns.countplot(x=column_name, data=dataframe)

def generate_pie_chart(dataframe, column_name):
    dataframe[column_name].value_counts().plot.pie()

def perform_tukey_hsd(dataframe, continuous_column, categorical_column):
    tukey_results = mc.MultiComparison(dataframe[continuous_column], dataframe[categorical_column]).tukeyhsd()
    return str(tukey_results)

def calculate_point_biserial_correlation(binary_column, categorical_column):
    point_biserial_correlation, p_value = stats.pointbiserialr(binary_column, pd.factorize(categorical_column)[0])
    return point_biserial_correlation, p_value

def calculate_gini_index(dataframe, column_name):
    value_counts = dataframe[column_name].value_counts()
    proportions = value_counts / len(dataframe)
    gini_index = 1 - (proportions ** 2).sum()
    return gini_index

def remove_trivial_columns(dataframe):
    return dataframe.loc[:, (dataframe.nunique() > 1)]

def calculate_missing_percentage(column):
    return column.isnull().mean() * 100

def calculate_non_null_count(column):
    return column.notnull().sum()

def count_null_values(column):
    return column.isnull().sum()

def calculate_empty_prevalence(column):
    return column.isna().mean()

def calculate_missing_values(column):
    return column.isna().mean()

def perform_pearson_correlation(dataframe, variable1, variable2):
    pearson_correlation_coefficient, p_value = stats.pearsonr(dataframe[variable1], dataframe[variable2])
    return pearson_correlation_coefficient, p_value

def calculate_entropy(dataframe, column_name):
    value_counts = dataframe[column_name].value_counts(normalize=True)
    entropy = -np.sum(value_counts * np.log2(value_counts))
    return entropy
