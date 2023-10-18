andas as pd
import numpy as np
from scipy.stats import chi2_contingency

def calculate_category_frequency(dataframe, column_name):
    """
    Calculate the frequency of each category in a column of a pandas dataframe.

    Parameters:
    dataframe (pandas.DataFrame): The input dataframe.
    column_name (str): The name of the column to calculate frequencies.

    Returns:
    pandas.DataFrame: A dataframe with two columns - 'Category' and 'Frequency'.
                      'Category' contains the unique categories in the column,
                      and 'Frequency' contains the count of each category.
    """
    
    # Calculate the frequency of each unique category
    category_counts = dataframe[column_name].value_counts().reset_index()

    # Rename the columns
    category_counts.columns = ['Category', 'Frequency']

    return category_counts



def calculate_category_proportions(dataframe, column_name):
    """
    Calculate the proportion of each category in a column of a pandas dataframe.

    Parameters:
        dataframe (pandas.DataFrame): The input dataframe.
        column_name (str): The name of the target column.

    Returns:
        pandas.DataFrame: A dataframe with two columns: 'Category' and 'Proportion'.
                          'Category' contains the unique categories in the specified column,
                          and 'Proportion' contains the calculated proportions.

    """

    # Check if the column exists in the dataframe
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the dataframe.")

    # Calculate the frequency count of each category
    category_counts = dataframe[column_name].value_counts(dropna=False)

    # Calculate the proportion of each category
    category_proportions = category_counts / len(dataframe)

    # Create a new dataframe to store the results
    result_dataframe = pd.DataFrame({'Category': category_proportions.index, 'Proportion': category_proportions.values})

    return result_dataframe



def calculate_category_percentage(column):
    """
    Function to calculate the percentage of each category in a column.

    Parameters:
    column (pandas.Series): The column for which to calculate the category percentages.

    Returns:
    pandas.DataFrame: A dataframe with two columns - 'category' and 'percentage'.

    """

    # Calculate the total count of values in the column
    total_count = len(column)

    # Calculate the count of each category in the column
    category_counts = column.value_counts()

    # Calculate the percentage of each category
    category_percentages = (category_counts / total_count) * 100

    # Create a dataframe with 'category' and 'percentage' columns
    result_df = pd.DataFrame({'category': category_percentages.index, 'percentage': category_percentages.values})

    return result_df



def calculate_cumulative_frequency(df, column_name):
    # Drop any rows with missing or null values in the specified column
    df = df.dropna(subset=[column_name])
    
    # Calculate the frequency count of each category in the column
    frequency_counts = df[column_name].value_counts()
    
    # Sort the frequency counts by category name
    sorted_counts = frequency_counts.sort_index()
    
    # Calculate the cumulative sum of the frequency counts
    cumulative_frequencies = sorted_counts.cumsum()
    
    return cumulative_frequencies



def calculate_cumulative_proportion(data, column_name):
    """
    Calculate the cumulative proportion of each category in a column of a pandas DataFrame.

    Parameters:
    data (pandas DataFrame): The input DataFrame.
    column_name (str): The name of the column to calculate the cumulative proportion for.

    Returns:
    pandas DataFrame: A DataFrame containing the cumulative proportion for each category in the specified column.
   """
   
   # Count the occurrences of each category in the specified column
    category_counts = data[column_name].value_counts()

    # Calculate the cumulative sum of the counts
    cumulative_counts = category_counts.cumsum()

    # Calculate the total count
    total_count = len(data)

    # Calculate the cumulative proportion for each category
    cumulative_proportions = cumulative_counts / total_count

    # Create a new DataFrame with the category names and their corresponding cumulative proportions
    result = pd.DataFrame({
        'Category': cumulative_proportions.index,
        'Cumulative Proportion': cumulative_proportions.values
    })

    return result



def calculate_cumulative_percentage(df, column_name):
    # Calculate the count of each category in the column
    category_counts = df[column_name].value_counts()
    
    # Calculate the cumulative sum of the category counts
    cumulative_counts = category_counts.cumsum()
    
    # Calculate the total count of all categories
    total_count = category_counts.sum()
    
    # Calculate the cumulative percentage of each category
    cumulative_percentage = cumulative_counts / total_count * 100
    
    return cumulative_percentage


andas as pd

def calculate_mode(column):
    """
    Function to calculate the mode(s) of a column.

    Parameters:
    column (pandas.Series): The input column to calculate the mode(s) for.

    Returns:
    list: A list of mode(s) of the column.
   """
   
   # Drop missing values from the column
   column = column.dropna()

   # Calculate the mode(s) using the value_counts() function
   modes = column.value_counts().index.tolist()

   return modes


def calculate_median(df, column_name):
    """
     Calculate the median(s) of a column in a pandas dataframe.
    
     Parameters:
         - df: Input pandas dataframe.
         - column_name: Name of the column for which to calculate the median(s).
        
     Returns:
         - List of medians for each unique category in the specified column.
    """
    medians = []
    
    # Group the dataframe by the specified column
    grouped_df = df.groupby(column_name)
    
    # Calculate median for each group
    for group_name, group_data in grouped_df:
        median = group_data[column_name].median()
        medians.append(median)
    
    return medians



def calculate_range(df, column_name):
    """
    Calculate the range of values in a column of a pandas dataframe.
    
    Parameters:
        - df: pandas dataframe
        - column_name: str, name of the column in the dataframe
    
    Returns:
        - range_values: tuple (min_value, max_value)
   """
   
   min_value = df[column_name].min()
   max_value = df[column_name].max()
    
   range_values = (min_value, max_value)
   return range_values



def calculate_minimum(dataframe, column_name):
    """
     Calculates the minimum value in a column of a pandas dataframe.

     Parameters:
         - dataframe: pandas dataframe
         - column_name: str, name of the column in the dataframe

     Returns:
         - minimum_value: float or int, the minimum value in the column
     """
     
     minimum_value = dataframe[column_name].min()
     return minimum_value



def calculate_max_value(df, column_name):
    """
    Function to calculate the maximum value in a column of a pandas dataframe.
    
    Parameters:
        - df (pd.DataFrame): Input pandas dataframe.
        - column_name (str): Name of the column to calculate the maximum value from.
    
    Returns:
        max_value: Maximum value in the specified column.
   """
   
   max_value = df[column_name].max()
   
   return max_value



def count_non_null_values(df, column_name):
    """
    Calculate the count of non-null values in a column of a pandas dataframe.

    Parameters:
        - df (pandas.DataFrame): The input dataframe.
        - column_name (str): The name of the column to calculate the count for.

    Returns:
        - int: The count of non-null values in the specified column.
   """
   
   return df[column_name].count()



def calculate_null_count(df, column_name):
    """
    Calculates the count of null values in a column of a pandas dataframe.

    Parameters:
        - df (pandas.DataFrame): The input dataframe.
        - column_name (str): The name of the column to calculate the null count for.

    Returns:
        int: The count of null values in the specified column.
   """
   
  return df[column_name].isnull().sum()



def calculate_unique_count(dataframe, column_name):
    """
     Calculate the count of unique values in a column of a pandas dataframe.
    
     Parameters:
         - dataframe: The input pandas dataframe.
         - column_name: The name of the column to calculate the count for.
    
     Returns:
         The count of unique values in the specified column.
     """
     
     return dataframe[column_name].nunique()



def count_trivial_values(column):
    """
     Calculates the count of trivial values (e.g., all zeros or all ones) in a column.
    
     Parameters:
         column (pandas.Series): The column to calculate the count of trivial values for.
    
     Returns:
         int: The count of trivial values in the column.
     """
     
     # Check if all values in the column are either 0 or 1
     if set(column.unique()) == {0, 1}:
         return column.value_counts().min()
     else:
         return 0



def calculate_missing_values(df, column):
    return df[column].isnull().sum()



def count_infinite_values(df, column_name):
    """
     Calculates the count of infinite values in a column of a pandas dataframe.
    
     Parameters:
         - df: pandas.DataFrame
             The input dataframe.
         - column_name: str
             The name of the column to calculate the count of infinite values from.
    
     Returns:
         - int: The count of infinite values in the specified column.
    """
    
    column = df[column_name]
    count = np.isinf(column).sum()
    return count



def check_missing_values(df):
    """
     Function to check if any missing values exist in a dataframe.

     Parameters:
         df (pandas.DataFrame): The input dataframe.

     Returns:
         bool: True if missing values exist, False otherwise.
     """
     
     return df.isnull().values.any()



def check_infinite_values(df):
    """
     Function to check if any infinite values exist in a dataframe.
     
     Parameters:
         df: pandas dataframe
     
     Returns:
         True if any infinite values exist, False otherwise.
     """
     
     return np.any(np.isinf(df.values))



def handle_missing_values_drop(df):
    # Drop rows with any missing values
    df.dropna(inplace=True)

# Usage:
handle_missing_values_drop(df)



def handle_infinite_values(dataframe, method='drop'):
    """
    Function to handle infinite values in a pandas dataframe.
    
    Parameters:
        - dataframe: pandas DataFrame
            Input dataframe with categorical data.
        - method: str, optional (default='drop')
            Method to handle infinite values. Possible options are:
            - 'drop': Drop rows containing infinite values.
            - 'impute_max': Impute infinite values with the maximum value of the column.
            - 'impute_min': Impute infinite values with the minimum value of the column.
            - 'impute_non_infinite': Impute infinite values with non-infinite values of the column.
    
    Returns:
        Pandas DataFrame with infinite values handled according to the specified method.
    """
    
    if method == 'drop':
        # Drop rows containing infinite values
        dataframe = dataframe.replace([np.inf, -np.inf], np.nan)
        dataframe = dataframe.dropna()
        
    elif method == 'impute_max':
        # Impute infinite values with maximum value of each column
        dataframe = dataframe.replace([np.inf, -np.inf], np.nan)
        max_values = dataframe.max()
        dataframe = dataframe.fillna(max_values)
        
    elif method == 'impute_min':
        # Impute infinite values with minimum value of each column
        dataframe = dataframe.replace([np.inf, -np.inf], np.nan)
        min_values = dataframe.min()
        dataframe = dataframe.fillna(min_values)
        
    elif method == 'impute_non_infinite':
        # Impute infinite values with non-infinite values of each column
        dataframe = dataframe.replace([np.inf, -np.inf], np.nan)
        non_infinite_values = dataframe[~dataframe.isin([np.nan, np.inf, -np.inf])].dropna()
        dataframe = dataframe.fillna(non_infinite_values)
        
    return dataframe



def handle_null_columns(df, replace_with):
    """
     Function to handle null columns by replacing them with constant values.

     Parameters:
     - df: pandas DataFrame - Input DataFrame
     - replace_with: any - The value to replace null values with

     Returns:
     - pandas DataFrame - DataFrame with null columns replaced
     """

     # Replace null values with the specified value
     df_filled = df.fillna(replace_with)

     return df_filled



def handle_trivial_columns(dataframe, constant_value):
    """
    Handle trivial columns by replacing them with a constant value.

    Parameters:
    dataframe (pandas.DataFrame): Input dataframe.
    constant_value: Value to replace trivial columns with.

    Returns:
    pandas.DataFrame: DataFrame with trivial columns replaced.
    """
    
    return dataframe.fillna(constant_value)



def handle_duplicate_columns(df, merge=True):
    """
     Function to handle duplicate columns by either dropping or merging into one unique column.

     Parameters:
         - df: pandas DataFrame
             Input dataframe with duplicate columns.
         - merge: bool, default=True
             Flag to indicate whether to merge duplicate columns into one or drop them.

     Returns:
         - df: pandas DataFrame
             Updated dataframe with no duplicate columns.
     """
     
     if merge:
         # Merge duplicate columns into one
         df = df.groupby(level=0, axis=1).first()
     else:
         # Drop duplicate columns
         df = df.loc[:, ~df.columns.duplicated()]
     
     return df



def calculate_chi_square_test(dataframe, column1, column2):
    # Select the two columns from the dataframe
    df = dataframe[[column1, column2]]

    # Calculate the contingency table
    contingency_table = pd.crosstab(df[column1], df[column2])

    # Perform the chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    return chi2, p_value, dof, expected



def calculate_chi_square_goodness_of_fit(df, column):
    """
     Calculate the chi-square goodness-of-fit test for a single categorical column in a pandas dataframe.

     Parameters:
         - df (pandas.DataFrame): Input dataframe.
         - column (str): Name of the categorical column to perform the test on.

     Returns:
         tuple: A tuple containing the chi-square statistic, p-value, degrees of freedom, and expected frequencies.
     """
     
     # Filter out missing and infinite data
     filtered_df = df[column].replace([np.inf, -np.inf], np.nan).dropna()

     # Get observed frequencies
     observed_frequencies = filtered_df.value_counts()

     # Calculate expected frequencies using uniform distribution assumption
     expected_frequencies = pd.Series(1 / len(observed_frequencies), index=observed_frequencies.index) * len(filtered_df)

     # Perform chi-square test
     chi2_statistic, p_value = chi2_contingency(pd.concat([observed_frequencies, expected_frequencies], axis=1))[:2]

     # Calculate degrees of freedom
     degrees_of_freedom = len(observed_frequencies) - 1

     return chi2_statistic, p_value, degrees_of_freedom, expected_frequencies



def calculate_fishers_exact(df, column1, column2):
    contingency_table = pd.crosstab(df[column1], df[column2])
    odds_ratio, p_value = fisher_exact(contingency_table)
    return odds_ratio, p_value



def calculate_g_test(dataframe, column1, column2):
    contingency_table = pd.crosstab(dataframe[column1], dataframe[column2])
    _, p_value, _, _ = chi2_contingency(contingency_table)
    return p_value


umpy as np
import pandas as pd
from scipy.stats import chi2_contingency

def cramers_v(column1, column2):
    contingency_table = pd.crosstab(column1, column2)
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = len(column1)
    r, k = contingency_table.shape
    v = np.sqrt(chi2 / (n * min(r-1, k-1)))
    return v



import pandas as pd

def calculate_contingency_table(df, col1, col2):
    contingency_table = pd.crosstab(df[col1], df[col2])
    return contingency_table

# Example usage:
# df is the pandas dataframe
# col1 and col2 are the names of the two categorical columns
contingency_table = calculate_contingency_table(df, 'column1', 'column2')
print(contingency_table


import pandas as pd
import matplotlib.pyplot as plt

def visualize_category_distribution(df, column):
    """
     Visualize the distribution of categories in a column using a bar plot.
    
     Parameters:
         df (pandas.DataFrame): The input dataframe.
         column (str): The name of the column to analyze.
     """
     
     df[column].value_counts().plot(kind='bar')
     plt.xlabel(column)
     plt.ylabel('Count')
     plt.title('Category Distribution')
     plt.show()


andas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic

def visualize_mosaic_plot(df, x, y):
    """
    Visualizes the relationship between two categorical variables using a mosaic plot.

    Parameters:
        df (pandas.DataFrame): The input dataframe containing the categorical variables.
        x (str): The name of the column representing the first categorical variable.
        y (str): The name of the column representing the second categorical variable.

    Returns:
        None
    """
    # Create a cross-tabulation of the two variables
    crosstab = pd.crosstab(df[x], df[y])

    # Calculate the proportions for each cell in the mosaic plot
    proportions = crosstab / crosstab.values.sum()

    # Create a mosaic plot
    mosaic(proportions.stack(), ax=plt.gca())

    # Set labels for x-axis and y-axis
    plt.xlabel(x)
    plt.ylabel(y)

    # Show the plot
    plt.show()


import pandas as pd

def calculate_entropy(dataframe, column_name):
    # Get the values in the specified column
    column_values = dataframe[column_name]
    
    # Count the frequency of each unique value in the column
    value_counts = column_values.value_counts()
    
    # Calculate the total number of samples
    total_samples = len(column_values)
    
    # Calculate the probabilities of each unique value
    probabilities = value_counts / total_samples
    
    # Calculate the entropy using the formula: -sum(p * log2(p))
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return entropy



import pandas as pd

def calculate_gini_index(column):
    # Calculate the total count of values in the column
    total_count = column.count()

    # Calculate the frequency of each unique value in the column
    value_counts = column.value_counts()

    # Calculate the probability of each unique value
    probabilities = value_counts / total_count

    # Calculate the Gini index
    gini_index = 1 - sum(probabilities ** 2)

    return gini_index



import pandas as pd

def calculate_concentration_ratio(dataframe, column_name):
     # Check if the column exists in the dataframe
     if column_name not in dataframe.columns:
         raise ValueError(f"Column '{column_name}' does not exist in the dataframe.")
     
     # Drop rows with missing or infinite values
     clean_dataframe = dataframe.dropna(subset=[column_name]).replace([np.inf, -np.inf], np.nan)
     
     # Calculate total count of non-null values in the column
     total_count = clean_dataframe[column_name].count()
     
     # Calculate counts for each unique value in the column
     value_counts = clean_dataframe[column_name].value_counts()
     
     # Calculate concentration ratio
     concentration_ratio = value_counts.max() / total_count
     
     return concentration_ratio


andas as pd
import numpy as np

def diversity_index(column):
    """
    Calculate the diversity index of a categorical column.
    
    Parameters:
    - column: pandas Series representing the categorical column
    
    Returns:
    - diversity index value
    """
    
    # Count the frequency of each category in the column
    frequencies = column.value_counts()

    # Calculate the total number of categories
    total_categories = len(frequencies)

    # Calculate the probability of each category
    probabilities = frequencies / len(column)

    # Calculate the diversity index using Shannon's entropy formula
    diversity_index = -np.sum(probabilities * np.log2(probabilities))

    return diversity_inde


import pandas as pd

def calculate_simpsons_index(df, column_name):
     # Calculate the frequency counts of each category in the column
     counts = df[column_name].value_counts()
     
     # Calculate the total number of observations
     total_count = counts.sum()
     
     # Calculate the probabilities of each category
     probabilities = counts / total_count
     
     # Calculate the Simpson's Index of Diversity
     simpsons_index = 1 / sum(probabilities ** 2)
     
     return simpsons_index



import pandas as pd
from scipy.spatial.distance import jaccard

def calculate_jaccard_similarity(df, column1, column2):
    """
    Calculate the Jaccard index of similarity between two categorical columns in a pandas DataFrame.

    Parameters:
        - df (pandas.DataFrame): Input DataFrame containing the two categorical columns.
        - column1 (str): Name of the first categorical column.
        - column2 (str): Name of the second categorical column.

    Returns:
        float: The Jaccard index of similarity between the two categorical columns.
    """
    # Extract the two columns from the DataFrame
    series1 = df[column1]
    series2 = df[column2]

    # Convert the series into sets for calculating Jaccard index
    set1 = set(series1)
    set2 = set(series2)

    # Calculate the Jaccard index of similarity
    jaccard_index = 1 - jaccard(set1, set2)

    return jaccard_index


andas as pd
import scipy.stats as stats

def perform_anova(dataframe, columns):
    """
     Perform one-way analysis of variance (ANOVA) for multiple categorical columns in a pandas dataframe.
    
     Parameters:
         - dataframe: pandas DataFrame containing the categorical data
         - columns: list of column names to perform ANOVA on
        
     Returns:
         - anova_results: pandas DataFrame containing the ANOVA results for each column
     """
     
     anova_results = pd.DataFrame(columns=['Column', 'F-statistic', 'p-value'])
     
     for column in columns:
         # Drop rows with missing or infinite values
         cleaned_data = dataframe.dropna(subset=[column]).replace([np.inf, -np.inf], np.nan)
         
         # Perform ANOVA test
         f_statistic, p_value = stats.f_oneway(*[cleaned_data[column][cleaned_data[column] == category] 
                                                 for category in cleaned_data[column].unique()])
         
         anova_results = anova_results.append({'Column': column, 'F-statistic': f_statistic, 'p-value': p_value}, 
                                              ignore_index=True)
     
     return anova_result


import statsmodels.stats.multicomp as mc

def perform_tukeyhsd(dataframe, target_column):
    # Extract the relevant data for analysis
    groups = dataframe[target_column]
    data = dataframe.drop(target_column, axis=1)

    # Perform ANOVA to obtain F-statistic and p-value
    fvalue, pvalue = stats.f_oneway(*data.values.T)

    # Perform Tukey HSD test with Bonferroni correction
    mc_results = mc.pairwise_tukeyhsd(data.values.ravel(), groups.values, alpha=0.05)
    
    return mc_results.summary(