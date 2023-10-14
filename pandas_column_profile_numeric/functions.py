andas as pd
import numpy as np
from scipy.stats import norm, lognorm, expon, gamma


def calculate_mean(column):
    """
    Calculate the mean of the input column.
    
    Parameters:
        column (pandas.Series): Input column
        
    Returns:
        float: Mean of the column
    """
    if column.isnull().all() or column.empty:
        return None
    
    mean = column.mean()
    
    return mean


def calculate_median(column):
    """
    Calculate the median of the input column.
    
    Parameters:
        column (pandas.Series): Numeric column for which median needs to be calculated
    
    Returns:
        float: Median of the input column
    """
    return column.median()


def calculate_mode(column):
    """
    Calculate the mode of the input column.
    
    Parameters:
        column (pandas.Series): Input column
        
    Returns:
        pandas.Series: Mode(s) of the input column
    """
    mode = column.mode()
    
    return mode


def calculate_standard_deviation(column):
    """
    Calculate the standard deviation of the input column.
    
    Parameters:
        column (pandas.Series): Input column
        
    Returns:
        float: Standard deviation of the column
    """
    return np.std(column)


def handle_missing_data(column):
    """
    Handle missing data in the input column by replacing them with the median value.
    
    Parameters:
        column (pandas.Series): Input column
        
     Returns:
         pandas.Series: Column with missing values replaced with median
     """
     if column.isnull().any():
         # Replace missing values with the median of the column
         column = column.fillna(column.median())
     
     return column


def handle_infinite_data(column):
     """
     Handle infinite data in the input column by replacing infinite values with NaN.
     
     Parameters:
         column (pandas.Series): Input column
         
     Returns:
         pandas.Series: Column with infinite values replaced with NaN
     """
     column.replace([np.inf, -np.inf], np.nan, inplace=True)
     
     # Count the number of infinite values
     num_infinite = column.isin([np.inf, -np.inf]).sum()
     
     return column, num_infinite


def check_null_columns(df):
    """
    Check for null columns in the dataframe.
    
    Parameters:
        df (pandas.DataFrame): The dataframe to check.
        
    Returns:
        list: A list of column names that contain null values.
    """
    null_columns = df.columns[df.isnull().any()].tolist()
    
    return null_columns


def check_trivial_columns(dataframe):
    """
    Check for trivial columns (columns with only one unique value) in the dataframe.
    
    Parameters:
        dataframe (pandas.DataFrame): The dataframe to check.
        
    Returns:
        list: A list of column names that contain trivial values.
    """
    trivial_columns = []
    
    for column in dataframe.columns:
        unique_values = dataframe[column].nunique()
        
        if unique_values == 1:
            trivial_columns.append(column)
    
    return trivial_columns


def calculate_missing_prevalence(column):
    """
    Calculate the prevalence of missing values in the input column.
    
    Parameters:
        column (pandas.Series): Input column to analyze.
        
    Returns:
        float: Prevalence of missing values in the input column.
    """
    missing_count = column.isnull().sum()
    total_count = len(column)
    missing_prevalence = missing_count / total_count
    
    return missing_prevalence


def calculate_zero_prevalence(column):
     """
     Calculate the prevalence of zero values in the input column.
     
     Parameters:
         - column: Pandas Series or dataframe column containing numeric values.
         
     Returns:
         - Prevalence of zero values as a float between 0 and 1.
     """
     if pd.isnull(column).all() or len(column.dropna()) == 0:
         raise ValueError("Invalid input column")
     
     num_zeros = (column == 0).sum()
     prevalence = num_zeros / len(column)
     
     return prevalence


def estimate_distribution(column):
    """
    Estimate the likely statistical distribution of the input column.
    
    Parameters:
        column (pandas.Series): Input column
        
    Returns:
        tuple: The name of the best fit distribution and its parameters
    """
    data = column.dropna().replace(0, np.nan)
    
    if len(data) < 3:
        return "Not enough non-trivial data points for estimation"
    
    distributions = [norm, lognorm, expon, gamma]
    best_fit = None
    best_fit_params = None
    best_fit_score = float('inf')
    
    for distribution in distributions:
        params = distribution.fit(data)
        
        pdf_values = distribution.pdf(data, *params[:-2], loc=params[-2], scale=params[-1])
        sse = np.sum(np.power((data - pdf_values), 2.0))
        
        if sse < best_fit_score:
            best_fit = distribution.name
            best_fit_params = params
            best_fit_score = sse
    
    return (best_fit, best_fit_params