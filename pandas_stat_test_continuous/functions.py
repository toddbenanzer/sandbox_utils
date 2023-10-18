umpy as np
import pandas as pd
from scipy import stats

def calculate_mean(column):
    """
    Calculate the mean of a column.

    Parameters:
        column (pandas.Series): The input column.

    Returns:
        float: The mean value of the column.
    """
    if column.isnull().all() or len(column.unique()) == 1:
        return np.nan

    return np.mean(column)

def calculate_median(dataframe, column):
    """
    Calculate the median of a column in a pandas dataframe.

    Parameters:
        dataframe (pandas.DataFrame): The input dataframe.
        column (str): The name of the column to calculate the median for.

    Returns:
        float: The median value of the column.
    """
    median = dataframe[column].median()
    return median

def calculate_mode(column):
    """
    Calculate the mode of a column in a pandas dataframe.

    Parameters:
        column (pandas.Series): The column for which to calculate the mode.

    Returns:
        pandas.Series: The mode(s) of the column.
    """
    return column.mode()

def calculate_standard_deviation(df, column):
    """
    Calculate the standard deviation of a column in a pandas DataFrame.

    Parameters:
        df (pandas.DataFrame): The input DataFrame.
        column (str): The name of the column to calculate the standard deviation for.

    Returns:
        float: The standard deviation of the specified column.
    """
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    filtered_column = df[column].replace([np.inf, -np.inf], np.nan).dropna()

    
   
   
 
       
     
   
   
     
                 
            
    

    
  
    

     
     
 

           
               
             
                
                   
                
            
        
         
 
            
             
              
        
   
  
            
                 
    
  

      
        
        
        
        

    std_dev = filtered_column.std()

    return std_dev

def calculate_variance(column):
    """
    Calculate the variance of a given column in a pandas dataframe.

    Parameters:
        column (pandas.Series): The column for which variance needs to be calculated.

    Returns:
        float: The variance of the column.
    """
    return column.var()

def calculate_range(df, column_name):
    """
    Calculate the range of a column in a pandas dataframe.

    Parameters:
        df (pandas.DataFrame): The input dataframe.
        column_name (str): The name of the column to calculate the range for.

    Returns:
        float: The range of the specified column.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the dataframe.")

    min_value = df[column_name].min()
    max_value = df[column_name].max()

    column_range = max_value - min_value

    return column_range

def calculate_minimum(df, column_name):
      """
      Calculates the minimum value of a column in a pandas dataframe.

      Parameters:
          df (pandas.DataFrame): Input dataframe.
          column_name (str): Name of the column to calculate the minimum value.

      Returns:
          float: Minimum value of the specified column.
      """
      return df[column_name].min()

def calculate_column_maximum(df, column_name):
  """
  Calculate the maximum value of a column in a pandas dataframe.

  Parameters:
  - df: pandas dataframe containing the data
  - column_name: name of the column for which to calculate the maximum

  Returns:
  - maximum value of the specified column
  """

  if column_name not in df.columns:
      raise ValueError(f"Column '{column_name}' does not exist in dataframe")

  
 
  return df[column_name].max()

def calculate_column_sum(dataframe, column_name):
    """
    Calculates the sum of a column in a pandas dataframe.

    Parameters:
        dataframe (pandas.DataFrame): pandas DataFrame object
        column_name (str): string, name of the column in the dataframe

    Returns:
        sum_value (float): float, sum of the values in the specified column
    """

    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the dataframe.")

    column = dataframe[column_name]
    sum_value = column.sum()

    return sum_value

def calculate_non_null_count(df, column_name):
    """
    Calculates the count of non-null values in a column of a pandas dataframe.
    
    Parameters:
        df (pandas.DataFrame): The input dataframe.
        column_name (str): The name of the column to calculate the count for.
    
    Returns:
        int: The count of non-null values in the column.
    """
    return df[column_name].count()

def calculate_null_count(dataframe, column_name):
  """
  Calculates the count of null values in a column of a pandas dataframe.

  Parameters:
      dataframe (pandas.Dataframe): Input dataframe.
      column_name (str): Name of the column to calculate the null count for.

  Returns:
      int: Count of null values in the specified column.
  """
  count = dataframe[column_name].isnull().sum()
  return count

def count_unique_values(df, column_name):
  """
  Calculates the count of unique values in a column of a pandas DataFrame.

  Parameters:
      df (pandas.DataFrame): The input DataFrame.
      column_name (str): The name of the column.

  Returns:
      int: The count of unique values in the specified column.
  """
  unique_values = df[column_name].nunique()
  return unique_values

import pandas as pd
import numpy as np

def calculate_skewness(column):
    """
    Calculate the skewness of a column in a pandas dataframe.
    
    Parameters:
    column (pandas Series): The column to calculate the skewness.
    
    Returns:
    float: The skewness value.
    """
    return column.skew()

import pandas as pd
import numpy as np

def calculate_kurtosis(column):
    """
    Calculate the kurtosis of a column in a pandas dataframe.

    Parameters:
        column (pandas.Series): The input column.

    Returns:
        float: The kurtosis value of the column.
    """
    return column.kurtosis()

def handle_missing_values(df, column, method='mean'):
    """
    Function to handle missing values with mean/median/mode imputation in a column.
    
    Parameters:
        - df: pandas DataFrame
            The input DataFrame containing the data.
        - column: str
            The name of the column with missing values to be imputed.
        - method: str (default: 'mean')
            The imputation method to be used. Can be one of 'mean', 'median', or 'mode'.
    
    Returns:
        - df: pandas DataFrame
            The DataFrame with missing values imputed in the specified column.
    """
    
    if method == 'mean':
        imputed_value = df[column].mean()
    elif method == 'median':
        imputed_value = df[column].median()
    elif method == 'mode':
        imputed_value = df[column].mode().values[0]
    else:
        raise ValueError("Invalid imputation method. Please choose one of 'mean', 'median', or 'mode'.")
    
    df[column] = df[column].fillna(imputed_value)
    
    return df

def drop_missing_values(df, columns):
  """
  Drop rows with missing values in any specified columns.

  Parameters:
      df (pandas.DataFrame): Input dataframe.
      columns (list): List of column names to check for missing values.

  Returns:
      pandas.DataFrame: Dataframe with rows containing missing values in the specified columns dropped.

  """
  df.dropna(subset=columns, inplace=True)
  return df

def handle_infinite_values(dataframe, columns, replacement_value=None):
    """
    Function to handle infinite values in specified columns of a pandas dataframe.
    
    Parameters:
    dataframe (pd.DataFrame): Input pandas dataframe.
    columns (list): List of column names to handle infinite values.
    replacement_value (float or None): Value to replace infinite values with. Default is None which replaces with NaN.
    
    Returns:
    pd.DataFrame: Updated pandas dataframe with replaced infinite values.
    """
    for column in columns:
        if replacement_value is not None:
            dataframe[column] = dataframe[column].replace([float('inf'), float('-inf')], replacement_value)
        else:
            dataframe[column] = dataframe[column].replace([float('inf'), float('-inf')], pd.NA)
    
    return dataframe

def check_for_infinite_values(dataframe, columns):
  """
  Function to check if any infinite values exist in any specified columns of a pandas dataframe.

  Parameters:
      dataframe: pandas DataFrame object.
      columns: List of column names to check for infinite values.

  Returns:
      Dictionary indicating whether infinite values exist in each specified column.
      The keys of the dictionary are the column names and the values are boolean values.
  """
  
  result = {}
  
  for col in columns:
      result[col] = np.any(np.isinf(dataframe[col]))
  
  return result

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def normalize_column(data, column, method='z-score'):
  """
  Normalize/standardize data in a column using different methods.

  Parameters:
      data (pd.DataFrame): Input data.
      column (str): Name of the column to normalize.
      method (str, optional): Normalization method. 
                              Options: 'z-score', 'min-max scaling'. Default is 'z-score'.

  Returns:
      pd.DataFrame: Dataframe with normalized column values.
  """
  if column not in data.columns:
    raise ValueError(f"Column '{column}' does not exist in the dataframe.")
  
  column_data = data[column]
  
  if column_data.isnull().sum() > 0:
    raise ValueError(f"Column '{column}' contains missing values.")
  
  if not np.isfinite(column_data).all():
    raise ValueError(f"Column '{column}' contains infinite values.")
  
  if method == 'z-score':
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(column_data.values.reshape(-1, 1))
  elif method == 'min-max scaling':
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(column_data.values.reshape(-1, 1))
  else:
    raise ValueError("Invalid normalization method. Available methods are 'z-score' and 'min-max scaling'.")
  
  normalized_column_name = f"{column}_{method.replace(' ', '_')}"
  data[normalized_column_name] = normalized_data.flatten()
  
  return data

import pandas as pd
from scipy.stats import zscore

def detect_outliers(df, column):
    """
    Detect outliers in a column using the z-score method.

    Parameters:
        df (pandas.DataFrame): The input DataFrame.
        column (str): The name of the column to analyze for outliers.

    Returns:
        list: A list of outlier values found in the specified column.
    """
    
    z_scores = zscore(df[column])
    outliers = df[abs(z_scores) > 3][column].tolist()
    
    return outliers

import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu

def hypothesis_testing_two_samples(df, sample1_col, sample2_col, test_type='t-test'):
    """
    Perform hypothesis testing on two samples using t-test or Mann-Whitney U test.

    Parameters:
        df (DataFrame): Input pandas DataFrame.
        sample1_col (str): Name of the column containing the first sample data.
        sample2_col (str): Name of the column containing the second sample data.
        test_type (str, optional): Type of test to perform. Can be 't-test' or 'mann-whitney'. Defaults to 't-test'.

    Returns:
        dict: Dictionary containing the test statistic and p-value.
    """

    sample1 = df[sample1_col]
    sample2 = df[sample2_col]

    if test_type == 't-test':
        statistic, pvalue = ttest_ind(sample1, sample2)
    elif test_type == 'mann-whitney':
        statistic, pvalue = mannwhitneyu(sample1, sample2)
    else:
        raise ValueError("Invalid test_type. Supported values are 't-test' and 'mann-whitney'.")

    result = {
        'test_statistic': statistic,
        'p_value': pvalue
    }

    return result

import pandas as pd
from scipy.stats import f_oneway, kruskal

def hypothesis_testing_multiple_samples(df, group_col, data_cols, test_type):
    """
    Perform hypothesis testing on multiple samples using ANOVA or Kruskal-Wallis test.

    Parameters:
        df (pandas.DataFrame): Input dataframe.
        group_col (str): Name of the column containing group labels.
        data_cols (list): List of column names containing continuous data.
        test_type (str): Type of test to perform. Either 'anova' or 'kruskal'.

    Returns:
        pandas.DataFrame: Dataframe with test results for each data column.

    """
    results = []
    
    for col in data_cols:
        groups = []
        for group in df[group_col].unique():
            groups.append(df[df[group_col] == group][col])
        
        if test_type == 'anova':
            stat, p_value = f_oneway(*groups)
        elif test_type == 'kruskal':
            stat, p_value = kruskal(*groups)
        
        result = {'Column': col, 'Test': test_type, 'Statistic': stat, 'P-value': p_value}
        results.append(result)
    
    return pd.DataFrame(results)

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

def correlation_analysis(df, variable1, variable2, method='pearson'):
    """
    Perform correlation analysis between two continuous variables in a pandas dataframe.
    
    Parameters:
    df (pandas.DataFrame): Input dataframe containing the variables.
    variable1 (str): Name of the first continuous variable.
    variable2 (str): Name of the second continuous variable.
    method (str): Correlation coefficient to be used. Options are 'pearson' (default) and 'spearman'.
    
    Returns:
    float: The correlation coefficient between the two variables.
    """
  
  if method not in ['pearson', 'spearman']:
      raise ValueError("Invalid method. Supported methods are 'pearson' and 'spearman'.")
  
  if method == 'pearson':
      correlation, _ = pearsonr(df[variable1], df[variable2])
  elif method == 'spearman':
      correlation, _ = spearmanr(df[variable1], df[variable2])
  
  return correlation

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def regression_analysis(df, x_col, y_col, model_type='linear'):
    """
    Perform regression analysis between two continuous variables in a pandas dataframe.

    Parameters:
        df (pandas.DataFrame): Input dataframe containing the variables.
        x_col (str): Name of the independent variable (X).
        y_col (str): Name of the dependent variable (Y).
        model_type (str): Type of regression model. Options are 'linear' (default) and 'non-linear'.

    Returns:
        dict: Dictionary containing the regression equation and coefficients.
    """
    # Check if columns exist in the dataframe
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError("Invalid column name(s).")

    # Check if columns contain numeric data
    if not pd.api.types.is_numeric_dtype(df[x_col]) or not pd.api.types.is_numeric_dtype(df[y_col]):
        raise TypeError("Columns must contain numeric data.")

    # Create X and y arrays for linear regression
    X = df[x_col].values.reshape(-1, 1)
    y = df[y_col].values

    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Get the coefficients and intercept
    coefficients = model.coef_
    intercept = model.intercept_

    # Create the regression equation
    equation = f"Y = {intercept:.3f} + {coefficients[0]:.3f} * X"

    result = {
        'equation': equation,
        'coefficients': coefficients.tolist(),
        'intercept': intercept
    }

    return result

import pandas as pd
from statsmodels.tsa.stattools import acf, pacf

def time_series_analysis(data, variable, method):
    """
    Perform time series analysis on a continuous variable using autocorrelation or moving average.

    Parameters:
        - data: pandas DataFrame containing the data
        - variable: string specifying the column name of the variable to analyze
        - method: string specifying the analysis method ('autocorrelation' or 'moving_average')

    Returns:
        - result: pandas DataFrame with the analysis results
    """

    # Check if the specified column exists in the DataFrame
    if variable not in data.columns:
        raise ValueError(f"Column '{variable}' does not exist in the input data.")

    # Check if the specified method is valid
    if method not in ['autocorrelation', 'moving_average']:
        raise ValueError("Invalid method. Supported methods are 'autocorrelation' and 'moving_average'.")

    # Select the variable to analyze
    series = data[variable]

    result = pd.DataFrame()

    if method == 'autocorrelation':
        # Calculate autocorrelation coefficients for different lags
        lags = range(1, len(series))
        autocorr_coeffs = [series.autocorr(lag=lag) for lag in lags]

        result['Lag'] = lags
        result['Autocorrelation Coefficient'] = autocorr_coeffs

    elif method == 'moving_average':
        # Calculate moving averages with different window sizes
        windows = range(2, len(series) + 1)
        moving_averages = [series.rolling(window=window).mean() for window in windows]

        for i, window in enumerate(windows):
            result[f'Moving Average (Window Size = {window})'] = moving_averages[i]

    return result

import matplotlib.pyplot as plt
import seaborn as sns

def visualize_distribution(dataframe, column):
    """
    Visualize the distribution of a continuous variable using histogram, box plot, and density plot.

    Parameters:
        dataframe (pd.DataFrame): The input dataframe.
        column (str): The name of the column to visualize.

    Returns:
        None
    """
    
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' does not exist in the dataframe.")
    
    if not pd.api.types.is_numeric_dtype(dataframe[column]):
        raise TypeError(f"Column '{column}' must contain numeric data.")
    
  
    
    
   
   
 
       
     
   
   
     
                 
            
    

    
  
    

     
     
 

           
               
             
                
                   
                
            
        
         
 
            
             
              
        
   
  
            
                 
    
  

      
        
        
        
        

   

    
    

    plt.figure(figsize=(12, 4))
    
    # Plot histogram
    plt.subplot(1, 3, 1)
    sns.histplot(data=dataframe, x=column)
    plt.title("Histogram")
    
    # Plot box plot
    plt.subplot(1, 3, 2)
    sns.boxplot(y=dataframe[column])
    plt.title("Box Plot")
    
    # Plot density plot
    plt.subplot(1, 3, 3)
    sns.kdeplot(data=dataframe, x=column)
    plt.title("Density Plot")
    
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt

def plot_continuous_variables(df, x, y):
  """
  Plot the relationship between two continuous variables using scatter plot or line plot.

  Parameters:
      df (pandas.DataFrame): Input dataframe.
      x (str): Name of the first continuous variable.
      y (str): Name of the second continuous variable.

  Returns:
      None
  """
  
  if x not in df.columns or y not in df.columns:
    raise ValueError("Invalid column name(s).")
  
  if not pd.api.types.is_numeric_dtype(df[x]) or not pd.api.types.is_numeric_dtype(df[y]):
    raise TypeError("Columns must contain numeric data.")
  
  unique_x = df[x].nunique()
  unique_y = df[y].nunique()
  
  if unique_x <= 25 and unique_y <= 25:
    plt.scatter(df[x], df[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"Scatter Plot: {x} vs {y}")
    plt.show()
  else:
    plt.plot(df[x], df[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"Line Plot: {x} vs {y}")
    plt.show()

import numpy as np

def calculate_confidence_interval(data, column, method='mean', confidence=0.95):
  """
  Calculate confidence intervals for the mean or median of a column in a pandas dataframe.

  Parameters:
      data (pandas.DataFrame): Input dataframe containing the data.
      column (str): Name of the column to calculate confidence intervals for.
      method (str, optional): Method to use for calculating confidence intervals. 
                              Options: 'mean' or 'median'. Default is 'mean'.
      confidence (float, optional): Desired level of confidence. Default is 0.95.

  Returns:
      tuple: Lower and upper bound of the confidence interval.
  """
  
  if column not in data.columns:
    raise ValueError(f"Column '{column}' does not exist in the dataframe.")
  
  if method not in ['mean', 'median']:
    raise ValueError("Invalid method specified. Method must be either 'mean' or 'median'.")
    
  
   
   
 
       
     
   
   
     
                 
            
    

    
  
    

     
     
 

           
               
             
                
                   
                
            
        
         
 
            
             
              
        
   
  
            
                 
    
  

      
        
        
        
        

    filtered_data = data[column].replace([np.inf, -np.inf], np.nan).dropna()
    
   
   
 
       
     
   
   
     
     


    if len(filtered_data) == 0:
        raise ValueError(f"No valid data found for column '{column}'.")
    
    n = len(filtered_data)
    
    if method == 'mean':
        sample_statistic = np.mean(filtered_data)
    else:
        sample_statistic = np.median(filtered_data)
    
    standard_error = stats.sem(filtered_data)
    
    z = stats.t.ppf((1 + confidence) / 2, n - 1)
    
    margin_of_error = z * standard_error
    
    lower_bound = sample_statistic - margin_of_error
    upper_bound = sample_statistic + margin_of_error
    
    return (lower_bound, upper_bound)

import pandas as pd
from scipy.stats import ttest_ind

def calculate_effect_size(sample1, sample2):
  """
  Calculate Cohen's d effect size for two samples.

  Parameters:
      sample1 (pd.Series): First sample.
      sample2 (pd.Series): Second sample.

  Returns:
      float: Effect size measure (Cohen's d).
  """
  
  mean_diff = sample1.mean() - sample2.mean()
  pooled_std = np.sqrt((sample1.std()**2 + sample2.std()**2) / 2)
  
  effect_size = mean_diff / pooled_std
  
  return effect_size

import numpy as np

def calculate_power(effect_size, significance_level, power):
  """
  Perform power analysis for sample size determination in hypothesis testing.

  Parameters:
      effect_size (float): The standardized effect size.
      significance_level (float): The desired significance level (alpha).
      power (float): The desired statistical power (1 - beta).

  Returns:
      int: The required sample size.
  """

  critical_value = stats.norm.ppf(1 - significance_level/2)

  noncentrality_parameter = stats.norm.ppf(power) + critical_value

  sample_size = (noncentrality_parameter / effect_size)**2

  return int(np.ceil(sample_size))

def handle_null_columns(dataframe):
    """
    Handle null columns by dropping them or filling them with specific values.

    Parameters:
        dataframe (pandas.DataFrame): The input dataframe.

    Returns:
        pandas.DataFrame: The dataframe after handling null columns.
    """
    dataframe = dataframe.dropna(axis=1)
    return dataframe

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def impute_missing_values(df, method='mean'):
    """
    Function to perform data imputation for missing values using different methods.

    Parameters:
        df (pandas.DataFrame): Input dataframe with missing values.
        method (str): Imputation method. Options: 'mean', 'median', 'most_frequent', 'constant'. Default is 'mean'.

    Returns:
        pandas.DataFrame: DataFrame with missing values imputed using the specified method.
    """
    imputer = SimpleImputer(strategy=method)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    return df_imputed

def handle_trivial_columns(df):
  """
  Handle trivial columns by dropping them or merging them with other columns based on specific conditions.

  Parameters:
      df (pandas.DataFrame): Input dataframe

  Returns:
      pandas.DataFrame: Updated dataframe after handling trivial columns
  """

  df = df.dropna(axis=1, how='all')

  
 
  df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all')

  trivial_cols = []
  for col in df.columns:
      if len(df[col].unique()) == 1:
          trivial_cols.append(col)

  
 
  for col in trivial_cols:
      if col.endswith('_trivial'):
          merge_col = col[:-8] 
          if merge_col in df.columns:
              df[merge_col] += df[col]
              df = df.drop(columns=[col])

  return df

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def handle_categorical(data):
    """
    Handle categorical variables in data by converting them into numerical representations.

    Parameters:
        data (pd.DataFrame): Input pandas dataframe with categorical columns

    Returns:
        pd.DataFrame: Dataframe with categorical columns replaced by numerical representations
    """
    categorical_cols = data.select_dtypes(include=['object']).columns

    label_encoder = LabelEncoder()
    data[label_cols] = data[label_cols].apply(lambda col: label_encoder.fit_transform(col))

    onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    onehot_encoded = onehot_encoder.fit_transform(data[onehot_cols])
    onehot_encoded_df = pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names())
    
    data_encoded = pd.concat([data.drop(columns=categorical_cols), onehot_encoded_df], axis=1)

    return data_encoded

import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, mutual_info_regression

def perform_feature_selection(df, target_column, method='correlation', k=5):
    """
    Perform feature selection on continuous variables using different methods.

    Parameters:
        - df: pandas DataFrame containing the input data.
        - target_column: string specifying the name of the target column.
        - method: string specifying the selection method (default: 'correlation').
                  Options: 'correlation' or 'mutual_information'.
        - k: integer specifying the number of top features to select (default: 5).

    Returns:
        - selected_features: list of column names of the selected features.
    """

    continuous_columns = df.select_dtypes(include='number').columns.tolist()

    continuous_columns = [col for col in continuous_columns if not df[col].isnull().all()]
    
    continuous_features = [col for col in continuous_columns if col != target_column]

    if len(continuous_features) == 0:
        raise ValueError("No suitable continuous features found.")

    if method == 'correlation':
        correlations = df[continuous_features + [target_column]].corr()
        correlations = correlations.dropna() 

        selected_features = correlations[target_column].abs().nlargest(k).index.tolist()
    
    elif method == 'mutual_information':
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
        selector.fit(df[continuous_features], df[target_column])

        selected_features = [continuous_features[i] for i in selector.get_support(indices=True)]
    
    else:
        raise ValueError("Invalid method. Supported methods: 'correlation', 'mutual_information'.")

    return selected_features

from sklearn.decomposition import PCA

def perform_pca(df, num_components):
    """
    Perform dimensionality reduction on continuous variables using PCA.

    Parameters:
        - df: pandas DataFrame containing the input data.
        - num_components: integer specifying the number of principal components to keep.

    Returns:
        - transformed_df: pandas DataFrame with transformed data.
    """
    
    df = df.dropna(axis=1, how='all')
    df = df.loc[:, (df != df.iloc[0]).any()]
    
    pca = PCA(n_components=num_components)
    transformed_data = pca.fit_transform(df)
    
    transformed_df = pd.DataFrame(data=transformed_data, 
                                  columns=[f'PC{i}' for i in range(1, num_components+1)])
    
    return transformed_df

import pandas as pd

def calculate_summary_statistics(data, group_column, target_column):
    """
    Calculates summary statistics for each group within a specified column based on another categorical column.

    Parameters:
        - data (pandas.DataFrame): The input dataframe.
        - group_column (str): The name of the categorical column used for grouping.
        - target_column (str): The name of the column to calculate summary statistics for.

    Returns:
        - summary_stats (pandas.DataFrame): The dataframe containing the calculated summary statistics.
    """
  
  grouped_data = data.groupby(group_column)

  summary_stats = grouped_data[target_column].agg(['mean', 'median', 'count'])

  return summary_stats

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

def calculate_r_squared(dataframe, target_column, predicted_column):
    """
    Calculate the coefficient of determination (R-squared) for regression models on a continuous variable.

    Parameters:
        dataframe (pandas.DataFrame): The input dataframe.
        target_column (str): The name of the column containing the actual values.
        predicted_column (str): The name of the column containing the predicted values.

    Returns:
        float: The R-squared value.
    """
  
  target = dataframe[target_column]
  predicted = dataframe[predicted_column]
  
  r_squared = r2_score(target, predicted)
  
  return r_squared

import numpy as np

def calculate_adjusted_r_squared(y, y_pred, num_predictors, sample_size):
    """
    Calculates the adjusted R-squared value for regression models on a continuous variable.
    
    Parameters:
    - y: Actual values (dependent variable)
    - y_pred: Predicted values (based on regression model)
    - num_predictors: Number of predictors in the model
    - sample_size: Sample size
    
    Returns:
    - adjusted_r_squared: Adjusted R-squared value
    """
  
  residuals = y - y_pred
  ssr = np.sum(residuals**2)  
  
  mean_y = np.mean(y)
  sst = np.sum((y - mean_y)**2)  
  
  r_squared = 1 - (ssr / sst)  
  
  adjusted_r_squared = 1 - ((1 - r_squared) * (sample_size - 1)) / (sample_size - num_predictors - 1)
  
  return adjusted_r_squared

import numpy as np

def generate_random_samples(distribution, mean, std_dev, size):
    """
    Function to generate random samples from a given distribution with specified mean and standard deviation.

    Parameters:
        distribution: str, the name of the desired distribution (e.g., 'normal', 'uniform')
        mean: float, the desired mean of the distribution
        std_dev: float, the desired standard deviation of the distribution
        size: int, the number of samples to generate

    Returns:
        samples: numpy array, random samples from the specified distribution
    """

    if distribution == 'normal':
        samples = np.random.normal(mean, std_dev, size)
    elif distribution == 'uniform':
        samples = np.random.uniform(mean-std_dev/2, mean+std_dev/2, size)

    return sample