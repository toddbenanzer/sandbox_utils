
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import kurtosis, ttest_1samp, ttest_ind, ttest_rel, chi2_contingency

def calculate_mean(df, column_name):
    """
    Function to calculate the mean of a numeric column in a pandas dataframe.
    
    Args:
        df (pandas.DataFrame): The input dataframe.
        column_name (str): The name of the column to calculate the mean for.
        
    Returns:
        float: The mean value of the specified column.
    """
    try:
        return df[column_name].mean()
    except KeyError:
        raise ValueError(f"Column '{column_name}' does not exist in the dataframe.")
    except TypeError:
        raise TypeError("The specified column is not numeric.")

def calculate_median(df, column_name):
    """
    Calculates the median of a numeric column in a pandas dataframe.
    
    Args:
        df (pd.DataFrame): The pandas dataframe containing the column.
        column_name (str): The name of the column to calculate the median for.
        
    Returns:
        float: The median value of the specified column.
    """
    return df[column_name].median()

def calculate_mode(df, column_name):
    """
    Calculates the mode of a numeric column in a pandas dataframe.
    
    Args:
        df (pd.DataFrame): The pandas dataframe containing the column.
        column_name (str): The name of the column to calculate the mode for.
        
    Returns:
        float: The mode value of the specified column. 
               If multiple modes are present, returns one of them.
               Returns None if no mode is found or if errors occur.
    """
    if df.empty or column_name not in df.columns:
        return None
    
    values = df[column_name].values
    mode_value = stats.mode(values)
    
    return mode_value.mode[0] if mode_value.mode.size > 0 else None

def calculate_quartiles(column):
    """
    Calculate the quartiles of a numeric column.

     Parameters:
         - data (pandas.Series): Numeric column to calculate quartiles.

     Returns:
         pandas.Series: Quartiles of the numeric column.
     """
     return column.quantile([0.25, 0.5, 0.75])

def calculate_range(data):
     """
     Calculate the range of a numeric column.

     Parameters:
         - data (pandas.Series): Numeric column to calculate range.

     Returns:
         float: Range of the numeric column.
     """
     return np.max(data) - np.min(data)

def calculate_standard_deviation(df, column_name):
     """
     Calculate the standard deviation of a numeric column in a pandas dataframe.

     Parameters:
         - df (pandas.DataFrame): Input dataframe
         - colname (str): Name of numeric columns whose standard deviation is calculated

      Returns:
          float: Standard deviation
      """
      col = df[colname]
      if pd.api.types.is_numeric_dtype(col):
          return np.std(col)
      else:
          raise ValueError("Column should be numeric")

 def calculate_variance(df, colname):
      """
      Function to calculate variance 
      
      Parameters
           - df(pandas.DataFrame)
           - colname(str)
           
       returns
            float: variance value
       """  
       if colname not in df.columns():
           raise ValueError("Column does not exist")
           
       values = df[colname].replace(np.inf,np.nan).dropna().values()
       if len(values)<2 : raise ValueError("Not enough valid values")
       
       return np.var(values)

 def calcuate_skewness(column) :
      """
      calculates skewness in given columns
      
      Parameter :
              Column(panda.Series) 
              
       returns : 
            Float : skewness value
       """   
       
       if not pd.api.types.is_numeric_dtype(column):
            raise ValueError("columns should be numeric")
      
      # drop missing values
      columns = columns.replace(np.inf,np.nan).dropna()
      
      #calculating skewness using panda's skew function 
      skewness = columns.skew()
      
       
          
