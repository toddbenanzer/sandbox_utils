
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis


def calculate_category_frequency(df, column_name):
    """
    Function to calculate the frequency of each category in a column of a pandas dataframe.

    Parameters:
    df (pandas.DataFrame): The input dataframe.
    column_name (str): The name of the column to calculate the frequency for.

    Returns:
    pandas.Series: A series containing the frequency count for each category in the specified column.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the dataframe.")
    
    return df[column_name].value_counts()


def calculate_category_percentage(df, column_name):
    """
    Calculates the percentage of each category in a column of a pandas dataframe.
    
    Parameters:
        - df: The pandas dataframe containing the data.
        - column_name: The name of the column to calculate the category percentages for.
        
    Returns:
        - A new pandas dataframe with two columns: 'Category' and 'Percentage'.
          The 'Category' column contains the unique categories from the input column,
          and the 'Percentage' column contains the corresponding percentage values. 
    """
    cleaned_column = df[column_name].replace([pd.NA, float('inf'), float('-inf')], pd.NA).dropna()
    category_counts = cleaned_column.value_counts()
    total_count = len(cleaned_column)
    
    category_percentages = (category_counts / total_count) * 100
    
    return pd.DataFrame({'Category': category_percentages.index, 'Percentage': category_percentages.values})


def calculate_all_columns_category_frequency(df):
    """
    Calculate the frequency of each category across all columns in a pandas dataframe.

    Parameters:
        df (pandas.DataFrame): The input dataframe.

    Returns:
        pandas.DataFrame: A dataframe containing the frequency of each category.
    """
    return df.apply(lambda col: col.value_counts()).fillna(0)


def calculate_all_columns_category_percentage(df):
    """
    Function to calculate the percentage of each category across all columns in a pandas dataframe.
    
    Parameters:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: Dataframe with the percentage of each category across all columns
    """
    
    return df.apply(lambda x: x.value_counts(normalize=True) * 100).fillna(0)


def handle_missing_data(df, method='exclude', value=None):
    """
    Function to handle missing data by excluding or imputing values.
    
    Parameters:
        - df: pandas dataframe containing the data
        - method: method to handle missing data, either 'exclude' or 'impute'. Default is 'exclude'.
                  If 'exclude', any rows with missing values will be excluded from analysis.
                  If 'impute', missing values will be imputed with the specified value.
        - value: value to impute missing data with. Only used if method is set to 'impute'. Default is None.
    
    Returns:
        - pd.DataFrame: Dataframe after handling missing data
    """
    
   return df.dropna() if method == 'exclude' else df.fillna(value)


def handle_infinite_data(df, method='exclude'):
   """
   Function to handle infinite data by excluding or imputing values.

   Parameters:
       - df: pandas DataFrame
       - method: str, optional (default='exclude')

   Returns:
       - result: pandas DataFrame
   """

   if method == 'exclude':
       result = df.replace([np.inf, -np.inf], np.nan).dropna()
   elif method == 'impute':
       result = df.replace([np.inf, -np.inf], np.nan)
       result.fillna(result.mean(), inplace=True)
   else:
       raise ValueError("Invalid method. Please choose 'exclude' or 'impute'.")
   
   return result


def remove_null_columns(df):
   """
   Drop columns with any null values

   Parameters:
       - df (pd.DataFrame): Input Dataframe

   Returns:
       pd.DataFrame: Dataframe without null columns
   """

   return df.dropna(axis=1)


def remove_trivial_columns(df):
   """
   Remove trivial columns from the dataframe.

   Parameters:
       - df (pd.DataFrame): Input Dataframe

   Returns:
       pd.DataFrame: Dataframe without trivial columns
   """

   return df.loc[:, df.nunique() > 1]


def calculate_category_count(column):
   """
   Calculate count of categories in a given column

   Parameters:
      - column(pd.Series): Column from which categories counts need calculated

     Returns:

      pd.Series : Series having count values   

  """

  return  column.value_counts()


def calculate_mode(df):
  """
  Function to calculate mode for given dataframe

  Parameters:

     pd.DataFrame : Given Input 

  Returns : 
     
     pd.Series : Series having mode 
   
  """

  return  df.mode()


def calculate_median(df):
  """
  Function to calculate median for given dataframe

  Parameter :

     pd.dataframe : Given input 

  Returns :

     pd.Series : Series having median 
  
  """

return  df.median()


def calculate_mean(df):

"""
Function to Calculate mean for given dataframe 

Parameters:

pd.dataframe : Given input 

Returns :

pd.series : Series having mean 

"""

return  df.mean()




def calculate_std(df):

"""
Function to Calculate standard deviation for given dataframes 

Parameters :

pd.dataframe : Given input 


Returns :


pd.series : Series having standard deviation 

"""

return  df.std()




def calculate_variance(df):

"""
Function to Calculate variance for given dataframes 


Parameters:

pd.dataframes : Given Input 

Returns:

pd.Series : Series having variance 


"""

return(variance)


 def calculate_range(df):

"""
Calculate range for given dataframes 


Parameters:

pd.dataframes : Given inputs 

Returns :

Dictionary having ranges   

"""


column_ranges={}

for col in range(df.columns):

if(df[col].dtype.name=='category') :

column_ranges[col]=len(dataframe[col].unique())

return(column_ranges)




 def calculate_column_min(DF):

"""
Calculate minimum value for given dataframes 


parameters:

pd.dataframes : Given inputs 


Returns :

PD.series having minimum value 


"""

return(DF.min(axis=0))



 def Calculate_max(Df):

"""
Calculate maximum value Of giveN DATAFRAMES


parameters:


Pd.dataframes : Givne inputs


Returns:


Pd.series Having Maximum Value

  
"""

RETURN Df.MAX()




 def Calculate_quartiles(Df):

"""
Calculate Quartiles Of GiveN DATAFRAMES


parameters:


Pd.dataframes : Givne inputs


Returns:


Pd.series Having Quartiles

  
"""


quartiles=df.quantile([0.25,0.5,0.75])

return quartiles



 def Calculate_interquartile_range(Df):

"""
Calculate Interquartile Range For GiveN DATAFRAMES


parameters:


Pd.dataframes : Givne inputs


Returns:


Pd.series Having Interquartile_range

  
"""

iqr_df=pd.dataframe(columns=['columns','Interquartile_range'])

for col in Df.columns():

try :
         
      Data= DF[col].replace([np.inf,-np.inf],np.nan).dropna()

      iqr=np.percentile(data,75)-np.percentile(data,25)

      iqr_df=iqr_df.append({'Column':col,'Interquartile Range':iqr},ignore_index=True)

except :
     
pass

return iqr_df




 def Calculate_skewness(Df):

"""
Calculate Skewness For GiveN DATAFRAMES


parameters:


Pd.dataframes:GivNe inputs 


returns :

 Pd.series Having skewness
   
"""  

return Df.apply(lambda x:skew(x.dropna()))



 def Calculate_kurtosis(Df):

"""
Calculate Kurtosis For GiveN DATAFRAMES


parameters:


Pd.dataframes:GivNe inputs 


returns :

 Pd.series Having kurtosis
   
"""  


Kurtosis_values=[]
for cols in Df.columns():
kurtosis_value=stats.kurtosis(Df[cols],nan_policy="omit")
kurtosi_values.append(kurtosis_value)
result=pd.dataframe({"Column":Df.columns,"Kurtosi":Kurtosi_values})
return(result)



