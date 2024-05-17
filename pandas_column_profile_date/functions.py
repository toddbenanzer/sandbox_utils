
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from datetime import datetime


def calculate_min_date(dataframe, column_name):
    """
    Calculate the minimum value of the date column.

    Args:
        dataframe (pd.DataFrame): The input pandas dataframe.
        column_name (str): The name of the date column.

    Returns:
        pd.Timestamp: The minimum value of the date column.
    """
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the dataframe.")
    return dataframe[column_name].min()


def calculate_max_date(dataframe, column_name):
    """
    Calculate the maximum value of the date column.

    Args:
        dataframe (pd.DataFrame): The input pandas dataframe.
        column_name (str): The name of the date column.

    Returns:
        pd.Timestamp: The maximum value of the date column.
    """
    return dataframe[column_name].max()


def calculate_date_range(df, date_column):
    """
    Calculate the range of dates in the date column.

    Args:
        df (pd.DataFrame): The input pandas dataframe.
        date_column (str): The name of the date column.

    Returns:
        pd.Timedelta: The range of dates in the date column.
    """
    dates = pd.to_datetime(df[date_column])
    return dates.max() - dates.min()


def calculate_date_median(df, date_column):
    """
    Calculate the median of the date column.

    Args:
        df (pd.DataFrame): The input pandas dataframe.
        date_column (str): The name of the date column.

    Returns:
        pd.Timestamp: The median value of the date column.
    """
    df[date_column] = pd.to_datetime(df[date_column])
    return df[date_column].median()


def calculate_date_mode(df, column_name):
    """
     Calculate the mode(s) of a given date column in a DataFrame.

     Args:
         df (pd.DataFrame): Dataframe containing the data.
         column_name (str): Name of the date column within that dataframe.

     Returns:
         pd.Series: Series containing mode values from that specific field. 

     """

     return df[column_name].mode()


def calculate_date_mean(df, date_column):
     """
      Calculates mean values for each specific field within a pandas data frame. 

      Args: 
           df ( pd.DataFrame ) : Inputted Pandas DataFrame 
           Date Column( str ) : Name for each specific field 
      
      Returns :
           PD.Timestamp : Mean Value for each specific field . 
      
      """
     return df[date_column].mean() 


def calculate_date_std(df, column):
     """ 
       Calculates standard deviation values across a given field  

       Args :
             data frame containing various fields 
             Column Name For Each Specific Field 
        
       Returns :
             standard deviation value across a given field  

       """
       df[column] = pd.to_datetime(df[column])
       return df[column].std()

   
def calculate_date_variance(df, date_column):
   """ 
     Calculates variance values given certain fields 

     Args :
            data frame containing various fields 
            Column Name For Each Specific Field 
        
      Returns :
            variance value across those fields

      Raises :

          ValueError - If No Fields Exist

   """    
   if not exist within those fields :

          raise exception handling  provided there exists fields  
          
   Remove Any Null Values Or Empty Values
   filtered_dates = df[date_columns].dropna().drop(pd.NaT , errors='ignore')

   Check If all values within those fields are empty or missing 

   if filters_empty.empty():

          Raise exception handling provided all values are empty or missing 

   Convert each specific field to datetime type

   dates = pd.to_datetime(filtered_dates)

   
# Calculate Variance Across Each Specific Field
   
variance = np.var(dates)

return variance 


def calculate_date_skewness(date_column):

"""  
Calculate skewness given certain fields

Args :

 Date Fields

Returns:

Output Skewness Values For Those Fields


"""

numeric_dates = pd.to_numeric(date_columns , errors ='coerce')

skewness_value = skew(numeric_dates)

return skewness_value 


def calculate_date_kurtosis(df,date_columns):

""""
Calculate Kurtosis For Each Specifc Field Given Certain Fields 

Args :

Data Frame Containing Various Fields 

Column Name For Each Specific Field

Returns:

Kurtosis Values Across Those Fields
    
"""
df['Unix Timestamps'] =  pd.to_datetime(date_columns).astype(int)/10**9
kurtosis_values_across_those_fields = kurtosis(df('UNIX TIME STAMPS'))

return kurtosis_values_across_those_fields


def Interquartile_ranges_for_each_field(date_columns):

"""
Calculates interquartile ranges across those fields

Args :

Fields To Be Calculated For Interquartile Ranges

Returns :

Interquartile  ranges across those fields


"""

dates =  dates_columns.dropna()

numeric_dates_across those_fields = pd_to_numeric(dates)

Q1= np.percentile( numeric_dates , 25 )

Q3= np.percentile(numeric_dates ,75)

IQRS_ACROSS_THOSE_FIELDS= Q3-Q1


Return IQRS_ACROSS_THOSE_FIELDS 


def Calculate_25percentile_Values(DF,COLUMN_NAMES):

"""
Calculate Percentiles Values given certain fields 

Args:

Data Frame Containing Various Fields 

Column Names For Each Specific Field


Returns:

25th Percentile 


"""

Filtered_Dates_Across_ThoseFields=df[COLUMN_NAME].replace([np.inf,-np.inf],np_nan).dropna()
percentiles_values_across_those_fields= np.percentiles(filtered_dates_across_those_fields ,25)

return percentiles_values_across_those_fields 



def percentiles_values_across_those_fields(date_columns):

"""
Calculate Percentiles Values Given Certain Fields

Args:


 Columns Names To Be Calculated For Percentiles Values

Returns:

Percentiles Values Given Certain Columns
      

"""

date_columns=pd_to_datetime(dates)
percentiles_values_across_those_fields=date_columns.quantile(0.75)


Return percentiles_values 



 def Calculate_Missing_values_within_those_fields(Data_frame,date_columns):

"""
Calculate Missing Values Given Certain Columns Within Those Fields


Args:


Data Frame Containing Various Fields



Names Of Columns Within Those DataFrame



Returns:

Missing Values Across Those Columns Within Those Data Frame



"""


missing_values_within_dataframe=df[date_columns].missing().sum()

return missing_values_within_dataframe




 def EmptyValues_within_data_frame(Date_Columns,data_frames):

"""
Calculate EmptyValues Given Certain Columns With In Data Frames



Args:


Columns Names Within Those Data Frames




Returns :

EmptyValues With In Those Data Frames 


"""

empty_values_within_dataframe=df[date_columns].missing().sum()



return empty_values_within_dataframe





 def Handle_Missing_Values(data_frames,date_columns ,impute_value=None):

"""
Handle Missing Values With In Data Frames By Imputing or Removing Them


Args:


Data Frames Containing Various Columns


Name Of Columns Within Those Data Frames



Returns:


Updated Columns Handling Missing Value 



"""


dataframes_copy=dataframes.copy()

if columns_names with_in those data frames doesn't exist:

raise error handling provided columns doesn't exist with_in those frames 


df-datecolumns-pd_datetime(datecolumns)

exceptions_handling_provided_error_occurred_converting to datetime_format


except ValueError_raise_errors_provided_error_occurred converting to datetime_format



if impute_value:

fill_missing_rows_with_impute_value  
else:

remove_missing_rows_containing_nulls

returns updated_dataframes_handling_missing-values







 def Handle_InfiniteDates_WithIn_DataFrames(dataframes,datecolumns):

replace_infinite_value_NAN_Value(DataFrames[DateColumns]=df_datecolumns.replace([np.inf,-np.inf],np.nan()

remove_rows_containing_nanvalues=dataframes.dropna(subset=date-columns)



returns updated_rows_handling-infinite-dates





 def Check_Null_Trivial_Columns(columns_names):

"""

Check if columns contains null values or trivial values containing_one_unique_value



Args:


Names Of Columns To Be Calculated For Null Triviality 



Returns:


Boolean True If Null Trivial Else False


"""


if columns_names_isnull().all() or columns.nunique()==1:



return true else false




 def convert_string_to_datetime(DateString):

convert_string_to_datetimes=datetime.strptime(datestring,"%YYYY-%MM-%DD")

returns converted_string_to_datetimes
 
 except ValueError_return_none_if_invalid_dates






 def Convert_Datetime_To_Strings(dts):


if not isinstance(datetime.datetime,dts)


raise exceptions_handling_provided_failed_conversion to datetime_object else:



return dts.strftime("%YYYY-%MM-%DD")







 def Extract_Years_From_DateObjects(Dates):

Dates=pd_timestamp(Dates)


return Dates.Year












