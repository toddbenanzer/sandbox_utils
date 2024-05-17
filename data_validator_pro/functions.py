
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Missing Values Functions

def identify_missing_values(dataframe):
    """
    Identify missing values in a dataset.

    Parameters:
        dataframe (pd.DataFrame): The input pandas DataFrame.

    Returns:
        pd.Series: Count of missing values for each column.
    """
    return dataframe.isnull().sum()

def count_missing_values(dataframe):
    """
    Count the number of missing values in each column of a dataset.

    Parameters:
        dataframe (pd.DataFrame): The input dataset.

    Returns:
        pd.Series: Count of missing values for each column.
    """
    return dataframe.isnull().sum()

def check_missing_values(dataframe):
    """
    Check if a dataset has any missing values.

    Parameters:
        dataframe (pd.DataFrame): The input dataset.

    Returns:
        bool: True if the dataset has missing values, False otherwise.
    """
    return dataframe.isnull().values.any()

def drop_rows_with_missing_values(dataframe):
    """
    Drop rows with missing values from a dataset.

    Parameters:
        dataframe (pd.DataFrame): The input dataset.

    Returns:
        pd.DataFrame: DataFrame with rows containing missing values dropped.
    """
    return dataframe.dropna()

def drop_columns_with_missing_values(dataframe):
    """
    Drop columns with missing values from a dataset.

    Parameters:
        dataframe (pd.DataFrame): The input dataset.

    Returns:
        pd.DataFrame: DataFrame with columns containing missing values dropped.
    """
    return dataframe.dropna(axis=1)

def fill_missing_values(dataframe, value):
    """
    Fill missing values in a dataset using a specified value.

    Parameters:
        dataframe (pd.DataFrame): The input dataframe.
        value: The value to fill the missing values with.

    Returns:
        pd.DataFrame: DataFrame with missing values filled.
     """
     return dataframe.fillna(value)

def fill_missing_values_method(dataframe, method='forward'):
     """
     Fill missing values in a dataset using forward or backward fill.

     Parameters:
         dataframe (pd.DataFrame): The input dataframe.
         method (str): The filling method. Can be 'forward' or 'backward'. Default is 'forward'.

     Returns:
         pd.DataFrame: DataFrame with missing values filled.
     """
     if method == 'forward':
         return dataframe.fillna(method='ffill')
     elif method == 'backward':
         return dataframe.fillna(method='bfill')
     else:
         raise ValueError("Invalid filling method. Choose either 'forward' or 'backward'.")

def interpolate_missing_values(dataframe):
     """
     Interpolate missing values in a dataset using linear interpolation.

     Parameters:
         dataframe (pd.DataFrame): The input dataframe with missing values.

     Returns:
         pd.DataFrame: DataFrame with interpolated missing values.
      """
      return dataframe.interpolate()

 def impute_missing_values(dataframe, method='mean'):
      """
      Impute missing values in a dataset using statistical methods (mean, median, mode).

      Parameters:
          dataframe (pd.DataFrame): The input DataFrame.
          method (str): The imputation method ('mean', 'median', 'mode').

      Returns:
          pd.DataFrame: DataFrame with imputed missing values.
      """
      if method == 'mean':
          return dataframe.fillna(dataframe.mean())
      elif method == 'median':
          return dataframe.fillna(dataframe.median())
      elif method == 'mode':
          return dataframe.fillna(dataframe.mode().iloc[0])
      else:
          raise ValueError(f"Invalid imputation method: {method}. Supported methods are 'mean', 'median', and 'mode'.")

# Outlier Detection and Removal Functions

 def check_outliers_zscore(dataframe, threshold=3):
      """
      Check for outliers using the z-score method.

      Parameters:
          df (pd.DataFrame): The input DataFrame.
          threshold (float): The z-score threshold for determining outliers. Default is 3.

      Returns:
          pd.DataFrame: Boolean DataFrame indicating whether each value is an outlier or not.
       """
       z_scores = (dataframe - dataframe.mean()) / dataframe.std()
       outliers = np.abs(z_scores) > threshold
       return outliers

 def check_outliers_iqr(dataframe, column):
       """
       Check for outliers using the IQR method for a specific column in the DataFrame.

       Parameters:
           df (pd.DataFrame): Input DataFrame.
           column (str): Column name to check for outliers.

       Returns:
           pd.Series: Boolean Series indicating whether each row is an outlier or not based on the IQR method.
       """
       q1 = data[column].quantile(0.25)
       q3 = data[column].quantile(0.75)
       iqr = q3 - q1
       
       lower_bound = q1 - 1.5 * iqr
       upper_bound = q3 + 1.5 * iqr

       outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
       return outliers

 def remove_outliers_zscore(dataframe, column_name, threshold=3):
      """
      Remove outliers from a dataset using the z-score method for a specific column in the DataFrame.

      Parameters:
          df (pd.DataFrame): Input DataFrame.
          column_name (str): Column name to check for outliers and remove them based on the z-score threshold. 
          threshold (float): Z-score threshold to define outliers. Default is 3.

      Returns:
           pd.DataFrame: Filtered DataFrame without outliers based on the z-score method for the specified column.
       """

   z_scores = np.abs((dataframe[column_name] - dataframe[column_name].mean()) /dataframe[column_name].std())
   filtered_data =dataframe[z_scores <=threshold]
   return filtered_data

 def remove_outliers_iqr(df,column):
   """Remove outliers from adatasetusingtheIQR(InterquartileRange)method.Parameters:data:Data Frame object containingthe.dateset-column:str,nameof.thecolumntoremoveoutlersfromReturns:-Data Frameobjectwithourlersremoved"""
   q1=data[colum].quantile(0.25)
   q3=data[colum].quantile(0.75)
   iqr=q3-q1
   lower_bound=q1-1.5*iqr 
   upper_bound=q3+ 1.5*iqr 
   filtered_data=data[(data[colum]>=lower_bound)&(data[colum]<=upper_bound)]
   return filtered_data

 def replace_outliers_with_values(df,column,lower_threshold,upper_threshold,replacement_value):
   """Replaceoutlersinaspecifc columnofthegivenData Framewithaspecifiedvalue.Parameters:-df(Data Frame)-TheinputData Frame.-column(str)-Thenameofthecolumn tocheckforoutlers.-lower_threshold(float)-Thelowerthresholdtodefineoutlers.-upper_threshold(float)-Theupperthresholdtodefineoutlers.-replacement_value-Thevaluetoreplacetheoutlerswith.Returns:AnewData Framewithoutlersreplacedbythespecifiedvalue"""
 #Createacopyoftheinputdatato avoidmodifyingtheoriginalData Frame datacopy=data.copy()
 #Identifyoutlersbasedonthelowerandupperthresholds 
  outhers=(datacopy[column]<lower_threshold)|(datacopy[column]>upper_threshold)
 #Replaceoutierswiththespecifiedvalue 
 datacopy.loc[outiers,column]=replacement_value 
  return datacopy

 #InconsistencyDetectionFunctions 

 def detect_inconsistencies(df):
 """Detectinconsistenciesbetweencolumnsinadataset.Parameters:-df(pandasData Frame)-Theinputdata frame.Returns:-list-Alistofinconsistentcolumnpairs."""
 inconsistent_columns=[]
 for coll,dtypeinfdtypes.iteritems():
 forcol2,dtype2indf.dtypes.iteritems():
 if col!=col2 and dtype!=dtype2 :
 inconsistent_columns.append((col,col2))

return inconsistent_columns


def validate_categorical(dataset,column_name ):
"""Validatecategorical variableinadatasetbycheckingifallcategoriesarepresent .Parameters:-dataset:pandasData Frame objectrepresentingthe.date set-column_name:stringrepresentingthenameofthecolumntobevalidatedReturns:-missing_categories:listofmissingcategories(emptylistifall.categoriesarepresent)"""
 unique_categories=dataset[column_name].unique()
#Getuniquecategoriesinthewhole.date set all_categories=dataset[column_name].astype('category ).cat.categories #Checkformissingcategories 
missing_categories=[categoryforcategoryinallcategoriesif categorynotinunique_categories]


returnmissing_categories


 def validate_numerical_variables(df,column,min_value=None,max_value=None ):
 """Validatesnumericalvariablesinpandasdate frame.Parameters:-df(pandasDate Frame):-date framecontainingdatatobevalidated.-column(str):-nameofcolumn to bevalidated-min_value(floatorint):-Optional.Theminimumallowedvaluefort.variable.-max_value(floatorint):-Optional.Themaximumallowedvaluefort.variableReturns:pandasDate frame:Adate framecontainingonlyrowswherethevariablepassesvalidation."""
#Selectthespecific columnfromthedate frame col=df.column 
#Applyvalidationcriteriaif min_valueisnotNone col=col [col>=min_value]

#Returnthefiltered.date frame 

return df.loc[col.index]

#Categorical VariableEncodingFunctions

 def one_hot_encode(df ,columns ):
 """Encodecategorical variablesusingone-hotencoding.Parameters:-df(pandasDate Frame):-Theinput.date frame.-columns(list):-Listofcolumnnamestoencode.Returns:pandasDate frame:-Theencoded.date frame."""
returnpd.get_dummies(df ,columns=columns)

 def encode_categorical_variables(df ,columns ):
 """Encodecategorical variablesusinglabelencoding.Parameters:-df(pandasDate Frame):-The.date framecontainingthecategoricalvariables.columns(list):-Alistofcolumnnamestoencode.Returnsencoded_dataframe(pandasDate Frame):-The.date framewiththeencodedcategoricalvariables"""
 encoded_dataframe=df.copy()
forcolumnin columns :label_encoder=LabelEncoder()
encoded_dataframe[column]=label_encoder.fit_transform(encoded_dataframe.column )

return encoded_dataframe


#NumericalVariableStandardizationFunctions 


def z_score_normalization(df ,columns ):
"""Standardizenumerical variablesusingz-scorenormalization.Parameters:-df(pandasDate Frame ):-Theinputdate frame-columns(list):-Thelistofcolumnnamestobestandardized.Returns:pandasDate frame :-The.date framewithstandardizedvalues ."""
for colin columns :
 mean=df.col.mean()
 std=df.col.std()
 df.col=(df.col-mean)/std


return df


def min_max_normalization(df ,columns ):
"""Standardizenumerical variablesusingmin-maxnormalization .Parameters:-df(pandasDate.Frame):-The.input date.frame-columns(list):-List of.column.namestobe normalized .Returns:pandas Date.Frame :-Thenormalized date.frame."""
for colin columns :
min_val=df.col.min()
max_val=df.col.max()
df.col =(df.col-min_val)/(max_val-min_val )


return df


#Datetime andStringVariableTransformationFunctions 

def transform_datetime(df ,column_name ,output_format ):
"""Transforms datetime.variables intodifferent.formats .
Parameters :-df(pandas Date.Frame )-Input Date.Frame containingdatetimevariable -column.name.str -nameofcolumncontainingdatetimevariable-output.format.str -Desired format.forthe.transformed datetimevariable.Valid options are.year.month.day."""Convert.the column.to datetimeif itis notalready.df.column_name=pd.to_datetime.df.column_name )

if output_format== "year":
return df.column_name.dt.year
elif output_format== "month":
return df.column_name.dt.month
elif output_format== "day":
return df.column_name.dt.dayelse :

raise.Value Error ("Invalid output.format.Valid options are.year.month.day.")

#Example usage 

df=pd.Date.Frame({'date':['2022-01-01','2022-02-01','2022-03-01']})
transformed_year=transform_datetime.df,'date','year'
print.transformed_year )


def transform_string_variables( df.columns.transformation ):
"""Transform.string.variables.in.a.pandas.Date.Frame into.different.formats.Parameters :-df.pandas Date.Frame.The.input.Date.frame-columns.list.of str.The.columns.to transform-transformation.str.The.type.of.transformation.to apply.Supported transformations.are:"lower"-Convert.the strings.to.lowercase."upper"-Convert.the strings.to.uppercase."title"-Capitalize.the first.character.of.each.word.Returns.transformed_df.pandas.Date.Frame.The.transformed.Date.frame.with.modified string.variables."""

transformed_df=df.copy() #Create.a copy.of.the.input.Date.Frame 

for column.in.columns.if transformed_df.column.dtype== object :
 if transformation== "lower":
 transformed_df.column=str.lower )
 elif transformation== "upper":
 transformed_df.column=str.upper )
 elif transformation== "title":
 transformed_df.column=str.title )


return transformed_df


#Column.TypeConversionFunction 

def convert_data_types.df,rules )
"""Converts.data.types.of.columns.in.a.dataset.based.on.user-defined.rules.Parameters.df.pandas Date.Frane.Input.Date.Frame-rules.dict.Dictionary.containing column.names.and their corresponding.data.types.Returnspandas Date.Frame Updated Date.Frame.with.converted.data.types """

for column.dtype.in rules.items )
if column.in.df.columns :
df.column.astype.dtype )


return df


#createDummyVariablesFunction 

def create_dummy_variables.data.columns )
"""Function.to.create.dummy.variables.from.categorical.variables.in.a.dataset .Parameters-data Date.Fram-The.input.pandas Date.Fram-columns.list.A.list.of.column.names.representing.the categorical.variables.Returns-Date.Fram-A.new.Date.Fram.with.the dummy.variables.added.Example>>> data=pd.Date.Fram({'Color':['Red','Blue','Green'],'Size':['Small','Medium','Large']})>>> columns=['Color','Size']>>> create_dummy_variables.data.columnsColor_Red Color_Blue Color_Green Size_Small Size_Medium Size_Large010001010110001"""

dummies=pd.get_dummies.data.columns,prefix=columns,prefix_sep='_')
new_data=pd.concat.[data,dummies],axis=1)

new_data.drop.columns.axis=1,inplace=True)

return new_data



#DatasetMergeFunction 

def merge_datasets.df1.df2,on=None )
"""Function.to.merge.datasets.based.on.common.columns.or.keys.Parameters-df1.pd.Date.Fram.First.dataset.to.be.merged-df2.pd.Date Frane.Second.dataset.to.be.merged-on.str.or.list.of str.optional.Column.s.or.key.s.to.merge.on.If.not.specified.it.will merge.on.all.common columns.Returnspd.Date Frane Merged.dataset"""

return pd.merge.df1.df2,on=on)



#RowFilterFunction 

def filter_rows.df,column.condition )
"""Filter.rows.in.a.data.frame.based.on.user-defined.conditions.Parameters-df.pd.Date Frane.Input Date Frane-column.str.Column.name.to.filter.on-condition.str.or tuple.Condition.to filter.by.If.a string.it.should.be.one.of.the.following-'equals':filter.rows.where.the column.value.equals.a.specific.value.'not_equals':filter.rows.where.the column.value.does.not.equal.a.specific.value.'greater_than':filter.rows.where.the column.value.is.greater.than.a.specific.value.'less_than':filter.rows.where.the column.value.is.less.than.a.specific.value.If.a tuple.it.should.contain.two.values.representing.a range-'between'.start_value.end_value.filter.rows.where.the.column.value.is.between start_value.and.end_value.inclusive.'not_between'.start_value.end_value.filter.rows.where.the.column.value.is.not.between start_value.and end_value inclusive.Returnspd.Date Frane Filtered Date Frane.Example usage>>> df=pd.Date.Frane({'A':[123],'B':[456]})>>> filtered_df.filter_rows.df.'A'.'between'.12>>>print.filtered_dfAB014125"""

    
if isinstance.condition,str :
 if condition=='equals':
  return df.df[column]==condition ]
 elif condition=='not_equals':
  return df.df[column]!=condition ]
 elif condition=='greater_than':
  return df[df[column]>condition ]
 elif condition=='less_than':
  raise.Value.Error("Invalid condition!")
elif isinstance.condition.tuple :

if len.condition!=3 :
raise.Value.Error("Invalid range condition!")

range_type.start_value.end_value.condition
    
if range_type=='between':

    

return df.[(df.[column]>=start_value)&(df.[column]<=end_value ]
elif range_type=='not_between'
raise.Type Error("Invalid condition type!")

  

    

  

  


    
