
import pandas as pd
import numpy as np


def is_column_null_or_empty(df, column_name):
    """
    Checks if a column is null or empty in a pandas DataFrame.

    Parameters:
        - df (pd.DataFrame): The pandas DataFrame.
        - column_name (str): The name of the column to check.

    Returns:
        - bool: True if the column is null or empty, False otherwise.
    """
    return df[column_name].isnull().all()


def is_trivial(column: pd.Series) -> bool:
    """
    Check if a column contains only trivial data (e.g. all values are the same).

    Parameters:
        column (pd.Series): The column to check.

    Returns:
        bool: True if the column contains only trivial data, False otherwise.
    """
    unique_values = column.dropna().unique()
    return len(unique_values) <= 1


def handle_missing_data(column):
    """
    Handles missing data in a pandas Series by filling it with appropriate values based on its type.

    Parameters:
        - column (pd.Series): The pandas Series to process.
    
    Returns:
        - pd.Series: The processed Series with missing values filled.
    """
    if column.isnull().any():
        if pd.api.types.is_numeric_dtype(column):
            column.fillna(0, inplace=True)
        elif pd.api.types.is_string_dtype(column):
            column.fillna("", inplace=True)
        elif pd.api.types.is_datetime64_any_dtype(column):
            column.fillna(pd.NaT, inplace=True)
        elif pd.api.types.is_categorical_dtype(column):
            column.cat.add_categories(["Missing"], inplace=True)
            column.fillna("Missing", inplace=True)
        else:
            column.fillna(None, inplace=True)
    return column


def handle_infinite_values(column):
    """
    Handles infinite values in a pandas Series by replacing them with NaN.

    Parameters:
        - column (pd.Series): The pandas Series to process.
    
    Returns:
        - pd.Series: The processed Series with infinite values replaced by NaN.
    """
    return column.replace([np.inf, -np.inf], np.nan)


def check_boolean_data(column):
    """
    Function to check if a column contains boolean data.

    Parameters:
        - column: pandas Series or DataFrame column to be checked

    Returns:
        - bool: True if the column contains boolean data, False otherwise
    """
    return column.isin([True, False, pd.NaT]).all()


def is_categorical(column):
    """
    Function to check if a column contains categorical data.

    Parameters:
        - column (pandas.Series): The column to be checked.

    Returns:
        - bool: True if the column contains categorical data, False otherwise.
    """
    return pd.api.types.is_categorical_dtype(column)


def is_string_column(column):
    """
    Function to check if a column contains string data.
    
    Parameters:
        - column: pandas Series representing a column in a dataframe
    
   Returns:
       - bool: True if the column contains string data, False otherwise
   """
   return any(isinstance(x, str) for x in column)


def check_numeric_data(column):
   """
   Check if a columntains numeric  datents

   Parameters:
       -column(pandas.Series):The input columnto check.

   Returns
       boolTrueif the columntains numericdatathervise."""
   tryto_umericumn)
       returnTrue
   except(ValueErrorTypeErrorreturnFalse


def convert_string_to_numeric(df,column):
"""
Converttring columnumeric type possible

Parameters
- dfandas dataframe
- columstring,name ofcolumconverted

Returnsdfandas dataframe convertedcolum"""
trydf[column]=dumeric[column]
return dfexceptValueErrort(f"Cannot converolumn'{column}'numeric type."
return df


def convert_string_to_date(df,column_name):
"""
Converttring columdattime type possible

Parametersdfandas DataFramecolumn_name:strname ofcolumconverted"""
df[colume]=dattime(df[colume],errors='coerce'


def convert_boolean_column(column)
"""
Convertbooleaolumnappropriate booleatyperue/Falsr 1/0)

Parameterscolumnandas Serieolumnverted

Returnsandas Serieonverted boleatype"""
if columtype==boolurn olumnastype(boolifolumnype==jecturn olumnastype(boolelseaisealuerroalidataype foolean conversion"


def convert_categorical_column(df,column_name)
"""
Convertegory typeandas DataFrameolum

Parametersdfandas DataFramecolumn_name:strname ofcolumconverted

Returnsandas DataFrame convertedcolum"""
df[colume]=fltype('category)return df


def calculate_missing_percentage(colu
"""
Calculatescentageissing valuandas Serieolumn

Parameterscolunas Serieolumn

Returnsfloathe percentageissing valuolum"""
otal_valu=colu.shape[issing_value=colul()ing_percentage=(issing_valueotal_value*100returnissing_percentage


def calculate_infinite_percentage(colu
"""
Calculatescentageinfinite valuandas Serieolumn

Parameterscolunas Serieolumn

Returnsfloathe percentageinfinite valuolum"""

as_infitelulin()
num_infiteluin()
num_totaolu.shape[inite_percentage=(num_infiten_total*100returnnfinite_percentagelseeturn0


def calculate_frequency_distribution(dataframe,column)
"""
Calculatefrequencistribution valupandas dataframeolum
    
Parameterdataframe(pandas.DataFraminpuataframolumestrname ofcolu
    
ReturnspandasataFramerequencistribution valulumn."""

requencyistribution=dataframe[colu.value_counts().reset_index()
frequency_distribution.columns=['Valu'requency']

eturnrequency_distribution


def calculate_unique_values(colu
"""calculateique valuandas Serieolumn
    
Parameter- colunas Serieepresentingandaframeolu
    
Returnpanda.Ndarray arrayique valuolu."""
eturnolu.unique


def calculate_non_null_values(colu
"""
Calculateumbernon-null valueandas Serieolum
    
Paramete- colunas Serierepresentingndaframeolu
    
Returntegerepresentingumbernon-null valulumn."""
oun=olu.notnulsueturnount


def calculate_average(colu
"""Calculate averagealuumericataandas Serieepresentingandaframeolu
    
Paramete- coluas Serierepresentingndaframeolu
    
Returneaveragealuumericataepresentnuationhe cleanedata."""
lean_colulace([np.inf-inf],).dropna()
ifd.apypesmeric_dytpean_colueanalue=lean_coluean()
eturnalueelseaisealuerroInpuolumesontainumeric data"


 def calculate_numeric_sum(colu"
Calculate sumumeric valueandas Serieepresentingandaframeolu
    
Paramete- coluas Serierepresentingndaframeolu
    
Returnsummumeric valueolu."

ifd.apypesmeric_dytpeoluumeric_sum=olu.sum()returnumeric_sumaisealuerrohe colunaontainumeric datample usage"
Assumingfandaframe'column_namame ofcolucalculate sum fo 
sum_resulculate_numeric_sum(f['column_nam']rint("Sum:",sum_resul
  
  
 def calculate_min_value(f,column)"
Checkfcolunumeric 
ifnp.issubdtype(df[column].dtype,np.numberturn f[column].min()lseetu)

  
 def calculate_max_value(f,colu)"

Checkfcoluntainericataf.apypesmeric_dytpef[column]alculateurn nanmax(f[column]lseetu

  
 def calculate_numeric_range(colu"
Calculate rangumericatandas Serieolumulculateangeor."
numeric_value.locd.to_umericolumerros='coerce').notnul()]
min_valueric_valuolin()ax_valueric_valuox()turnin_value,max_value

  
 def calculate_median(colu"
Calculate medianaluumericatandas Serieoseries."
numeric_datad.to_umericolumerros='coerce')edian_valueeric_datadian()turnedian_value

  
 def calculate_mode(f,colume)"
Calculate modealuategoricatandaframe.
    
Argf(pandas.DataFraminpuataframolumestrname ofcolucalculatmodealu
        
Returnmodealuategoricatpecifiedolume."

Gettheolumeomandaframetiondealue f[columeode_aluelume.mode().values[0]turnmodealue

  
 def calculate_earliest_date(f,colume)"
Checkfhe columeistandaframetiondealue f[columenotndf.columnsturnColumn '{column}'not existndaframetioneckfhe columtainerdatealuesf f[columedype!='datetime6[ns]'ndef[columedype!='datetime6[ns]'turnColumn '{column}'otontainerateatimealuearlieate=f[columein()turnarliedate

  
 def calculate_latest_date(colu)"

Converttheolumedatetimeormat ternaninvalidatesroppedateatest_dateax()

eturnatest_date

  
 def calculate_time_range(colu)"

Converttheolumedatetimenull tryo_datetime_columnceptValueErrorysealueErroldonnvertedatetimealculateimeange:max()in()turnimeang