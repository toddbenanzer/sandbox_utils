
import os
import pandas as pd
from tableauhyperapi import HyperProcess, Connection, TableName, Telemetry

def convert_to_csv(dataframe, filename):
    """
    Convert a dataframe into a .csv file.

    Args:
        dataframe (pd.DataFrame): The dataframe to be converted.
        filename (str): The name of the output .csv file.
    """
    dataframe.to_csv(filename, index=False)


def convert_to_excel(dataframe, filename):
    """
    Convert a dataframe into an .xlsx file.

    Args:
        dataframe (pd.DataFrame): The dataframe to be converted.
        filename (str): The name of the output .xlsx file.
    """
    dataframe.to_excel(filename, index=False)


def convert_dataframe_to_tde(dataframe, output_file):
    """
    Convert a dataframe into a .tde file.

    Args:
        dataframe (pd.DataFrame): The dataframe to be converted.
        output_file (str): The name of the output .tde file.
    """
    from tableauhyperapi import HyperProcess, Connection, TableDefinition

    with HyperProcess(telemetry=Telemetry.SEND_USAGE_DATA_TO_TABLEAU) as hyper:
        with Connection(endpoint=hyper.endpoint, database=output_file) as connection:
            table_definition = TableDefinition(
                TableName("Extract", "Table"),
                [(col_name, col_dtype) for col_name, col_dtype in zip(dataframe.columns, dataframe.dtypes)]
            )
            connection.catalog.create_table(table_definition)
            with connection.execute_query(f"INSERT INTO {table_definition.table_name} SELECT * FROM {dataframe}") as result:
                result.write_pandas(dataframe)


def export_dataframes_to_csv(dataframes, output_directory):
    """
    Export multiple dataframes into separate .csv files.

    Parameters:
        dataframes (dict): A dictionary where keys are the filenames and values are the dataframes.
        output_directory (str): Directory path where the .csv files will be saved.
    
    Returns:
        None
    """
    os.makedirs(output_directory, exist_ok=True)
    
    for filename, dataframe in dataframes.items():
        filepath = os.path.join(output_directory, f"{filename}.csv")
        dataframe.to_csv(filepath, index=False)


def export_dataframes_to_excel(dataframes, filenames):
    """
    Export multiple dataframes into separate .xlsx files.

    Parameters:
        dataframes (list): List of pandas dataframes to be exported.
        filenames (list): List of filenames for the exported files. Number of filenames should match the number of dataframes.

    Returns:
        None
    """
    
    for df, filename in zip(dataframes, filenames):
        df.to_excel(filename, index=False)


def export_dataframes_to_tde(dataframes, output_dir):
    """
    Export multiple dataframes into separate .tde files.

    Parameters:
        dataframes (dict): Dictionary where keys are filenames and values are pandas DataFrames.
        output_dir (str): Directory path where the .tde files will be saved.

    Returns:
        None
    """
    
   from tableausdk import Extract
    
   os.makedirs(output_dir, exist_ok=True)
   
   for name, df in dataframes.items():
       tde_path = os.path.join(output_dir, f"{name}.tde")
       extract = Extract(tde_path)
       
       # Define table schema and insert rows
       table_def = create_table_definition(df)
       table = extract.add_table("Data", table_def)
       insert_dataframe_rows(table, df)
       
       extract.close()


def create_table_definition(df):
   from tableausdk import Extract
   
   table_def = Extract.TableDefinition()
   
   for column_name,dtype in zip(df.columns,dff.dtypes):
       
       tableau_dtype=get_tableau_data_type(dtype)
       
       table_def.addColumn(column_name,
tableau_dtype)

return table_def


def get_tableau_data_type(dtype):
  from tableausdk import Extract
  
  tableau_dtypes={
  "int64":Extract.Type.INTEGER,"float64":Extract.Type.DOUBLE,"object":Extract.Type.UNICODE_STRING}
  
return tableau_dtypes.get(str(dtype),Extract.Type.UNICODE_STRING)


def insert_dataframe_rows(table,
df):

with table.open()as t:for row in df.itertuples(index=False):

tableau_row=[value if not pd.isnull(value)else None for value in row] t.insert(tableau_row)


def append_dataframe_to_csv(df,file_path):

"""
Appends a dataframe to an existing .csv file.


Parameters:

df(pandas.DataFrame):

The DataFrame to append.


file_path(str):

The path of the.csv file.


Returns:

None

"""
df.to_csv(file_path,
mode='a',header=False,index=False)


def append_dataframe_to_excel(dataframe,file_path,sheet_name):

"""
Appends a DataFrame to an existing.xlsx file.


Parameters:

dataframe(pandas.DataFrame):

The DataFrame to append.


file_path(str):

Path to the existing.xlsx file.


sheet_name(str):

Name of the sheet where the DataFrame should be appended.


Returns:

None

"""
with pd.ExcelWriter(file_path,
mode='a',engine='openpyxl',if_sheet_exists='overlay')as writer:

dataframe.to_excel(writer,sheet_name=sheet_name,index=False)


def merge_and_export_dataframes_to_csv(dataframes,
output_file):

"""
Merge multiple dataframes and export as a.single.csv file.


Parameters:

dataframes(list):

List of pandas DataFrames to be merged.


output_file(str):

File path for the merged.csv file.


Returns:

None

"""
merged_df=pd.concat(dataframes) merged_df.to_csv(output_file,index=False)



def merge_and_export_dataframes_to_excel(dataframes,
output_file):

"""
Merge multipleDataFramesandexportasa single.xlsxfile.

Parameters:


data frames(list):

ListofpandasData Framestobemerged.

output_filestrFilepathforthemerged.xlsxfile.ReturnsNone





merged_df=pd.concat(DataFrames) merged_df.to_excel(output_file,index=false) deffil

ter_columns(DataFrame columns:listfiltered_df=data frame[columns] returnfiltered_fndf


# Example usage defrename_columns(df,column_mapping:dictrenamed_df=df.rename(columns=column_mapping return rename columns


data={'col1':[1,

2,

3],'col2':['a',

'b',

'c'],'col3':[True,

False,

True]} df=pd.Data Frame(data)

metadata=extract_metadata(df)

printmetadata)}