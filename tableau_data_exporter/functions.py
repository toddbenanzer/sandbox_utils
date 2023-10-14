andas as pd
from tableausdk import *
from tableausdk.Extract import *


def convert_to_tde(dataframe, output_file):
    # Create a new extract file
    ExtractAPI.Initialize()
    extract = Extract(output_file)

    # Define the table schema based on the dataframe columns
    table_definition = TableDefinition()
    for column in dataframe.columns:
        if dataframe[column].dtype == 'object':
            table_definition.addColumn(column, Type.UNICODE_STRING)
        elif dataframe[column].dtype == 'int64':
            table_definition.addColumn(column, Type.INTEGER)
        elif dataframe[column].dtype == 'float64':
            table_definition.addColumn(column, Type.DOUBLE)
        elif dataframe[column].dtype == 'bool':
            table_definition.addColumn(column, Type.BOOLEAN)

    # Create the new table in the extract file
    extract_table = extract.addTable("Extract", table_definition)

    # Add data rows to the table
    with extract_table.open() as t:
        for _, row in dataframe.iterrows():
            new_row = Row(table_definition)
            for column in dataframe.columns:
                if dataframe[column].dtype == 'object':
                    new_row.setCharString(column, str(row[column]))
                elif dataframe[column].dtype == 'int64':
                    new_row.setInteger(column, int(row[column]))
                elif dataframe[column].dtype == 'float64':
                    new_row.setDouble(column, float(row[column]))
                elif dataframe[column].dtype == 'bool':
                    new_row.setBoolean(column, bool(row[column]))
            t.insert(new_row)

    # Save and close the extract file
    extract.close()
    ExtractAPI.Cleanup()

# Usage example:
df = pd.DataFrame({'Name': ['John', 'Jane', 'Mike'],
                   'Age': [25, 30, 35],
                   'Height': [170.5, 165.2, 180.3],
                   'IsStudent': [False, True, True]})

convert_to_tde(df, 'data.tde'