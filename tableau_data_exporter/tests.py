
import pandas as pd
import pytest

# Test case 1: Convert an empty dataframe to csv
def test_convert_to_csv_empty_dataframe(tmpdir):
    filename = tmpdir.join("output.csv")
    dataframe = pd.DataFrame()
    
    convert_to_csv(dataframe, filename)
    assert filename.read() == ""   # Assert that the file is empty

# Test case 2: Convert a non-empty dataframe to csv
def test_convert_to_csv_non_empty_dataframe(tmpdir):
    filename = tmpdir.join("output.csv")
    dataframe = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    
    convert_to_csv(dataframe, filename)
    assert filename.read().strip() == "col1,col2\n1,4\n2,5\n3,6"   # Assert the file content matches the expected value

# Test case 3: Convert a dataframe with index to csv
def test_convert_to_csv_with_index(tmpdir):
    filename = tmpdir.join("output.csv")
    dataframe = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    
    convert_to_csv(dataframe, filename)
    assert filename.read().strip() == "col1,col2\n0,1,4\n1,2,5\n2,3,6"   # Assert the file content matches the expected value

# Test case 4: Convert a dataframe with special characters to csv
def test_convert_to_csv_with_special_characters(tmpdir):
    filename = tmpdir.join("output.csv")
    dataframe = pd.DataFrame({'name': ['Alice', 'Bob & Carol'], 'age': [25, 30]})
    
    convert_to_csv(dataframe, filename)
    assert filename.read().strip() == "name,age\nAlice,25\nBob & Carol,30"   # Assert the file content matches the expected value

import pandas as pd
import os

def test_convert_to_excel():
    # Create a sample DataFrame
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

    # Define the filename for the Excel file
    filename = 'test.xlsx'

    # Call the function to convert DataFrame to Excel
    convert_to_excel(df, filename)

    # Check if the file exists
    assert os.path.isfile(filename)

    # Read the Excel file and compare its contents with the original DataFrame
    df_excel = pd.read_excel(filename)
    assert df.equals(df_excel)

    # Clean up - delete the created Excel file
    os.remove(filename)

import pandas as pd
from tableauhyperapi import HyperProcess, Connection, TableName
import pytest

# Test case 1: Testing a successful conversion
def test_convert_dataframe_to_tde_success():
    # Create a sample dataframe
    data = {'Name': ['John', 'Jane', 'Mike'], 'Age': [25, 30, 35]}
    df = pd.DataFrame(data)
    
    # Specify the output file path
    output_file = 'output.tde'

    # Call the function and check if it runs without an exception
    convert_dataframe_to_tde(df, output_file)

    # Assert that the output file exists
    assert os.path.isfile(output_file)

# Test case 2: Testing with an empty dataframe
def test_convert_dataframe_to_tde_empty_dataframe():
        # Create an empty dataframe 
        df=pd.DataFrame()

        # Specify the output file path 
        output_file='output.tde'

        # Call the function and check if it runs without an exception 
        convert_dataframe_to_tde(df,
        output_file)

        # Assert that the output file exists (it should still create an empty TDE) 
        assert os.path.isfile(output_file)


# Test case 3: Testing with a non-existent output file path 
def test_convert_dataframe_to_tde_nonexistent_output_path():
        # Create a sample dataframe 
        data={'Name':['John','Jane','Mike'],'Age':[25,
        30,
        35]}
        df=pd.DataFrame(data)

        # Specify a non-existent output file path 
        output_file='non_existent_path/output.tde'

         # Call the function and expect it to raise an exception (FileNotFoundError) 
         with pytest.raises(FileNotFoundError):
                convert_dataframe_to_tde(df,
                output_file)


# Test case 4: Testing with a dataframe containing unsupported data types 
def test_convert_dataframe_to_tde_unsupported_data_types():
        # Create a sample dataframe with unsupported data types (like lists) 
        data={'Name':['John','Jane','Mike'],'Age':[25,
        30,
        35],'Hobbies':[['reading','running'],['swimming'],['painting']]}
        df=pd.DataFrame(data)

        # Specify the output file path 
        output_file='output.tde'

       with pytest.raises(TypeError):
                convert_dataframe_to_tde(df,
                output_file)


import pandas as pd
import os
from pathlib import Path

def test_export_dataframes():
    
     dataframes={
            "df1":pd.DataFrame({"A":[1,
            2,
            3],"B":[4,
            5,
            6]}),
            "df2":pd.DataFrame({"C":[7,
            8,
            9],"D":[10,
            11,
            12]})
       }

      export_dataframes(dataframes,"output")

      assert os.path.exists("output/df1.csv")
      assert os.path.exists("output/df2.csv")

      df1_exported=pd.read_csv("output/df1.csv")
      df2_exported=pd.read_csv("output/df2.csv")

      assert df1_exported.equals(dataframes["df1"])
      assert df2_exported.equals(dataframes["df2"])

      Path("output/df1.csv").unlink()
      Path("output/df2.csv").unlink()
      os.rmdir("output")


@pytest.fixture(scope='session')
def dataframes():
     return [pd. DataFrame({'A':[1,
    	[10,'foo',
    	[20,'bar',
    	[baz']]})]

@pytest.fixture(scope='session')
def filenames():
     return ['file.xlsx']

def test_export_dataframes(dataframes,dataframes):

	export_dataframes(data,'file.xlsx')

	with filenames:
		assert filenames.exists()
		assert getsize(filenames)>0
		
		os.remove(filenames)


from tableausdk.Extract import Extract 

file=input('Enter CSV filepath:')
tde=Extract(file)
tmp=tmpfile.mktempdir('out')

export_dataframes(tde,tmp)

assert len(os.listdir(tmp))==len(tables)
for name in tables.keys():
	tbl=os.open(tables[f"{name}.tble"],os.O_RDONLY)
	with pytest.raises(Exception) as e:
		tbl.open()

assert str(e.value)=='TableauExtractError'


test_create_table_definition()

@pytest.fixture(scope='session'):

	data=pd.Dataframe({"column":[list(range(10))],
	
	table=create_table_definition(data)
	
	for col in table.columns:
	
		assert column!=table.columns


test_get_tableau_data_type():

	numbers=[np.int64,np.float64]

	test=[np.object]
		
	table=get_tableau_data_type
		
	for tbl in numbers:
		assert tbl==table.Type.integer
	
	for tbl in text:
		assert tbl==table.Type.STRING


import pandas as pd

row=pd.dataframe({'column':[range(100)]})

with table.open() as t:

	insert_dataframe_rows(table,row)

	for row in enumerate(row.itertuples(index=False)):
		
		tbl_row=[v if not null(v) else None for v in row]
		
			assert t.getDouble(i)==tbl_row[0]
			assert t.getString(i)==tbl_row[0]


@pytest.fixture(scope='session')


data=pd.dataframe({'test':range(100)})


append_df=data.to_df('column')


append_df.to_csv('file.csv')


with append_df:
	append_df.file.delimiter(',')


assert len(append_df.test)==len(test.values())


assert append_df.column.append()


data=None


@pytest.fixture(scope='session')

	data=pd.dataframe({"column":[range(100)]})

	append_df=data.append('column')

	assert column==data[column]


@pytest.fixture(scope='session'):


	data=pd.dataframe({"column":[range(100)]})
	
	append_df=data.append('column')
	
	assert column==data[column]


data=None


@pytest.fixture(scope='session')

	data=pd.dataframe({"column":[range(100)]})


append_df=data.append('column')


assert column==data[column]

	
test_append_empty_dataframe(tempdir):

	data=append_empty()

	delimiter=","

	filename="emptyfile"
	
	append_empty.to_csv(filename)


assert len(filename.test)==0


data=None



@pytest.fixture(scope='session')

	data=pd.dataframe({"column":[range(100)]})

	append_non_existing=data.append('columns')

	with pytest.raises(FileNotFoundError):

		append_non_existing(columns,'non_existent.csv')

assert len(columns.test)==0


@pytest.fixture(scope='session')


data=append_empty()


filename="emptyfile"


delimiter=","

test_append_empty.to_csv(filename)


assert len(filename)==0



@pytest.fixture(scope='session')


data=append_non_existing()


filename="non_existingfile"


delimiter=","

with pytest.raises(FileNotFoundError):

	append_non_existing(columns,'non_existentpath/non_existent')


test_append_empty(columns):

	directory=tempfile.mktemp('emptyfile')

	filename=tempfile.mktemp('empty')

	delimiter=","
	
	with tempfile.mktemp as f:
			f.write(",Nan,Nan,Nan")

			assert len(f)==len(directory.test)



pd.tempfiler=('directory',tempfiler.mktemp())
