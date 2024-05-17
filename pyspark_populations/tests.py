
import pytest
from pyspark.sql import SparkSession
import os
import shutil

# Mock file path for testing
file_path = "test.csv"

@pytest.fixture(scope="session")
def spark():
    spark = SparkSession.builder.getOrCreate()
    yield spark
    spark.stop()

def test_read_csv_to_dataframe(spark):
    df = read_csv_to_dataframe(file_path)
    
    assert not df.isEmpty()
    assert 'column1' in df.schema.fieldNames()
    assert 'column2' in df.schema.fieldNames()
    assert df.count() == 10
    assert df.select('column1').dtypes[0][1] == 'integer'
    assert df.filter(df.column2 == 'value').count() > 0

@pytest.fixture(scope="session")
def spark_session():
    spark = SparkSession.builder.getOrCreate()
    yield spark
    spark.stop()

def test_read_json_file_valid(spark_session):
    temp_dir = "./temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_file_path = os.path.join(temp_dir, "test.json")
    
    try:
        with open(temp_file_path, "w") as f:
            f.write('{"name": "John", "age": 30}')
        
        df = read_json_file(temp_file_path)
        
        assert not df.isEmpty()
        assert "name" in df.columns
        assert "age" in df.columns
        
    finally:
        shutil.rmtree(temp_dir)

def test_read_json_file_invalid(spark_session):
    temp_dir = "./temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_file_path = os.path.join(temp_dir, "test.json")
    
    try:
        with open(temp_file_path, "w") as f:
            f.write('{"name": "John", "age": 30')
        
        df = read_json_file(temp_file_path)
        
        assert df.isEmpty()
        
    finally:
        shutil.rmtree(temp_dir)

def test_read_parquet_file_valid(spark_session):
    data = [(1, "apple"), (2, "banana"), (3, "orange")]
    columns = ["id", "fruit"]
    
    df = spark_session.createDataFrame(data, columns)
    file_path = "test_file.parquet"
    
    df.write.parquet(file_path)
    
    result_df = read_parquet_file(file_path)
    
    assert isinstance(result_df, type(df))
    
def test_read_parquet_file_invalid():
   with pytest.raises(Exception):
       read_parquet_file("invalid_path.parquet")

def test_read_dataframe_from_table(spark_session):
   url = "jdbc:postgresql://localhost:5432/mydatabase"
   table = "mytable"
   properties = {
       "user": "myusername",
       "password": "mypassword"
   }
   
   df = read_dataframe_from_table(url, table, properties)
   
   assert df.count() > 0
   assert len(df.columns) == 3
   
   expected_columns = ["column1", "column2", "column3"]
   assert df.columns == expected_columns
   
   expected_data_types = ["string", "integer", "double"]
   assert [str(field.dataType) for field in df.schema.fields] == expected_data_types

@pytest.fixture(scope="session")
def test_data(spark):
   data = [("John", 25), ("Alice", 30), ("Bob", 35)]
   schema = ["name", "age"]
   return spark.createDataFrame(data, schema)

def test_filter_dataframe_with_condition(test_data):
   condition = "age > 25"
   
   filtered_df = filter_dataframe(test_data, condition)
   
   assert filtered_df.count() == 2

def test_filter_dataframe_without_matching_condition(test_data):
   condition = "age > 40"
   
   filtered_df = filter_dataframe(test_data, condition)
   
   assert filtered_df.count() == 0

def test_filter_dataframe_with_invalid_condition(test_data):
   condition = "invalid_column > 30"
   
   with pytest.raises(Exception) as e:
       filter_dataframe(test_data, condition)
       
       assert str(e.value) == "cannot resolve 'invalid_column'"

def test_select_columns(spark_session):
   dataframe = spark_session.createDataFrame([(1, 'Alice', 25), (2, 'Bob', 30)], ['id', 'name', 'age'])
   
   selected_df = select_columns(dataframe, ['name', 'age'])
   
   assert selected_df.columns == ['name', 'age']
    
def test_select_columns_no_columns(spark_session):
   dataframe= spark_session.createDataFrame([(1,'Alice',25),(2,'Bob',30)],['id','name','age'])
   
   selected_df= select_columns(dataframe,[])
   
   assert len(selected_df.columns)==0
    
def test_select_columns_invalid_columns(spark_session):
  
 dataframe= spark_session.createDataFrame([(1,'Alice',25),(2,'Bob',30)],['id','name','age'])
 
 with pytest.raises(AttributeError): 
     select_columns(dataframe,['id','invalid_column'])

@pytest.fixture(scope="session")
def join_test_data(spark):
 
 data_1= [(1,"John"),(2,"Jane"),(3,"Alice")]
 data_2= [(1,"USA"),(2,"Canada"),(3,"UK")]
 
 return (
     spark.createDataFrame(data_1,['id','name']),
     spark.createDataFrame(data_2,['id','country'])
 )

 def test_join_dataframes(join_test_data):

 data_1,data_2= join_test_data
 
 joined_df= join_dataframes(data_1,data_2,"id")
 
 expected= [(1,"John","USA"),(2,"Jane","Canada"),(3,"Alice","UK")]
 
 result= joined_df.collect()
 
 for row_expected,row_result in zip(expected,result): 
     for value_expected,value_result in zip(row_expected,row_result): 
         assert value_expected==value_result

 def create_spark():
     return SparkSession.builder.master("local").appName("pytest-pyspark-local-testing").getOrCreate()

 def stop_spark(session): 
     session.stop()

@pytest.fixture(scope="session")
 def create_spark_fixture(): 
     session=create_spark() yield session stop_spark(session)

 @pytest.fixture(scope="module")

 def group_by_test_data(create_spark_fixture):

 data=[(1,'A',10),(2,'B',20),(3,'C',30)] 

 return create_spark_fixture.createDataFrame(data,['id','category','value'])

 def test_group_by_columns(group_by_test_data): 

grouped_df= group_by_columns(group_by_test_data,['category']) 

assert len(grouped_df.columns)==1 grouped_df=
group_by_columns(group_by_test_data,['category','value']) 

assert len(grouped_df.columns)== 2

 def cache_or_persist(create_spark_fixture):

 data=[(i,)for i in range (5)] 

schema=['value'] 

return create_spark_fixture.createDataFrame(data,schema)

@pytest.mark.parametrize("storage_level",
 [
"MEMORY_ONLY",
"MEMORY_AND_DISK",
"DISK_ONLY"])

 def test_cache_or_persist(cache_or_persist_storage_level): 

df=cache_or_persist cache_or_persist(df)

assert getattr(df.storageLevel,f"use{storage_level.replace('_AND_','And')}",False