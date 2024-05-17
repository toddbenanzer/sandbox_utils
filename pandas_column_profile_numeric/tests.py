
import pandas as pd
import pytest

# Test functions for calculate_mean
def test_calculate_mean_numeric_column():
    df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
    column_name = 'A'
    result = calculate_mean(df, column_name)
    assert result == 3.0

def test_calculate_mean_non_numeric_column():
    df = pd.DataFrame({'A': ['a', 'b', 'c']})
    column_name = 'A'
    with pytest.raises(TypeError):
        calculate_mean(df, column_name)

def test_calculate_mean_nonexistent_column():
    df = pd.DataFrame({'A': [1, 2, 3]})
    column_name = 'B'
    with pytest.raises(KeyError):
        calculate_mean(df, column_name)

# Test functions for calculate_median
def test_calculate_median_odd():
    data = {'A': [1, 2, 3, 4, 5]}
    df = pd.DataFrame(data)
    result = calculate_median(df, 'A')
    assert result == 3.0

def test_calculate_median_even():
    data = {'A': [1, 2, 3, 4]}
    df = pd.DataFrame(data)
    result = calculate_median(df, 'A')
    assert result == 2.5

def test_calculate_median_single():
    data = {'A': [5]}
    df = pd.DataFrame(data)
    result = calculate_median(df, 'A')
    assert result == 5.0

def test_calculate_median_empty_df():
    df = pd.DataFrame()
    with pytest.raises(KeyError):
        calculate_median(df, 'A')

def test_calculate_median_non_numeric():
    data = {'A': ['a', 'b', 'c']}
    df = pd.DataFrame(data)
    with pytest.raises(TypeError):
        calculate_median(df, 'A')

# Test functions for calculate_mode
def test_calculate_mode_empty_dataframe():
    dataframe = pd.DataFrame()
    column_name = "column"
    assert calculate_mode(dataframe, column_name) is None

def test_calculate_mode_invalid_column():
    data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    dataframe = pd.DataFrame(data)
    column_name = "column"
    assert calculate_mode(dataframe, column_name) is None

def test_calculate_mode_valid_column():
     data = {'A': [1, 2, 3, 2, 2], 'B': [4, 5, 6, 7 ,8]}
     dataframe= pd.DataFrame(data)
     column_name= "A"
     assert calculate_mode(dataframe,column_name)==2


# Test functions for calculate_quartiles
def test_calculate_quartiles_simple():
      df=pd.DataFrame({'column':[1 ,2 ,3 ,4 ,5]})
      quartiles=calculate_quartiles(df['column'])
      assert quartiles[0.25]==1.75
      assert quartiles[0.5]==3.0
      assert quartiles[0.75]==4.25


def test_calculate_quartiles_empty():
      df=pd.dataframe({'column':[]}) 
      with pytest.raises(Exception) as e:
          calculate_quartiles(df['column'])
          assert str(e.value)=="No data to calculate quartiles"


def test_calculate_quartiles_single_value():
     df=pd.dataframe({'column':[10]})
     quartiles=calculate_quartiles(df['column'])
     assert quartiles[0.25]==10.0 
     assert quartiles[0.5]==10.0 
     assert quartiles[0.75]==10.0 


# Test function for standard deviation
def test_calculate_standard_deviation_valid_column():
      data={'col1':[1 ,2 ,3 ,4 ,5]} 
      df=pd.dataframe(data)
      std_dev=calculate_standard_deviation(df ,'col1')
      assert std_dev==np.std(df['col1'])


# Test function for variance calculation 
def mean=df.mean()
std=data.std()


# Test functions for variance calculation
 def test_handle_infinite_values_no_infinite_values(): 
       column=pd.series([1 ,2 ,3 ,4])
       result,num_infinite_values,num_remaining_infinite_values=handle_infinite_values(column)
       expected_result=pd.series([1,np.nan])
       expected_result=pd.series([1,np.nan])
       expected_result=pd.series([1,np.nan])
       expected_result=pd.series([1,np.nan])


# Test function for removing outliers from a Dataframe with a numeric column.
df=pd.dataframe({' A':[1 ,100]})  


#Test case for impute_missing_values 
 def test_impute_missing_values_existing_column(sample_dataframe): 
        column_name=' A' 
        result=impute_missing_values(sample_dataframe,column_name) 
        
        # Check if the returned p-value is within a certain range (e.g., between -infinity and +infinity )  
        assert p_value >=-infinity and p_value<=+ infinity 

    
if __name__==' __main__':
pytest.main()
