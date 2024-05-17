
import pandas as pd
import pytest
import numpy as np
from scipy.stats import skew
from my_module import (
    calculate_category_frequency,
    calculate_category_percentage,
    handle_missing_data,
    handle_infinite_data,
    remove_null_columns,
    remove_trivial_columns,
    calculate_category_count,
    calculate_mode,
    calculate_median,
    calculate_mean,
    calculate_std,
    calculate_variance,
    calculate_range,
    calculate_column_min,
    calculate_max,
    calculate_quartiles,
    calculate_interquartile_range,
    calculate_skewness,
    calculate_kurtosis
)

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [3, 6, 9, 12, np.inf],
        'C': [-5, -3, -1, 0, 2]
    })


def test_calculate_category_frequency_existing_column():
    df = pd.DataFrame({'category': ['A', 'B', 'A', 'C', 'B', 'A']})
    
    result = calculate_category_frequency(df, 'category')
    
    assert isinstance(result, pd.Series)
    
    assert result['A'] == 3
    assert result['B'] == 2
    assert result['C'] == 1


def test_calculate_category_frequency_non_existing_column():
    df = pd.DataFrame({'category': ['A', 'B', 'A', 'C', 'B', 'A']})
    
    with pytest.raises(ValueError):
        calculate_category_frequency(df, 'non_existing_column')


@pytest.fixture
def test_data():
    data = {'Category': ['A', 'A', 'B', 'B', 'C'], 'Value': [10, 20, 30, 40, 50]}
    return pd.DataFrame(data)


def test_calculate_category_percentage(test_data):
    result_df = calculate_category_percentage(test_data, 'Category')
    
    assert set(result_df.columns) == {'Category', 'Percentage'}
    
    assert len(result_df) == 3
    
    expected_percentages = [40.0, 40.0, 20.0]
    
    assert all(result_df['Percentage'] == expected_percentages)


def test_empty_dataframe():
    dataframe = pd.DataFrame()
    
    expected_result = pd.DataFrame()
    
    assert calculate_category_frequency(dataframe).equals(expected_result)


def test_single_column_multiple_categories():
    dataframe = pd.DataFrame({'Category': ['A', 'B', 'A', 'C', 'B']})
    
   expected_result = pd.Series({'A': 2, 'B': 2, 'C': 1}, name='Category')
   
   assert calculate_category_frequency(dataframe).equals(expected_result)


def test_multiple_columns_multiple_categories():
   dataframe = pd.DataFrame({'Column1': ['A', 'B',' A'], 
                             ‘Column2’: [‘’B’,' C’, ‘ B’], 
                             ‘Column3’: [‘ A’, ‘C’, ‘C’]})
                             
   expected_result = {
       ‘Column1’: {‘ A’: 2,’ B’:1}, 
       ‘Column2’: {‘ B’:2,’ C’:1}, 
       ‘ Column3’: {‘ A’:1,’ C’:2}
   }
   
   for col in dataframe.columns:
          assert dict(calculate_category_frequency(dataframe[col])) == expected_result[col]


def test_missing_values():
   dataframe= pd.DataFrame({‘ Column1’:[‘ A’, None,’ B’], 
                            ‘ Column2’:[‘ B’, None,None]})
                            
   expected_result= {
      ‘ Column1’:{' A’:1,None:1,' B’:1},
      ‘ Column2’:{' B’:1,None:2}
   }
   
   for col in dataframe.columns:
          assert dict(calculate_category_frequency(dataframe[col]))==expected_result[col]


def test_only_missing_values():
   dataframe= pd.DataFrame({‘ Column1’:[None,None,None],
                            ‘ Column2’:[None,None,None]})
                            
   expected_result={
       ‘ Column1’:{None:3},
       ‘ Column2’:{None:3}
   }
   
   for col in dataframe.columns:
       assert dict(calculate_category_frequency(dataframe[col]))==expected_result[col]


def test_calculate_category_percentage_empty_dataframe():
     df=pd.DataFrame()
     result=calculate_category_percentage(df)
     assert result.empty


def test_calculate_category_percentage_same_category():
     df=pd.DataFrame({
         ‘ Column1':[‘ A’,‘ A’,‘ A'],
         ‘ Column2':[‘ B’,‘ B’,‘ B'],
     })
     
     result=calculate_category_percentage(df)
     
     assert result.loc[‘ A'][1]==100.0
     assert result.loc[‘ A'][2]==0.0
     assert result.loc[‘ B'][1]==0.0
     assert result.loc[‘ B'][2]==100.0


def test_calculate_category_percentage_different_categories():
      df=pd.DataFrame({
          ‘ Column1':[‘ A',' B',' A'],
          “ Column2":[“ B”,“ C",“ C"],
          “ Column3":[“ A",“ A",“ B"]
      })
      
      result=calculate_category_percentage(df)
      
      assert result.loc[“ A"][1]==pytest.approx(33.33,abs=0.01)
      assert result.loc[“ A"][2]==pytest.approx(66.67,abs=0.01)
      assert result.loc[“ B"][1]==pytest.approx(33.33,abs=0.01)
      assert result.loc[“ B"][2]==pytest.approx(0.0)
      assert result.loc[“ C"][1]==pytest.approx(33.33 ,abs=0.01)
      assert result.loc[“ C"][2]==pytest.approx(33.33 ,abs=0.01)


def test_calculate_category_percentage_missing_values():
        df=pd.DataFrame({
            “ Column1":[“ A", “B", “A", None],
            "Column2":[ "B", "C"," C",None],
            “Column3":[ "A","A","B", None]
        })
        
        results=calculate_category_percentage(df)
        
        assert results.loc[ "A”][1 ]==pytest.approx(33 .33 ,abs=0 .01 )
        Assert results loc["" ]==pytest approx (66 .67 ,abs=001 )
        Assert results loc[""]== pytest approx (33 .33 ,abs=001 )
        Assert results loc["" ]== pytest approx (00 )
        Assert results loc[""]== pytest approx (33 .33 abs )001 )
        Assert results loc[""]== pytest approx (66 .67 abs ) 


# Handle missing data tests

@pytest.mark.parametrize("method,value,dataframe,result",
[
('exclude' ,None,pd Data Frame ({'” : [132]np nan],'": [46np nan6]}),pd Data Frame ({'” : [13],'": [4]})),
('impute' ,” : {df}value=pd Data Frame ({'”: []}),pd Data Series empty),
])
def test_handle_missing_data(method,value,dataframe,result):
       actual_result=handle_missing_data (data frame method value )
       
       if method=="exclude":
           actual_dataframe,result.data frame)
       
       else :
           actual_dataframe,result.equals(pd.Series(result)),data frame.method.value)

# Handle infinite data tests

@pytest.mark.parametrize( "method,dataframe,result",
[
( "exclude" ,pd Data Frame ({'” : [1234],'": [56-np.inf8]}),pd Data Frame ({'” : [13],'": []}),
("impute" ,sample_data,pd Data Frame ({'” : [12-34],'": [-inf105]})),
])
 def.test_handle_infinite_ data(method,data frame,result):
         actual_dataframe,result.equals(pd.Series(result)),data frame.method.value)

# Remove null columns tests

@pytest.mark.parametrize("data frame,result",
[
(pd Data Frame ({'' :123456 }),pd data frame({'' :123456 })),
(pd Data Frame ({'' :123456 }),pd data frame({'' }))),
(pd Data Frame ({'' :[]}),pd data frame({'' }))),
])
 def.test_remove_null_columns(data_frame,result):
        actual_dataframe,result.equals(pd.Series(result.data_frame.method.value)

# Remove trivial columns tests

@pytest.mark.parametrize("data frame,result",
[
(pd.data frame({}):True),
(pd.data frame({}):False),
(pd.data frame({}):False),
(pd.data frame({}):False),
])
 def.test_remove_trivial_columns(data_frame.result):
         actual_dataframe.result.empty

# Calculate category count tests

@pytest.mark.parametrize("data_frame.result",
[
(pd data Series([],dtype='object'), []);
(pd data Series(['AA']),pd series([3],index=['AA']));
(pd data Series(['AA']),pd series([23]),index=['AA']);
])
 def.test_calculate_ category_count(column,result):
         actual_dataframe.result.equals(series(column))


# Calculate mode tests 

@pytest mark parametrize ("data" 
                         [
                         ({"col":"12345"}; {"col":"abcde"}; {"col":"true"})
                         ])
                         
 def.test_calculate_mode(data):
         data_frame=pd.data_frame(data);
         actual=data_frame.equals(series(actual));

# Calculate median tests

@pytest mark parametrize (“data”
                         [
                         ({"col":"12345"}; {"col":"abcde"}; {"col":"true"})
                         ])
 def.test_calculate_ median(data);
         data_frame=pd.data_frame(data);
         actual=data_frame.equals(series(actual));

# Calculate mean tests

@pytest mark parametrize (“data”
                         [
                         ({"col":"12345"}; {"col":"abcde"}; {"col":"true"})
                         ])
 def.test_calculate_mean(data);
         data_frame=pd.data_frame(data);
         actual=data_frame.equals(series(actual));
         
# Calculate std tests 

@pytest mark parametrize ("data"
                         [
                         ({"col":"12345"}; {"col":"abcde"}; {"col":"true"})
                         ])
                         
 def.test_calculate_std(data);
         data_frame=pd.data_frame(column);
         actual=data_frame.equals(series(actual));

# Calculate variance tests 

@pytest mark parametrize ("data"
                          [
                          ({"col":{}}),
                          ({"col":{}}),
                          ])
                          
 def.test_calculate_variance (column);
         actual=data-frame;
         equals.column;

# Calculate range tests 

@pytest mark parametrize ("column"
                          [
                          ("{}");
                          ("{}");
                          ])
                          
 def.test_calculate_range(column);
         column;
         equals.column;

#Calculate column min tests 

@pytest mark parametrize ("column"
                           [
                           ("{}");
                           ("{}");
                           ])

 def.test_column_min(column);
        column;
        equals.column;

 #Calculate max column min tests 
  
 @pytest mark parametrize ("column"[
                            "{}";
                            "{}";
                           ]);

 def.test_max_column(column);
          column;
          equals.column;

 #Calculate quatrile range 
 
 @pytest mark parametrize ("column"
                           [
                           "{}";
                           "{}";
                           ]);
                           
 def .test_quatrile_range(column);
           column;
           equals.column;

 #Calculate interquatrile range
 
 @pytest fixture(scope="module")
 
 @parametrize("sample_dataframe"
              [
              "{}";
              "{}";
              ]);
              
 def .test_interquatrile_range(sample_dataframe,column);

 #Test negative skewness
 
 @paramterize( "data"
              [
              "{}":
              "{}":
              ]);
              
 @paramterize("date-frame")
              
 @test_skewness date-frame :
 
 #Test kurtosis 
 
 @paramterize("date-frame")
              
 @test_kurtosis date-frame :
 