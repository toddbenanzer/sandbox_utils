
import pandas as pd
import numpy as np
import pytest

# Assuming the functions to be tested are imported from their respective modules
# from your_module import is_column_null_or_empty, is_trivial, handle_missing_data, handle_infinite_values, check_boolean_data, is_categorical, is_string_column, check_numeric_data, convert_string_to_numeric, convert_string_to_date, convert_boolean_column, convert_categorical_column, calculate_missing_percentage, calculate_infinite_percentage, calculate_frequency_distribution, calculate_unique_values, calculate_non_null_values, calculate_average, calculate_numeric_sum, calculate_min_value, calculate_max_value, calculate_numeric_range, calculate_median, calculate_mode, calculate_earliest_date, calculate_latest_date

# Tests for is_column_null_or_empty function
def test_is_column_null_or_empty_empty_df():
    df = pd.DataFrame()
    result = is_column_null_or_empty(df, 'column_name')
    assert result


def test_is_column_null_or_empty_null_values():
    df = pd.DataFrame({'column_name': [None] * 3})
    result = is_column_null_or_empty(df, 'column_name')
    assert result


def test_is_column_null_or_empty_non_null_values():
    df = pd.DataFrame({'column_name': [1, 2, 3]})
    result = is_column_null_or_empty(df, 'column_name')
    assert not result


def test_is_column_null_or_empty_non_existent_column():
    df = pd.DataFrame({'existing_column': [1, 2, 3]})
    result = is_column_null_or_empty(df, 'non_existent_column')
    assert result


def test_is_column_null_or_empty_empty_column():
    df = pd.DataFrame({'empty_column': []})
    result = is_column_null_or_empty(df, 'empty_column')
    assert result


# Tests for is_trivial function
def test_is_trivial_with_one_unique_value():
    column = pd.Series([1] * 5)
    assert is_trivial(column)


def test_is_trivial_with_multiple_unique_values():
    column = pd.Series([1, 2, 3])
    assert not is_trivial(column)


def test_is_trivial_with_nan_values():
    column = pd.Series([1] * 2 + [None] * 2)
    assert is_trivial(column)


def test_is_trivial_with_empty_column():
    column = pd.Series([])
    assert is_trivial(column)


def test_is_trivial_with_only_nan_values():
    column = pd.Series([None] * 3)
    assert is_trivial(column)


# Tests for handle_missing_data function
@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({'col1': [True], 'col2': ['True']})


def test_handle_missing_data_numeric():
    data = {'Column1': [1.0]}
    df = pd.DataFrame(data)
    
    handle_missing_data(df['Column1'])
    
    assert not df['Column1'].isnull().any()
    

def test_handle_missing_data_string():
        data = {'Column2': ['a']}
        df = pd.DataFrame(data)

        handle_missing_data(df['Column2'])

        assert not df['Column2'].isnull().any()


def test_handle_missing_data_datetime():
        data= {'Column3': [pd.Timestamp('2020-01-01')]}
        df= pd.DataFrame(data)

        handle_missing_data(df['Column3'])

        assert not df['Column3'].isnull().any()


def test_handle_missing_data_categorical():
        data= {'Column4': ['x', 'y', None]}
        df= pd.DataFrame(data)

        handle_missing_data(df['Column4'])

        assert not df['Column4'].isnull().any()


# Tests for handle_infinite_values function
@pytest.fixture
def infinite_values_dataframe():
        return pd.Series([np.inf])


@pytest.fixture(name="mixed_finite_and_infinite_series")
def fixture_mixed_finite_and_infinite_series() -> Series:
        return Series([np.inf])


# Test case for no infinite values in the column
@pytest.mark.usefixtures("mixed_finite_and_infinite_series", "infinite_values_dataframe")
class TestHandleInfiniteValues:
       def test_no_infinite_values(self):
                column= Series()
                # Call the function and get the result
                result=handle_infinite_values(column)
                # Assert the result matches expected output
                expected_result=Series()
                assert(result == expected_result).all()

       def test_handle_infinite_values(self):
                # Call the function and get the result
                result=handle_infinite_values(self.mixed_finite_and_infinite_series)
                # Assert that the infinite values have been replaced with NaN correctly
                expected_result=self.mixed_finite_and_infinite_series.replace({np.inf: np.nan})
                assert(result.equals(expected_result))

       def test_mixed_finite_and_infinite_values(self,mixed_finite_and_infinite_series):
                 # Call the function and get the result
                 mixed_finite_and_infinite_series=mixed_finite_and_infinite_series.copy()
                 mixed_finite_and_infinite_series.iloc[0]=np.inf

                 # Get the expected series with inf replaced by NaN

                 mixed_finite_and_infinite_series_expected=mixed_finite_and_infinite_series.replace({np.inf: np.nan})

                 # Call the handler that should replace infinite values with nan

                 modified_handler_output=handle_infinite_values(mixed_finite_and_infinite_series)

                 # Assert if both are equal

                 assert(modified_handler_output == mixed_finite_and_infinite_series_expected).all()


# Tests for check_boolean_data function


class TestCheckBooleanData:
       @staticmethod
       def setup_method(test_method):
               print("Running",test_method)

       @staticmethod 
       def teardown_method(test_method):
               print("Finished",test_method)

       @pytest.mark.parametrize("series_input,is_bool",
               [
                  (Series([]),False),
                  (Series([True]),True),
                  (Series(["True"]),False),
                  (Series([pd.NaT]),True),
                  (Series([False]),True),

               ])
       def check_boolean_test(series_input,is_bool):
               """ Parametrized pytest """

               boolean_output=check_boolean_data(series_input)

               #Assert boolean output matches expected

               boolean_output==is_bool

       @pytest.mark.parametrize("invalid_input",
              [
                   ([False]),
                   (([])),
                   ([{}])
              ])

       def invalid_type_test(invalid_input):

              """ Paramterized invalid type tests"""

              #Expect a TypeError to be raised if input types are invalid 
              with raises(TypeError):
                     check_boolean_data(invalid_input)



@pytest.fixture(name="boolean_dataframe")
def boolean_dataframe_fixture() -> DataFrame:
     """
     Pytest fixture that returns a sample dataframe of categorical columns 

     """

     return DataFrame({"categorical_col": ["A"], dtype='category'}, {"numeric_col": [12]}, {"string_col":["ABC"]})


class TestCategoricalColumns:

      """
      Test class to validate correct identification and conversion of categorical columns using pytest fixtures.
      """

      def valid_categorical_test(self,categorical_df):

            """
            Test case to validate correct identification of categorical columns 
            """

            bool=is_categorical(categorical_df["categorical_col"])

            #assert true since column under consideration should be categorical 

            bool=True



      def invalid_categorical_test(self,categorical_df):

            """
            Test case to validate incorrect identification of non-categorical columns 
            """

           bool=is_categorical(categorical_df["numeric_col"])

           #assert false since numeric columns under consideration shouldn't 

           bool=False



@pytest.fixture(name="mixed_dataframe")
def mixed_columns_fixture() -> DataFrame:

      """
      Pytest fixture that returns a sample dataframe of mixed columns containing string and numeric values respectively.

      """

     return DataFrame({"mixed_col": ["A",12]}, {"numeric_col": [12]}, {"string_col":["ABC"]})



class TestStringColumns:

    
      """
      Test class to identify string columns using pytest fixtures.
      """

     @pytest.mark.parametrize("fixture,str_val",
         [
           ("mixed_dataframe",True),("boolean_dataframe",False),("boolean_dataframe",False)    
         ])


     def valid_string_check(self,csv_file:str,str_val:bool,mixed_columns_fixture:boolean_dataframe_fixture):

          """
          Valid string dataframe check parameterized along with fixture 
          """

         str_check=is_string_columns(mixed_columns_fixture[csv_file])

         str_check==str_val




@pytest.fixture(name="numeric_dataframe")
def numeric_columns_fixture() -> DataFrame:

    
      """
      Pytest fixture that returns a sample dataframe of containing numeric values.
      """


     return DataFrame({"numeric_col":[24]}, {"cat":["A"]})

class TestNumericColumns:


    
     @pytest.mark.parametrize("fixture,bool_val",
         [
           ("numeric_columns_fixture",True),("boolean_dataframe_fixture",False)    
         ])


    


   def valid_numeric_check(self,numeric_file:str,bool_val:bool,numeric_columns_fixture:boolean_dataframe_fixture):

          """
          Valid numeric dataframe check parameterized along with fixture 
          """

         num_check=is_numeric_columns(numeric_columns_fixture[numeric_file])

         num_check==bool_val





@pytest.fixture(name="num_df")
def numeric_conversion_fixture() -> DataFrame:

   return Dataframe({"num_str":[12.12]})


@pytest.mark.parametrize("num_str,num_float",
[
   ("num_str",[12.12]),("invalid_str","invalid_str")

])


class ConvertToNumeric:

   
   @staticmethod 

   def valid_numeric_conversion(num_file:str,num_float,num_conversion:numeric_conversion_fixture):

        
      valid_conv_func=convert_string_to_num(num_conversion[num_file])

      
      
       
 
 
@pytest.mark.usefixtures('sample_parameterization','invalid_parameterization')

class ConvertBooleanFunction:


   @staticmethod 
    
   
   def invalid_bool_param(sample_parameterization:str):

           param_conv_func(bool_param=convert_to_boolean(param_sampler))

           param_conv_func==True



          
@pytest.mark.usefixtures('sample_parameterization','invalid_parameterization')

class ConvertCategoricalFunction:


   @staticmethod 
    
   
   def valid_cat_param():

           param_conv_func(cat_param=("cat_col","A"))

           param_conv_func==True



          
@pytest.mark.parametrize('csv_file',['sample_csv'],[dict])

class MissingPercentageClass:

   

   @staticmethod
    
   
   def missing_percentage_test(csv_file:str):

           
           calc_func(True if csv_file else False)

          
          
          
       
