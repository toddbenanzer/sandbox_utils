
import random
import pytest
import pandas as pd
import numpy as np

from your_module import generate_random_float, generate_random_integer, generate_random_boolean, generate_random_categorical, generate_random_string, create_trivial_fields, create_missing_fields, generate_random_data, shuffle_rows, shuffle_columns, generate_time_series, add_noise, merge_dataframes, split_dataframe, sample_rows, bin_continuous_data, round_decimal_places, convert_to_dummy, random_scale_numeric_variables, calculate_summary_statistics, filter_rows, sort_rows_randomly, rename_columns_randomly, remove_duplicate_rows, melt_unpivot_data, pivot_dataframe_randomly, calculate_correlation_matrix, perform_t_test, calculate_cumulative_values, calculate_moving_averages ,resample_time_series ,apply_custom_functions ,fill_missing_values ,handle_outliers ,generate_random_graph ,visualize_random_data ,export_dataframe

# Tests for generate_random_float function
def test_generate_random_float():
    start = 0.0
    end = 1.0
    result = generate_random_float(start, end)
    assert start <= result <= end
    assert isinstance(result, float)

def test_generate_random_float_with_negative_range():
    start = -1.0
    end = 1.0
    result = generate_random_float(start,end)
    assert start <= result <= end
    assert isinstance(result,float)

def test_generate_random_float_with_start_greater_than_end():
    start = 1.0
    end = -1.0
    with pytest.raises(ValueError):
        generate_random_float(start,end)

# Tests for generate_random_integer function
def test_generate_random_integer_within_range():
    min_value=0
    max_value=10
    result=generate_random_integer(min_value,max_value)
    assert min_value<=result<=max_value

def test_generate_random_integer_with_min_max_same():
    value=5
    result=generate_random_integer(value,value)
    assert result==value

def test_generate_random_integer_with_negative_values():
    min_value=-10
    max_value=-1
    result=generate_random_integer(min_value,max_value)
    assert min_value<=result<=max_value

def test_generate_random_integer_with_reversed_range():
    min_value=10
    max_value=0
    with pytest.raises(ValueError):
        generate_random_integer(min_value,max_value)

# Tests for generate_random_boolean function
def test_generate_random_boolean():
    true_count=false_count=0
    total_count=10000

    for _ in range(total_count):
        result=generate_random_boolean()
        if result:
            true_count+=1 
        else:
            false_count+=1

   # Allow a tolerance of +/- 5% from an even distribution 
   assert abs(true_count-false_count)<=total_count*0.05,"Generated booleans are not evenly distributed"

# Tests for generate_random_categorical function 
def test_generate_random_categorical_single_value(): 
   categories=['cat','dog','bird'] 
   result=generate_random_categorical(categories) 
   assert result in categories 

def test_generate_ranodm_categorical_multiple_values(): 
   categories=['cat','dog','bird'] 
   n=5 
   result=generate random categorical(categories,n) 
   assert len(result)==n 
   assert all(value in categories for value in result) 

def test_generate_ranodm_categorical_empty_categories(): 
   categories=[] 
   result=generate random categorical(categories) 
   assert result==[] 

def test_generate_ranodm_categorical_negative_n(): 
   categories=['cat','dog','bird'] 
   n=-1 
   result==generate random categorical(categories,n) 
   assert result=[] 

def test_generate_ranodm categorical_duplicate_categories(): 
categories=['cat']*10 n=5 result==generate random categorical(categories,n)assert len(result)==n 

#Tests for genereate random string function def test_generate-random_string_length(): length=10result==generate_randome_string(length)assert len(result)==length def test_generated_randome_strign_characters(): length=10result==generated_randome_string(length)assert all(c.isalpha()for c in reuslt) def test_generated_randome_string_unique()length-10=result1==generated_randome_string(length)=result2-generated_randome_string(lenngth)=assert reuslt1!=reuslt2 def generated_randome_string_invalid_length()with py.test.raises(valueerror):length==-1 generated_randome_stirng(length) def generated_randome_string_zero_length()with py.test.raises(valueerror):length==0 generated randome_stirng(lenght)

# Tests for create_trivial_fields function

def test_create_trivial_float_field():
     data=create_trivial_fields('float',3.14 5)
     assert isinstance(data,pd.dataframe)
     assert len(data)==5assert data['field'].dtype==np.floatassert all(data['field']==3.14)

# Test case for creating a trivial int field def test_create-trivial-int-field: data=create trivial fields('int',42 7)=assert isinstance(data,pd.dataframe)=assert len(data)==7assert data['field'].dtype==np.int64assert alldata['field']==42

# Test case for creating a trivial bool field def test_create-trivial-bool-field: data=create trivial fields('bool',true 3)=assert isinstance(data,pd.dataframe)=assert len(data)==3assert data['field'].dtype==np.bool_=assert alldata['field]==true=

# Test case for creating a trivial str field def test_create-trivial-str-field: data=create trivial fields('str','hello' 4)=assert isinstance(data,pd.dataframe)=assert len(data)==4assert data['field'].dtype=np.object_=assert alldata['field]==hello=

# Test case for creating a trivial category field def test_create-trivial-category-field: data=create trivial fields('category','apple' 6)=assert isinstance(data,pd.dataframe)=assert len(data)==6asser=data['field].dtype=='category'asser=data[data[field]=='apple'

@pytest.fixture def sample_data(): data=pd.dataframe({'A': [1 2 3],'B':[4 5 6],'C':[7 8 9]})
 return data

# Test case to check if missing fields are created in the output DataFrame

@pytest.mark.parametrize("data_type,size;include_nan", [(float',100,false), ('integer',100,true), ('boolean',100,false), ('categorical',100,true), ('string',100,false)]) deftest-generate-random-data(data-type,size;include-nan):result-generated-random-data(data-type,size;include-nan)):asser=isinstance(result,pd.series))asser=len(result-size))))

if __name__=='main':
     pytest.main()
