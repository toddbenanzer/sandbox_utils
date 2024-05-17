
import pandas as pd
import pytest

@pytest.fixture
def sample_df():
    data = {'col1': [True, False, True, True, False],
            'col2': [False, False, True, False, True]}
    return pd.DataFrame(data)

def test_count_true_values(sample_df):
    assert count_true_values(sample_df, 'col1') == 3
    assert count_true_values(sample_df, 'col2') == 2

def test_count_true_values_empty_df():
    df = pd.DataFrame()
    with pytest.raises(KeyError):
        count_true_values(df, 'col1')

def test_count_true_values_non_bool_column(sample_df):
    with pytest.raises(AttributeError):
        count_true_values(sample_df, 'col4')

def test_count_false_values_empty_column():
    column = pd.Series([])
    assert count_false_values(column) == 0

def test_count_false_values_no_false_values():
    column = pd.Series([True, True, True])
    assert count_false_values(column) == 0

def test_count_false_values_all_false_values():
    column = pd.Series([False, False, False])
    assert count_false_values(column) == 3

def test_count_false_values_mixed_true_and_false_values():
    column = pd.Series([True, False, True, False])
    assert count_false_values(column) == 2

def test_count_false_values_non_boolean_column():
    column = pd.Series([1, 2, 3])
    with pytest.raises(AttributeError):
        count_false_values(column)

def test_calculate_missing_values():
    column = pd.Series(dtype=bool)
    assert calculate_missing_values(column) == 0
    column = pd.Series([True, False, True])
    assert calculate_missing_values(column) == 0
    column = pd.Series([True, None, True])
    assert calculate_missing_values(column) == 1
    column = pd.Series([None, None, None])
    assert calculate_missing_values(column) == 3
    column = pd.Series([None] * 10000)
    assert calculate_missing_values(column) == 10000
    column = pd.Series([True] * 10000)
    assert calculate_missing_values(column) == 0

def test_count_empty_values_with_empty_column():
    column = pd.Series([], dtype=bool)
    assert count_empty_values(column) == 0

def test_count_empty_values_with_nonempty_column():
    column = pd.Series([True, False, np.nan, True])
    assert count_empty_values(column) == 1

def test_count_empty_values_with_inf_column():
    column = pd.Series([np.inf, -np.inf, True, False])
    assert count_empty_values(column) == 2

def test_count_empty_values_with_no_nan_or_inf_column():
    column = pd.Series([True, False, True, False])
    assert count_empty_values(column) == 0
    
def test_count_empty_values_with_mixed_column():
    column = pd.Series([True, False, np.nan, np.inf])
    assert count_empty_values(column) == 2

df_sample = pd.DataFrame({"col": [True, False, True,np.nan ,False]})

def test_calculate_missing_prevalence_no_missing_vals(df_sample):
   column=df_sample["col"]
   prevalence=calculate_missing_prevalence(column)

   #Assert
   assrt prevalence==0.4
   
   
import pandas as pd
import pytest

# Test case for an empty DataFrame
def test_calculate_boolean_variance_empty_df():
      df=pd.DataFrame({'boolean_column':[True ,False ,np.nan ,False ]})
      expected_result=calculate_boolean_variance(df,'boolean_column')
      
      #Assert
      expected_result==pytest.approx(0.54)

