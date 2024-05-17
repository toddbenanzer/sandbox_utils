
import pandas as pd
import pytest

# Test case 1: column with no unique categories
def test_calculate_unique_categories_no_unique():
    column = pd.Series(['A', 'A', 'A'])
    assert calculate_unique_categories(column) == 1

# Test case 2: column with one unique category
def test_calculate_unique_categories_one_unique():
    column = pd.Series(['A', 'A', 'B'])
    assert calculate_unique_categories(column) == 2

# Test case 3: column with multiple unique categories
def test_calculate_unique_categories_multiple_unique():
    column = pd.Series(['A', 'B', 'C'])
    assert calculate_unique_categories(column) == 3

# Test case 4: empty column
def test_calculate_unique_categories_empty():
    column = pd.Series([])
    assert calculate_unique_categories(column) == 0

# Test case 5: column with mixed data types
def test_calculate_unique_categories_mixed_data_types():
    column = pd.Series(['A', 1, True])
    with pytest.raises(TypeError):
        calculate_unique_categories(column)

@pytest.fixture
def sample_data():
    return pd.Series(['apple', 'banana', 'apple', 'orange', 'banana', 'banana'])

# Test case to check if the output is correct for a given input
def test_calculate_category_count(sample_data):
    expected_output = pd.Series([2, 3, 1], index=['banana', 'apple', 'orange'])
    assert calculate_category_count(sample_data).equals(expected_output)

# Test case to check if the output is a pandas Series
def test_calculate_category_count_output_type(sample_data):
    assert isinstance(calculate_category_count(sample_data), pd.Series)

# Test case to check if the output is empty for an empty input
def test_calculate_category_count_empty_input():
    empty_data = pd.Series([])
    assert calculate_category_count(empty_data).empty

# Test case to check if the output is correct for a single category input
def test_calculate_category_count_single_category():
    single_category = pd.Series(['apple'] * 10)
    expected_output = pd.Series([10], index=['apple'])
    assert calculate_category_count(single_category).equals(expected_output)

# Test case 1: Test with a column containing only one category
def test_calculate_category_percentage_single_category():
    # Create a column with a single category
    column = pd.Series(['A', 'A', 'A'])

    # Call the function
    result = calculate_category_percentage(column)

    # Check if the result is correct
    assert len(result) == 1  # There should be only one row in the result
    assert result['Category'].iloc[0] == 'A'  # The category should be 'A'
    assert result['Percentage'].iloc[0] == 100.0  # The percentage should be 100%

# Test case 2: Test with a column containing multiple categories
def test_calculate_category_percentage_multiple_categories():
    # Create a column with multiple categories
    column = pd.Series(['A', 'B', 'A', 'C', 'B'])

    # Call the function
    result = calculate_category_percentage(column)

    # Check if the result is correct
    assert len(result) == 3  # There should be three rows in the result
    assert sorted(result['Category'].tolist()) == ['A', 'B', 'C']  # The categories should be in alphabetical order
    assert sorted(result['Percentage'].tolist()) == [40.0, 40.0, 20.0]  # The percentages should be correctly calculated

# Test case 3: Test with an empty column
def test_calculate_category_percentage_empty_column():
    # Create an empty column
    column = pd.Series([])

    # Call the function
        result = calculate_category_percentage(column)

        # Check if the result is correct 
        assert len(result) == 0   # The result should be an empty dataframe 

        # Test case   :Test with a   containing NaN values 
        def   () -> None : 
        """Test handling of NaN values."""  
        df_ : DataFrame  
        df_["X"].fillna(df_[ "X"].mean(), inplace=True)
        
        def   (): 
            """Test that function correctly replaces NaN values."""  
            df_ : DataFrame  
            df_["Y"].replace(to_replace=np.nan, value="unknown", inplace=True)
