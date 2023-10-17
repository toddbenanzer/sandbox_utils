andas as pd
import pytest

from your_module import calculate_total_observations, calculate_missing_values, count_non_missing_values, calculate_true_percentage, calculate_false_values, calculate_most_common_values, calculate_missing_prevalence, is_trivial_column, handle_missing_data, handle_infinite_data

# Test case 1: When the dataframe is empty
def test_calculate_total_observations_empty_df():
    df = pd.DataFrame()
    column_name = 'boolean_column'
    result = calculate_total_observations(df, column_name)
    assert result == 0

# Test case 2: When the boolean column doesn't exist in the dataframe
def test_calculate_total_observations_column_not_exist():
    df = pd.DataFrame({'other_column': [True, False, True]})
    column_name = 'boolean_column'
    result = calculate_total_observations(df, column_name)
    assert result == 0

# Test case 3: When the boolean column has no observations (all NaN values)
def test_calculate_total_observations_no_observations():
    df = pd.DataFrame({'boolean_column': [float('nan'), float('nan'), float('nan')]})
    column_name = 'boolean_column'
    result = calculate_total_observations(df, column_name)
    assert result == 0

# Test case 4: When the boolean column has some observations (True or False values)
def test_calculate_total_observations_with_observations():
    df = pd.DataFrame({'boolean_column': [True, False, True]})
    column_name = 'boolean_column'
    result = calculate_total_observations(df, column_name)
    assert result == 3
    
# Test case 5: When the boolean column has some observations (including NaN values)
def test_calculate_total_observations_with_nan_values():
    df = pd.DataFrame({'boolean_column': [True, False, float('nan')]})
    column_name = 'boolean_column'
    result = calculate_total_observations(df, column_name)
    assert result == 2


# Test case 1: Test with a dataframe containing missing values in the specified column
def test_calculate_missing_values_with_missing_values():
    df = pd.DataFrame({'col1': [True, False, None, False]})
    column_name = 'col1'
    
    assert calculate_missing_values(df, column_name) == 1

# Test case 2: Test with a dataframe without any missing values in the specified column
def test_calculate_missing_values_without_missing_values():
    df = pd.DataFrame({'col1': [True, False, True, False]})
    column_name = 'col1'
    
    assert calculate_missing_values(df, column_name) == 0

# Test case 3: Test with a dataframe that does not contain the specified column
def test_calculate_missing_values_column_not_found():
    df = pd.DataFrame({'col1': [True, False, True, False]})
    column_name = 'col2'

    with pytest.raises(ValueError):
        calculate_missing_values(df, column_name)


def test_count_non_missing_values():
    # Test case 1: column with no missing values
    column_1 = pd.Series([1, 2, 3, 4, 5])
    assert count_non_missing_values(column_1) == 5

    # Test case 2: column with some missing values
    column_2 = pd.Series([1, None, 3, None, 5])
    assert count_non_missing_values(column_2) == 3

    # Test case 3: column with all missing values
    column_3 = pd.Series([None, None, None, None, None])
    assert count_non_missing_values(column_3) == 0


def test_calculate_true_percentage_valid_column():
    # Create a boolean column with some True and False values
    column = pd.Series([True, False, True, True, False])

    # Call the function and get the true count and true percentage
    true_count, true_percentage = calculate_true_percentage(column)

    # Check if the true count is calculated correctly
    assert true_count == 3

    # Check if the true percentage is calculated correctly
    assert true_percentage == 60.0


def test_calculate_true_percentage_invalid_column():
    # Create a column with non-boolean values
    column = pd.Series([1, 2, 3, 4, 5])

    # Check if the function raises a ValueError when given an invalid column
    with pytest.raises(ValueError):
        calculate_true_percentage(column)



# Test case for a boolean column with no False values
def test_calculate_false_values_no_false():
    column = pd.Series([True, True, True])
    num_false, percentage_false = calculate_false_values(column)
    assert num_false == 0
    assert percentage_false == 0

# Test case for a boolean column with some False values
def test_calculate_false_values_some_false():
    column = pd.Series([True, True, False])
    num_false, percentage_false = calculate_false_values(column)
    assert num_false == 1
    assert percentage_false == pytest.approx(33.3333, rel=1e-4)

# Test case for a boolean column with all False values
def test_calculate_false_values_all_false():
    column = pd.Series([False, False, False])
    num_false, percentage_false = calculate_false_values(column)
    assert num_false == 3
    assert percentage_false == 100

# Test case for a non-boolean column
def test_calculate_false_values_non_boolean():
    column = pd.Series([1, 2, 3])
    with pytest.raises(ValueError):
        calculate_false_values(column)


# Test case 1: When all values in the column are the same
def test_calculate_most_common_values_all_same():
    df = pd.DataFrame({'column': [True, True, True, True]})
    expected_result = [True]
    
    assert calculate_most_common_values(df['column']) == expected_result

# Test case 2: When all values in the column are different
def test_calculate_most_common_values_all_different():
    df = pd.DataFrame({'column': [True, False, True, False]})
    expected_result = [True, False]
    
    assert calculate_most_common_values(df['column']) == expected_result

# Test case 3: When there are multiple values with the same maximum frequency
def test_calculate_most_common_values_multiple_max():
    df = pd.DataFrame({'column': [True, False, False, True, True]})
    expected_result = [True, False]
    
    assert calculate_most_common_values(df['column']) == expected_result

# Test case 4: When the column is empty
def test_calculate_most_common_values_empty_column():
    df = pd.DataFrame({'column': []})
    expected_result = []
    
    assert calculate_most_common_values(df['column']) == expected_result

# Test case 5: When the column has only one value
def test_calculate_most_common_values_single_value():
    df = pd.DataFrame({'column': [False]})
    expected_result = [False]
    
    assert calculate_most_common_values(df['column']) == expected_result


def test_calculate_missing_prevalence():
    # Test case 1: Empty column
    column = pd.Series([])
    assert calculate_missing_prevalence(column) == 0.0

    # Test case 2: No missing values
    column = pd.Series([True, False, True])
    assert calculate_missing_prevalence(column) == 0.0

    # Test case 3: All missing values
    column = pd.Series([None, None, None])
    assert calculate_missing_prevalence(column) == 1.0

    # Test case 4: Mixed missing and non-missing values
    column = pd.Series([True, None, False, None])
    assert calculate_missing_prevalence(column) == 0.5

    # Test case 5: Non-boolean values in column
    column = pd.Series(['A', 'B', 'C'])
    assert calculate_missing_prevalence(column) == 0.0


# Test case 1: Non-trivial column
def test_is_trivial_column_nontrivial():
    column = pd.Series([True, False, True])
    assert is_trivial_column(column) == False

# Test case 2: Trivial column with all True values
def test_is_trivial_column_all_true():
    column = pd.Series([True, True, True])
    assert is_trivial_column(column) == True

# Test case 3: Trivial column with all False values
def test_is_trivial_column_all_false():
    column = pd.Series([False, False, False])
    assert is_trivial_column(column) == True

# Test case 4: Empty column
def test_is_trivial_column_empty():
    column = pd.Series([])
    assert is_trivial_column(column) == False

# Test case 5: Column with mixed values
def test_is_trivial_column_mixed_values():
    column = pd.Series([True, False, True, False])
    assert is_trivial_column(column) == False

# Test case 6: Column with single value 'True'
def test_is_trivial_column_single_value_true():
    column = pd.Series([True])
    assert is_trivial_column(column) == True

# Test case 7: Column with single value 'False'
def test_is_trivial_column_single_value_false():
    column = pd.Series([False])
    assert is_trivial_column(column) == True


# Test case 1: When the dataframe is empty
def test_handle_missing_data_empty_df():
    df = pd.DataFrame()
    column_name = 'col1'
    impute_value = False
    result = handle_missing_data(df, column_name, impute_value)
    assert result.empty

# Test case 2: When the specified column does not exist in the dataframe
def test_handle_missing_data_column_not_exist():
    df = pd.DataFrame({'other_column': [True, False, True]})
    column_name = 'col1'
    impute_value = False
    with pytest.raises(KeyError):
        handle_missing_data(df, column_name, impute_value)

# Test case 3: When there are missing values in the specified column and impute value is False
def test_handle_missing_data_with_missing_values():
    df = pd.DataFrame({'col1': [True, False, None, False]})
    column_name = 'col1'
    impute_value = False
    expected_result = pd.Series([True, False, None, False])
    result = handle_missing_data(df, column_name, impute_value)
    assert result.equals(expected_result)

# Test case 4: When there are missing values in the specified column and impute value is True
def test_handle_missing_data_with_impute_values():
    df = pd.DataFrame({'col1': [True, False, None, False]})
    column_name = 'col1'
    impute_value = True
    expected_result = pd.Series([True, False, True, False])
    result = handle_missing_data(df, column_name, impute_value)
    assert result.equals(expected_result)

# Test case 5: When there are no missing values in the specified column
def test_handle_missing_data_no_missing_values():
    df = pd.DataFrame({'col1': [True, False, True]})
    column_name = 'col1'
    impute_value = False
    expected_result = df['col1']
    result = handle_missing_data(df, column_name, impute_value)
    assert result.equals(expected_result)


# Test case 1: Missing values in the specified column are replaced correctly with the impute value
def test_handle_infinite_data():
    # Create a test dataframe with infinite values
    df = pd.DataFrame({'A': [1, 2, float('inf'), float('-inf'), 5]})

    # Test case 1: Impute positive infinite value with 10
    expected_output_1 = pd.Series([1, 2, 10, float('-inf'), 5])
    assert handle_infinite_data(df['A'], 10).equals(expected_output_1)

    # Test case 2: Impute negative infinite value with -5
    expected_output_2 = pd.Series([1, 2, float('inf'), -5, 5])
    assert handle_infinite_data(df['A'], -5).equals(expected_output_2)

    # Test case 3: Impute both positive and negative infinite values with 0
    expected_output_3 = pd.Series([1, 2, 0, 0, 5])
    assert handle_infinite_data(df['A'], 0).equals(expected_output_3