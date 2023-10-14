andas as pd
import pytest

@pytest.fixture
def sample_dataframe():
    # Create a sample dataframe with missing values
    data = {'A': [1, 2, None, 4, 5],
            'B': [None, 2, 3, None, 5],
            'C': [1, 2, 3, 4, None]}
    df = pd.DataFrame(data)
    return df

def test_count_nonempty_strings(sample_dataframe):
    # Test with a column that has non-empty strings
    column = sample_dataframe['A']
    assert count_nonempty_strings(column) == 5

    # Test with a column that has empty strings
    column = sample_dataframe['B']
    assert count_nonempty_strings(column) == 2

    # Test with an empty column
    column = pd.Series([])
    assert count_nonempty_strings(column) == 0

    # Test with a column that has null values
    column = sample_dataframe['C']
    assert count_nonempty_strings(column) == 3


def test_calculate_percentage_empty_strings(sample_dataframe):
    # Test with a column that has no empty strings
    column = sample_dataframe['A']
    assert calculate_percentage_empty_strings(column) == 0.0

    # Test with a column that has all empty strings
    column = sample_dataframe['B']
    assert calculate_percentage_empty_strings(column) == 100.0

    # Test with a column that has a mix of empty and non-empty strings
    column = sample_dataframe['C']
    assert calculate_percentage_empty_strings(column) == 20.0


def test_calculate_min_string_length(sample_dataframe):
     # Test with a dataframe that has non-empty string values
     df = pd.DataFrame({'Column1': ['Apple', 'Banana', 'Cherry']})
     assert calculate_min_string_length(df['Column1']) == 5

     # Test with a dataframe that has empty string values
     df = pd.DataFrame({'Column1': ['', '', '']})
     assert calculate_min_string_length(df['Column1']) == 0

     # Test with a dataframe that has null values
     df = pd.DataFrame({'Column1': [None, None, None]})
     assert calculate_min_string_length(df['Column1']) is None

     # Test with an empty dataframe
     df = pd.DataFrame()
     assert calculate_min_string_length(df['Column1']) is None


def test_calculate_max_string_length(sample_dataframe):
    # Test with a dataframe that has non-empty string values
    df = pd.DataFrame({'Column1': ['Apple', 'Banana', 'Cherry']})
    assert calculate_max_string_length(df['Column1']) == 6

    # Test with a dataframe that has empty string values
    df = pd.DataFrame({'Column1': ['', '', '']})
    assert calculate_max_string_length(df['Column1']) == 0

    # Test with a dataframe that has null values
    df = pd.DataFrame({'Column1': [None, None, None]})
    assert calculate_max_string_length(df['Column1']) is None

    # Test with an empty dataframe
    df = pd.DataFrame()
    assert calculate_max_string_length(df['Column1']) is None


def test_calculate_average_string_length(sample_dataframe):
    # Test with a column that has non-empty strings
    column = sample_dataframe['A']
    assert calculate_average_string_length(column) == 5.666666666666667

    # Test with a column that has empty strings
    column = sample_dataframe['B']
    assert calculate_average_string_length(column) == 0.0

    # Test with an empty column
    column = pd.Series([])
    assert calculate_average_string_length(column) == 0.0

    # Test with a column that has null values
    column = sample_dataframe['C']
    assert calculate_average_string_length(column) == pytest.approx(2.0)


def test_calculate_non_empty_percentage(sample_dataframe):
    # Test with a column that has non-empty strings
    column = sample_dataframe['A']
    assert calculate_non_empty_percentage(column) == 100.0

    # Test with a column that has empty strings
    column = sample_dataframe['B']
    assert calculate_non_empty_percentage(column) == 60.0

    # Test with an empty column
    column = pd.Series([])
    assert calculate_non_empty_percentage(column) == 0.0

    # Test with a column that has null values
    column = sample_dataframe['C']
    assert calculate_non_empty_percentage(column) == pytest.approx(80.0)


def test_count_unique_values():
     # Create test data
     data = {'Column1': ['Apple', 'Banana', 'Apple', 'Orange', 'Banana']}
     df = pd.DataFrame(data)
    
     # Call the function under test
     unique_count = count_unique_values(df['Column1'])
    
     # Check the result
     assert unique_count == 3


def test_calculate_most_common_values(sample_dataframe):
    # Test with a column that has non-empty values
    column = sample_dataframe['A']
    assert calculate_most_common_values(column) == [5, 4, None, 2, 1]

    # Test with a column that has empty values
    column = sample_dataframe['B']
    assert calculate_most_common_values(column) == [5, 3, None, 2]

    # Test with an empty column
    column = pd.Series([])
    assert calculate_most_common_values(column) == []

    # Test with a column that has null values
    column = sample_dataframe['C']
    assert calculate_most_common_values(column) == [4, 3, 2, None, 1]


def test_calculate_missing_prevalence(sample_dataframe):
    # Test with a column that has non-empty values
    column = sample_dataframe['A']
    assert calculate_missing_prevalence(column) == 40.0

    # Test with a column that has no missing values
    column = sample_dataframe['C']
    assert calculate_missing_prevalence(column) == 0.0

    # Test with an empty column
    column = pd.Series([])
    assert calculate_missing_prevalence(column) == 0.0

    # Test with a column that has all missing values
    column = sample_dataframe['B']
    assert calculate_missing_prevalence(column) == 60.0


def test_calculate_empty_string_prevalence(sample_dataframe):
     # Test with a column that has non-empty values
     column = sample_dataframe['A']
     assert calculate_empty_string_prevalence(column) == 0.0

     # Test with a column that has no empty strings
     column = sample_dataframe['C']
     assert calculate_empty_string_prevalence(column) == 0.0

     # Test with an empty column
     column = pd.Series([])
     with pytest.raises(ValueError):
         calculate_empty_string_prevalence(column)

     # Test with a column that has empty strings
     column = sample_dataframe['B']
     assert calculate_empty_string_prevalence(column) == 40.0


def test_calculate_min_string_length(sample_dataframe):
    # Test with a dataframe that has non-empty string values
    df = pd.DataFrame({'Column1': ['Apple', 'Banana', 'Cherry']})
    assert calculate_min_string_length(df['Column1']) == 5

    # Test with a dataframe that has empty string values
    df = pd.DataFrame({'Column1': ['', '', '']})
    assert calculate_min_string_length(df['Column1']) == 0

    # Test with a dataframe that has null values
    df = pd.DataFrame({'Column1': [None, None, None]})
    assert calculate_min_string_length(df['Column1']) is None

    # Test with an empty dataframe
    df = pd.DataFrame()
    assert calculate_min_string_length(df['Column1']) is None


def test_calculate_max_string_length(sample_dataframe):
    # Test with a dataframe that has non-empty string values
    df = pd.DataFrame({'Column1': ['Apple', 'Banana', 'Cherry']})
    assert calculate_max_string_length(df['Column1']) == 6

    # Test with a dataframe that has empty string values
    df = pd.DataFrame({'Column1': ['', '', '']})
    assert calculate_max_string_length(df['Column1']) == 0

    # Test with a dataframe that has null values
    df = pd.DataFrame({'Column1': [None, None, None]})
    assert calculate_max_string_length(df['Column1']) is None

    # Test with an empty dataframe
    df = pd.DataFrame()
    assert calculate_max_string_length(df['Column1']) is None


def test_calculate_average_string_length(sample_dataframe):
     # Test with a column that has non-empty strings
     column = sample_dataframe['A']
     assert calculate_average_string_length(column) == 5.666666666666667

     # Test with a column that has empty strings
     column = sample_dataframe['B']
     assert calculate_average_string_length(column) == 0.0

     # Test with an empty column
     column = pd.Series([])
     assert calculate_average_string_length(column) == 0.0

     # Test with a column that has null values
     column = sample_dataframe['C']
     assert calculate_average_string_length(column) == pytest.approx(2.0)


def test_calculate_non_empty_percentage(sample_dataframe):
    # Test with a column that has non-empty strings
    column = sample_dataframe['A']
    assert calculate_non_empty_percentage(column) == 100.0

    # Test with a column that has empty strings
    column = sample_dataframe['B']
    assert calculate_non_empty_percentage(column) == 60.0

    # Test with an empty column
    column = pd.Series([])
    assert calculate_non_empty_percentage(column) == 0.0

    # Test with a column that has null values
    column = sample_dataframe['C']
    assert calculate_non_empty_percentage(column) == pytest.approx(80.0)


def test_count_unique_values():
     # Create test data
     data = {'Column1': ['Apple', 'Banana', 'Apple', 'Orange', 'Banana']}
     df = pd.DataFrame(data)
    
     # Call the function under test
     unique_count = count_unique_values(df['Column1'])
    
     # Check the result
     assert unique_count == 3


def test_calculate_most_common_values(sample_dataframe):
    # Test with a column that has non-empty values
    column = sample_dataframe['A']
    assert calculate_most_common_values(column) == [5, 4, None, 2, 1]

    # Test with a column that has empty values
    column = sample_dataframe['B']
    assert calculate_most_common_values(column) == [5, 3, None, 2]

    # Test with an empty column
    column = pd.Series([])
    assert calculate_most_common_values(column) == []

    # Test with a column that has null values
    column = sample_dataframe['C']
    assert calculate_most_common_values(column) == [4, 3, 2, None, 1]


def test_calculate_missing_prevalence(sample_dataframe):
    # Test with a column that has non-empty values
    column = sample_dataframe['A']
    assert calculate_missing_prevalence(column) == 40.0

    # Test with a column that has no missing values
    column = sample_dataframe['C']
    assert calculate_missing_prevalence(column) == 0.0

    # Test with an empty column
    column = pd.Series([])
    assert calculate_missing_prevalence(column) == 0.0

    # Test with a column that has all missing values
    column = sample_dataframe['B']
    assert calculate_missing_prevalence(column) == 60.0


def test_calculate_empty_string_prevalence(sample_dataframe):
     # Test with a column that has non-empty values
     column = sample_dataframe['A']
     assert calculate_empty_string_prevalence(column) == 0.0

     # Test with a column that has no empty strings
     column = sample_dataframe['C']
     assert calculate_empty_string_prevalence(column) == 0.0

     # Test with an empty column
     column = pd.Series([])
     with pytest.raises(ValueError):
         calculate_empty_string_prevalence(column)

     # Test with a column that has empty strings
     column = sample_dataframe['B']
     assert calculate_empty_string_prevalence(column) == 40.0


def test_calculate_percentage_unique_values(sample_dataframe):
    # Test with a dataframe that has non-empty string values
    df = pd.DataFrame({'Column1': ['Apple', 'Banana', 'Cherry']})
    assert calculate_percentage_unique_values(df, 'Column1') == 100.0

    # Test with a dataframe that has empty string values
    df = pd.DataFrame({'Column1': ['', '', '']})
    assert calculate_percentage_unique_values(df, 'Column1') == 0.0

    # Test with a dataframe that has null values
    df = pd.DataFrame({'Column1': [None, None, None]})
    assert calculate_percentage_unique_values(df, 'Column1') == 0.0

    # Test with an empty dataframe
    df = pd.DataFrame()
    assert calculate_percentage_unique_values(df, 'Column1') == 0.0


def test_calculate_string_statistics(sample_dataframe):
     # Test with a column that has non-empty string values
     column = sample_dataframe['A']
     result = calculate_string_statistics(column)
    
     assert isinstance(result, dict)
     assert 'min_length' in result
     assert 'max_length' in result
     assert 'avg_length' in result
     assert 'most_common_values' in result
     assert 'most_common_frequencies' in result
     assert 'std_deviation' in result
     assert 'median_length' in result
     assert 'first_quartile' in result
     assert 'third_quartile' in result

def test_calculate_string_statistics_empty_column():
    # Test with an empty column
    column = pd.Series([])
    result = calculate_string_statistics(column)
    
    assert result is None


def test_calculate_string_statistics_all_missing_values():
    # Test with a column that has all missing values
    column = pd.Series([None, None, None])
    result = calculate_string_statistics(column)
    
    assert result is None


def test_calculate_string_statistics_valid_input():
    # Test with a column that has valid input
    column = pd.Series(['apple', 'banana', 'orange'])
    result = calculate_string_statistics(column)
    
    assert isinstance(result, dict)
    assert 'min_length' in result
    assert 'max_length' in result
    assert 'avg_length' in result
    assert 'most_common_values' in result
    assert 'most_common_frequencies' in result
    assert 'std_deviation' in result
    assert 'median_length' in result
    assert 'first_quartile' in result
    assert 'third_quartile' in result


def test_calculate_string_statistics_missing_values_removed():
    # Test with a column that has missing values and non-missing values
    column = pd.Series(['apple', None, 'banana'])
    
    result = calculate_string_statistics(column)
    
    assert isinstance(result, dict)
    
    # Check if missing values were removed from the column before calculating statistics
    assert len(column.dropna()) == len(result['most_common_values'])


def test_calculate_string_statistics_multiple_most_common_values():
     # Test with a column that has multiple most common values
     column = pd.Series(['apple', 'banana', 'banana', 'orange'])
     result = calculate_string_statistics(column)
    
     assert isinstance(result, dict)
    
     # Check if the correct values and frequencies are returned for most common values
     expected_most_common_values = ['banana']
     expected_most_common_frequencies = [2]
     assert result['most_common_values'] == expected_most_common_values
     assert result['most_common_frequencies'] == expected_most_common_frequencies


def test_calculate_string_statistics_statistical_measurements():
     # Test with a column that has valid input
     column = pd.Series(['apple', 'banana', 'orange'])
     result = calculate_string_statistics(column)
    
     assert isinstance(result, dict)
    
     # Check if the statistical measurements are calculated correctly
     assert result['std_deviation'] is not None
     assert result['median_length'] is not None
     assert result['first_quartile'] is not None
     assert result['third_quartile'] is not None


def test_calculate_string_statistics_no_statistical_measurements():
     # Test with a column that has only one value
     column = pd.Series(['apple'])
     result = calculate_string_statistics(column)
    
     assert result is Non