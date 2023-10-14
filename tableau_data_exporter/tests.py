ored Code

import pandas as pd
import pytest

from my_module import calculate_custom_metric, apply_filters, format_data


# Test case for calculate_custom_metric function
def test_calculate_custom_metric():
    # Create a test dataframe
    data = {'column1': [1, 2, 3], 'column2': [4, 5, 6]}
    df = pd.DataFrame(data)

    # Apply the custom calculation function
    df = calculate_custom_metric(df)

    # Check if the custom metric column was added correctly
    assert 'custom_metric' in df.columns

    # Check if custom metric values are calculated correctly
    assert (df['custom_metric'] == [9, 12, 15]).all()


# Test case for apply_filters function
def test_apply_filters_single_column():
    filtered_df = apply_filters(df, ['Country'])
    expected_df = pd.DataFrame({'Name': ['Alice', 'Charlie'], 'Age': [25, 35], 'Country': ['USA', 'USA']})
    assert filtered_df.equals(expected_df)


def test_apply_filters_multiple_columns():
    filtered_df = apply_filters(df, ['Country', 'Age'])
    expected_df = pd.DataFrame({'Name': ['Charlie'], 'Age': [35], 'Country': ['USA']})
    assert filtered_df.equals(expected_df)


def test_apply_filters_no_match():
    filtered_df = apply_filters(df, ['Age'])
    expected_df = pd.DataFrame({'Name': [], 'Age': [], 'Country': []})
    assert filtered_df.equals(expected_df)


def test_apply_filters_empty_dataframe():
    empty_df = pd.DataFrame()
    filtered_df = apply_filters(empty_df, ['Country'])
    expected_df = pd.DataFrame()
    assert filtered_df.equals(expected_df)


def test_apply_filters_invalid_column():
    with pytest.raises(KeyError):
        apply_filters(df, ['InvalidColumn'])


# Test case for format_data function
def test_format_data():
    # Sample data and column formats
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4.0, 5.0, 6.0]})
    column_formats = {'A': lambda x: f"Value: {x}", 'B': lambda x: f"Number: {x:.2f}"}

    # Expected output with applied formatting options
    expected_output = pd.DataFrame({'A': ['Value: 1', 'Value: 2', 'Value: 3'],
                                    'B': ['Number: 4.00', 'Number: 5.00', 'Number: 6.00']})

    # Call the format_data function with the sample data
    formatted_data = format_data(data, column_formats)

    # Check if the formatted data matches the expected output
    assert formatted_data.equals(expected_output)


if __name__ == "__main__":
    pytest.main(