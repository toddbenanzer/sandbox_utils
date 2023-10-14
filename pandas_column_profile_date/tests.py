andas as pd
import pytest

# Test case 1: column exists in the dataframe
def test_check_column_exists():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    assert check_column_exists(df, 'A') == True

# Test case 2: column does not exist in the dataframe
def test_check_column_exists_missing_column():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    assert check_column_exists(df, 'C') == False

# Test case 3: empty dataframe
def test_check_column_exists_empty_df():
    df = pd.DataFrame()
    assert check_column_exists(df, 'A') == Fals