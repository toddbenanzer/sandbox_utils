
import pandas as pd
import pytest

@pytest.fixture
def df1():
    return pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

@pytest.fixture
def df2():
    return pd.DataFrame({'A': [1, 2, 3], 'C': [7, 8, 9]})

def test_merge_datasets_all_columns(df1, df2):
    merged_df = merge_datasets(df1, df2)
    assert list(merged_df.columns) == ['A', 'B', 'C']
    assert merged_df.shape == (3, 3)

def test_merge_datasets_specific_column(df1, df2):
    merged_df = merge_datasets(df1, df2, on='A')
    assert list(merged_df.columns) == ['A', 'B', 'C']
    assert merged_df.shape == (3, 3)

def test_merge_datasets_specific_columns(df1, df2):
    merged_df = merge_datasets(df1, df2, on=['A'])
    assert list(merged_df.columns) == ['A', 'B', 'C']
    assert merged_df.shape == (3, 3)
