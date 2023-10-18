andas as pd
import numpy as np
import pytest
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def test_perform_tukeyhsd():
    # Create a sample dataframe for testing
    df = pd.DataFrame({
        'Group': ['A', 'A', 'B', 'B', 'C', 'C'],
        'Value': [1, 2, 3, 4, 5, 6]
    })

    # Test case 1: Perform Tukey HSD test and compare the result with the expected output
    result = perform_tukeyhsd(df, 'Group')
    mc_results = pairwise_tukeyhsd(df['Value'], df['Group'])
    assert str(mc_results) == result

    # Test case 2: Test with an empty dataframe
    empty_df = pd.DataFrame(columns=['Group', 'Value'])
    with pytest.raises(ValueError):
        perform_tukeyhsd(empty_df, 'Group')

    # Test case 3: Test with a non-existent column name
    with pytest.raises(KeyError):
        perform_tukeyhsd(df, 'NonExistentColumn'