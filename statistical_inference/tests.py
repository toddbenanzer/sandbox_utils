ytest
import numpy as np
from scipy.stats import ttest_ind

def t_test(array1, array2):
    if np.var(array1) == 0 or np.var(array2) == 0:
        raise ValueError("Arrays have zero variance")
    elif np.mean(array1) == np.mean(array2):
        raise ValueError("Arrays have constant values")
    else:
        t_statistic, p_value = ttest_ind(array1, array2)
        return {
            "t_statistic": t_statistic,
            "p_value": p_value,
            "array1_mean": np.mean(array1),
            "array2_mean": np.mean(array2),
            "array1_std": np.std(array1),
            "array2_std": np.std(array2)
        }

def test_t_test():
    # Test case 1: Arrays have zero variance
    array1 = [1, 1, 1, 1]
    array2 = [2, 2, 2, 2]
    
    with pytest.raises(ValueError) as e:
        t_test(array1, array2)
    
    assert str(e.value) == "Arrays have zero variance"

    # Test case 2: Arrays have constant values
    array3 = [3, 3, 3, 3]
    array4 = [4, 4, 4, 4]
    
    with pytest.raises(ValueError) as e:
        t_test(array3, array4)
    
    assert str(e.value) == "Arrays have constant values"

    # Test case 3: Normal arrays
    array5 = [0.5, 0.6, 0.7, 0.8]
    array6 = [0.4, 0.3, 0.2, 0.1]
    
    result = t_test(array5, array6)
    
    assert result["t_statistic"] == ttest_ind(array5, array6)[0]
    assert result["p_value"] == ttest_ind(array5, array6)[1]
    assert result["array1_mean"] == np.mean(array5)
    assert result["array2_mean"] == np.mean(array6)
    assert result["array1_std"] == np.std(array5)
    assert result["array2_std"] == np.std(array6