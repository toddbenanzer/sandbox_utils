andas as pd
import numpy as np
from sklearn.datasets import make_classification

# Test generate_classification_data function
def test_generate_classification_data():
    # Test with default parameters
    X, y = generate_classification_data()
    
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == y.shape[0]
    
    # Test with specific parameters
    X, y = generate_classification_data(n_samples=100, n_features=5, n_informative=3, random_state=42)
    
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (100, 5)
    assert y.shape == (100,)
    
    # Test with invalid parameters
    with pytest.raises(ValueError):
        generate_classification_data(n_samples=-1