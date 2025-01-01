from data_generator import DataGenerator  # Assume the class is in data_generator.py
from module import set_seed  # Assume the function is in module.py
from module import validate_parameters  # Assume the function is in module.py
import numpy as np
import pytest
import random


def test_generate_float_column():
    gen = DataGenerator(1000)
    floats = gen.generate_float_column(min_value=0, max_value=10)
    assert len(floats) == 1000
    assert all(0 <= x <= 10 for x in floats)

def test_generate_integer_column():
    gen = DataGenerator(1000)
    integers = gen.generate_integer_column(min_value=0, max_value=100)
    assert len(integers) == 1000
    assert all(0 <= x <= 100 for x in integers)

def test_generate_boolean_column():
    gen = DataGenerator(1000)
    booleans = gen.generate_boolean_column(true_probability=0.7)
    assert len(booleans) == 1000
    true_count = sum(booleans)
    assert 600 <= true_count <= 800  # Roughly considering a probability distribution

def test_generate_categorical_column():
    gen = DataGenerator(1000)
    categories = ['A', 'B', 'C']
    categorical = gen.generate_categorical_column(categories=categories)
    assert len(categorical) == 1000
    assert all(x in categories for x in categorical)

def test_generate_string_column():
    gen = DataGenerator(1000)
    strings = gen.generate_string_column(length=5)
    assert len(strings) == 1000
    assert all(len(s) == 5 for s in strings)

def test_generate_single_value_column():
    gen = DataGenerator(1000)
    single_values = gen.generate_single_value_column(value=42)
    assert len(single_values) == 1000
    assert all(x == 42 for x in single_values)

def test_generate_missing_values():
    gen = DataGenerator(1000)
    data = list(range(1000))
    data = gen.generate_missing_values(data, percentage=0.1)
    assert len(data) == 1000
    missing_count = sum(x is None for x in data)
    assert 90 <= missing_count <= 110  # Considering some variation

def test_include_inf_nan():
    gen = DataGenerator(1000)
    data = list(range(1000))
    data = gen.include_inf_nan(data, inf_percentage=0.05, nan_percentage=0.05)
    assert len(data) == 1000
    inf_count = sum(x == float('inf') for x in data)
    nan_count = sum(x != x for x in data)  # NaNs don't equal themselves
    assert 40 <= inf_count <= 60  # Considering some variation
    assert 40 <= nan_count <= 60  # Considering some variation

def test_to_dataframe():
    gen = DataGenerator(1000)
    gen.data['floats'] = gen.generate_float_column()
    gen.data['ints'] = gen.generate_integer_column()
    df = gen.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1000, 2)



def test_set_seed_reproducibility():
    set_seed(42)
    random_values1 = [random.random() for _ in range(5)]
    np_random_values1 = np.random.rand(5)

    set_seed(42)
    random_values2 = [random.random() for _ in range(5)]
    np_random_values2 = np.random.rand(5)

    assert random_values1 == random_values2, "Random values with set seed should be identical"
    assert np.array_equal(np_random_values1, np_random_values2), "NumPy random values with set seed should be identical"

def test_set_seed_different_seed():
    set_seed(42)
    random_values1 = [random.random() for _ in range(5)]

    set_seed(43)
    random_values2 = [random.random() for _ in range(5)]

    assert random_values1 != random_values2, "Random values with different seeds should not be identical"



def test_valid_parameters():
    params = {
        'num_records': 100,
        'min_value': 1.0,
        'max_value': 10.0,
        'true_probability': 0.7,
        'categories': ['A', 'B', 'C'],
        'length': 5,
        'inf_percentage': 0.05,
        'nan_percentage': 0.05,
        'percentage': 0.1
    }
    try:
        validate_parameters(params)
    except Exception as e:
        pytest.fail(f"valid parameters should not raise an exception: {e}")

def test_invalid_num_records():
    params = {'num_records': -1}
    with pytest.raises(ValueError):
        validate_parameters(params)

def test_non_integer_num_records():
    params = {'num_records': '100'}
    with pytest.raises(TypeError):
        validate_parameters(params)

def test_min_greater_than_max():
    params = {'min_value': 10.0, 'max_value': 1.0}
    with pytest.raises(ValueError):
        validate_parameters(params)

def test_invalid_true_probability():
    params = {'true_probability': 1.5}
    with pytest.raises(ValueError):
        validate_parameters(params)

def test_non_list_categories():
    params = {'categories': 'notalist'}
    with pytest.raises(TypeError):
        validate_parameters(params)

def test_empty_categories():
    params = {'categories': []}
    with pytest.raises(ValueError):
        validate_parameters(params)

def test_invalid_length():
    params = {'length': 0}
    with pytest.raises(ValueError):
        validate_parameters(params)

def test_invalid_float_percentage():
    params = {'inf_percentage': 1.5}
    with pytest.raises(ValueError):
        validate_parameters(params)
