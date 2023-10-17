andom
import pytest

from your_module import generate_random_float, generate_random_integer, generate_random_boolean, generate_random_categorical, generate_random_string, create_trivial_field, create_missing_fields, generate_random_data


def test_generate_random_float_returns_float():
    result = generate_random_float(0, 1)
    assert isinstance(result, float)


def test_generate_random_float_within_range():
    start = 0
    end = 10
    result = generate_random_float(start, end)
    assert start <= result <= end


def test_generate_random_float_with_negative_range():
    start = -10
    end = -5
    result = generate_random_float(start, end)
    assert start <= result <= end


def test_generate_random_float_with_zero_range():
    start = 5
    end = 5
    result = generate_random_float(start, end)
    assert result == start == end


def test_generate_random_float_with_reverse_range():
    start = 10
    end = 5
    with pytest.raises(ValueError):
        generate_random_float(start, end)


def test_generate_random_integer_within_range():
    start = 0
    end = 10
    generated = generate_random_integer(start, end)
    assert start <= generated <= end


def test_generate_random_integer_is_integer():
    start = 0
    end = 10
    generated = generate_random_integer(start, end)
    assert isinstance(generated, int)


def test_generate_random_boolean():
    # Test that the function returns a boolean value
    result = generate_random_boolean()
    assert isinstance(result, bool)

    # Test that the function returns either True or False
    assert result in [True, False]


def test_generate_random_categorical():
    # Test when categories list is empty
    with pytest.raises(IndexError):
        generate_random_categorical([])

    # Test when categories list has one category
    category = "category"
    assert generate_random_categorical([category]) == category

    # Test when categories list has multiple categories
    categories = ["cat1", "cat2", "cat3"]
    generated_category = generate_random_categorical(categories)
    assert generated_category in categories

    # Test that the function generates a random category
    random.seed(0)  # Set seed for reproducibility
    generated_categories = set()
    for _ in range(100):
        generated_categories.add(generate_random_categorical(categories))

    assert len(generated_categories) > 1


def test_generate_random_string_returns_string():
    result = generate_random_string(10)
    assert isinstance(result, str)


def test_generate_random_string_returns_correct_length():
    length = 8
    result = generate_random_string(length)
    assert len(result) == length


def test_generate_random_string_returns_different_strings():
    string1 = generate_random_string(10)
    string2 = generate_random_string(10)
    assert string1 != string2


def test_create_trivial_field_float():
    result = create_trivial_field('float', 3.14)
    assert isinstance(result, pd.Series)
    assert result.dtype == 'float64'
    assert result[0] == 3.14


def test_create_trivial_field_int():
    result = create_trivial_field('int', 42)
    assert isinstance(result, pd.Series)
    assert result.dtype == 'int64'
    assert result[0] == 42


def test_create_trivial_field_bool():
    result = create_trivial_field('bool', True)
    assert isinstance(result, pd.Series)
    assert result.dtype == 'bool'
    assert result[0] is True


def test_create_trivial_field_str():
    result = create_trivial_field('str', 'Hello')
    assert isinstance(result, pd.Series)
    assert result.dtype == 'object'
    assert result[0] == 'Hello'


def test_create_trivial_field_invalid_type():
    with pytest.raises(ValueError):
        create_trivial_field('invalid_type', 123)


@pytest.fixture
def sample_data():
    # Create a sample DataFrame for testing
    data = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': ['foo', 'bar', 'baz', 'qux'],
        'C': [True, False, True, False]
    })
    return data


def test_create_missing_fields(sample_data):
    # Call the function with default parameters
    modified_data = create_missing_fields(sample_data)

    # Assert that the modified_data has the same shape as the original data
    assert modified_data.shape == sample_data.shape

    # Assert that the modified_data contains null or None values in the missing fields
    for col in modified_data.columns:
        missing_values = modified_data[col].isnull().sum()
        assert missing_values > 0


def test_create_missing_fields_custom_ratio(sample_data):
    # Call the function with a custom null_ratio
    null_ratio = 0.5
    modified_data = create_missing_fields(sample_data, null_ratio=null_ratio)

    # Assert that the modified_data has the same shape as the original data
    assert modified_data.shape == sample_data.shape

    # Assert that the modified_data contains approximately half of its values as null or None 
    for col in modified_data.columns:
        missing_values = modified_data[col].isnull().sum()
        expected_missing_values = int(len(sample_data) * null_ratio)
        assert missing_values == expected_missing_values


def test_generate_random_data():
    # Test case 1: Check if the generated data has the correct number of rows
    n = 100
    data = generate_random_data(n)
    assert len(data) == n

    # Test case 2: Check if all columns are present in the generated data
    expected_columns = ['float_data', 'integer_data', 'boolean_data', 'categorical_data', 'string_data']
    assert set(data.columns) == set(expected_columns)

    # Test case 3: Check if all columns have the correct data types
    assert data['float_data'].dtype == np.float64
    assert data['integer_data'].dtype == np.int64
    assert data['boolean_data'].dtype == np.bool_
    assert data['categorical_data'].dtype == 'category'
    assert data['string_data'].dtype == object


def test_generate_random_data_with_inf_nan():
    # Test case 1: Check if inf and nan values are included in the generated data
    n = 100
    include_inf_nan = True
    data = generate_random_data(n, include_inf_nan=include_inf_nan)

    assert np.isinf(data['float_data']).any()
    assert np.isnan(data['float_data']).any()

    # Test case 2: Check if the number of inf and nan values is correct
    inf_count = np.isinf(data['float_data']).sum()
    nan_count = np.isnan(data['float_data']).sum()

    expected_inf_count = int(n / 10)
    expected_nan_count = int(n / 10)

    assert inf_count == expected_inf_count
    assert nan_count == expected_nan_coun