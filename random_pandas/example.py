

# Example 1: Generating a DataFrame with mixed data types
gen = DataGenerator(num_records=100)
gen.data['integers'] = gen.generate_integer_column(min_value=0, max_value=50)
gen.data['floats'] = gen.generate_float_column(min_value=0.0, max_value=10.0)
gen.data['booleans'] = gen.generate_boolean_column(true_probability=0.25)
gen.data['categories'] = gen.generate_categorical_column(categories=['red', 'green', 'blue'])
gen.data['strings'] = gen.generate_string_column(length=8)
df = gen.to_dataframe()
print(df.head())

# Example 2: Adding missing values to an integer column
integers = gen.generate_integer_column(min_value=1, max_value=100)
missing_integers = gen.generate_missing_values(integers, percentage=0.1)
gen.data['integers_with_missing'] = missing_integers
df = gen.to_dataframe()
print(df.head())

# Example 3: Introducing 'inf' and 'nan' into a float column
floats = gen.generate_float_column(min_value=-10.0, max_value=10.0)
floats_with_inf_nan = gen.include_inf_nan(floats, inf_percentage=0.05, nan_percentage=0.05)
gen.data['floats_with_inf_nan'] = floats_with_inf_nan
df = gen.to_dataframe()
print(df.head())

# Example 4: Generating a column of identical values
single_value_col = gen.generate_single_value_column(value=42)
gen.data['single_value'] = single_value_col
df = gen.to_dataframe()
print(df.head())


# Example 1: Ensuring reproducibility with random.random
set_seed(42)
print([random.random() for _ in range(3)])  # [0.6394267984578837, 0.025010755222666936, 0.27502931836911926]

set_seed(42)
print([random.random() for _ in range(3)])  # [0.6394267984578837, 0.025010755222666936, 0.27502931836911926]

# Example 2: Ensuring reproducibility with numpy.random
set_seed(42)
print(np.random.rand(3))  # [0.37454012 0.95071431 0.73199394]

set_seed(42)
print(np.random.rand(3))  # [0.37454012 0.95071431 0.73199394]

# Example 3: Different seeds produce different results
set_seed(42)
print([random.random() for _ in range(3)])  # [0.6394267984578837, 0.025010755222666936, 0.27502931836911926]

set_seed(43)
print([random.random() for _ in range(3)])  # [0.038551839337380045, 0.6962243226370521, 0.08693883262941615]


# Example 1: Valid parameter set
params = {
    'num_records': 100,
    'min_value': 0.0,
    'max_value': 10.0,
    'true_probability': 0.5,
    'categories': ['A', 'B', 'C'],
    'length': 8,
    'inf_percentage': 0.1,
    'nan_percentage': 0.1,
    'percentage': 0.05
}

try:
    validate_parameters(params)
    print("Parameters are valid.")
except Exception as e:
    print(f"Validation failed: {e}")

# Example 2: Invalid `num_records`
params = {'num_records': -10}

try:
    validate_parameters(params)
except ValueError as e:
    print(e)  # Output: Parameter 'num_records' must be greater than zero.

# Example 3: Invalid type for `true_probability`
params = {'true_probability': '0.5'}

try:
    validate_parameters(params)
except TypeError as e:
    print(e)  # Output: Parameter 'true_probability' must be a float.

# Example 4: Empty `categories` list
params = {'categories': []}

try:
    validate_parameters(params)
except ValueError as e:
    print(e)  # Output: Parameter 'categories' must not be an empty list.

# Example 5: `min_value` greater than `max_value`
params = {'min_value': 10, 'max_value': 5}

try:
    validate_parameters(params)
except ValueError as e:
    print(e)  # Output: Parameter 'min_value' cannot be greater than 'max_value'.
