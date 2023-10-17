# Random Data Generator

This package provides functions to generate random data of various types and formats. It can be used for a variety of purposes, such as testing, simulation, or generating sample data for analysis.

## Installation

To install the package, simply run the following command:

```bash
pip install random-data-generator
```

## Usage

Import the package into your Python script or Jupyter notebook using the following line of code:

```python
import random_data_generator as rdg
```

The package provides the following functions:

### `generate_random_float(start, end)`

Generate a random float value within a specified range.

```python
start = 0.0
end = 1.0

random_float = rdg.generate_random_float(start, end)
```

### `generate_random_integer(start, end)`

Generate a random integer value within a specified range.

```python
start = 0
end = 10

random_integer = rdg.generate_random_integer(start, end)
```

### `generate_random_boolean()`

Generate a random boolean value.

```python
random_boolean = rdg.generate_random_boolean()
```

### `generate_random_categorical(categories)`

Generate a random categorical value from a given list of categories.

```python
categories = ['A', 'B', 'C']

random_categorical = rdg.generate_random_categorical(categories)
```

### `generate_random_string(length)`

Generate a random string of specified length.

```python
length = 5

random_string = rdg.generate_random_string(length)
```

### `create_trivial_field(data_type, value)`

Create a trivial field with a single value.

Supported data types: 'float', 'int', 'bool', 'str'

```python
data_type = 'float'
value = 1.5

trivial_field = rdg.create_trivial_field(data_type, value)
```

### `create_missing_fields(data, null_ratio=0.1)`

Create missing fields with null or None values in the given pandas DataFrame.

```python
import pandas as pd

data = pd.DataFrame({
    'column1': [1, 2, 3],
    'column2': ['a', 'b', 'c']
})

null_ratio = 0.2

missing_data = rdg.create_missing_fields(data, null_ratio)
```

### `generate_random_data(n, include_inf_nan=False)`

Generate random data of specified length.

```python
n = 100
include_inf_nan = True

random_data = rdg.generate_random_data(n, include_inf_nan)
```

## Examples

Here are some examples to demonstrate the usage of the package:

```python
import random_data_generator as rdg

# Generate a random float value between 0 and 1
random_float = rdg.generate_random_float(0.0, 1.0)

# Generate a random integer value between 0 and 10
random_integer = rdg.generate_random_integer(0, 10)

# Generate a random boolean value
random_boolean = rdg.generate_random_boolean()

# Generate a random categorical value from a list of categories
categories = ['A', 'B', 'C']
random_categorical = rdg.generate_random_categorical(categories)

# Generate a random string of length 5
random_string = rdg.generate_random_string(5)

# Create a trivial field with a float value of 1.5
trivial_field = rdg.create_trivial_field('float', 1.5)

# Create missing fields in a DataFrame with a null ratio of 0.2
import pandas as pd

data = pd.DataFrame({
    'column1': [1, 2, 3],
    'column2': ['a', 'b', 'c']
})

null_ratio = 0.2

missing_data = rdg.create_missing_fields(data, null_ratio)

# Generate a DataFrame with 100 rows of random data, including inf and nan values
n = 100
include_inf_nan = True

random_data = rdg.generate_random_data(n, include_inf_nan)
```

These examples demonstrate the basic usage of the package and highlight some of its key features. Feel free to explore the package further and customize it to suit your specific needs.