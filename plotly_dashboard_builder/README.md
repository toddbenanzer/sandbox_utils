# Python Data Analysis Package

This package provides various functionalities for data analysis in Python. It includes functions for reading data from different file formats, cleaning and preprocessing data, filtering and aggregating data, calculating summary statistics, and creating visualizations using Plotly.

## Installation

To install the package, use the following command:

```
pip install data_analysis_package
```

## Usage

Import the package using the following code:

```python
import data_analysis_package as dap
```

### Read Data from CSV File

Use the `read_csv_file` function to read data from a CSV file and return it as a pandas DataFrame.

```python
data = dap.read_csv_file(file_path)
```

### Read Data from JSON File

Use the `read_json_file` function to read data from a JSON file and return it as a dictionary.

```python
data = dap.read_json_file(file_path)
```

### Read Data from Database

Use the `read_data_from_database` function to read data from a database using a given query.

```python
database_url = "your-database-url"
query = "SELECT * FROM table"
data = dap.read_data_from_database(database_url, query)
```

### Clean and Preprocess Data

Use the `clean_and_preprocess_data` function to clean and preprocess the data.

```python
cleaned_data = dap.clean_and_preprocess_data(data)
```

### Filter Data

Use the `filter_data` function to filter data based on certain conditions.

```python
filtered_data = dap.filter_data(data, condition)
```

### Aggregate Data

Use the `aggregate_data` function to aggregate data based on different variables.

```python
aggregated_data = dap.aggregate_data(data, group_by, agg_func)
```

### Calculate Summary Statistics

Use the `calculate_summary_statistics` function to calculate summary statistics for a given dataset.

```python
summary_stats = dap.calculate_summary_statistics(data)
```

### Create Visualizations

The package provides various functions to create different types of visualizations using Plotly.

#### Bar Plot

Use the `create_bar_plot` function to create a bar plot.

```python
x = [1, 2, 3]
y = [4, 5, 6]
title = "Bar Plot"
dap.create_bar_plot(x, y, title)
```

#### Line Plot

Use the `create_line_plot` function to create a line plot.

```python
x_values = [1, 2, 3]
y_values = [4, 5, 6]
title = "Line Plot"
dap.create_line_plot(x_values, y_values, title)
```

#### Scatter Plot

Use the `create_scatter_plot` function to create a scatter plot.

```python
x = "x_column"
y = "y_column"
title = "Scatter Plot"
dap.create_scatter_plot(data, x, y, title)
```

#### Pie Chart

Use the `create_pie_chart` function to create a pie chart.

```python
labels = ["A", "B", "C"]
values = [10, 20, 30]
dap.create_pie_chart(labels, values)
```

#### Histogram

Use the `create_histogram` function to create a histogram.

```python
data = [1, 2, 3]
x_label = "Values"
title = "Histogram"
dap.create_histogram(data, x_label, title)
```

### Examples

Here are some examples of how to use the package for data analysis:

1. Read data from a CSV file:

```python
data = dap.read_csv_file("data.csv")
```

2. Clean and preprocess the data:

```python
cleaned_data = dap.clean_and_preprocess_data(data)
```

3. Calculate summary statistics:

```python
summary_stats = dap.calculate_summary_statistics(data)
```

4. Create a bar plot:

```python
x = [1, 2, 3]
y = [4, 5, 6]
title = "Bar Plot"
dap.create_bar_plot(x, y, title)
```

5. Create a scatter plot:

```python
x = "x_column"
y = "y_column"
title = "Scatter Plot"
dap.create_scatter_plot(data, x, y, title)
```

## Contributing

Contributions to this package are welcome. Please submit any bug reports, feature requests, or pull requests through the GitHub repository.

## License

This package is licensed under the MIT License. See the LICENSE file for more information.