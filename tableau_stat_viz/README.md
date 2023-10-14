# Package Name

This package provides a set of functions for data cleaning, preprocessing, and visualization using the pandas and seaborn libraries.

## Table of Contents
- [Overview](#overview)
- [Usage](#usage)
- [Examples](#examples)

## Overview<a name="overview"></a>

The functions in this package are designed to help with common data cleaning, preprocessing, and visualization tasks. The main features of this package include:

- Reading a csv file and returning it as a pandas DataFrame.
- Cleaning and preprocessing data by removing missing values, duplicate rows, converting categorical variables to numerical, and normalizing numerical variables.
- Visualizing different types of distributions using histograms, box plots, and density plots.
- Calculating and visualizing correlations between variables using correlation matrices and scatter plots.
- Creating statistical summaries and visualizations for categorical variables using bar charts and pie charts.
- Creating heatmaps to explore complex relationships in the data.
- Generating interactive visualizations for Tableau using Tableau code generation.
- Exporting visualizations as Tableau workbooks or images.
- Customizing the appearance of Tableau visualizations including color schemes, labels, titles, and annotations.
- Handling missing values in the data by excluding rows or filling missing values with mean, median, or mode of respective columns.
- Handling large datasets efficiently by optimizing memory usage and processing speed.
- Getting a list of Tableau projects using Tableau's APIs for seamless interaction with Tableau Desktop or Tableau Server.

## Usage<a name="usage"></a>

To use this package, you need to have the following dependencies installed:

- pandas
- matplotlib.pyplot
- seaborn
- os

You can install these dependencies using pip:

```bash
pip install pandas matplotlib seaborn
```

Once you have installed the required dependencies, you can import the package in your Python script or notebook:

```python
import dataprep as dp
```

## Examples<a name="examples"></a>

### Reading a CSV file

```python
file_path = 'data.csv'
df = dp.read_csv_file(file_path)
```

This function reads a csv file and returns a pandas DataFrame.

### Cleaning and preprocessing data

```python
cleaned_data = dp.clean_and_preprocess_data(df)
```

This function cleans and preprocesses the data by removing missing values, duplicate rows, converting categorical variables to numerical, and normalizing numerical variables.

### Visualizing distributions

```python
dp.visualize_distribution(df['column'], 'histogram')
```

This function calculates and visualizes different types of distributions including histograms, box plots, and density plots.

### Visualizing correlations

```python
dp.visualize_correlations(df)
```

This function calculates and visualizes correlations between variables using correlation matrices and scatter plots.

### Creating categorical visualizations

```python
dp.create_categorical_visualization(df, 'column')
```

This function creates statistical summaries and visualizations for categorical variables including bar charts and pie charts.

### Creating heatmaps

```python
dp.create_heatmap(df, 'x_column', 'y_column', 'values_column')
```

This function creates heatmaps to explore complex relationships in the data.

### Generating interactive visualizations for Tableau

```python
tableau_code = dp.create_interactive_visualization(df, filters=['filter1', 'filter2'], tooltips=['tooltip1', 'tooltip2'], highlight='highlight_column')
```

This function generates interactive visualizations for Tableau using Tableau code generation.

### Exporting visualizations as Tableau workbooks or images

```python
visualizations = ['visualization1.twbx', 'visualization2.png']
output_path = 'output'
dp.tableau_export(visualizations, output_path)
```

This function exports visualizations as Tableau workbooks or images to the specified output directory.

### Customizing the appearance of Tableau visualizations

```python
dp.customize_visualization('visualization', color_scheme='blue', labels={'x_axis': 'X Axis', 'y_axis': 'Y Axis'}, title='Custom Title', annotations=['annotation1', 'annotation2'])
```

This function customizes the appearance of a Tableau visualization including color schemes, labels, titles, and annotations.

### Handling missing values

```python
cleaned_data = dp.handle_missing_values(df, method='mean')
```

This function handles missing values in the data by excluding rows or filling missing values with mean, median, or mode of respective columns.

### Handling large datasets efficiently

```python
final_result = dp.handle_large_dataset(file_path)
```

This function handles large datasets efficiently by optimizing memory usage and processing speed.

### Getting Tableau projects using Tableau's APIs

```python
server_url = 'https://tableau_server_url'
username = 'username'
password = 'password'
projects = dp.get_tableau_projects(server_url, username, password)
```

This function gets a list of Tableau projects using Tableau's APIs for seamless interaction with Tableau Desktop or Tableau Server.