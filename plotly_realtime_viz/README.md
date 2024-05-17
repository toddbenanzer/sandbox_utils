# Real-Time Data Analysis Package

This package provides functionality for fetching, preprocessing, aggregating, filtering, transforming, and visualizing real-time data. It includes methods for retrieving data from RESTful APIs, websockets, and databases. The package also offers various data preprocessing techniques such as handling missing values and outliers, normalizing data, and calculating descriptive statistics. Additionally, it provides tools for creating real-time line charts, bar charts, scatter plots, area charts, pie charts, and heatmaps.

## Installation

To install the package, use the following command:

```
pip install real-time-data-analysis
```

## Usage

Import the package in your Python script:

```python
import real_time_data_analysis as rtda
```

## Examples

### Fetching Real-Time Data from RESTful API

```python
url = "https://api.example.com/data"
data = rtda.fetch_realtime_data(url)
```

### Fetching Real-Time Data from WebSocket

```python
url = "wss://ws.example.com/data"
rtda.fetch_realtime_data_ws(url)
```

### Fetching Real-Time Data from PostgreSQL Database

```python
data = rtda.fetch_realtime_data_db()
```

### Preprocessing and Cleaning Data

```python
preprocessed_data = rtda.preprocess_data(data)
```

### Aggregating Data over a Specified Time Interval

```python
aggregated_value = rtda.aggregate_data(data, interval_minutes=5)
```

### Filtering Real-Time Data based on Specific Conditions

```python
filtered_data = rtda.filter_realtime_data(data, condition=lambda x: x['value'] > 10)
```

### Transforming Real-Time Data into a Desired Format

```python
transformed_data = rtda.transform_data(real_time_data)
```

### Handling Missing or Null Values in Real-Time Data

```python
cleaned_data = rtda.handle_missing_values(data)
```

### Handling Outliers in Real-Time Data using Z-Score Method

```python
processed_data = rtda.handle_outliers(data, threshold=3)
```

### Normalizing or Scaling Real-Time Data using Min-Max Scaling Technique

```python
normalized_data = rtda.normalize_data(data)
```

### Calculating Descriptive Statistics of Real-Time Data

```python
statistics = rtda.calculate_statistics(data)
```

### Creating Real-Time Line Charts

```python
rtda.create_realtime_line_chart()
```

### Creating Real-Time Bar Charts

```python
rtda.create_realtime_bar_chart()
```

### Creating Real-Time Scatter Plots

```python
rtda.create_realtime_scatter_plot(x_data=[1,2], y_data=[4,5])
```

### Creating Real-Time Area Charts

```python
rtda.create_realtime_area_chart(x_data=[1,2,3], y_data=[10,20,30], title="Real-Time Area Chart")
```

### Creating Real-Time Pie Charts

```python
data = {'Label 1': 10, 'Label 2': 20, 'Label 3': 30}
rtda.create_pie_chart_realtime(data)
```

### Creating Real-Time Heatmaps

```python
rtda.create_realtime_heatmap()
```

## Contributions

Contributions are welcome! If you have any ideas or suggestions for improving this package, please open an issue or submit a pull request on the GitHub repository.

## License

This package is licensed under the MIT License. See the [LICENSE](https://github.com/your-username/real-time-data-analysis/blob/main/LICENSE) file for more information.