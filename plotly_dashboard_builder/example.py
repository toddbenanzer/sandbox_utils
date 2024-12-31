from your_module_path import export_dashboard
from your_module_path import import_data
from your_module_path import setup_config
from your_module_path import setup_logging
from your_module_path.cohort_analysis import CohortAnalysis
from your_module_path.dashboard_creator import DashboardCreator
from your_module_path.data_handler import DataHandler
from your_module_path.funnel_plot import FunnelPlot
from your_module_path.plotly_template import PlotlyTemplate
from your_module_path.time_series_plot import TimeSeriesPlot
import logging
import os
import pandas as pd
import plotly.graph_objs as go


# Example 1: Initializing with a DataFrame
df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
dashboard_creator = DashboardCreator(df)

# Example 2: Creating a simple scatter plot dashboard
scatter = go.Scatter(x=df['x'], y=df['y'], mode='lines+markers')
dashboard = dashboard_creator.create_dashboard(scatter=scatter)
dashboard.show()

# Example 3: Creating a dashboard with multiple components
bar = go.Bar(x=df['x'], y=df['y'])
layout = [{'name': 'scatter'}, {'name': 'bar'}]
dashboard = dashboard_creator.create_dashboard(layout=layout, scatter=scatter, bar=bar)
dashboard.show()

# Example 4: Previewing the dashboard
dashboard_creator.preview_dashboard()

# Example 5: Handling invalid data type
try:
    invalid_data_dashboard = DashboardCreator("invalid data type")
except TypeError as e:
    print(e)



# Example 1: Using a Funnel Template
funnel_template = PlotlyTemplate('funnel')
funnel_figure = funnel_template.load_template()
funnel_figure.show()

# Example 2: Customizing a Times Series Template
time_series_template = PlotlyTemplate('time_series')
time_series_figure = time_series_template.load_template()
customized_time_series_figure = time_series_template.customize_template(title='Customized Time Series', xaxis_title='Date', yaxis_title='Value')
customized_time_series_figure.show()

# Example 3: Using Cohort Analysis Template and apply customizations
cohort_template = PlotlyTemplate('cohort_analysis')
cohort_figure = cohort_template.load_template()
customized_cohort_figure = cohort_template.customize_template(title='Cohort Analysis', paper_bgcolor='lightgray')
customized_cohort_figure.show()

# Example 4: Handling Unsupported Template Type
try:
    unsupported_template = PlotlyTemplate('unsupported')
except ValueError as e:
    print(e)



# Example 1: Creating a Funnel Plot with a DataFrame
data_df = pd.DataFrame({'x': ['Stage 1', 'Stage 2', 'Stage 3'], 'y': [100, 50, 20]})
funnel_plot = FunnelPlot(data_df)
figure = funnel_plot.generate_plot(x='x', y='y', title='Funnel Plot Example with DataFrame')
figure.show()

# Example 2: Creating a Funnel Plot with a Dictionary
data_dict = {'x': ['Step 1', 'Step 2', 'Step 3'], 'y': [120, 80, 30]}
funnel_plot = FunnelPlot(data_dict)
figure = funnel_plot.generate_plot(x=data_dict['x'], y=data_dict['y'], title='Funnel Plot Example with Dict')
figure.show()

# Example 3: Customizing a Funnel Plot with a Title
data_df = pd.DataFrame({'x': ['Phase A', 'Phase B'], 'y': [200, 100]})
funnel_plot = FunnelPlot(data_df)
figure = funnel_plot.generate_plot(x='x', y='y', title='Customized Funnel Plot with Title')
figure.show()

# Example 4: Handling Invalid Data Type
try:
    invalid_funnel_plot = FunnelPlot("invalid data")
except TypeError as e:
    print(e)



# Example 1: Perform Cohort Analysis and Generate Default Plot
data = pd.DataFrame({'cohort': ['2021-Q1', '2021-Q2', '2021-Q1', '2021-Q3', '2021-Q2'], 'value': [1, 2, 3, 4, 5]})
cohort_analysis = CohortAnalysis(data)
analysis_results = cohort_analysis.perform_analysis()
print("Analysis Results:")
print(analysis_results)
plot = cohort_analysis.generate_plot()
plot.show()

# Example 2: Customizing the Cohort Analysis Plot with a Title
cohort_analysis = CohortAnalysis(data)
cohort_analysis.perform_analysis()
custom_plot = cohort_analysis.generate_plot(title="Customized Cohort Analysis Plot")
custom_plot.show()

# Example 3: Handling Non-DataFrame Input
try:
    invalid_cohort_analysis = CohortAnalysis("invalid data type")
except TypeError as e:
    print(e)

# Example 4: Generate Plot Without Performing Analysis
try:
    cohort_analysis_without_analysis = CohortAnalysis(data)
    cohort_analysis_without_analysis.generate_plot()
except ValueError as e:
    print(e)



# Example 1: Create a daily time series plot
dates = pd.date_range(start='2022-01-01', periods=10, freq='D')
data = pd.DataFrame({'sales': [150, 200, 250, 300, 350, 400, 300, 250, 200, 150]}, index=dates)
time_series_plot = TimeSeriesPlot(data)
figure = time_series_plot.generate_plot('D')
figure.show()

# Example 2: Create a monthly time series plot with a title
dates = pd.date_range(start='2021-01-01', periods=12, freq='M')
monthly_data = pd.DataFrame({'revenue': [1200, 1500, 1350, 1450, 1600, 1700, 1650, 1750, 1800, 1950, 2000, 2100]}, index=dates)
monthly_time_series_plot = TimeSeriesPlot(monthly_data)
monthly_figure = monthly_time_series_plot.generate_plot('M', title='Monthly Revenue Over Time')
monthly_figure.show()

# Example 3: Handling non-DataFrame input
try:
    invalid_data = [[1, 2, 3], [4, 5, 6]]
    invalid_time_series_plot = TimeSeriesPlot(invalid_data)
except TypeError as e:
    print(e)

# Example 4: Handling non-DatetimeIndex
try:
    non_datetime_index_data = pd.DataFrame({'value': [100, 200]}, index=[1, 2])
    non_datetime_time_series_plot = TimeSeriesPlot(non_datetime_index_data)
except ValueError as e:
    print(e)



# Example 1: Load Data from a CSV File
handler_csv = DataHandler("example_data.csv")
data_csv = handler_csv.load_data()
print(data_csv.head())

# Example 2: Load Data from an Excel File
handler_excel = DataHandler("example_data.xlsx")
data_excel = handler_excel.load_data()
print(data_excel.head())

# Example 3: Apply a Single Transformation Function
handler_csv = DataHandler("example_data.csv")
handler_csv.load_data()
transformed_data_single = handler_csv.transform_data(lambda df: df.dropna())
print(transformed_data_single.head())

# Example 4: Apply Multiple Transformation Functions
handler_csv = DataHandler("example_data.csv")
handler_csv.load_data()
transformations = [
    lambda df: df.dropna(),
    lambda df: df[df['column'] > 0],
]
transformed_data_multiple = handler_csv.transform_data(transformations)
print(transformed_data_multiple.head())

# Example 5: Handle Unsupported File Format
try:
    handler_unsupported = DataHandler("example_data.txt")
    handler_unsupported.load_data()
except ValueError as e:
    print(e)

# Example 6: Handle Nonexistent File
try:
    handler_nonexistent = DataHandler("nonexistent.csv")
    handler_nonexistent.load_data()
except FileNotFoundError as e:
    print(e)



# Example 1: Importing a CSV File
try:
    csv_data = import_data("data/sales_data.csv", "csv")
    print(csv_data.head())
except Exception as e:
    print(e)

# Example 2: Importing an Excel File
try:
    excel_data = import_data("data/sales_data.xlsx", "xlsx")
    print(excel_data.head())
except Exception as e:
    print(e)

# Example 3: Importing a JSON File
try:
    json_data = import_data("data/sales_data.json", "json")
    print(json_data.head())
except Exception as e:
    print(e)

# Example 4: Handling Unsupported File Type
try:
    unsupported_data = import_data("data/sales_data.txt", "txt")
except ValueError as e:
    print(e)

# Example 5: Handling Nonexistent File
try:
    nonexistent_data = import_data("data/nonexistent_file.csv", "csv")
except FileNotFoundError as e:
    print(e)



# Ensure output directory exists
os.makedirs('output', exist_ok=True)

# Example 1: Export a simple line chart as an HTML file
fig_html = go.Figure(data=go.Scatter(x=[0, 1, 2], y=[2, 1, 0]))
export_dashboard(fig_html, 'html', 'output/line_chart.html')

# Example 2: Export a bar chart as a PNG file
fig_png = go.Figure(data=go.Bar(x=['A', 'B', 'C'], y=[4, 7, 3]))
export_dashboard(fig_png, 'png', 'output/bar_chart.png')

# Example 3: Export a pie chart as a PDF file
fig_pdf = go.Figure(data=go.Pie(labels=['Apples', 'Oranges'], values=[50, 50]))
export_dashboard(fig_pdf, 'pdf', 'output/pie_chart.pdf')

# Example 4: Attempt to export with an unsupported format (will raise ValueError)
try:
    export_dashboard(fig_html, 'txt', 'output/line_chart.txt')
except ValueError as e:
    print(e)

# Example 5: Attempt to export to a non-existent directory (will raise FileNotFoundError)
try:
    export_dashboard(fig_html, 'html', 'nonexistent_dir/line_chart.html')
except FileNotFoundError as e:
    print(e)



# Example 1: Setting up logging with a string level
setup_logging('DEBUG')
logging.debug("This is a debug message.")
logging.info("This is an info message.")
logging.warning("This is a warning message.")

# Example 2: Setting up logging with an integer level
setup_logging(20)  # Equivalent to INFO
logging.info("This info message will be displayed.")
logging.debug("This debug message will not be displayed.")  # Lower than INFO

# Example 3: Invalid string logging level (will raise ValueError)
try:
    setup_logging('INVALID')
except ValueError as e:
    print(e)

# Example 4: Invalid integer logging level (will raise ValueError)
try:
    setup_logging(99)  # Not a recognized logging level
except ValueError as e:
    print(e)

# Example 5: Using a TypeError for unsupported logging level type
try:
    setup_logging(3.5)  # Float type is not valid
except TypeError as e:
    print(e)



# Example 1: Loading configuration from a valid JSON file
try:
    config = setup_config("config/valid_config.json")
    print(config)
except Exception as e:
    print(e)

# Example 2: Handling a nonexistent configuration file
try:
    config = setup_config("config/nonexistent_config.json")
except FileNotFoundError as e:
    print(e)

# Example 3: Handling an invalid JSON formulation
try:
    config = setup_config("config/invalid_config.json")
except ValueError as e:
    print(e)

# Example 4: Handling an unexpected error during file read
try:
    config = setup_config("config/broken_config.json")
except Exception as e:
    print(e)
