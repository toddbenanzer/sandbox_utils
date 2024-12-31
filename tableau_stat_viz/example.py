from correlation_visualizer import CorrelationVisualizer
from data_handler import DataHandler
from distribution_visualizer import DistributionVisualizer
from setup_logging import setup_logging
from setup_visualization_style import setup_visualization_style
from statistical_analyzer import StatisticalAnalyzer
from tableau_exporter import TableauExporter
import logging
import matplotlib.pyplot as plt
import pandas as pd


# Sample data
data = pd.DataFrame({
    'A': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'B': [9, 8, 7, 6, 5, 4, 3, 2, 1]
})

# Initialize the visualizer with the dataset
visualizer = DistributionVisualizer(data=data)

# Create a histogram for column 'A'
hist_fig = visualizer.create_histogram(column='A', bins=5, color='skyblue')
plt.show(hist_fig)

# Create a histogram for column 'B'
hist_fig_b = visualizer.create_histogram(column='B', bins=3, color='coral')
plt.show(hist_fig_b)

# Create a box plot for columns 'A' and 'B'
box_fig = visualizer.create_box_plot(columns=['A', 'B'], grid=False)
plt.show(box_fig)



# Sample data
data = pd.DataFrame({
    'A': [1, 2, 3, 4, 5, 6],
    'B': [6, 5, 4, 3, 2, 1],
    'C': [2, 3, 4, 5, 6, 7]
})

# Initialize the visualizer with the dataset
visualizer = CorrelationVisualizer(data=data)

# Create a correlation matrix heatmap
correlation_fig = visualizer.create_correlation_matrix(cmap='coolwarm', annot=True)
plt.show(correlation_fig)

# Create a scatter plot between columns 'A' and 'B'
scatter_fig = visualizer.create_scatter_plot(x='A', y='B', color='blue')
plt.show(scatter_fig)

# Create a scatter plot between columns 'A' and 'C' with a trendline
scatter_trendline_fig = visualizer.create_scatter_plot(x='A', y='C', with_trendline=True, color='green')
plt.show(scatter_trendline_fig)



# Initialize the DataHandler
handler = DataHandler()

# Example 1: Importing a CSV file
csv_data = handler.import_data('path/to/data.csv', 'csv')
print("CSV Data:\n", csv_data)

# Example 2: Importing an Excel file
excel_data = handler.import_data('path/to/data.xlsx', 'xlsx')
print("Excel Data:\n", excel_data)

# Example 3: Importing a JSON file
json_data = handler.import_data('path/to/data.json', 'json')
print("JSON Data:\n", json_data)

# Example 4: Preprocessing - Filling NaN values
filled_data = handler.preprocess_data({'fillna': {'value': 0}})
print("Filled Data:\n", filled_data)

# Example 5: Preprocessing - Dropping NaN values
dropped_data = handler.preprocess_data({'dropna': {}})
print("Dropped Data:\n", dropped_data)

# Example 6: Preprocessing - Renaming columns
renamed_data = handler.preprocess_data({'rename': {'columns': {'old_name': 'new_name'}}})
print("Renamed Data:\n", renamed_data)



# Sample data
data = pd.DataFrame({
    'Height': [160, 165, 170, 175, 180],
    'Weight': [55, 60, 65, 70, 75],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female'],
    'Smoker': ['No', 'Yes', 'No', 'No', 'Yes']
})

# Initialize the StatisticalAnalyzer with the dataset
analyzer = StatisticalAnalyzer(data=data)

# Example 1: Compute basic statistics for 'Height' and 'Weight'
basic_stats = analyzer.compute_basic_statistics(['Height', 'Weight'])
print("Basic Statistics:\n", basic_stats)

# Example 2: Perform a t-test between 'Height' and 'Weight'
t_test_result = analyzer.perform_hypothesis_tests(test_type='t-test', columns=['Height', 'Weight'])
print("T-test Result:\n", t_test_result)

# Example 3: Perform a chi-squared test between 'Gender' and 'Smoker'
chi_squared_result = analyzer.perform_hypothesis_tests(test_type='chi-squared', columns=['Gender', 'Smoker'])
print("Chi-squared Test Result:\n", chi_squared_result)



# Example 1: Basic Line Plot Export
fig, ax = plt.subplots()
ax.plot([0, 1, 2], [0, 1, 4], label='Line Plot')
ax.set_title('Simple Line Plot')
ax.legend()

exporter = TableauExporter(fig)
exporter.export_to_tableau_format('line_plot.png')

# Example 2: Bar Chart Export
fig, ax = plt.subplots()
ax.bar(['A', 'B', 'C'], [5, 7, 9], color='skyblue')
ax.set_title('Bar Chart')

exporter = TableauExporter(fig)
exporter.export_to_tableau_format('bar_chart.png')

# Example 3: Scatter Plot Export
fig, ax = plt.subplots()
ax.scatter([10, 20, 30], [3, 6, 9], color='green', label='Scatter')
ax.set_title('Scatter Plot')
ax.legend()

exporter = TableauExporter(fig)
exporter.export_to_tableau_format('scatter_plot.png')



# Example 1: Set a specific color palette and font size
style_options = {
    'color_palette': 'ggplot',
    'font_size': 10
}
setup_visualization_style(style_options)
plt.plot([0, 1, 2], [0, 1, 4])
plt.title("Styled Plot with GGPlot Palette")
plt.show()

# Example 2: Configure line style and figure size
style_options = {
    'line_style': '--',
    'figure_size': (10, 5)
}
setup_visualization_style(style_options)
plt.plot([0, 1, 2], [0, 1, 4])
plt.title("Styled Plot with Dashed Lines")
plt.show()

# Example 3: Change background color of axes
style_options = {
    'background_color': 'lightgrey'
}
setup_visualization_style(style_options)
plt.plot([0, 1, 2], [0, 1, 4])
plt.title("Styled Plot with Light Grey Background")
plt.show()



# Example 1: Setup logging with 'DEBUG' level
setup_logging('DEBUG')
logging.debug("This is a debug message.")
logging.info("Debug level set, info message should appear.")
logging.warning("This is a warning message.")
logging.error("This is an error message.")
logging.critical("This is a critical message.")

# Example 2: Setup logging with 'INFO' level
setup_logging('INFO')
logging.debug("This debug message should not appear.")
logging.info("Info level set, this info message should appear.")
logging.warning("This is a warning message.")
logging.error("This is an error message.")
logging.critical("This is a critical message.")

# Example 3: Setup logging with 'ERROR' level
setup_logging('ERROR')
logging.debug("This debug message should not appear.")
logging.info("This info message should not appear.")
logging.warning("This warning message should not appear.")
logging.error("Error level set, this error message should appear.")
logging.critical("This is a critical message.")
