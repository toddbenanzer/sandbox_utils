from data_analyzer import DataAnalyzer
from data_preprocessor import DataPreprocessor
from data_visualizer import DataVisualizer
from report_generator import ReportGenerator
from setup_logging_function import setup_logging
import logging
import numpy as np
import pandas as pd
import time

# Example 1: Fetching data from an API
api_config = {
    'url': 'https://jsonplaceholder.typicode.com/todos',
    'params': {}
}

fetcher_api = DataFetcher('API', api_config)
api_data = fetcher_api.fetch_data()
print("API Data:", api_data)

# Example 2: Fetching data from a SQLite database
database_config = {
    'dbname': 'my_database.db',
    'query': 'SELECT * FROM my_table'
}

fetcher_db = DataFetcher('database', database_config)
db_data = fetcher_db.fetch_data()
print("Database Data:\n", db_data)

# Example 3: Fetching data from a CSV file
csv_config = {
    'file_path': 'data.csv'
}

fetcher_csv = DataFetcher('CSV', csv_config)
csv_data = fetcher_csv.fetch_data()
print("CSV Data:\n", csv_data)

# Example 4: Scheduling data refresh from a CSV file every 10 minutes
fetcher_csv.schedule_refresh(10)

# To stop the scheduled refreshes
fetcher_csv.stop_refresh()



# Example 1: Handle missing values by filling with the mean
data = pd.DataFrame({"A": [1, np.nan, 3], "B": [4, 5, np.nan]})
preprocessor = DataPreprocessor(data)
filled_data = preprocessor.handle_missing_values("fill_mean")
print("Filled Data with Mean:\n", filled_data)

# Example 2: Normalize data using min-max normalization
data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
preprocessor = DataPreprocessor(data)
normalized_data = preprocessor.normalize_data("min-max")
print("Min-Max Normalized Data:\n", normalized_data)

# Example 3: Apply square root transformation
data = pd.DataFrame({"A": [0, 4, 9], "B": [16, 25, 36]})
preprocessor = DataPreprocessor(data)
sqrt_transformed_data = preprocessor.transform_data("sqrt")
print("Square Root Transformed Data:\n", sqrt_transformed_data)

# Example 4: Encode categorical variables
data = pd.DataFrame({"A": ["cat", "dog", "cat"]})
preprocessor = DataPreprocessor(data)
encoded_data = preprocessor.transform_data("encode_categorical")
print("Encoded Categorical Data:\n", encoded_data)



# Example 1: Descriptive Statistics
data = pd.DataFrame({
    "Height": [150, 160, 170, 180, 190],
    "Weight": [50, 60, 70, 80, 90]
})
analyzer = DataAnalyzer(data)
descriptive_stats = analyzer.descriptive_statistics()
print("Descriptive Statistics:\n", descriptive_stats)

# Example 2: Exploratory Data Analysis
eda_results = analyzer.exploratory_data_analysis()
print("Exploratory Data Analysis (Correlation Matrix):\n", eda_results)

# Example 3: Linear Regression Analysis
data_linear = pd.DataFrame({
    "X": [1, 2, 3, 4, 5],
    "y": [2, 4, 6, 8, 10]
})
analyzer_linear = DataAnalyzer(data_linear)
linear_regression_results = analyzer_linear.regression_analysis("linear")
print("Linear Regression Results:\n", linear_regression_results)

# Example 4: Logistic Regression Analysis
data_logistic = pd.DataFrame({
    "Age": [22, 25, 47, 52, 46],
    "Purchased": [0, 0, 1, 1, 1]
})
analyzer_logistic = DataAnalyzer(data_logistic)
logistic_regression_results = analyzer_logistic.regression_analysis("logistic")
print("Logistic Regression Results:\n", logistic_regression_results)



# Example 1: Creating a Bar Plot
data = pd.DataFrame({
    'Fruits': ['Apple', 'Banana', 'Cherry'],
    'Sales': [100, 150, 50]
})
visualizer = DataVisualizer(data)
visualizer.generate_plot('bar', x='Fruits', y='Sales')

# Example 2: Customizing the Plot
visualizer.customize_plot(title='Fruit Sales', xlabel='Type of Fruit', ylabel='Number Sold', grid=True)

# Example 3: Exporting the Plot to a PNG file
visualizer.export_visualization(format='png', file_name='fruit_sales')

# Example 4: Creating a Line Plot
data_line = pd.DataFrame({
    'Year': [2018, 2019, 2020],
    'Revenue': [400, 450, 500]
})
line_visualizer = DataVisualizer(data_line)
line_visualizer.generate_plot('line', x='Year', y='Revenue')

# Example 5: Creating a Scatter Plot
data_scatter = pd.DataFrame({
    'Height': [150, 160, 170, 180],
    'Weight': [50, 60, 70, 80]
})
scatter_visualizer = DataVisualizer(data_scatter)
scatter_visualizer.generate_plot('scatter', x='Height', y='Weight')



# Example 1: Creating a PDF Report
analysis_results = {
    "Summary": "The data indicates a steady increase in sales.",
    "Conclusion": "Implementing new marketing strategies has proven effective."
}
report_gen = ReportGenerator(analysis_results)
pdf_report_path = report_gen.create_report('pdf')
print(f"PDF report created at: {pdf_report_path}")

# Example 2: Creating an Excel Report
df_results = pd.DataFrame({
    "A": [1, 2, 3],
    "B": [4, 5, 6]
})
excel_report_gen = ReportGenerator(df_results)
xlsx_report_path = excel_report_gen.create_report('xlsx')
print(f"Excel report created at: {xlsx_report_path}")

# Example 3: Scheduling Report Generation
report_gen.schedule_report(1, 'local')
print("Scheduled report generation every hour. Report will be saved locally.")

# Example 4: Stopping Scheduled Report Generation
time.sleep(10)  # Wait for demonstration purposes
report_gen.stop_scheduling()
print("Stopped scheduled report generation.")


# Example 1: Load Configuration from a JSON File
config_json_path = 'config.json'
# Example JSON content:
# {
#     "database": {
#         "host": "localhost",
#         "port": 3306,
#         "user": "admin",
#         "password": "password"
#     }
# }
config_data_json = load_config(config_json_path)
print("JSON Config:", config_data_json)

# Example 2: Load Configuration from a YAML File
config_yaml_path = 'config.yaml'
# Example YAML content:
# database:
#   host: localhost
#   port: 3306
#   user: admin
#   password: password
config_data_yaml = load_config(config_yaml_path)
print("YAML Config:", config_data_yaml)


# Example 1: Set up the environment with common libraries
dependencies = ['numpy', 'pandas', 'matplotlib']
setup_environment(dependencies)
# Expected: Either confirms the packages are installed or installs them if missing.

# Example 2: Set up the environment with a single package
setup_environment(['requests'])
# Expected: Confirms if 'requests' is installed or installs it if missing.

# Example 3: Attempt to set up the environment with an invalid package name
setup_environment(['unlikely-package-name'])
# Expected: Attempts to install the package and reports an error if it fails.



# Example 1: Validate a correctly formatted DataFrame
data_valid = pd.DataFrame({
    'column1': [1, 2, 3],
    'column2': [0.1, 0.2, 0.3],
    'column3': ['A', 'B', 'C']
})
is_valid, message = validate_data(data_valid)
print("Is valid:", is_valid)  # Expected: True
print("Message:", message)    # Expected: None

# Example 2: Validate a DataFrame with missing values
data_with_missing = pd.DataFrame({
    'column1': [1, 2, None],  # Missing value in column1
    'column2': [0.1, 0.2, 0.3],
    'column3': ['A', 'B', 'C']
})
is_valid, message = validate_data(data_with_missing)
print("Is valid:", is_valid)  # Expected: False
print("Message:", message)    # Expected: "The dataset contains missing values."

# Example 3: Validate a DataFrame with missing columns
data_missing_columns = pd.DataFrame({
    'column1': [1, 2, 3],
    'column3': ['A', 'B', 'C']  # Missing column2
})
is_valid, message = validate_data(data_missing_columns)
print("Is valid:", is_valid)  # Expected: False
print("Message:", message)    # Expected: "Missing required columns: column2"

# Example 4: Validate a DataFrame with incorrect column types
data_wrong_types = pd.DataFrame({
    'column1': [1, 2, 3],
    'column2': [1, 2, 3],  # Should be float64, not int64
    'column3': ['A', 'B', 'C']
})
is_valid, message = validate_data(data_wrong_types)
print("Is valid:", is_valid)  # Expected: False
print("Message:", message)    # Expected: "Column 'column2' should be of type float64, but found int64."



# Example 1: Set logging to DEBUG level
setup_logging('DEBUG')
logging.debug("This is a debug message.")
logging.info("This is an info message.")
# Expected: Both messages will be printed.

# Example 2: Set logging to INFO level
setup_logging('INFO')
logging.info("This is another info message.")
logging.debug("This message will not be printed.")
# Expected: Only the info message is printed.

# Example 3: Set logging to WARNING level
setup_logging('WARNING')
logging.warning("This is a warning message.")
logging.error("This is an error message.")
logging.info("This message will not be printed.")
# Expected: Warning and error messages are printed.

# Example 4: Attempt to set an invalid logging level
try:
    setup_logging('INVALID')
except ValueError as e:
    print(e)
# Expected: Raises ValueError and prints the error message.
