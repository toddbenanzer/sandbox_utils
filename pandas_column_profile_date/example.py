from date_statistics import DateStatisticsCalculator  # Assuming this is the correct import path
from mymodule import DataValidator  # Replace `mymodule` with the actual module name
from mymodule import load_dataframe  # Replace `mymodule` with the actual module name
from mymodule import save_descriptive_statistics  # Replace `mymodule` with the actual module name
from mymodule import setup_logging  # Replace `mymodule` with the actual module name
import json
import logging
import numpy as np
import pandas as pd


# Example 1: Basic date range analysis
data1 = pd.DataFrame({
    'event_dates': pd.to_datetime(['2022-01-01', '2022-02-01', '2022-03-01', '2022-01-01'])
})
calculator1 = DateStatisticsCalculator(data1, 'event_dates')
print("Example 1 Results:", calculator1.analyze_dates())

# Example 2: Handling missing and empty date values
data2 = pd.DataFrame({
    'dates_with_missing': [pd.NaT, '2022-04-01', '', '2022-04-02']
})
calculator2 = DateStatisticsCalculator(data2, 'dates_with_missing')
print("Example 2 Results:", calculator2.analyze_dates())

# Example 3: Checking for trivial columns
data3 = pd.DataFrame({
    'same_dates': pd.to_datetime(['2022-05-05', '2022-05-05', '2022-05-05'])
})
calculator3 = DateStatisticsCalculator(data3, 'same_dates')
print("Example 3 Results:", calculator3.analyze_dates())

# Example 4: Most common dates analysis
data4 = pd.DataFrame({
    'event_timestamps': pd.to_datetime(['2022-06-01', '2022-06-01', '2022-06-02', '2022-06-01', '2022-06-03'])
})
calculator4 = DateStatisticsCalculator(data4, 'event_timestamps')
print("Example 4 Results:", calculator4.analyze_dates())



# Example 1: Validating a date column
data1 = pd.DataFrame({
    'dates': ['2021-01-01', 'invalid_date', '2021-01-03']
})
validator1 = DataValidator(data1, 'dates')
is_valid_dates = validator1.validate_date_column()
print("Is the date column valid?:", is_valid_dates)

# Example 2: Handling missing values by dropping them
data2 = pd.DataFrame({
    'dates': [pd.NaT, '2021-01-01', pd.NaT, '2021-01-03']
})
validator2 = DataValidator(data2, 'dates')
cleaned_dates = validator2.handle_missing_values(strategy='drop')
print("Dates after dropping missing values:\n", cleaned_dates)

# Example 3: Handling missing values by filling with a specific date
data3 = pd.DataFrame({
    'dates': [pd.NaT, '2021-01-01', pd.NaT, '2021-01-03']
})
validator3 = DataValidator(data3, 'dates')
filled_dates = validator3.handle_missing_values(strategy='fill', fill_value=pd.Timestamp('2020-01-01'))
print("Dates after filling missing values:\n", filled_dates)

# Example 4: Handling infinite values in a numerical column
data4 = pd.DataFrame({
    'values': [1, np.inf, -np.inf, 3]
})
validator4 = DataValidator(data4, 'values')
finite_values = validator4.handle_infinite_values()
print("Values after handling infinite values:\n", finite_values)



# Example 1: Successfully load a CSV file
try:
    df = load_dataframe('data/sales_data.csv')
    print("DataFrame loaded successfully.")
    print(df.head())
except Exception as e:
    print(f"Error: {e}")

# Example 2: Attempt to load a non-existent file
try:
    df = load_dataframe('data/non_existent.csv')
except FileNotFoundError as e:
    print(f"Error: {e}")

# Example 3: Load a CSV file and check its structure
try:
    df = load_dataframe('data/product_data.csv')
    print(f"DataFrame columns: {df.columns}")
    print(f"DataFrame shape: {df.shape}")
except Exception as e:
    print(f"Error: {e}")

# Example 4: Handle a CSV file that may be empty
try:
    df = load_dataframe('data/empty_sales.csv')
except ValueError as e:
    print(f"Error: {e}")



# Example 1: Save statistics as a JSON file
statistics1 = {
    "date_range": {"min_date": "2022-01-01", "max_date": "2022-12-31"},
    "distinct_count": 365,
    "most_common_dates": [("2022-07-04", 1)],
    "missing_and_empty_count": {"missing_values_count": 0}
}
save_descriptive_statistics(statistics1, 'output/statistics.json')

# Example 2: Save statistics as a CSV file
statistics2 = {
    "date_range": {"min_date": "2021-01-01", "max_date": "2021-12-31"},
    "distinct_count": 365,
    "most_common_dates": [("2021-07-04", 1)],
    "missing_and_empty_count": {"missing_values_count": 0}
}
save_descriptive_statistics(statistics2, 'output/statistics.csv')

# Example 3: Attempt to save statistics with unsupported file extension
try:
    save_descriptive_statistics(statistics1, 'output/statistics.txt')
except ValueError as e:
    print(f"Error: {e}")

# Example 4: Check for creating necessary directories and saving as JSON
statistics3 = {
    "date_analysis": {"min_date": "2023-01-01", "max_date": "2023-12-31"},
    "count_distinct": 365,
    "common_dates": [("2023-07-04", 1)],
    "missing_count": {"missing_values": 0}
}
save_descriptive_statistics(statistics3, 'output/newfolder/statistics.json')



# Example 1: Basic setup with default logging level (INFO)
setup_logging()
logging.info("This is an info message.")  # This will be logged
logging.debug("This is a debug message.")  # This will not be logged

# Example 2: Change logging level to DEBUG
setup_logging(level=logging.DEBUG)
logging.debug("This is a debug message.")  # This will be logged

# Example 3: Logging a warning message
setup_logging()
logging.warning("This is a warning message.")

# Example 4: Configure additional details for error messages
try:
    x = 1 / 0
except ZeroDivisionError:
    logging.error("An error occurred: Division by zero", exc_info=True)
