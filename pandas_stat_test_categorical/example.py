from mypackage.data_frame_handler import DataFrameHandler
from mypackage.data_loader import load_dataframe
from mypackage.descriptive_statistics import DescriptiveStatistics
from mypackage.exceptions import DataFrameValidationError
from mypackage.logging_config import configure_logging
from mypackage.output_manager import OutputManager
from mypackage.output_manager import save_summary
from mypackage.statistical_tests import StatisticalTests
import logging
import pandas as pd


# Example 1: Basic Usage with Valid DataFrame
try:
    df = pd.DataFrame({
        'Category1': pd.Categorical(['Apple', 'Banana', 'Apple', None, 'Banana']),
        'Category2': pd.Categorical(['Red', 'Yellow', 'Red', 'Yellow', float('inf')])
    })
    handler = DataFrameHandler(df, ['Category1', 'Category2'])
    processed_df = handler.handle_missing_infinite_values()

    print("Processed DataFrame:")
    print(processed_df)
except DataFrameValidationError as e:
    print(f"Error: {e}")

# Example 2: Initialization with an Empty DataFrame
try:
    empty_df = pd.DataFrame()
    handler_empty = DataFrameHandler(empty_df, [])
except DataFrameValidationError as e:
    print(f"Error: {e}")

# Example 3: DataFrame with Missing Column
try:
    df_missing_column = pd.DataFrame({
        'Category1': pd.Categorical(['Dog', 'Cat', 'Dog']),
    })
    handler_missing = DataFrameHandler(df_missing_column, ['Category1', 'Category2'])
except DataFrameValidationError as e:
    print(f"Error: {e}")

# Example 4: Handling Non-Categorical Column
try:
    df_non_cat = pd.DataFrame({
        'Category1': pd.Categorical(['Circle', 'Square', 'Circle']),
        'Numeric': [1, 2, 3]
    })
    handler_non_cat = DataFrameHandler(df_non_cat, ['Category1', 'Numeric'])
except DataFrameValidationError as e:
    print(f"Error: {e}")



# Creating a sample DataFrame with categorical data
df = pd.DataFrame({
    'Category1': pd.Categorical(['Apple', 'Banana', 'Apple', 'Cherry', 'Banana']),
    'Category2': pd.Categorical(['Red', 'Yellow', 'Green', 'Red', 'Yellow']),
})

# Initializing DescriptiveStatistics
stats = DescriptiveStatistics(df, ['Category1', 'Category2'])

# Example 1: Calculate category frequencies
frequencies = stats.calculate_frequencies()
print("Frequencies:")
for column, freq in frequencies.items():
    print(f"{column}:\n{freq}\n")

# Example 2: Compute the mode of the categorical data
modes = stats.compute_mode()
print("Modes:")
for column, mode in modes.items():
    print(f"{column}:\n{mode}\n")

# Example 3: Generate a contingency table for two columns
contingency_table = stats.generate_contingency_table('Category1', 'Category2')
print("Contingency Table:")
print(contingency_table)



# Creating a sample DataFrame with categorical data
df = pd.DataFrame({
    'Fruit': pd.Categorical(['Apple', 'Apple', 'Banana', 'Banana', 'Cherry', 'Cherry']),
    'Color': pd.Categorical(['Red', 'Green', 'Yellow', 'Green', 'Red', 'Yellow']),
})

# Initializing StatisticalTests
stats_tests = StatisticalTests(df, ['Fruit', 'Color'])

# Example 1: Perform Chi-squared test
chi2_statistic, p_value, dof, expected_frequencies = stats_tests.perform_chi_squared_test('Fruit', 'Color')
print(f"Chi-squared Test:\nStatistic={chi2_statistic}, p-value={p_value}, dof={dof}")
print(f"Expected Frequencies:\n{expected_frequencies}\n")

# Example 2: Perform Fisher's exact test
contingency_table = pd.DataFrame([[8, 2], [1, 5]], index=['Row1', 'Row2'], columns=['Col1', 'Col2'])
oddsratio, fisher_p_value = stats_tests.perform_fishers_exact_test(contingency_table)
print(f"Fisher's Exact Test:\nOdds ratio={oddsratio}, p-value={fisher_p_value}\n")



# Example 1: Exporting DataFrame to CSV
df = pd.DataFrame({
    'Item': ['Apple', 'Banana', 'Pear'],
    'Quantity': [10, 5, 15]
})

output_manager = OutputManager(df)
output_manager.export_to_csv('inventory.csv')  # Exports the results to 'inventory.csv'

# Example 2: Exporting DataFrame to Excel
output_manager.export_to_excel('inventory.xlsx')  # Exports the results to 'inventory.xlsx'

# Example 3: Generating a Bar Chart
output_manager.generate_visualization('bar')  # Displays a bar chart of the DataFrame

# Example 4: Generating a Pie Chart
output_manager.generate_visualization('pie')  # Displays a pie chart of the DataFrame



# Example 1: Load data from a CSV file
csv_df = load_dataframe('data.csv')
print("CSV DataFrame:")
print(csv_df)

# Example 2: Load data from an Excel file
excel_df = load_dataframe('data.xlsx')
print("\nExcel DataFrame:")
print(excel_df)

# Example 3: Handling a non-existent file
try:
    missing_df = load_dataframe('missing_file.csv')
except FileNotFoundError as e:
    print(f"\nError: {e}")

# Example 4: Handling unsupported file format
try:
    unsupported_df = load_dataframe('data.txt')
except ValueError as e:
    print(f"\nError: {e}")



# Example 1: Save DataFrame as CSV
data = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35]
})
save_summary(data, 'people.csv', 'csv')

# Example 2: Save DataFrame as Excel
save_summary(data, 'people.xlsx', 'excel')

# Example 3: Handling unsupported format
try:
    save_summary(data, 'people.txt', 'txt')
except ValueError as e:
    print(f"Error: {e}")

# Example 4: Save using case-insensitive format
save_summary(data, 'people_case_insensitive.csv', 'CSV')



# Example 1: Configure logging with INFO level
configure_logging('INFO')
logging.info("This is an info message.")

# Example 2: Configure logging with DEBUG level
configure_logging('DEBUG')
logging.debug("This is a debug message.")

# Example 3: Configure logging with ERROR level
configure_logging('ERROR')
logging.error("This is an error message.")

# Example 4: Handling invalid log level
try:
    configure_logging('VERBOSE')  # Invalid level
except ValueError as e:
    print(f"Error: {e}")

# Example 5: Case-insensitivity in log level
configure_logging('critical')
logging.critical("This is a critical message.")
