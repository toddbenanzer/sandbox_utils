from data_exporter import DataExporter
from export_as_csv import export_as_csv
from export_as_tdsx import export_as_tdsx
from read_user_config import read_user_config
from setup_logging import setup_logging
from validate_dataframe import validate_dataframe
import json
import logging
import pandas as pd


# Example 1: Simple CSV Export
data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35]}
df = pd.DataFrame(data)
exporter = DataExporter(df)
exporter.to_tableau_csv('output_simple.csv')

# Example 2: CSV Export with Custom Delimiter
exporter.to_tableau_csv('output_custom_delim.csv', sep=';')

# Example 3: Attaching Metadata
metadata = {'description': 'Sample dataset', 'created_by': 'Data Team'}
exporter.attach_metadata(metadata)
print(exporter.metadata)

# Example 4: Applying User Configuration
config = {'include_index': False, 'date_format': 'YYYY-MM-DD'}
exporter.apply_user_config(config)

# Example 5: Handling Export Error
try:
    # Intentionally cause an error by passing an invalid path
    exporter.to_tableau_csv('/invalid_path/output.csv')
except Exception as e:
    exporter.handle_errors(e)



# Example 1: Valid DataFrame
df_valid = pd.DataFrame({
    'Name': ['Alice', 'Bob'],
    'Age': [30, 25],
    'Joined': pd.to_datetime(['2021-01-01', '2021-02-01'])
})

try:
    if validate_dataframe(df_valid):
        print("DataFrame is valid for export.")
except ValueError as e:
    print(f"Validation error: {e}")

# Example 2: DataFrame with Empty Data
df_empty = pd.DataFrame()

try:
    if validate_dataframe(df_empty):
        print("DataFrame is valid for export.")
except ValueError as e:
    print(f"Validation error: {e}")

# Example 3: DataFrame with Null Values
df_with_nulls = pd.DataFrame({
    'Name': ['Alice', None],
    'Age': [30, 25]
})

try:
    if validate_dataframe(df_with_nulls):
        print("DataFrame is valid for export.")
except ValueError as e:
    print(f"Validation error: {e}")

# Example 4: DataFrame with Invalid Column Names
df_invalid_columns = pd.DataFrame({
    'Invalid Column@': [1, 2],
    'ValidColumn': [3, 4]
})

try:
    if validate_dataframe(df_invalid_columns):
        print("DataFrame is valid for export.")
except ValueError as e:
    print(f"Validation error: {e}")

# Example 5: DataFrame with Unsupported Data Types
df_unsupported_types = pd.DataFrame({
    'Name': ['Alice', 'Bob'],
    'ComplexNumbers': [1+1j, 2+2j]
})

try:
    if validate_dataframe(df_unsupported_types):
        print("DataFrame is valid for export.")
except ValueError as e:
    print(f"Validation error: {e}")



# Enable logging
logging.basicConfig(level=logging.INFO)

# Example 1: Basic CSV Export
data = {'Name': ['Alice', 'Bob'], 'Age': [30, 25]}
df = pd.DataFrame(data)
export_as_csv(df, 'output_basic.csv')

# Example 2: CSV Export Without Index
export_as_csv(df, 'output_no_index.csv', index=False)

# Example 3: CSV Export with Custom Separator
export_as_csv(df, 'output_custom_sep.csv', sep=';')

# Example 4: Handling Export Failure (Invalid Path)
try:
    export_as_csv(df, '/invalid_path/output.csv')
except Exception as e:
    print(f"Error occurred: {e}")



# Enable logging
logging.basicConfig(level=logging.INFO)

# Example 1: Export DataFrame as TDSX using Hyper
data = {'Name': ['Alice', 'Bob'], 'Age': [30, 25]}
df = pd.DataFrame(data)
success = export_as_tdsx(df, 'output_hyper.tdsx', use_hyper=True)
print("Export successful:", success)

# Example 2: Export DataFrame as TDSX falling back to CSV
df2 = pd.DataFrame({'Product': ['A', 'B'], 'Sales': [100, 200]})
success = export_as_tdsx(df2, 'output_fallback.tdsx')
print("Export successful:", success)

# Example 3: Attempting export when TSC is not installed
try:
    import tableauserverclient as TSC
except ImportError:
    print("Tableau Server Client not installed.")

# Example 4: Handling Export Error
try:
    export_as_tdsx(df, '/invalid_path/output_invalid.tdsx')
except Exception as e:
    logging.error(f"Expected error: {e}")



# Example 1: Set up logging at INFO level
setup_logging(logging.INFO)
logging.info("This is an informational message.")
logging.debug("This debug message will not be shown at INFO level.")

# Example 2: Set up logging at DEBUG level
setup_logging(logging.DEBUG)
logging.info("Debugging enabled, this message and all other levels will show.")
logging.debug("This debug message will now be visible.")

# Example 3: Attempt setting an invalid logging level, observe fallback handling
try:
    setup_logging(9999)  # An invalid logging level
except Exception as e:
    logging.error(f"Expected logging configuration error: {e}")



# Enable logging
logging.basicConfig(level=logging.INFO)

# Example 1: Successfully loading a valid JSON configuration file
try:
    config = read_user_config("config_valid.json")
    print("Configuration Loaded:", config)
except Exception as e:
    print(f"Error: {e}")

# Example 2: Trying to load a non-existent configuration file
try:
    config = read_user_config("non_existent_config.json")
except FileNotFoundError as e:
    print(f"Error: {e}")

# Example 3: Handling invalid JSON format in configuration file
try:
    config = read_user_config("config_invalid.json")
except ValueError as e:
    print(f"Error: {e}")

# Example 4: Using the configuration settings
try:
    config = read_user_config("config_valid.json")
    if "setting1" in config:
        print(f"Setting1: {config['setting1']}")
except Exception as e:
    print(f"Error: {e}")


# Example 1: Generate and Print Sample Code
sample_code = generate_sample_code()
print(sample_code)

# Example 2: Write Sample Code to a File
sample_code = generate_sample_code()
with open("sample_code.txt", "w") as file:
    file.write(sample_code)
print("Sample code saved to sample_code.txt")

# Example 3: Check for Specific Code Section
sample_code = generate_sample_code()
if "# Example: Validating a DataFrame" in sample_code:
    print("Sample code includes DataFrame validation example.")
