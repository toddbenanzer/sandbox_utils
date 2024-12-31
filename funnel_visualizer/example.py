

# Example 1: Initializing a Funnel with predefined stages
initial_stages = [
    {'name': 'Awareness', 'metrics': {'visitors': 1000}},
    {'name': 'Interest', 'metrics': {'engaged_users': 750}}
]

funnel = Funnel(initial_stages)
print(funnel.get_stages())

# Example 2: Adding a new stage to the Funnel
funnel.add_stage('Consideration', {'users_checked_product': 500})
print(funnel.get_stages())

# Example 3: Removing an existing stage from the Funnel
funnel.remove_stage('Interest')
print(funnel.get_stages())

# Example 4: Attempting to add a duplicate stage (will raise an error)
try:
    funnel.add_stage('Awareness', {'visitors': 1000})
except ValueError as e:
    print(e)

# Example 5: Retrieving stages from the Funnel
stages = funnel.get_stages()
for stage in stages:
    print(f"Stage: {stage['name']}, Metrics: {stage['metrics']}")


# Example 1: Loading CSV data
csv_source = 'path/to/funnel_data.csv'
data_handler = DataHandler(csv_source)
loaded_data = data_handler.load_data('csv')
print("Loaded CSV Data:", loaded_data)

# Example 2: Loading JSON data
json_source = 'path/to/funnel_data.json'
data_handler = DataHandler(json_source)
loaded_data = data_handler.load_data('json')
print("Loaded JSON Data:", loaded_data)

# Example 3: Filtering data by specified stage
csv_source = 'path/to/funnel_data.csv'
data_handler = DataHandler(csv_source)
data_handler.load_data('csv')
awareness_data = data_handler.filter_data_by_stage('Awareness')
print("Filtered Data for Awareness Stage:", awareness_data)

# Example 4: Handling unsupported data format
try:
    data_handler.load_data('xml')
except ValueError as e:
    print("Error:", e)

# Example 5: Attempting to filter without loading data
try:
    empty_handler = DataHandler('path/to/missing.csv')
    empty_handler.filter_data_by_stage('Consideration')
except ValueError as e:
    print("Error:", e)


# Example 1: Initializing the Visualization with funnel data
funnel_data = [
    {'name': 'Awareness', 'metrics': {'user_count': 1000, 'conversion_rate': 70}},
    {'name': 'Interest', 'metrics': {'user_count': 700, 'conversion_rate': 60}},
    {'name': 'Consideration', 'metrics': {'user_count': 420, 'conversion_rate': 50}},
]

visualization = Visualization(funnel_data)

# Example 2: Creating a funnel chart
visualization.create_funnel_chart(title='Marketing Funnel Chart', palette='Greens_d')

# Example 3: Creating a conversion chart
visualization.create_conversion_chart(title='Stage Conversion Rates Chart', color='red')

# Example 4: Attempting to export a visualization (will raise an error)
try:
    visualization.export_visualization(file_type='pdf')
except NotImplementedError as e:
    print(e)


# Example 1: Initializing MetricsCalculator with funnel data
funnel_data = [
    {'name': 'Awareness', 'metrics': {'user_count': 1000, 'conversion_rate': 70}},
    {'name': 'Interest', 'metrics': {'user_count': 700, 'conversion_rate': 60}},
    {'name': 'Consideration', 'metrics': {'user_count': 420, 'conversion_rate': 50}},
]
calculator = MetricsCalculator(funnel_data)

# Example 2: Calculating conversion rate between stages
conversion_rate = calculator.calculate_conversion_rate('Awareness', 'Interest')
print(f"Conversion Rate from Awareness to Interest: {conversion_rate}%")

# Example 3: Calculating drop-off rate for a stage
drop_off_rate = calculator.calculate_drop_off('Interest')
print(f"Drop-off Rate at Interest Stage: {drop_off_rate}%")

# Example 4: Getting summary statistics for the funnel
summary_statistics = calculator.get_summary_statistics()
print("Summary Statistics:", summary_statistics)

# Example 5: Handling errors for unknown stages
try:
    calculator.calculate_conversion_rate('Unknown', 'Interest')
except ValueError as e:
    print("Error:", e)


# Example 1: Starting the CLI and displaying help
cli = CLI()
print(cli.parse_command("help"))

# Example 2: Loading data from a CSV file
cli = CLI()
try:
    result = cli.parse_command("load_data path/to/data.csv csv")
    print(result)
except Exception as e:
    print("Error loading data:", e)

# Example 3: Attempting to display a funnel chart without initializing visualization
try:
    cli.parse_command("funnel_chart")
except ValueError as e:
    print("Visualization error:", e)

# Example 4: Calculating metrics without initializing the MetricsCalculator
try:
    cli.parse_command("calculate_metrics")
except ValueError as e:
    print("MetricsCalculator error:", e)

# Example 5: Handling an unknown command
try:
    cli.parse_command("some_unknown_command")
except ValueError as e:
    print("Command error:", e)


# Example 1: Exporting data to CSV format
data_list = [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]
print(export_to_format(data_list, 'csv', 'output/data.csv'))

# Example 2: Exporting data to JSON format
data_dict = {'name': 'Alice', 'age': 30}
print(export_to_format(data_dict, 'json', 'output/data.json'))

# Example 3: Exporting a DataFrame to Excel format
data_df = pd.DataFrame([{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}])
print(export_to_format(data_df, 'xlsx', 'output/data.xlsx'))

# Example 4: Handling unsupported export format
try:
    print(export_to_format(data_list, 'xml', 'output/data.xml'))
except ValueError as e:
    print("Error:", e)

# Example 5: Handling file IO errors
try:
    print(export_to_format(data_list, 'csv', '/root/protected/data.csv'))  # Attempting to write to a protected location
except IOError as e:
    print("Error:", e)


# Example 1: Setting up logging at the INFO level
setup_logging('INFO')
logging.info("This is an info message.")
logging.debug("This debug message will not appear, as the level is set to INFO.")

# Example 2: Setting up logging at the DEBUG level
setup_logging('DEBUG')
logging.debug("This debug message will appear, as the level is set to DEBUG.")

# Example 3: Changing logging to WARNING level
setup_logging('WARNING')
logging.info("This info message will not appear, as the level is set to WARNING.")
logging.warning("This is a warning message.")

# Example 4: Attempting to set an unsupported logging level
try:
    setup_logging('VERBOSE')
except ValueError as e:
    print("Error:", e)


# Example 1: Loading a valid configuration file
config_file_path = 'config/settings.json'
setup_config(config_file_path)  # Assuming settings.json exists and is correctly formatted

# Example 2: Handling a missing configuration file
try:
    setup_config('config/missing_config.json')  # Non-existent file
except FileNotFoundError as e:
    print("Error:", e)

# Example 3: Handling a configuration file with invalid JSON
invalid_config_path = 'config/invalid_config.json'
# Suppose invalid_config.json contains incorrect JSON format
try:
    setup_config(invalid_config_path)
except ValueError as e:
    print("Error:", e)

# Example 4: Handling a generic error during configuration loading
try:
    setup_config('config/generic_error_config.json')  # Simulate a generic error scenario
except Exception as e:
    print("Error:", e)

# Example 5: Applying configuration settings
# Assume 'config/app_config.json' contains: {"app_mode": "production", "max_users": 100}
setup_config('config/app_config.json')  # This will print: app_mode: production \n max_users: 100
