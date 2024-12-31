from data_analyzer import DataAnalyzer
import logging
import pandas as pd

# Example 1: Define groups for an A/B test
designer = ExperimentDesigner()
group_sizes = designer.define_groups(control_size=120, experiment_size=180)
print(f"Group Sizes: {group_sizes}")

# Example 2: Randomize participants into control and experimental groups
participants = ['user1', 'user2', 'user3', 'user4', 'user5', 'user6']
allocation = designer.randomize_participants(participants)
print(f"Control Group: {allocation['control']}")
print(f"Experiment Group: {allocation['experiment']}")

# Example 3: Set factors for the experiment
factors = {
    'button_color': ['green', 'red'],
    'layout': ['A', 'B']
}
designer.set_factors(factors)
print(f"Factors: {designer.factors}")


# Example 1: Capture metrics from a data source
collector = DataCollector()
metrics_collected = collector.capture_metrics('api_source', ['views', 'engagement'])
print(metrics_collected)

# Example 2: Successful integration with a marketing platform
platform_info = {'name': 'Facebook Ads', 'API_key': 'secure_key123'}
integration_status = collector.integrate_with_platform(platform_info)
print(f"Integration successful: {integration_status}")

# Example 3: Failed integration with missing API key
platform_info_incomplete = {'name': 'Facebook Ads'}
integration_status_failed = collector.integrate_with_platform(platform_info_incomplete)
print(f"Integration successful: {integration_status_failed}")



# Example Dataset
data = pd.DataFrame({
    'A': [10, 20, 30, 40],
    'B': [100, 200, 300, 400]
})

# Initialize DataAnalyzer
analyzer = DataAnalyzer(data)

# Example 1: Perform t-test
t_test_result = analyzer.perform_statistical_tests('t-test')
print("T-Test Result:", t_test_result)

# Example 2: Perform chi-squared test
chi_squared_result = analyzer.perform_statistical_tests('chi-squared')
print("Chi-Squared Result:", chi_squared_result)

# Example 3: Attempt to perform an invalid test
invalid_test_result = analyzer.perform_statistical_tests('invalid_test')
print("Invalid Test Result:", invalid_test_result)

# Example 4: Visualize data using a bar chart
analyzer.visualize_data(data, 'bar')

# Example 5: Visualize data using a line chart
analyzer.visualize_data(data, 'line')

# Example 6: Attempt to visualize data using an invalid chart type
analyzer.visualize_data(data, 'invalid')


# Example 1: Generate a summary from analysis results
analysis_results = {
    'p-value': 0.04,
    'statistic': 3.89,
    'test_type': 'chi-squared'
}
report_generator = ReportGenerator(analysis_results)
summary = report_generator.generate_summary()

# Example 2: Create a bar chart visual report
visualization_details = {
    'charts': [
        {'data': {'Category 1': 100, 'Category 2': 150, 'Category 3': 200}, 'type': 'bar', 'title': 'Sales by Category'}
    ]
}
report_generator.create_visual_report(visualization_details)

# Example 3: Create a pie chart visual report
visualization_details = {
    'charts': [
        {'data': {'Segment A': 40, 'Segment B': 60}, 'type': 'pie', 'title': 'Market Share'}
    ]
}
report_generator.create_visual_report(visualization_details)


# Example 1: Initialize UserInterface
ui = UserInterface()

# Example 2: Launch command-line interface
ui.launch_cli()

# Example 3: Launch graphical user interface
ui.launch_gui()

# Example 4: Process input parameters
input_params = {'experiment_name': 'Test A/B Experiment', 'duration_days': 30, 'metrics': ['clicks', 'conversions']}
processed_params = ui.process_input_parameters(input_params)
print(f"Returned Processed Parameters: {processed_params}")


# Example 1: Load data from a CSV file
try:
    csv_data = load_data_from_source('example.csv')
    print(csv_data.head())
except Exception as e:
    print(e)

# Example 2: Load data from an Excel file
try:
    excel_data = load_data_from_source('example.xlsx')
    print(excel_data.head())
except Exception as e:
    print(e)

# Example 3: Attempt to load data from an unsupported file format
try:
    unsupported_data = load_data_from_source('example.txt')
except Exception as e:
    print(e)

# Example 4: Attempt to load data from a non-existent file
try:
    nonexistent_data = load_data_from_source('missing.csv')
except Exception as e:
    print(e)



# Example 1: Save DataFrame to CSV
data = pd.DataFrame({'Name': ['Alice', 'Bob'], 'Age': [30, 24]})
save_results_to_file(data, 'output/results.csv')

# Example 2: Save DataFrame to JSON
data = [{'Name': 'Charlie', 'Age': 29}, {'Name': 'David', 'Age': 35}]
save_results_to_file(data, 'output/results.json')

# Example 3: Save DataFrame to Excel
data = pd.DataFrame({'Product': ['Apples', 'Bananas'], 'Quantity': [100, 150]})
save_results_to_file(data, 'output/inventory.xlsx')

# Example 4: Attempt to save to an unsupported file format
try:
    data = pd.DataFrame({'Metric': ['Speed', 'Accuracy'], 'Value': [100, 95]})
    save_results_to_file(data, 'output/results.txt')
except ValueError as e:
    print(e)



# Example 1: Set logging level to DEBUG
setup_logging_config('DEBUG')
logging.debug("This is a debug message.")
logging.info("This is an info message.")

# Example 2: Set logging level to INFO
setup_logging_config('INFO')
logging.info("Info level logging started.")
logging.warning("This is a warning message.")

# Example 3: Set logging level to ERROR
setup_logging_config('ERROR')
logging.error("This is an error message.")
logging.critical("This is a critical message.")

# Example 4: Use an invalid logging level
setup_logging_config('INVALID')
logging.warning("Default level set to WARNING due to invalid input.")


# Example 1: Load a JSON configuration file
try:
    config = parse_experiment_config('experiment_config.json')
    print(config)
except Exception as e:
    print(e)

# Example 2: Load a YAML configuration file
try:
    config = parse_experiment_config('experiment_config.yaml')
    print(config)
except Exception as e:
    print(e)

# Example 3: Attempt to load a non-existing configuration file
try:
    config = parse_experiment_config('nonexistent_config.json')
except Exception as e:
    print(e)

# Example 4: Attempt to load an unsupported file format
try:
    config = parse_experiment_config('experiment_config.txt')
except Exception as e:
    print(e)
