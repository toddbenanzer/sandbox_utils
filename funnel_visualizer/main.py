import json
import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

class Funnel:
    """
    A class representing a marketing funnel with multiple stages, 
    allowing the addition, removal, and retrieval of stages and their metrics.
    """

    def __init__(self, stages):
        """
        Initializes the Funnel object with a given list of stages.

        Args:
            stages (list of dicts): Each stage is represented as a dictionary containing the stage name and metrics.
        """
        if not isinstance(stages, list) or not all(isinstance(stage, dict) for stage in stages):
            raise ValueError("Stages should be a list of dictionaries.")
        self.stages = stages

    def add_stage(self, stage_name, metrics):
        """
        Adds a new stage to the funnel with specified metrics.

        Args:
            stage_name (str): The name of the new stage to add.
            metrics (dict): A dictionary containing metrics relevant to the stage.

        Raises:
            ValueError: If the stage already exists or if the metrics are not in dictionary format.
        """
        if any(stage['name'] == stage_name for stage in self.stages):
            raise ValueError(f"Stage '{stage_name}' already exists.")
        if not isinstance(metrics, dict):
            raise ValueError("Metrics should be a dictionary.")
        
        self.stages.append({'name': stage_name, 'metrics': metrics})

    def remove_stage(self, stage_name):
        """
        Removes an existing stage from the funnel.

        Args:
            stage_name (str): The name of the stage to remove.

        Raises:
            ValueError: If the stage does not exist.
        """
        stage_index = next((i for i, stage in enumerate(self.stages) if stage['name'] == stage_name), None)
        if stage_index is None:
            raise ValueError(f"Stage '{stage_name}' does not exist.")
        
        self.stages.pop(stage_index)

    def get_stages(self):
        """
        Retrieves the list of all stages currently defined in the funnel.

        Returns:
            list of dicts: A list where each item is a dictionary representing a funnel stage and its associated metrics.
        """
        return self.stages



class DataHandler:
    """
    A class for handling data operations including loading and filtering data for funnel analysis.
    """

    def __init__(self, source):
        """
        Initializes the DataHandler object with a data source.

        Args:
            source (str or connection object): The data source path or connection object to load the data from.
        """
        self.source = source
        self.data = []

    def load_data(self, format):
        """
        Loads data from the specified source in the given format.

        Args:
            format (str): The data format to load (e.g., 'csv', 'json', 'database').

        Returns:
            list of dicts: A loaded list where each item is a dictionary representing a row of data.

        Raises:
            ValueError: If the specified format is not supported.
        """
        if format == 'csv':
            self.data = pd.read_csv(self.source).to_dict(orient='records')
        elif format == 'json':
            self.data = pd.read_json(self.source).to_dict(orient='records')
        # Add further handling if there is a database connection or other formats
        else:
            raise ValueError(f"Unsupported format '{format}'. Supported formats are: 'csv', 'json'.")
        return self.data

    def filter_data_by_stage(self, stage_name):
        """
        Filters the loaded data to only include records relevant to the specified funnel stage.

        Args:
            stage_name (str): The name of the funnel stage to filter data by.

        Returns:
            list of dicts: A list where each item is a dictionary representing a row of filtered data specific to the stage.

        Raises:
            ValueError: If data is not loaded before filtering.
        """
        if not self.data:
            raise ValueError("Data is not loaded. Please load data before filtering.")

        # Example filtering logic, assuming 'stage' is a field in the data
        filtered_data = [record for record in self.data if record.get('stage') == stage_name]
        return filtered_data



class Visualization:
    """
    A class to handle visualization of funnel data, including funnel charts and conversion charts.
    """

    def __init__(self, funnel_data):
        """
        Initializes the Visualization object with the provided funnel data.

        Args:
            funnel_data (list of dicts): The funnel data used for creating visualizations.
        """
        if not isinstance(funnel_data, list) or not all(isinstance(stage, dict) for stage in funnel_data):
            raise ValueError("Funnel data should be a list of dictionaries.")
        self.funnel_data = funnel_data

    def create_funnel_chart(self, **kwargs):
        """
        Creates a funnel chart to visualize the stages in the funnel and their respective metrics.

        Args:
            **kwargs: Additional parameters for customizing the appearance of the funnel chart.

        Returns:
            The funnel chart display.
        """
        stages = [stage['name'] for stage in self.funnel_data]
        values = [stage['metrics'].get('user_count', 0) for stage in self.funnel_data]
        
        plt.figure(figsize=kwargs.get('figsize', (8, 6)))
        sns.barplot(x=values, y=stages, palette=kwargs.get('palette', 'Blues_d'))
        plt.title(kwargs.get('title', 'Funnel Chart'))
        plt.xlabel('Users')
        plt.ylabel('Stages')
        plt.show()

    def create_conversion_chart(self, **kwargs):
        """
        Creates a conversion chart to visualize conversion rates between different stages in the funnel.

        Args:
            **kwargs: Additional parameters for customizing the appearance of the conversion chart.

        Returns:
            The conversion chart display.
        """
        stages = [stage['name'] for stage in self.funnel_data]
        conversion_rates = [stage['metrics'].get('conversion_rate', 0) for stage in self.funnel_data]

        plt.figure(figsize=kwargs.get('figsize', (8, 6)))
        sns.lineplot(x=stages, y=conversion_rates, marker='o', linestyle='-', 
                     color=kwargs.get('color', 'blue'))
        plt.title(kwargs.get('title', 'Conversion Chart'))
        plt.xlabel('Stages')
        plt.ylabel('Conversion Rate %')
        plt.ylim(0, 100)
        plt.show()

    def export_visualization(self, file_type='png'):
        """
        Exports the generated visualizations to a file in the specified format.

        Args:
            file_type (str): The file format to export the visualization (e.g., 'png', 'pdf', 'svg').

        Returns:
            None; the function saves the file to the designated location.
        """
        # Implement export logic as needed. This could involve capturing current plots and saving them.
        raise NotImplementedError("Export logic is not implemented yet.")


class MetricsCalculator:
    """
    A class to perform metrics calculations on a funnel, such as conversion rates and drop-off rates.
    """

    def __init__(self, funnel_data):
        """
        Initializes the MetricsCalculator object with the provided funnel data.

        Args:
            funnel_data (list of dicts): The funnel data used for calculating metrics.
        """
        if not isinstance(funnel_data, list) or not all(isinstance(stage, dict) for stage in funnel_data):
            raise ValueError("Funnel data should be a list of dictionaries.")
        self.funnel_data = funnel_data

    def calculate_conversion_rate(self, from_stage, to_stage):
        """
        Calculates the conversion rate from one stage to the next.

        Args:
            from_stage (str): The name of the initial stage from which the conversion starts.
            to_stage (str): The name of the target stage to which the conversion is measured.

        Returns:
            float: The conversion rate percentage between the two specified stages.

        Raises:
            ValueError: If either stage is not found in the funnel data.
        """
        from_stage_data = next((stage for stage in self.funnel_data if stage['name'] == from_stage), None)
        to_stage_data = next((stage for stage in self.funnel_data if stage['name'] == to_stage), None)

        if from_stage_data is None or to_stage_data is None:
            raise ValueError(f"Stages '{from_stage}' or '{to_stage}' not found in funnel data.")

        from_users = from_stage_data['metrics'].get('user_count', 0)
        to_users = to_stage_data['metrics'].get('user_count', 0)

        if from_users == 0:
            return 0.0

        conversion_rate = (to_users / from_users) * 100
        return conversion_rate

    def calculate_drop_off(self, stage_name):
        """
        Calculates the drop-off rate at a given stage in the funnel.

        Args:
            stage_name (str): The name of the stage for which the drop-off rate is calculated.

        Returns:
            float: The drop-off rate percentage for the specified stage.

        Raises:
            ValueError: If the stage is not found in the funnel data.
        """
        stage_data = next((stage for stage in self.funnel_data if stage['name'] == stage_name), None)

        if stage_data is None:
            raise ValueError(f"Stage '{stage_name}' not found in funnel data.")

        user_count = stage_data['metrics'].get('user_count', 0)
        conversion_rate = stage_data['metrics'].get('conversion_rate', 0)

        drop_off_rate = 100 - conversion_rate
        return drop_off_rate

    def get_summary_statistics(self):
        """
        Provides summary statistics for the entire funnel, summarizing total users, conversions, and other relevant metrics.

        Returns:
            dict: A dictionary containing summarized statistics.
        """
        total_users = sum(stage['metrics'].get('user_count', 0) for stage in self.funnel_data)
        total_conversions = sum((stage['metrics'].get('conversion_rate', 0) / 100) * stage['metrics'].get('user_count', 0) for stage in self.funnel_data)
        
        if not self.funnel_data:
            average_conversion_rate = 0.0
        else:
            average_conversion_rate = sum(stage['metrics'].get('conversion_rate', 0) for stage in self.funnel_data) / len(self.funnel_data)

        statistics = {
            'total_users': total_users,
            'total_conversions': total_conversions,
            'average_conversion_rate': average_conversion_rate
        }
        return statistics


class CLI:
    """
    A command-line interface for interacting with the funnel visualization and analysis tool.
    """

    def __init__(self):
        """
        Initializes the CLI object for interacting with the tool.
        """
        self.data_handler = None
        self.funnel = None
        self.visualization = None
        self.metrics_calculator = None

    def start(self):
        """
        Initiates the command-line interface, processing user inputs in a loop.
        """
        print("Welcome to the Funnel Analysis Tool. Type 'help' to see available commands.")
        while True:
            try:
                command = input("> ")
                if command.lower() in ['exit', 'quit']:
                    print("Exiting the CLI. Goodbye!")
                    break
                response = self.parse_command(command)
                if response:
                    print(response)
            except Exception as e:
                print(f"Error: {e}")

    def parse_command(self, command):
        """
        Parses and executes a given command entered by the user.

        Args:
            command (str): A command string input by the CLI user.

        Returns:
            str or dict: Result of the command execution.

        Raises:
            ValueError: If the command is not recognized or has invalid syntax.
        """
        parts = command.strip().split()
        if not parts:
            raise ValueError("No command entered.")

        cmd = parts[0].lower()
        args = parts[1:]

        if cmd == 'load_data':
            if len(args) != 2:
                raise ValueError("Usage: load_data <source> <format>")
            source, format = args
            self.data_handler = DataHandler(source)
            data = self.data_handler.load_data(format)
            return f"Data loaded from {source} in {format} format."

        elif cmd == 'funnel_chart':
            if not self.visualization:
                raise ValueError("Visualization object not initialized. Load and prepare data first.")
            self.visualization.create_funnel_chart(title='Funnel Chart')
            return "Funnel chart displayed."

        elif cmd == 'calculate_metrics':
            if not self.metrics_calculator:
                raise ValueError("MetricsCalculator not initialized. Load and prepare data first.")
            stats = self.metrics_calculator.get_summary_statistics()
            return f"Summary Statistics: {stats}"

        elif cmd == 'help':
            return ("Available commands:\n"
                    "  load_data <source> <format> - Load data from a specified source\n"
                    "  funnel_chart - Display funnel chart of the loaded data\n"
                    "  calculate_metrics - Display summary statistics of the funnel\n"
                    "  help - Show this help message\n"
                    "  exit, quit - Exit the CLI")

        else:
            raise ValueError(f"Unknown command: {cmd}")



def export_to_format(data, format, file_path):
    """
    Exports the given data to a specified file format.

    Args:
        data: The data to be exported. This can be a list, dictionary, DataFrame, or any serializable object.
        format: str - The file format to export the data to, such as 'csv', 'json', or 'xlsx'.
        file_path: str - The path where the exported file should be saved.

    Returns:
        str: A message indicating the success of the export operation or the path/location of the exported file.

    Raises:
        ValueError: If the specified format is not supported or if there is an error in data serialization for the given format.
        IOError: If there is an issue in writing the data to a file, such as permission issues or disk space errors.
    """
    try:
        if format == 'csv':
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            data.to_csv(file_path, index=False)
        elif format == 'json':
            with open(file_path, 'w') as json_file:
                json.dump(data, json_file)
        elif format == 'xlsx':
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            data.to_excel(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format '{format}'. Supported formats are: 'csv', 'json', 'xlsx'.")
        return f"Data successfully exported to {file_path}."
    except Exception as e:
        raise IOError(f"An error occurred while writing to file: {e}")



def setup_logging(level):
    """
    Configures the logging settings for the application.

    Args:
        level: str - The logging level to be set. Common levels include 'DEBUG', 'INFO', 'WARNING', 'ERROR', and 'CRITICAL'.

    Returns:
        None; sets up the global logging configuration.

    Raises:
        ValueError: If the specified logging level is not recognized or supported.
        Exception: For unexpected errors during logging configuration.
    """
    level = level.upper()

    if level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        raise ValueError(f"Unsupported logging level '{level}'. Supported levels are: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.")

    try:
        logging.basicConfig(level=getattr(logging, level),
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logging.info(f"Logging configured for level: {level}")
    except Exception as e:
        raise Exception(f"Failed to configure logging: {e}")



def setup_config(config_file):
    """
    Loads and sets up the configuration settings for the application from a specified configuration file.

    Args:
        config_file (str): The path to the configuration file, which contains the settings to be loaded.

    Returns:
        None; the function initializes and applies configuration settings globally for the application.

    Raises:
        FileNotFoundError: If the specified configuration file cannot be found.
        ValueError: If there is an error in parsing the configuration file due to incorrect format or contents.
        Exception: For any generic error that occurs during the loading or applying of configuration settings.
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"The configuration file '{config_file}' does not exist.")

    try:
        with open(config_file, 'r') as file:
            config = json.load(file)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing configuration file '{config_file}': {e}")
    except Exception as e:
        raise Exception(f"An error occurred while loading the configuration file: {e}")

    # Here you can apply the configuration settings globally
    # For demonstration, we just print the configuration settings
    for key, value in config.items():
        print(f"{key}: {value}")
