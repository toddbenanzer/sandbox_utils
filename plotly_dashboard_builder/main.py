from typing import Any, Callable, List, Union
from typing import Any, Dict
from typing import Any, Dict, List, Optional
from typing import Dict
from typing import Union
import json
import logging
import os
import pandas as pd
import plotly.graph_objs as go


class DashboardCreator:
    """
    A class to create interactive dashboards using Plotly.
    """

    def __init__(self, data: Any):
        """
        Initializes the DashboardCreator with the provided data.

        Args:
            data (Any): The dataset to be used in the dashboard. Typically a pandas DataFrame.
        """
        if not isinstance(data, (pd.DataFrame, dict, list)):
            raise TypeError("Data should be a pandas DataFrame, dict, or list.")
        
        self.data = data
        self.components = []

    def create_dashboard(self, layout: Optional[List[Dict]] = None, **components):
        """
        Create a dashboard with a specified layout and set of components.

        Args:
            layout (Optional[List[Dict]]): A list of dictionaries specifying the component layout setup.
            **components: Arbitrary keyword arguments representing dashboard components (e.g., charts).

        Returns:
            go.Figure: A Plotly dashboard figure object.
        """
        self.components = components
        
        fig = go.Figure()
        
        if layout:
            for comp_name, comp_props in layout:
                if comp_name in components:
                    fig.add_trace(components[comp_name])
                else:
                    raise ValueError(f"Component '{comp_name}' not found in provided components.")
        else:
            for comp in components.values():
                fig.add_trace(comp)
        
        return fig

    def preview_dashboard(self):
        """
        Render a preview of the dashboard.

        Returns:
            go.Figure: An interactive Plotly dashboard preview.
        """
        if not self.components:
            raise ValueError("No components have been added to the dashboard.")
        
        fig = self.create_dashboard()
        fig.show()



class PlotlyTemplate:
    """
    A class to manage Plotly templates for common visualization types.
    """

    def __init__(self, template_type: str):
        """
        Initializes the PlotlyTemplate with the specified template type.

        Args:
            template_type (str): The type of template to be used (e.g., 'funnel', 'cohort analysis', 'time series').

        Raises:
            ValueError: If the template_type is not supported.
        """
        valid_templates = ['funnel', 'cohort_analysis', 'time_series']
        if template_type not in valid_templates:
            raise ValueError(f"Unsupported template type '{template_type}'. Supported types: {', '.join(valid_templates)}.")
        
        self.template_type = template_type
        self.template = None

    def load_template(self):
        """
        Loads the specified Plotly template for the given type.

        Returns:
            go.Figure: A Plotly figure that represents the loaded template.
        """
        if self.template_type == 'funnel':
            self.template = go.Figure(go.Funnel())
        elif self.template_type == 'cohort_analysis':
            self.template = go.Figure()  # Simulating loading of cohort analysis template
        elif self.template_type == 'time_series':
            self.template = go.Figure()  # Simulating loading of time series template
        else:
            raise ValueError("Invalid template type")

        return self.template

    def customize_template(self, **customizations: Dict):
        """
        Applies customizations to the loaded template.

        Args:
            **customizations: Arbitrary keyword arguments specifying customization parameters.

        Returns:
            go.Figure: The modified Plotly figure with customizations applied.
        """
        if self.template is None:
            raise ValueError("Template has not been loaded. Call 'load_template()' first.")

        for key, value in customizations.items():
            if key in self.template.layout:
                self.template.layout[key] = value
            else:
                # For additional configurations that might not be direct layout properties
                self.template.update_layout({key: value})

        return self.template



class FunnelPlot:
    """
    A class to generate funnel plots using Plotly.
    """

    def __init__(self, data: Any):
        """
        Initializes the FunnelPlot with the given dataset.

        Args:
            data (Any): The dataset to be used in the funnel plot, typically a pandas DataFrame, dict, or list.
        
        Raises:
            TypeError: If the data is not a pandas DataFrame, dict, or list.
        """
        if not isinstance(data, (pd.DataFrame, dict, list)):
            raise TypeError("Data should be a pandas DataFrame, dict, or list.")
        
        self.data = data

    def generate_plot(self, **kwargs: Dict) -> go.Figure:
        """
        Generates a funnel plot based on the provided data and customization parameters.

        Args:
            **kwargs: Arbitrary keyword arguments for customizing the funnel plot (e.g., title, colors).

        Returns:
            go.Figure: A Plotly graph object representing the funnel plot.
        """
        if isinstance(self.data, pd.DataFrame):
            if 'x' not in kwargs or 'y' not in kwargs:
                raise ValueError("DataFrame must have 'x' and 'y' columns specified in kwargs.")
            x = self.data[kwargs.get('x')]
            y = self.data[kwargs.get('y')]
        elif isinstance(self.data, dict) or isinstance(self.data, list):
            if 'x' not in kwargs or 'y' not in kwargs:
                raise ValueError("Both 'x' and 'y' keys must be present in kwargs for dict or list.")
            x = kwargs.get('x')
            y = kwargs.get('y')
        else:
            raise ValueError("Unsupported data format for funnel plot generation.")

        funnel = go.Figure(go.Funnel(x=x, y=y))

        # Apply additional customizations
        funnel.update_layout(**kwargs)

        return funnel



class CohortAnalysis:
    """
    A class to perform cohort analysis and generate visualizations using Plotly.
    """

    def __init__(self, data: Any):
        """
        Initializes the CohortAnalysis with the given dataset.

        Args:
            data (Any): The dataset to be used for cohort analysis, typically a pandas DataFrame.

        Raises:
            TypeError: If the data is not a pandas DataFrame.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data should be a pandas DataFrame.")
        
        self.data = data
        self.analysis_results = None

    def perform_analysis(self) -> pd.DataFrame:
        """
        Processes the dataset to extract cohort-related metrics and insights.

        Returns:
            pd.DataFrame: The processed data containing the results of the cohort analysis.
        """
        # Example processing: Group by 'cohort' and calculate the retention
        if 'cohort' not in self.data.columns:
            raise ValueError("DataFrame must contain a 'cohort' column for analysis.")

        # Dummy processing: Calculate the size of each cohort
        self.analysis_results = self.data.groupby('cohort').size().reset_index(name='size')

        return self.analysis_results

    def generate_plot(self, **kwargs: Dict) -> go.Figure:
        """
        Generates a visualization based on the cohort analysis results.

        Args:
            **kwargs: Arbitrary keyword arguments for customizing the cohort plot (e.g., title, colors).

        Returns:
            go.Figure: A Plotly graph object representing the cohort analysis visualization.

        Raises:
            ValueError: If cohort analysis has not been performed.
        """
        if self.analysis_results is None:
            raise ValueError("Cohort analysis has not been performed. Call 'perform_analysis()' first.")

        cohort_plot = go.Figure()
        
        # Example plot: Bar chart of cohort sizes
        cohort_plot.add_trace(go.Bar(
            x=self.analysis_results['cohort'],
            y=self.analysis_results['size']
        ))

        # Apply additional customizations
        cohort_plot.update_layout(**kwargs)

        return cohort_plot



class TimeSeriesPlot:
    """
    A class to generate time series plots using Plotly.
    """

    def __init__(self, data: Any):
        """
        Initializes the TimeSeriesPlot with the given dataset.

        Args:
            data (Any): The dataset to be used in the time series plot, typically a pandas DataFrame.

        Raises:
            TypeError: If the data is not a pandas DataFrame.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data should be a pandas DataFrame.")
        
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex for the index.")
        
        self.data = data

    def generate_plot(self, frequency: str, **kwargs: Dict) -> go.Figure:
        """
        Generates a time series plot based on the provided data, frequency, and customization parameters.

        Args:
            frequency (str): The frequency of the time series data (e.g., 'D' for daily, 'M' for monthly).
            **kwargs: Arbitrary keyword arguments for customizing the time series plot (e.g., title, axis labels).

        Returns:
            go.Figure: A Plotly graph object representing the time series visualization.
        """

        # Resample data based on the frequency
        resampled_data = self.data.resample(frequency).mean()

        # Create time series plot
        time_series_plot = go.Figure()

        for column in resampled_data.columns:
            time_series_plot.add_trace(go.Scatter(
                x=resampled_data.index,
                y=resampled_data[column],
                mode='lines',
                name=column
            ))

        # Apply additional customizations
        time_series_plot.update_layout(**kwargs)

        return time_series_plot



class DataHandler:
    """
    A class to handle data loading and transformation processes.
    """

    def __init__(self, file_path: str):
        """
        Initializes the DataHandler with the path to the data file.

        Args:
            file_path (str): A string path to the data file to be loaded.

        Raises:
            FileNotFoundError: If the file path is invalid or the file does not exist.
        """
        self.file_path = file_path
        self.data = None

    def load_data(self) -> pd.DataFrame:
        """
        Loads data from the specified file path.

        Returns:
            pd.DataFrame: The loaded data.

        Raises:
            ValueError: If the file format is unsupported.
        """
        try:
            if self.file_path.endswith('.csv'):
                self.data = pd.read_csv(self.file_path)
            elif self.file_path.endswith('.xlsx'):
                self.data = pd.read_excel(self.file_path)
            else:
                raise ValueError("Unsupported file format. Supported formats: .csv, .xlsx")
            return self.data
        except Exception as e:
            raise FileNotFoundError(f"Error loading file: {e}")

    def transform_data(self, transformations: Union[List[Callable], Callable]) -> pd.DataFrame:
        """
        Applies a series of transformations to the loaded data.

        Args:
            transformations (Union[List[Callable], Callable]): Transformations to be applied to the data.

        Returns:
            pd.DataFrame: The transformed data.

        Raises:
            ValueError: If transformations are not provided or the data has not been loaded.
        """
        if self.data is None:
            raise ValueError("Data has not been loaded. Call 'load_data()' first.")
        
        if isinstance(transformations, list):
            for transform in transformations:
                self.data = transform(self.data)
        elif callable(transformations):
            self.data = transformations(self.data)
        else:
            raise ValueError("Transformations must be a callable or a list of callables.")
        
        return self.data



def import_data(file_path: str, file_type: str):
    """
    Imports data from the specified file path based on the file type.

    Args:
        file_path (str): The path to the file to be imported.
        file_type (str): The type of the file, specifying the import method (e.g., 'csv', 'xlsx', 'json').

    Returns:
        pandas.DataFrame: The imported data.

    Raises:
        ValueError: If the specified file_type is unsupported.
        FileNotFoundError: If the file path does not exist or cannot be accessed.
        Exception: For any unexpected issues during the data import process.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' was not found.")

    try:
        if file_type == 'csv':
            data = pd.read_csv(file_path)
        elif file_type == 'xlsx':
            data = pd.read_excel(file_path)
        elif file_type == 'json':
            data = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file_type '{file_type}'. Supported types are: 'csv', 'xlsx', 'json'.")
        return data
    except Exception as e:
        raise Exception(f"Error importing data: {e}")



def export_dashboard(dashboard: go.Figure, output_format: str, file_path: str) -> None:
    """
    Exports the given dashboard into the specified file format.

    Args:
        dashboard (go.Figure): The dashboard object to be exported.
        output_format (str): The file format to export the dashboard into (e.g., 'html', 'png', 'pdf').
        file_path (str): The file path or directory where the dashboard should be saved.

    Raises:
        ValueError: If the specified output_format is unsupported.
        FileNotFoundError: If the directory does not exist.
        Exception: For any unexpected issues during the export process.
    """
    if not isinstance(dashboard, go.Figure):
        raise TypeError("The dashboard must be an instance of plotly.graph_objs.Figure.")

    supported_formats = ['html', 'png', 'pdf']
    if output_format not in supported_formats:
        raise ValueError(f"Unsupported output_format '{output_format}'. Supported formats are: {', '.join(supported_formats)}.")

    if not os.path.exists(os.path.dirname(file_path)):
        raise FileNotFoundError(f"The specified directory '{os.path.dirname(file_path)}' does not exist.")

    try:
        if output_format == 'html':
            dashboard.write_html(file_path)
        elif output_format == 'png':
            dashboard.write_image(file_path)
        elif output_format == 'pdf':
            dashboard.write_image(file_path, format='pdf')
    except Exception as e:
        raise Exception(f"Error exporting dashboard: {e}")



def setup_logging(level: Union[str, int]) -> None:
    """
    Configures the logging settings for the application.

    Args:
        level (Union[str, int]): The logging level to set for the logger.
            Valid string options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL',
            or corresponding integer levels.

    Raises:
        ValueError: If an invalid logging level is provided.
        Exception: For any other issues encountered during logging setup.
    """
    # Validate the logging level
    if isinstance(level, str):
        level = level.upper()
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if level not in valid_levels:
            raise ValueError(f"Invalid logging level: '{level}'. Choose from {valid_levels}.")
    elif isinstance(level, int):
        valid_level_numbers = [10, 20, 30, 40, 50]
        if level not in valid_level_numbers:
            raise ValueError(f"Invalid logging level: {level}. Choose from {valid_level_numbers}.")
    else:
        raise TypeError("Logging level must be a string or an integer.")

    # Setup basic logging configuration
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
            # Additional handlers, such as FileHandler, can be added here if needed
        ]
    )

    # Optional: Log a message to confirm logging setup
    logging.info(f"Logging is set to level: {level}")



def setup_config(config_file: str) -> Dict:
    """
    Reads and parses configuration settings from the specified config file.

    Args:
        config_file (str): The path to the configuration file to be loaded.

    Returns:
        Dict: A dictionary containing the parsed configuration settings.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If there is an error in parsing the configuration file.
        Exception: For any other unexpected issues.
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"The configuration file '{config_file}' does not exist.")

    try:
        with open(config_file, 'r') as file:
            config = json.load(file)
        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing the configuration file: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error loading configuration: {e}")
