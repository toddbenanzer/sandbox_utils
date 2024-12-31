from apscheduler.schedulers.background import BackgroundScheduler
from fpdf import FPDF
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Dict, Any
from typing import List
from typing import Tuple, Optional
from typing import Union
from typing import Union, Dict
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import requests
import seaborn as sns
import sqlite3
import subprocess
import sys
import time
import yaml


class DataFetcher:
    """
    A class to fetch data from various sources and refresh it at scheduled intervals.
    """

    def __init__(self, source_type: str, config: dict):
        """
        Initialize the DataFetcher with source type and configuration.

        Args:
            source_type (str): The type of data source ('API', 'database', 'CSV').
            config (dict): Configuration parameters for connecting to the data source.
        """
        self.source_type = source_type
        self.config = config
        self.data = None
        self.scheduler = BackgroundScheduler()
        self.scheduler.start()

    def fetch_data(self) -> Union[pd.DataFrame, dict]:
        """
        Fetch data from the specified source.

        Returns:
            Union[pd.DataFrame, dict]: The fetched data as a DataFrame or dictionary.
        """
        if self.source_type == 'API':
            response = requests.get(self.config['url'], params=self.config.get('params', {}))
            if response.status_code == 200:
                self.data = response.json()
            else:
                raise ConnectionError(f"Failed to fetch data from API: {response.status_code}")
        
        elif self.source_type == 'database':
            connection = sqlite3.connect(self.config['dbname'])
            query = self.config['query']
            self.data = pd.read_sql_query(query, connection)
            connection.close()
        
        elif self.source_type == 'CSV':
            self.data = pd.read_csv(self.config['file_path'])
        
        else:
            raise ValueError(f"Unsupported source type: {self.source_type}")
        
        return self.data

    def schedule_refresh(self, interval: int):
        """
        Schedule regular data refreshes at a specified interval.

        Args:
            interval (int): The interval in minutes for refreshing data.
        """
        self.scheduler.add_job(self.fetch_data, 'interval', minutes=interval)

    def stop_refresh(self):
        """
        Stop the scheduled data refreshes.
        """
        self.scheduler.shutdown()



class DataPreprocessor:
    """
    A class to handle data preprocessing, including missing value imputation, data normalization, and transformation.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the DataPreprocessor with a dataset.

        Args:
            data (DataFrame): The dataset to be preprocessed.
        """
        self.data = data

    def handle_missing_values(self, method: str) -> pd.DataFrame:
        """
        Handle missing values in the dataset using the specified method.

        Args:
            method (str): The method for handling missing values ('drop', 'fill_mean', 'fill_median', 'fill_mode').

        Returns:
            DataFrame: The dataset with missing values handled.
        """
        if method == 'drop':
            return self.data.dropna()
        
        elif method in ['fill_mean', 'fill_median', 'fill_mode']:
            strategy = method.split('_')[1]
            imputer = SimpleImputer(strategy=strategy)
            self.data[:] = imputer.fit_transform(self.data)
            return self.data
        
        else:
            raise ValueError(f"Unsupported method: {method}")

    def normalize_data(self, method: str) -> pd.DataFrame:
        """
        Normalize or standardize the dataset using the specified method.

        Args:
            method (str): The normalization method ('min-max', 'z-score').

        Returns:
            DataFrame: The normalized dataset.
        """
        if method == 'min-max':
            scaler = MinMaxScaler()
        
        elif method == 'z-score':
            scaler = StandardScaler()
        
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        self.data[:] = scaler.fit_transform(self.data)
        return self.data

    def transform_data(self, transformation: str) -> pd.DataFrame:
        """
        Apply specified transformations to the dataset.

        Args:
            transformation (str): The transformation to apply ('log', 'sqrt', 'encode_categorical').

        Returns:
            DataFrame: The transformed dataset.
        """
        if transformation == 'log':
            self.data = self.data.applymap(lambda x: np.log(x) if x > 0 else x)
        
        elif transformation == 'sqrt':
            self.data = self.data.applymap(lambda x: np.sqrt(x) if x >= 0 else x)
        
        elif transformation == 'encode_categorical':
            self.data = pd.get_dummies(self.data)
        
        else:
            raise ValueError(f"Unsupported transformation: {transformation}")

        return self.data



class DataAnalyzer:
    """
    A class to perform data analysis including descriptive statistics, exploratory data analysis, and regression analysis.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the DataAnalyzer with a dataset.

        Args:
            data (DataFrame): The dataset to be analyzed.
        """
        self.data = data

    def descriptive_statistics(self) -> Dict[str, pd.Series]:
        """
        Compute basic descriptive statistics for the dataset.

        Returns:
            dict: A dictionary containing mean, median, mode, and standard deviation for each numeric column.
        """
        stats = {
            'mean': self.data.mean(),
            'median': self.data.median(),
            'mode': self.data.mode().iloc[0],
            'std_dev': self.data.std()
        }
        return stats

    def exploratory_data_analysis(self) -> pd.DataFrame:
        """
        Perform exploratory data analysis on the dataset.

        Returns:
            DataFrame: A DataFrame containing the correlation matrix of the dataset.
        """
        # Here, we're using a correlation matrix to uncover relationships
        correlation_matrix = self.data.corr()
        return correlation_matrix

    def regression_analysis(self, model_type: str) -> Union[Dict[str, float], Dict[str, np.ndarray]]:
        """
        Conduct regression analysis using the specified model type.

        Args:
            model_type (str): The type of regression model ('linear', 'logistic').

        Returns:
            dict: A dictionary containing model coefficients and performance metrics.
        """
        X = self.data.iloc[:, :-1]  # Feature variables
        y = self.data.iloc[:, -1]   # Target variable

        if model_type == 'linear':
            model = LinearRegression()
            model.fit(X, y)
            predictions = model.predict(X)
            metrics = {
                'coefficients': model.coef_,
                'intercept': model.intercept_,
                'r2_score': r2_score(y, predictions),
                'rmse': mean_squared_error(y, predictions, squared=False)
            }
        
        elif model_type == 'logistic':
            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)
            predictions = model.predict(X)
            metrics = {
                'coefficients': model.coef_,
                'intercept': model.intercept_,
                'accuracy': accuracy_score(y, predictions),
                'confusion_matrix': confusion_matrix(y, predictions)
            }
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return metrics



class DataVisualizer:
    """
    A class to visualize data using various types of charts and to allow customization and export options.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the DataVisualizer with a dataset.

        Args:
            data (DataFrame): The dataset to be visualized.
        """
        self.data = data
        self.current_plot = None

    def generate_plot(self, plot_type: str, **kwargs):
        """
        Generate a plot of the specified type using the data provided.

        Args:
            plot_type (str): The type of plot to generate ('bar', 'line', 'scatter', etc.).
            **kwargs: Additional keyword arguments for plot details like columns, titles, etc.
        """
        plt.figure()
        if plot_type == 'bar':
            self.current_plot = sns.barplot(data=self.data, **kwargs)
        elif plot_type == 'line':
            self.current_plot = sns.lineplot(data=self.data, **kwargs)
        elif plot_type == 'scatter':
            x = kwargs.get('x')
            y = kwargs.get('y')
            self.current_plot = sns.scatterplot(data=self.data, x=x, y=y, **kwargs)
        elif plot_type == 'histogram':
            column = kwargs.get('column')
            self.current_plot = sns.histplot(data=self.data, x=column, **kwargs)
        elif plot_type == 'box':
            self.current_plot = sns.boxplot(data=self.data, **kwargs)
        elif plot_type == 'heatmap':
            if 'annot' not in kwargs:
                kwargs['annot'] = True
            self.current_plot = sns.heatmap(self.data.corr(), **kwargs)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
        plt.show()

    def customize_plot(self, **kwargs):
        """
        Customize the current plot with details like title, labels, etc.

        Args:
            **kwargs: Customization options such as title, xlabel, ylabel, etc.
        """
        if self.current_plot is not None:
            title = kwargs.get('title')
            xlabel = kwargs.get('xlabel')
            ylabel = kwargs.get('ylabel')
            legend = kwargs.get('legend', False)
            grid = kwargs.get('grid', False)
            
            if title:
                self.current_plot.set_title(title)
            if xlabel:
                self.current_plot.set_xlabel(xlabel)
            if ylabel:
                self.current_plot.set_ylabel(ylabel)
            if legend:
                self.current_plot.legend()
            if grid:
                plt.grid(grid)
            plt.draw()
        else:
            raise RuntimeError("No plot has been generated to customize.")

    def export_visualization(self, format: str, file_name="visualization"):
        """
        Export the current visualization to a specified format.

        Args:
            format (str): The export format ('png', 'jpeg', 'pdf', 'svg').
            file_name (str): The name of the file to save. Default is 'visualization'.
        """
        if self.current_plot is not None:
            plt.savefig(f"{file_name}.{format}", format=format)
        else:
            raise RuntimeError("No plot has been generated to export.")



class ReportGenerator:
    """
    A class to generate and schedule reports based on analysis results.
    """
    
    def __init__(self, analysis_results):
        """
        Initialize the ReportGenerator with analysis results.

        Args:
            analysis_results (dict or DataFrame): The data or analyses to use in generating reports.
        """
        self.analysis_results = analysis_results
        self.scheduler = BackgroundScheduler()
        self.scheduler.start()

    def create_report(self, format: str) -> str:
        """
        Generate a report from the analysis results in the specified format.

        Args:
            format (str): The format in which to create the report ('pdf', 'html', 'xlsx', 'md').

        Returns:
            str: The file path to the created report.
        """
        file_path = f"report.{format}"
        
        if format == 'pdf':
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font('Arial', size=12)
            for key, value in self.analysis_results.items():
                pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
            pdf.output(file_path)
        
        elif format == 'html':
            with open(file_path, 'w') as f:
                f.write('<html><body><h1>Analysis Report</h1>')
                for key, value in self.analysis_results.items():
                    f.write(f"<p><strong>{key}</strong>: {value}</p>")
                f.write('</body></html>')
        
        elif format == 'xlsx':
            if isinstance(self.analysis_results, pd.DataFrame):
                self.analysis_results.to_excel(file_path, index=False)
            else:
                pd.DataFrame(self.analysis_results).to_excel(file_path, index=False)
        
        elif format == 'md':
            with open(file_path, 'w') as f:
                for key, value in self.analysis_results.items():
                    f.write(f"**{key}**: {value}\n\n")
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return file_path

    def schedule_report(self, interval: int, distribution_method: str):
        """
        Schedule the automatic generation and distribution of reports at a specified interval.

        Args:
            interval (int): The time interval in hours at which to generate and distribute reports.
            distribution_method (str): The method for distributing the report ('email', 'cloud', 'local').
        """
        self.scheduler.add_job(self._generate_and_distribute_report, 'interval', hours=interval, args=[distribution_method])

    def _generate_and_distribute_report(self, distribution_method: str):
        """
        Generate the report and distribute it using the specified method.

        Args:
            distribution_method (str): The method for distributing the report.
        """
        report_path = self.create_report('pdf')
        
        if distribution_method == 'email':
            # Functionality to send the report via email
            print("Sending report via email...")
        
        elif distribution_method == 'cloud':
            # Functionality to upload the report to a cloud storage
            print("Uploading report to cloud...")
        
        elif distribution_method == 'local':
            # The report is already saved locally
            print(f"Report saved locally at {report_path}")
        
        else:
            raise ValueError(f"Unsupported distribution method: {distribution_method}")

    def stop_scheduling(self):
        """
        Stop the scheduled report generation.
        """
        self.scheduler.shutdown()



def load_config(file_path: str) -> Dict[str, Any]:
    """
    Load configuration settings from the specified file.

    Args:
        file_path (str): The path to the configuration file. Supported formats are JSON and YAML.

    Returns:
        dict: A dictionary containing the configuration parameters.

    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
        ValueError: If the configuration file format is unsupported or an error occurs in reading the file.
    """
    try:
        with open(file_path, 'r') as file:
            if file_path.endswith('.json'):
                # Parse JSON file
                config = json.load(file)
            elif file_path.endswith(('.yaml', '.yml')):
                # Parse YAML file
                config = yaml.safe_load(file)
            else:
                raise ValueError(f"Unsupported configuration file format: {file_path}")
        
        return config

    except FileNotFoundError:
        raise FileNotFoundError(f"The configuration file '{file_path}' was not found.")
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        raise ValueError(f"Error parsing the configuration file '{file_path}': {e}")



def setup_environment(dependencies: List[str]) -> None:
    """
    Set up the programming environment by ensuring that all specified dependencies are installed.

    Args:
        dependencies (list): A list of package names as strings that are required for the project.
    
    Returns:
        None
    """
    for package in dependencies:
        try:
            # Check if the package is already installed
            subprocess.check_call([sys.executable, '-m', 'pip', 'show', package], stdout=subprocess.DEVNULL)
            print(f"Package '{package}' is already installed.")
        except subprocess.CalledProcessError:
            print(f"Package '{package}' is not installed. Installing now...")
            try:
                # Install the package using pip
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"Successfully installed '{package}'.")
            except subprocess.CalledProcessError as e:
                print(f"Failed to install '{package}'. Error: {e}")




def validate_data(data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Validate the provided dataset to ensure it meets criteria for further processing.

    Args:
        data (DataFrame): The dataset to be validated.

    Returns:
        Tuple[bool, Optional[str]]: A tuple where the first element is a boolean indicating whether the data is valid,
                                    and the second element is an optional string message detailing any validation errors.
    """
    # Check if the DataFrame is empty
    if data.empty:
        return False, "The dataset is empty."

    # Check for missing values
    if data.isnull().values.any():
        return False, "The dataset contains missing values."

    # Example schema: Define expected columns and types
    expected_columns = {
        'column1': 'int64',
        'column2': 'float64',
        'column3': 'object'  # For strings or categorical data
    }

    # Check for required columns
    missing_columns = [col for col in expected_columns if col not in data.columns]
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Validate data types of the columns
    for column, expected_type in expected_columns.items():
        if column in data.columns:
            if data[column].dtype != expected_type:
                return False, f"Column '{column}' should be of type {expected_type}, but found {data[column].dtype}."
    
    # Additional checks can be added here, such as checking ranges or categorical values

    return True, None  # Data is valid



def setup_logging(level: str) -> None:
    """
    Set up the logging configuration based on the specified logging level.

    Args:
        level (str): The logging level to be set. Expected values include 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
    
    Returns:
        None
    """
    # Dictionary for mapping string levels to logging module levels
    level_mapping = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    # Check if the provided level is valid
    if level not in level_mapping:
        raise ValueError(f"Invalid logging level: {level}. Expected one of {list(level_mapping.keys())}.")

    # Set up the basic configuration for logging
    logging.basicConfig(
        level=level_mapping[level],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()  # Outputting log messages to the console
        ]
    )

    logging.info(f"Logging is set to {level} level.")
