from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col
from typing import Dict
from typing import Dict, List
from typing import List
import logging


class PopulationManager:
    """
    A class to manage and manipulate defined populations using PySpark.
    """

    def __init__(self, spark_session: SparkSession):
        """
        Initializes the PopulationManager with a Spark session.

        Args:
            spark_session (SparkSession): An active Spark session.
        """
        self.spark_session = spark_session
        self.populations: Dict[str, DataFrame] = {}

    def define_population(self, population_name: str, sql_query: str) -> None:
        """
        Defines a population based on the provided SQL query and stores it with the specified name.

        Args:
            population_name (str): The name for the defined population.
            sql_query (str): SQL query to define the population.

        Raises:
            ValueError: If the population name already exists or if the SQL query is invalid.
        """
        if population_name in self.populations:
            raise ValueError(f"Population '{population_name}' is already defined.")
        
        try:
            population_df = self.spark_session.sql(sql_query)
            self.populations[population_name] = population_df
        except Exception as e:
            raise ValueError(f"Failed to define population '{population_name}': {str(e)}")

    def get_population(self, population_name: str) -> DataFrame:
        """
        Retrieves the DataFrame corresponding to the specified population name.

        Args:
            population_name (str): The name of the population to retrieve.

        Returns:
            DataFrame: The DataFrame of the specified population.

        Raises:
            ValueError: If the population name is not found.
        """
        if population_name not in self.populations:
            raise ValueError(f"Population '{population_name}' not found.")
        
        return self.populations[population_name]



class OverlapAnalyzer:
    """
    A class to analyze overlap among defined populations using PySpark.
    """

    def __init__(self, population_manager):
        """
        Initializes the OverlapAnalyzer with an instance of PopulationManager.

        Args:
            population_manager (PopulationManager): The instance to manage and access the defined populations.
        """
        self.population_manager = population_manager

    def analyze_overlap(self, population_names: List[str]) -> DataFrame:
        """
        Analyzes the overlap among the specified populations and produces a DataFrame detailing the intersections.

        Args:
            population_names (List[str]): A list containing the names of the populations to be analyzed for overlap.

        Returns:
            DataFrame: A DataFrame containing the overlap information between specified populations, including details such as size of intersections.

        Raises:
            ValueError: If any population name is invalid or not defined.
        """
        if len(population_names) < 2:
            raise ValueError("At least two populations must be specified for overlap analysis.")
        
        # Retrieve DataFrames for each population
        try:
            population_dataframes = [
                self.population_manager.get_population(name) for name in population_names
            ]
        except ValueError as e:
            raise ValueError(f"Error retrieving populations: {e}")

        # Perform intersection on the populations based on a common identifier
        overlap_df = population_dataframes[0]
        for df in population_dataframes[1:]:
            overlap_df = overlap_df.join(df, on=[col("id")], how="inner")

        return overlap_df



class ProfileCalculator:
    """
    A class to calculate profiles for defined populations based on specified metrics using PySpark.
    """

    def __init__(self, spark_session: SparkSession):
        """
        Initializes the ProfileCalculator with a Spark session.

        Args:
            spark_session (SparkSession): An active Spark session.
        """
        self.spark_session = spark_session
        self.metrics: Dict[str, DataFrame] = {}

    def define_metrics(self, metric_name: str, sql_query: str) -> None:
        """
        Defines a metric by executing a SQL query and stores the resultant DataFrame under the given metric name.

        Args:
            metric_name (str): The name to assign to the metric.
            sql_query (str): SQL query used to calculate the metric.

        Raises:
            ValueError: If the metric name already exists or if the SQL query is invalid.
        """
        if metric_name in self.metrics:
            raise ValueError(f"Metric '{metric_name}' is already defined.")

        try:
            metric_df = self.spark_session.sql(sql_query)
            self.metrics[metric_name] = metric_df
        except Exception as e:
            raise ValueError(f"Failed to define metric '{metric_name}': {str(e)}")

    def calculate_profiles(self, population_name: str, metric_names: List[str]) -> DataFrame:
        """
        Calculates profiles for a given population based on specified metrics and returns a summary DataFrame.

        Args:
            population_name (str): The name of the population.
            metric_names (List[str]): A list of metric names to include in the profile.

        Returns:
            DataFrame: A DataFrame summarizing the profiles for the specified population against the selected metrics.

        Raises:
            ValueError: If the population or any metric is not found.
        """
        try:
            population_df = self.spark_session.sql(f"SELECT * FROM {population_name}")
        except Exception:
            raise ValueError(f"Population '{population_name}' not found or could not be accessed.")

        metric_dfs = []
        for metric_name in metric_names:
            if metric_name not in self.metrics:
                raise ValueError(f"Metric '{metric_name}' not found.")
            metric_dfs.append(self.metrics[metric_name])

        # Joining metric data to the population data
        profile_df = population_df
        for metric_df in metric_dfs:
            profile_df = profile_df.join(metric_df, on=["id"], how="left")

        return profile_df



def load_data(spark_session: SparkSession, file_path: str, file_format: str) -> DataFrame:
    """
    Loads data from a specified file path into a Spark DataFrame, using the specified file format.

    Args:
        spark_session (SparkSession): An active Spark session used to facilitate the data loading process.
        file_path (str): The path to the file containing the data to be loaded.
        file_format (str): The format of the file to be loaded. Common formats include 'csv', 'json', 'parquet', etc.

    Returns:
        DataFrame: A Spark DataFrame containing the data loaded from the specified file.

    Raises:
        IOError: If the file cannot be accessed or read.
        ValueError: If the file format is unsupported or invalid.
    """
    try:
        if file_format.lower() == 'csv':
            return spark_session.read.csv(file_path, header=True, inferSchema=True)
        elif file_format.lower() == 'json':
            return spark_session.read.json(file_path)
        elif file_format.lower() == 'parquet':
            return spark_session.read.parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    except Exception as e:
        raise IOError(f"Failed to load data from {file_path}: {str(e)}")



def save_results(dataframe: DataFrame, file_path: str, file_format: str) -> None:
    """
    Saves a given Spark DataFrame to a specified file path in the designated file format.

    Args:
        dataframe (DataFrame): The Spark DataFrame to be saved.
        file_path (str): The destination file path where the DataFrame will be saved.
        file_format (str): The format in which to save the DataFrame. Common formats include 'csv', 'json', 'parquet', etc.

    Raises:
        IOError: If the file path is invalid or the DataFrame cannot be saved.
        ValueError: If the file format is unsupported or invalid.
    """
    try:
        if file_format.lower() == 'csv':
            dataframe.write.csv(file_path, header=True, mode='overwrite')
        elif file_format.lower() == 'json':
            dataframe.write.json(file_path, mode='overwrite')
        elif file_format.lower() == 'parquet':
            dataframe.write.parquet(file_path, mode='overwrite')
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    except Exception as e:
        raise IOError(f"Failed to save DataFrame to {file_path}: {str(e)}")



def validate_sql_query(spark_session: SparkSession, sql_query: str) -> bool:
    """
    Validates a given SQL query to ensure it is syntactically correct and ready for execution.

    Args:
        spark_session (SparkSession): An active Spark session used for validation purposes.
        sql_query (str): The SQL query string to be validated for correctness.

    Returns:
        bool: A boolean value indicating whether the SQL query is valid (True) or not (False).

    Raises:
        ValueError: If the input SQL query is empty or null.
    """
    if not sql_query or not sql_query.strip():
        raise ValueError("The SQL query must be a non-empty string.")
    
    try:
        # Create a temporary view to validate the SQL syntax without executing on real data
        temp_view = "temp_validation_view"
        spark_session.sql(f"CREATE OR REPLACE TEMP VIEW {temp_view} AS {sql_query} LIMIT 0")
        spark_session.sql(f"DROP VIEW IF EXISTS {temp_view}")
        return True
    except Exception as e:
        return False



def setup_logging(level: str) -> None:
    """
    Configures the logging settings for the application, setting the specified logging level globally.

    Args:
        level (str): The logging level to be set. Common levels include 'DEBUG', 'INFO', 'WARNING', 'ERROR', and 'CRITICAL'.

    Raises:
        ValueError: If the provided logging level is invalid or not recognized by the logging module.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid logging level: {level}")

    logging.basicConfig(level=numeric_level, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
