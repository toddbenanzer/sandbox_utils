from .exceptions import DataAnalyzerError, DataFrameValidationError
from scipy.stats import chi2_contingency, fisher_exact
from typing import Any
from typing import List, Dict
from typing import List, Optional
from typing import List, Tuple
import logging
import matplotlib.pyplot as plt
import pandas as pd


class DataFrameHandler:
    """
    A class to handle and preprocess a pandas DataFrame for categorical data analysis.
    """

    def __init__(self, dataframe: pd.DataFrame, columns: List[str]):
        """
        Initializes the DataFrameHandler with the specified DataFrame and columns.

        Args:
            dataframe (pd.DataFrame): The DataFrame to be processed.
            columns (List[str]): List of column names to be included in the analysis.
        
        Raises:
            DataFrameValidationError: If validation of the DataFrame fails.
        """
        self.dataframe = dataframe
        self.columns = columns
        if not self.validate_dataframe():
            raise DataFrameValidationError("DataFrame validation failed.")

    def validate_dataframe(self) -> bool:
        """
        Validates the structure and content of the DataFrame.

        Returns:
            bool: True if validation is successful, otherwise raises an exception.
        
        Raises:
            DataFrameValidationError: If validation checks fail.
        """
        # Check if columns exist in the DataFrame
        for column in self.columns:
            if column not in self.dataframe.columns:
                raise DataFrameValidationError(f"Column '{column}' not found in DataFrame.")
        
        # Check if DataFrame is not empty
        if self.dataframe.empty:
            raise DataFrameValidationError("DataFrame is empty.")
        
        # Validate that columns are categorical
        for column in self.columns:
            if not pd.api.types.is_categorical_dtype(self.dataframe[column]):
                raise DataFrameValidationError(f"Column '{column}' is not of categorical type.")

        return True

    def handle_missing_infinite_values(self) -> pd.DataFrame:
        """
        Handles missing and infinite values in the specified columns of the DataFrame.

        Returns:
            pd.DataFrame: The preprocessed DataFrame ready for analysis.
        """
        # Fill missing values with the mode of the column
        for column in self.columns:
            mode_value = self.dataframe[column].mode().iloc[0] if not self.dataframe[column].mode().empty else None
            self.dataframe[column].fillna(mode_value, inplace=True)

        # Replace infinite values with NaN and drop rows with NaN values
        self.dataframe.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
        self.dataframe.dropna(subset=self.columns, inplace=True)

        return self.dataframe



class DescriptiveStatistics:
    """
    A class to compute descriptive statistics for categorical data in a pandas DataFrame.
    """

    def __init__(self, dataframe: pd.DataFrame, columns: List[str]):
        """
        Initializes the DescriptiveStatistics object with the specified DataFrame and columns.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the categorical data.
            columns (List[str]): List of column names for which to compute descriptive statistics.
        
        Raises:
            ValueError: If columns specified do not exist in the DataFrame.
        """
        self.dataframe = dataframe
        self.columns = columns
        self._validate_columns()

    def _validate_columns(self) -> None:
        """
        Validates that specified columns exist in the DataFrame and are categorical.

        Raises:
            ValueError: If any of the specified columns are not found or are not categorical.
        """
        for column in self.columns:
            if column not in self.dataframe.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame.")
            if not pd.api.types.is_categorical_dtype(self.dataframe[column]):
                raise ValueError(f"Column '{column}' is not of categorical type.")

    def calculate_frequencies(self) -> Dict[str, pd.Series]:
        """
        Calculates the frequency of each category in the specified columns.

        Returns:
            Dict[str, pd.Series]: Dictionary of frequency counts for each column.
        """
        frequencies = {}
        for column in self.columns:
            frequencies[column] = self.dataframe[column].value_counts()
        return frequencies

    def compute_mode(self) -> Dict[str, pd.Series]:
        """
        Computes the mode (most frequent category) for each specified column.

        Returns:
            Dict[str, pd.Series]: Dictionary of modes for each column.
        """
        modes = {}
        for column in self.columns:
            modes[column] = self.dataframe[column].mode()
        return modes

    def generate_contingency_table(self, column1: str, column2: str) -> pd.DataFrame:
        """
        Generates a contingency table representing the cross-tabulation of two columns.

        Args:
            column1 (str): The first column for cross-tabulation.
            column2 (str): The second column for cross-tabulation.

        Returns:
            pd.DataFrame: Contingency table with counts for each combination of categories.

        Raises:
            ValueError: If either column does not exist or is not categorical.
        """
        if column1 not in self.dataframe.columns or column2 not in self.dataframe.columns:
            raise ValueError(f"One or both columns '{column1}', '{column2}' not found in DataFrame.")
        if not pd.api.types.is_categorical_dtype(self.dataframe[column1]) or not pd.api.types.is_categorical_dtype(self.dataframe[column2]):
            raise ValueError(f"One or both columns '{column1}', '{column2}' are not of categorical type.")

        return pd.crosstab(self.dataframe[column1], self.dataframe[column2])



class StatisticalTests:
    """
    A class to perform statistical tests on categorical data in a pandas DataFrame.
    """

    def __init__(self, dataframe: pd.DataFrame, columns: List[str]):
        """
        Initializes the StatisticalTests object with the specified DataFrame and columns.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the categorical data.
            columns (List[str]): A list of column names for statistical tests.
        
        Raises:
            ValueError: If columns specified do not exist or are not categorical.
        """
        self.dataframe = dataframe
        self.columns = columns
        self._validate_columns()

    def _validate_columns(self) -> None:
        """
        Validates that specified columns exist in the DataFrame and are categorical.

        Raises:
            ValueError: If any of the specified columns are not found or are not categorical.
        """
        for column in self.columns:
            if column not in self.dataframe.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame.")
            if not pd.api.types.is_categorical_dtype(self.dataframe[column]):
                raise ValueError(f"Column '{column}' is not of categorical type.")

    def perform_chi_squared_test(self, column1: str, column2: str) -> Tuple[float, float, int, pd.DataFrame]:
        """
        Performs the Chi-squared test for independence between two categorical columns.

        Args:
            column1 (str): The name of the first column for the chi-squared test.
            column2 (str): The name of the second column for the chi-squared test.

        Returns:
            Tuple[float, float, int, pd.DataFrame]: Returns statistic, p-value,
            degrees of freedom, and expected frequencies.
        
        Raises:
            ValueError: If columns are not valid for the test.
        """
        if column1 not in self.dataframe.columns or column2 not in self.dataframe.columns:
            raise ValueError(f"One or both columns '{column1}', '{column2}' not found in DataFrame.")
        if not pd.api.types.is_categorical_dtype(self.dataframe[column1]) or not pd.api.types.is_categorical_dtype(self.dataframe[column2]):
            raise ValueError(f"One or both columns '{column1}', '{column2}' are not of categorical type.")

        contingency_table = pd.crosstab(self.dataframe[column1], self.dataframe[column2])
        chi2, p, dof, expected = chi2_contingency(contingency_table)

        return chi2, p, dof, pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)

    def perform_fishers_exact_test(self, table: pd.DataFrame) -> Tuple[float, float]:
        """
        Performs Fisher's exact test on a 2x2 contingency table.

        Args:
            table (pd.DataFrame): A 2x2 contingency table.

        Returns:
            Tuple[float, float]: Returns odds ratio and p-value.
        
        Raises:
            ValueError: If the table is not a valid 2x2 table.
        """
        if table.shape != (2, 2):
            raise ValueError("Fisher's exact test is only applicable to 2x2 contingency tables.")

        oddsratio, p_value = fisher_exact(table)

        return oddsratio, p_value



class OutputManager:
    """
    A class to manage the output of statistical analysis results, including exporting to files and generating visualizations.
    """

    def __init__(self, results: Any):
        """
        Initializes the OutputManager with analysis results.

        Args:
            results (Any): The analysis results to be managed, formatted, and outputted.
        """
        self.results = results

    def export_to_csv(self, file_path: str) -> None:
        """
        Exports the results to a CSV file.

        Args:
            file_path (str): The file path where the CSV will be saved.
        
        Raises:
            ValueError: If the results cannot be converted to a CSV-compatible format.
        """
        if isinstance(self.results, pd.DataFrame):
            self.results.to_csv(file_path, index=False)
        else:
            raise ValueError("Results must be a DataFrame to export to CSV.")

    def export_to_excel(self, file_path: str) -> None:
        """
        Exports the results to an Excel file.

        Args:
            file_path (str): The file path where the Excel file will be saved.
        
        Raises:
            ValueError: If the results cannot be converted to an Excel-compatible format.
        """
        if isinstance(self.results, pd.DataFrame):
            self.results.to_excel(file_path, index=False)
        else:
            raise ValueError("Results must be a DataFrame to export to Excel.")

    def generate_visualization(self, chart_type: str) -> None:
        """
        Generates a visualization of the results.

        Args:
            chart_type (str): A string specifying the type of chart to generate (e.g., 'bar', 'pie').
        
        Raises:
            ValueError: If the chart type is unsupported or if visualization generation fails.
        """
        if not isinstance(self.results, pd.DataFrame):
            raise ValueError("Results must be a DataFrame to generate visualizations.")

        if chart_type == 'bar':
            self.results.plot(kind='bar')
        elif chart_type == 'pie':
            self.results.plot(kind='pie', subplots=True, figsize=(8, 8), autopct='%1.1f%%')
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")

        plt.title('Results Visualization')
        plt.show()



def load_dataframe(file_path: str) -> pd.DataFrame:
    """
    Loads data from a specified file path into a pandas DataFrame.

    Args:
        file_path (str): The file path to the data source.

    Returns:
        pd.DataFrame: A DataFrame containing the data loaded from the specified file.
    
    Raises:
        FileNotFoundError: If the file is not found at the specified path.
        ValueError: If the file format is unsupported or reading the file fails.
    """
    try:
        # Determine the file extension
        file_extension = file_path.split('.')[-1].lower()

        # Load the file based on its extension
        if file_extension == 'csv':
            return pd.read_csv(file_path)
        elif file_extension in ['xls', 'xlsx']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise ValueError(f"An error occurred while reading the file: {str(e)}")



def save_summary(results: Any, file_path: str, format: str) -> None:
    """
    Saves the analysis results to the specified file path in the given format.

    Args:
        results (Any): The analysis results to be saved. Should be compatible with the specified format.
        file_path (str): The file path where the results will be saved.
        format (str): The format in which to save the results ('csv', 'excel').

    Raises:
        ValueError: If the format is unsupported or if results are incompatible with the format.
    """
    # Convert format to lowercase to handle case-insensitivity
    format = format.lower()

    try:
        if format == 'csv':
            if isinstance(results, pd.DataFrame):
                results.to_csv(file_path, index=False)
            else:
                raise ValueError("Results are not a DataFrame and cannot be saved as CSV.")
        elif format == 'excel':
            if isinstance(results, pd.DataFrame):
                results.to_excel(file_path, index=False)
            else:
                raise ValueError("Results are not a DataFrame and cannot be saved as Excel.")
        else:
            raise ValueError(f"Unsupported format: {format}")
    except Exception as e:
        raise ValueError(f"An error occurred while saving the summary: {str(e)}")



def configure_logging(level: str) -> None:
    """
    Configures the logging settings for the application.

    Args:
        level (str): The log level to set (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').

    Raises:
        ValueError: If the provided log level is not valid.
    """
    # Convert level to uppercase to handle case-insensitivity
    level = level.upper()

    # Validate the log level
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if level not in valid_levels:
        raise ValueError(f"Invalid log level: '{level}'. Must be one of {valid_levels}.")

    # Configure logging with the specified level
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
