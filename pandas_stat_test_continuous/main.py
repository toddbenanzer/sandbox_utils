from scipy.stats import ttest_1samp, ttest_ind, ttest_rel, f_oneway, chi2_contingency
from typing import Dict, Any
from typing import List, Dict, Any
from typing import List, Optional, Dict
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import subprocess


class DataAnalyzer:
    """
    A class to analyze a pandas DataFrame for continuous data.
    It calculates descriptive statistics and handles missing, infinite, and trivial data.
    """

    def __init__(self, dataframe: pd.DataFrame, columns: List[str]):
        """
        Initializes the DataAnalyzer with dataframe and columns.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing data to be analyzed.
            columns (List[str]): A list of column names to analyze.
        """
        self.dataframe = dataframe
        self.columns = columns

    def calculate_descriptive_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Calculates descriptive statistics for the specified columns.

        Returns:
            Dict[str, Dict[str, float]]: A dictionary with column names as keys and 
                                           their statistics as values.
        """
        statistics = {}
        for column in self.columns:
            data = self.dataframe[column]
            stats = {
                'mean': data.mean(),
                'median': data.median(),
                'mode': data.mode()[0],  # Take the first mode
                'variance': data.var(),
                'std_dev': data.std(),
                'range': data.max() - data.min(),
                'iqr': data.quantile(0.75) - data.quantile(0.25),
                'skewness': data.skew(),
                'kurtosis': data.kurt()
            }
            statistics[column] = stats
        return statistics

    def detect_and_handle_missing_data(self, method: str = 'drop') -> None:
        """
        Detects and handles missing data in the specified columns.

        Args:
            method (str): The method to handle missing data: 'drop', 'fill_mean', or 'fill_median'.
                          Defaults to 'drop'.
        """
        if method == 'drop':
            self.dataframe.dropna(subset=self.columns, inplace=True)
        elif method == 'fill_mean':
            for column in self.columns:
                self.dataframe[column].fillna(self.dataframe[column].mean(), inplace=True)
        elif method == 'fill_median':
            for column in self.columns:
                self.dataframe[column].fillna(self.dataframe[column].median(), inplace=True)
        else:
            raise ValueError(f"Unsupported method '{method}'. Choose 'drop', 'fill_mean', or 'fill_median'.")

    def detect_and_handle_infinite_data(self, method: str = 'replace_nan') -> None:
        """
        Detects and handles infinite data in the specified columns.

        Args:
            method (str): The method to handle infinite data: 'drop', 'replace_nan'.
                          Defaults to 'replace_nan'.
        """
        if method == 'drop':
            self.dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
            self.dataframe.dropna(subset=self.columns, inplace=True)
        elif method == 'replace_nan':
            self.dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
        else:
            raise ValueError(f"Unsupported method '{method}'. Choose 'drop' or 'replace_nan'.")

    def exclude_null_and_trivial_columns(self) -> List[str]:
        """
        Excludes columns that are completely null or trivial.

        Returns:
            List[str]: A list of column names that were excluded.
        """
        excluded_columns = []
        for column in self.columns:
            if self.dataframe[column].isnull().all():
                excluded_columns.append(column)
                self.columns.remove(column)
            elif self.dataframe[column].nunique() == 1:
                excluded_columns.append(column)
                self.columns.remove(column)
        return excluded_columns



class StatisticalTests:
    """
    A class to perform statistical tests on a pandas DataFrame.
    It provides methods for t-tests, ANOVA, and chi-squared tests.
    """

    def __init__(self, dataframe: pd.DataFrame, columns: List[str]):
        """
        Initializes the StatisticalTests with dataframe and columns.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing data to be analyzed.
            columns (List[str]): A list of column names to include in the analysis.
        """
        self.dataframe = dataframe
        self.columns = columns

    def perform_t_tests(self, test_type: str, **kwargs) -> Dict[str, float]:
        """
        Performs t-tests on the specified columns or between specified columns.

        Args:
            test_type (str): The type of t-test to perform, 'one-sample', 'independent', or 'paired'.
            **kwargs: Additional parameters such as 'popmean', 'group_column', etc.

        Returns:
            dict: A dictionary containing test statistics and p-values for the performed t-tests.
        """
        results = {}
        if test_type == 'one-sample':
            popmean = kwargs.get('popmean', 0)
            for column in self.columns:
                stat, pval = ttest_1samp(self.dataframe[column].dropna(), popmean)
                results[column] = {'t-statistic': stat, 'p-value': pval}
        
        elif test_type == 'independent':
            group_column = kwargs.get('group_column')
            if group_column:
                groups = self.dataframe[group_column].dropna().unique()
                if len(groups) != 2:
                    raise ValueError("Independent t-test requires exactly two groups.")
                data1 = self.dataframe[self.dataframe[group_column] == groups[0]][self.columns[0]].dropna()
                data2 = self.dataframe[self.dataframe[group_column] == groups[1]][self.columns[0]].dropna()
                stat, pval = ttest_ind(data1, data2)
                results['independent'] = {'t-statistic': stat, 'p-value': pval}
        
        elif test_type == 'paired':
            data1 = self.dataframe[self.columns[0]].dropna()
            data2 = self.dataframe[self.columns[1]].dropna()
            stat, pval = ttest_rel(data1, data2)
            results['paired'] = {'t-statistic': stat, 'p-value': pval}
        
        else:
            raise ValueError("Unsupported t-test type. Choose 'one-sample', 'independent', or 'paired'.")
        
        return results

    def perform_anova(self, **kwargs) -> Dict[str, float]:
        """
        Conducts ANOVA tests on the specified columns or among groups defined in a column.

        Args:
            **kwargs: Additional parameters such as 'group_column'.

        Returns:
            dict: A dictionary containing ANOVA statistics and p-values.
        """
        group_column = kwargs.get('group_column')
        if group_column:
            grouped_data = [group.dropna() for name, group in self.dataframe.groupby(group_column)[self.columns[0]]]
            stat, pval = f_oneway(*grouped_data)
            return {'F-statistic': stat, 'p-value': pval}
        else:
            raise ValueError("ANOVA requires a 'group_column' parameter.")

    def perform_chi_squared_test(self, **kwargs) -> Dict[str, float]:
        """
        Performs chi-squared tests on the data.

        Args:
            **kwargs: Additional parameters such as 'columns', specific columns for the test.

        Returns:
            dict: A dictionary containing chi-squared test statistics and p-values.
        """
        test_columns = kwargs.get('columns', self.columns)
        if len(test_columns) != 2:
            raise ValueError("Chi-squared test requires exactly two columns for a contingency table.")
        
        contingency_table = pd.crosstab(self.dataframe[test_columns[0]], self.dataframe[test_columns[1]])
        stat, pval, dof, expected = chi2_contingency(contingency_table)
        return {'chi2-statistic': stat, 'p-value': pval}


def generate_summary_report(data: dict) -> str:
    """
    Generates a summary report from statistical data.

    Args:
        data (dict): A dictionary containing statistical data to be summarized. 
                     Keys represent metric names and values contain numerical results.

    Returns:
        str: A formatted string representing the summary report.
    """
    report_lines = []
    report_lines.append("Statistical Summary Report")
    report_lines.append("=" * 50)

    for key, value in data.items():
        report_lines.append(f"\n{key}:")

        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, float):
                    sub_value = f"{sub_value:.4f}"
                report_lines.append(f"  {sub_key}: {sub_value}")
        else:
            report_lines.append(f"  Value: {value}")

    report_lines.append("\n" + "=" * 50)
    return "\n".join(report_lines)

# Example usage
if __name__ == "__main__":
    example_data = {
        'Descriptive Stats': {
            'Mean': 5.123456,
            'Median': 5.0,
            'Variance': 1.2345
        },
        'T-test Results': {
            't-statistic': -1.456,
            'p-value': 0.1234
        }
    }

    report = generate_summary_report(example_data)
    print(report)



def visualize_statistics(statistics_data: Dict[str, Any], **kwargs) -> None:
    """
    Creates visualizations from statistical data.

    Args:
        statistics_data (dict): A dictionary containing the statistical data to be visualized.
        **kwargs: Additional parameters to customize the visualizations.

    """
    # Retrieve optional parameters from kwargs
    plot_type = kwargs.get('plot_type', 'bar')
    title = kwargs.get('title', 'Statistical Visualization')
    xlabel = kwargs.get('xlabel', '')
    ylabel = kwargs.get('ylabel', '')
    figsize = kwargs.get('figsize', (10, 6))
    
    plt.figure(figsize=figsize)
    plt.title(title)

    if plot_type == 'bar':
        # Create bar plots
        for key, value in statistics_data.items():
            if isinstance(value, dict):
                plt.bar(value.keys(), value.values(), label=key)
            else:
                plt.bar(key, value)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        
    elif plot_type == 'scatter':
        # Assumes statistics_data is a dictionary of two lists for scatter plot
        x_values = next(iter(statistics_data.values()))  # Get first key's value
        y_values = next(iter(statistics_data.values()))  # Get second key's value
        
        if len(statistics_data) != 2:
            raise ValueError("Scatter plot requires exactly two sets of data.")
        
        plt.scatter(x_values, y_values)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    elif plot_type == 'histogram':
        # Create histograms
        for key, value in statistics_data.items():
            plt.hist(value, alpha=0.5, label=key)
        plt.xlabel(xlabel)
        plt.ylabel('Frequency')
        plt.legend()
        
    else:
        raise ValueError("Unsupported plot type. Choose 'bar', 'scatter', or 'histogram'.")

    plt.show()

# Example usage
if __name__ == "__main__":
    example_data = {
        'Category1': [10, 20, 30],
        'Category2': [15, 25, 35]
    }
    visualize_statistics(example_data, plot_type='bar', title='Example Bar Plot', xlabel='Categories', ylabel='Values')



def install_dependencies(file_path: str) -> None:
    """
    Installs Python package dependencies listed in a requirements file.

    Args:
        file_path (str): The path to the requirements file containing package specifications.

    Raises:
        FileNotFoundError: If the specified requirements file does not exist.
        Exception: If package installation fails.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', file_path])
        print("All dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        raise Exception("There was an error installing the dependencies. Please check the requirements file.") from e

# Example usage
if __name__ == "__main__":
    try:
        install_dependencies('requirements.txt')
    except Exception as e:
        print(e)
