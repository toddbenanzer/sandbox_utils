from scipy import stats
from typing import Any
from typing import Any, Dict
from typing import Dict, Any
from typing import List, Dict, Any
import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class DistributionVisualizer:
    """
    A class to create distribution visualizations such as histograms and box plots.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initializes the DistributionVisualizer with the dataset.

        Args:
            data (pd.DataFrame): The dataset for visualization.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")
        self.data = data

    def create_histogram(self, **kwargs: Dict[str, Any]) -> plt.Figure:
        """
        Generates a histogram for the specified data.

        Args:
            **kwargs: Optional parameters for matplotlib's hist function,
                      e.g., bins, color, edgecolor.

        Returns:
            plt.Figure: The matplotlib figure object containing the histogram.
        """
        column = kwargs.pop('column', None)
        if column is None or column not in self.data.columns:
            raise ValueError(f"Please specify a valid column from the dataset. Available columns: {self.data.columns}")

        fig, ax = plt.subplots()
        self.data[column].plot.hist(ax=ax, **kwargs)
        ax.set_title(f'Histogram of {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')

        return fig

    def create_box_plot(self, **kwargs: Dict[str, Any]) -> plt.Figure:
        """
        Generates a box plot for the specified data.

        Args:
            **kwargs: Optional parameters for pandas' boxplot function,
                      e.g., column, color, grid.

        Returns:
            plt.Figure: The matplotlib figure object containing the box plot.
        """
        columns = kwargs.pop('columns', None)
        if columns is None:
            raise ValueError("Please specify the columns for the box plot.")
        if not all(col in self.data.columns for col in columns):
            raise ValueError(f"Some specified columns are not in the dataset. Available columns: {self.data.columns}")

        fig, ax = plt.subplots()
        self.data.boxplot(column=columns, ax=ax, **kwargs)
        ax.set_title(f'Box Plot of {", ".join(columns)}')
        ax.set_ylabel('Values')

        return fig



class CorrelationVisualizer:
    """
    A class to create visualizations to explore correlations between variables in a dataset.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initializes the CorrelationVisualizer with the dataset.

        Args:
            data (pd.DataFrame): The dataset for visualization.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")
        self.data = data

    def create_correlation_matrix(self, **kwargs: Dict[str, Any]) -> plt.Figure:
        """
        Generates a heatmap of the correlation matrix for the dataset.

        Args:
            **kwargs: Optional parameters for seaborn's heatmap function,
                      e.g., cmap, annot, fmt.

        Returns:
            plt.Figure: The matplotlib figure object containing the correlation matrix heatmap.
        """
        correlation_matrix = self.data.corr()
        fig, ax = plt.subplots()
        sns.heatmap(correlation_matrix, ax=ax, **kwargs)
        ax.set_title('Correlation Matrix')

        return fig

    def create_scatter_plot(self, x: str, y: str, with_trendline: bool = False, **kwargs: Dict[str, Any]) -> plt.Figure:
        """
        Generates a scatter plot for the specified variables with an optional trendline.

        Args:
            x (str): The name of the column to be used as the x-axis variable.
            y (str): The name of the column to be used as the y-axis variable.
            with_trendline (bool): Whether to include a trendline on the scatter plot. Default is False.
            **kwargs: Additional optional parameters for seaborn's scatterplot function, e.g., hue, style.

        Returns:
            plt.Figure: The matplotlib figure object containing the scatter plot.
        """
        if x not in self.data.columns or y not in self.data.columns:
            raise ValueError(f"Specified columns {x}, {y} must be in the dataset. Available columns: {self.data.columns}")

        fig, ax = plt.subplots()
        sns.scatterplot(data=self.data, x=x, y=y, ax=ax, **kwargs)
        if with_trendline:
            sns.regplot(data=self.data, x=x, y=y, ax=ax, scatter=False, color='red', truncate=False)

        ax.set_title(f'Scatter Plot of {x} vs {y}')

        return fig



class DataHandler:
    """
    A class to handle the import and preprocessing of datasets from various formats.
    """

    def __init__(self):
        """
        Initializes the DataHandler object without any dataset.
        """
        self.data = None

    def import_data(self, file_path: str, file_type: str) -> pd.DataFrame:
        """
        Imports data from a specified file path and type.

        Args:
            file_path (str): The file path to the data source.
            file_type (str): The type of file being imported (e.g., 'csv', 'xlsx', 'json').

        Returns:
            pd.DataFrame: A Pandas DataFrame containing the imported data.

        Raises:
            ValueError: If the file type is unsupported or if there is an error reading the file.
        """
        try:
            if file_type.lower() == 'csv':
                self.data = pd.read_csv(file_path)
            elif file_type.lower() == 'xlsx':
                self.data = pd.read_excel(file_path)
            elif file_type.lower() == 'json':
                self.data = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            raise ValueError(f"Error reading {file_type} file: {e}")

        return self.data

    def preprocess_data(self, strategies: Dict[str, Any]) -> pd.DataFrame:
        """
        Applies specified preprocessing strategies to the dataset.

        Args:
            strategies (dict): A dictionary where keys are preprocessing methods
                               (e.g., 'dropna', 'fillna') and values are the
                               parameters for those methods.

        Returns:
            pd.DataFrame: A Pandas DataFrame after applying the preprocessing strategies.

        Raises:
            ValueError: If a specified preprocessing strategy is not recognized or fails.
        """
        if self.data is None:
            raise ValueError("No data available. Please import data first.")

        for strategy, params in strategies.items():
            try:
                if strategy == 'dropna':
                    self.data = self.data.dropna(**params)
                elif strategy == 'fillna':
                    self.data = self.data.fillna(**params)
                elif strategy == 'rename':
                    self.data = self.data.rename(**params)
                else:
                    raise ValueError(f"Unrecognized preprocessing strategy: {strategy}")
            except Exception as e:
                raise ValueError(f"Failed to apply strategy {strategy}: {e}")

        return self.data



class StatisticalAnalyzer:
    """
    A class to perform statistical analysis, including basic statistics and hypothesis testing.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initializes the StatisticalAnalyzer with the dataset.

        Args:
            data (pd.DataFrame): The dataset for statistical analysis.
        
        Raises:
            ValueError: If the provided data is not a pandas DataFrame.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")
        self.data = data

    def compute_basic_statistics(self, columns: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Computes basic statistics for specified columns in the dataset.

        Args:
            columns (list of str): A list of column names to compute statistics for.

        Returns:
            dict: A dictionary of basic statistics for the specified columns.
        """
        statistics = {}
        for col in columns:
            if col not in self.data.columns:
                raise ValueError(f"Column {col} does not exist in the dataset.")
            statistics[col] = {
                'mean': self.data[col].mean(),
                'median': self.data[col].median(),
                'mode': self.data[col].mode()[0] if not self.data[col].mode().empty else None,
                'variance': self.data[col].var(),
                'std_dev': self.data[col].std()
            }
        return statistics

    def perform_hypothesis_tests(self, test_type: str, columns: List[str], **kwargs) -> Dict[str, Any]:
        """
        Performs specified hypothesis tests on the dataset.

        Args:
            test_type (str): The type of hypothesis test to perform (e.g., 't-test', 'chi-squared').
            columns (list of str): The columns to include in the hypothesis test.
            **kwargs: Additional parameters depending on the type of test.

        Returns:
            dict: A dictionary containing the results of the hypothesis test.

        Raises:
            ValueError: If an unsupported test type is specified or if the test cannot be performed.
        """
        if test_type == 't-test':
            if len(columns) != 2:
                raise ValueError("T-test requires exactly two columns.")
            sample1 = self.data[columns[0]].dropna()
            sample2 = self.data[columns[1]].dropna()
            t_stat, p_value = stats.ttest_ind(sample1, sample2)
            return {
                't_statistic': t_stat,
                'p_value': p_value
            }

        elif test_type == 'chi-squared':
            if len(columns) != 2:
                raise ValueError("Chi-squared test requires exactly two categorical columns.")
            contingency_table = pd.crosstab(self.data[columns[0]], self.data[columns[1]])
            chi2, p, _, _ = stats.chi2_contingency(contingency_table)
            return {
                'chi2_statistic': chi2,
                'p_value': p
            }

        else:
            raise ValueError(f"Unsupported test type: {test_type}")



class TableauExporter:
    """
    A class to export visualization objects into a Tableau-compatible format.
    """

    def __init__(self, visualization: Any):
        """
        Initializes the TableauExporter with a given visualization object.

        Args:
            visualization (Any): The visualization object to be exported.

        Raises:
            ValueError: If the provided visualization object is not compatible or supported for export.
        """
        if not isinstance(visualization, plt.Figure):
            raise ValueError("Currently only matplotlib Figure objects are supported for export.")
        self.visualization = visualization

    def export_to_tableau_format(self, output_path: str) -> None:
        """
        Exports the visualization object to a file format that can be utilized by Tableau.

        Args:
            output_path (str): The file path where the exported tableau-compatible file will be saved.

        Raises:
            OSError: If there is an issue with writing to the specified output path.
            ValueError: If the specified output format or path is not compatible with Tableau.
        """
        try:
            # Saving as PNG image, which can be imported into Tableau
            self.visualization.savefig(output_path, format='png')
            print(f"Visualization exported successfully to {output_path}")
        except ValueError as e:
            raise ValueError(f"Error in saving file: {e}")
        except OSError as e:
            raise OSError(f"Could not write to the specified path: {e}")



def setup_visualization_style(style_options: Dict[str, Any]) -> None:
    """
    Configures visual aesthetics and style properties for visualizations.

    Args:
        style_options (dict): A dictionary containing style parameters and their values, such as:
            - 'color_palette': Name of the color palette to use (e.g., 'viridis', 'plasma').
            - 'font_size': Default font size for text elements.
            - 'line_style': Style of lines in plot (e.g., '-', '--', '-.', ':').
            - 'figure_size': A tuple representing the width and height of the figure in inches.
            - 'background_color': Background color of the visualization (e.g., 'white', 'black').

    Raises:
        ValueError: If an unknown style option is provided.
    """
    try:
        if 'color_palette' in style_options:
            plt.style.use(style_options['color_palette'])

        if 'font_size' in style_options:
            plt.rcParams.update({'font.size': style_options['font_size']})

        if 'line_style' in style_options:
            plt.rcParams.update({'lines.linestyle': style_options['line_style']})

        if 'figure_size' in style_options:
            plt.rcParams.update({'figure.figsize': style_options['figure_size']})

        if 'background_color' in style_options:
            plt.rcParams.update({'axes.facecolor': style_options['background_color']})

    except Exception as e:
        raise ValueError(f"An error occurred while setting the style: {e}")



def setup_logging(level: str) -> None:
    """
    Configures the logging system to capture and display messages based on the specified log level.

    Args:
        level (str): The desired logging level (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').

    Raises:
        ValueError: If an unsupported or invalid logging level is provided.
    """
    levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }

    if level not in levels:
        raise ValueError(f"Invalid logging level: {level}. Supported levels are: {list(levels.keys())}")

    logging.basicConfig(
        level=levels[level],
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
