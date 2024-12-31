from scipy import stats
from scipy.stats import t
from typing import Any
from typing import Dict, Tuple
from typing import List, Dict, Any
from typing import Union
import importlib
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class StatisticalTests:
    """
    A class to perform common statistical tests used in marketing analytics.
    """

    def perform_t_test(self, data_a: np.ndarray, data_b: np.ndarray, paired: bool = False) -> Dict[str, Any]:
        """
        Performs a t-test to compare the means of two samples.

        Args:
            data_a (np.ndarray): First set of data for the t-test.
            data_b (np.ndarray): Second set of data for the t-test.
            paired (bool): If True, performs a paired t-test. Default is False.

        Returns:
            Dict[str, Any]: A dictionary containing the t-value, p-value, and degrees of freedom.
        """
        if paired:
            t_stat, p_value = stats.ttest_rel(data_a, data_b)
        else:
            t_stat, p_value = stats.ttest_ind(data_a, data_b)

        df = len(data_a) + len(data_b) - 2
        return {"t_stat": t_stat, "p_value": p_value, "degrees_of_freedom": df}

    def perform_anova(self, *groups: np.ndarray) -> Dict[str, Any]:
        """
        Performs an ANOVA test to determine if there are significant differences between group means.

        Args:
            groups (np.ndarray): Multiple arrays, each corresponding to a group.

        Returns:
            Dict[str, Any]: A dictionary containing the F-statistic, p-value, and degrees of freedom.
        """
        f_stat, p_value = stats.f_oneway(*groups)

        df_between = len(groups) - 1
        df_within = sum([len(group) for group in groups]) - len(groups)

        return {"f_stat": f_stat, "p_value": p_value, "df_between": df_between, "df_within": df_within}

    def perform_chi_square(self, observed: np.ndarray, expected: np.ndarray) -> Dict[str, Any]:
        """
        Performs a chi-square test for goodness of fit.

        Args:
            observed (np.ndarray): Observed frequency counts.
            expected (np.ndarray): Expected frequency counts.

        Returns:
            Dict[str, Any]: A dictionary containing the chi-square statistic and p-value.
        """
        chi_stat, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
        return {"chi_stat": chi_stat, "p_value": p_value}

    def perform_regression_analysis(self, data: pd.DataFrame, independent_vars: List[str], dependent_var: str) -> Dict[str, Any]:
        """
        Performs a regression analysis to model the relationship between independent and dependent variables.

        Args:
            data (pd.DataFrame): Dataset containing independent and dependent variables.
            independent_vars (List[str]): Column names in data representing independent variables.
            dependent_var (str): Column name in data representing the dependent variable.

        Returns:
            Dict[str, Any]: A dictionary containing regression coefficients, R-squared, standard error, and p-values.
        """
        X = data[independent_vars]
        X = np.hstack([np.ones((X.shape[0], 1)), X])  # Add intercept term
        y = data[dependent_var]

        # Performing Ordinary Least Squares Regression
        beta_hat = np.linalg.inv(X.T @ X) @ (X.T @ y)
        predictions = X @ beta_hat

        residuals = y - predictions
        rss = residuals.T @ residuals  # Residual Sum of Squares
        tss = ((y - y.mean()) ** 2).sum()  # Total Sum of Squares

        r_squared = 1 - (rss / tss)
        standard_error = np.sqrt(rss / (len(y) - len(beta_hat)))

        # Calculating p-values
        var_beta_hat = np.linalg.inv(X.T @ X) * standard_error ** 2
        standard_errors = np.sqrt(np.diag(var_beta_hat))
        t_stats = beta_hat / standard_errors
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=len(y) - len(beta_hat)))

        return {"coefficients": beta_hat, "r_squared": r_squared, "standard_error": standard_error, "p_values": p_values}



class DataHandler:
    """
    A class for managing data handling tasks, such as loading, cleansing, and preprocessing data.
    """

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Loads data from the specified file path into a pandas DataFrame.

        Args:
            file_path (str): Path to the data file.

        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.
        """
        try:
            if file_path.endswith('.csv'):
                data_frame = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                data_frame = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
            return data_frame
        except Exception as e:
            raise ValueError(f"An error occurred while loading the data: {e}")

    def cleanse_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cleanses the data by handling missing values and removing duplicates.

        Args:
            data (pd.DataFrame): The input data to cleanse.

        Returns:
            pd.DataFrame: The cleansed data.
        """
        # Handle missing values: simple imputation with mean for numeric columns
        data = data.fillna(data.mean())

        # Remove duplicates
        data = data.drop_duplicates()

        # Convert data types if necessary
        # Here you can add specific type corrections as needed for your dataset
        return data

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the data by normalizing numeric values and encoding categorical features.

        Args:
            data (pd.DataFrame): Input data for preprocessing.

        Returns:
            pd.DataFrame: The preprocessed data.
        """
        # Normalize numerical features
        numerical_cols = data.select_dtypes(include=['number']).columns
        data[numerical_cols] = (data[numerical_cols] - data[numerical_cols].mean()) / data[numerical_cols].std()

        # Encode categorical features
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

        return data



class ResultsInterpreter:
    """
    A class for interpreting statistical test results, calculating effect sizes, and confidence intervals.
    """

    def interpret_t_test(self, result: Dict[str, float]) -> str:
        """
        Interprets t-test results.

        Args:
            result (Dict[str, float]): A dictionary containing 't_stat', 'p_value', and 'degrees_of_freedom'.

        Returns:
            str: A human-readable interpretation of the t-test results.
        """
        t_stat = result['t_stat']
        p_value = result['p_value']
        df = result['degrees_of_freedom']

        interpretation = f"T-test: t-statistic = {t_stat:.2f}, p-value = {p_value:.4f}, df = {df}."
        if p_value < 0.05:
            interpretation += " Result is statistically significant; reject the null hypothesis."
        else:
            interpretation += " Result is not statistically significant; fail to reject the null hypothesis."
        return interpretation

    def interpret_anova(self, result: Dict[str, float]) -> str:
        """
        Interprets ANOVA results.

        Args:
            result (Dict[str, float]): A dictionary containing 'f_stat', 'p_value', 'df_between', and 'df_within'.

        Returns:
            str: A human-readable interpretation of the ANOVA results.
        """
        f_stat = result['f_stat']
        p_value = result['p_value']
        df_between = result['df_between']
        df_within = result['df_within']

        interpretation = f"ANOVA: F-statistic = {f_stat:.2f}, p-value = {p_value:.4f}, df_between = {df_between}, df_within = {df_within}."
        if p_value < 0.05:
            interpretation += " Significant differences between groups; reject the null hypothesis."
        else:
            interpretation += " No significant differences between groups; fail to reject the null hypothesis."
        return interpretation

    def calculate_effect_size(self, test_result: Dict[str, float]) -> float:
        """
        Calculates effect size for a t-test (Cohen's d).

        Args:
            test_result (Dict[str, float]): A t-test result dictionary with 't_stat' and 'degrees_of_freedom'.

        Returns:
            float: The calculated effect size.
        """
        t_stat = test_result['t_stat']
        df = test_result['degrees_of_freedom']

        # Calculate Cohen's d
        cohen_d = t_stat / np.sqrt(df + 1)
        return cohen_d

    def compute_confidence_interval(self, test_result: Dict[str, float], confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Computes the confidence interval for the mean difference.

        Args:
            test_result (Dict[str, float]): A t-test result dictionary with 't_stat' and 'degrees_of_freedom'.
            confidence_level (float): The confidence level for the interval.

        Returns:
            Tuple[float, float]: The lower and upper bounds of the confidence interval.
        """
        t_stat = test_result['t_stat']
        se = test_result.get('standard_error', 1) # Assume 1 as default if not provided
        df = test_result['degrees_of_freedom']

        # Calculate critical t-value
        alpha = 1 - confidence_level
        t_crit = t.ppf(1 - alpha / 2, df)

        # Calculate margin of error and confidence interval
        margin_of_error = t_crit * se
        lower_bound = t_stat - margin_of_error
        upper_bound = t_stat + margin_of_error

        return (lower_bound, upper_bound)



class Visualizer:
    """
    A class for visualizing data and statistical test results.
    """

    def plot_distribution(self, data: Union[np.ndarray, pd.Series], test_type: str):
        """
        Plots the distribution of the given data.

        Args:
            data (Union[np.ndarray, pd.Series]): The dataset for which to plot the distribution.
            test_type (str): Specifies the type of distribution to plot (e.g., 't-test', 'ANOVA').

        Returns:
            None
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(data, kde=True, bins=30)
        plt.title(f'Distribution Plot for {test_type}')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.show()

    def create_result_summary_plot(self, results: dict):
        """
        Creates a plot summarizing statistical test results.

        Args:
            results (dict): Contains statistical test outcomes and relevant metrics to visualize.

        Returns:
            None
        """
        # This example assumes results contain 'means', 'conf_int_lower', 'conf_int_upper'
        categories = results.get('categories', [])
        means = results.get('means', [])
        conf_int_lower = results.get('conf_int_lower', [])
        conf_int_upper = results.get('conf_int_upper', [])
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(categories, means, yerr=[conf_int_lower, conf_int_upper], fmt='o', ecolor='g', capsize=5)
        plt.title('Result Summary Plot')
        plt.xlabel('Categories')
        plt.ylabel('Means')
        plt.show()

    def generate_correlation_matrix(self, data: pd.DataFrame):
        """
        Generates and visualizes a correlation matrix of the dataset.

        Args:
            data (pd.DataFrame): The dataset for which to compute and visualize correlations.

        Returns:
            None
        """
        correlation_matrix = data.corr()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.show()



def setup_logging(level: str):
    """
    Configures the logging settings for the module.

    Args:
        level (str): The desired logging level (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").

    Returns:
        None
    """
    # Define the valid logging levels
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')

    # Configure the logging format and level
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()  # Log to console by default
        ]
    )



def configure_integration(libraries):
    """
    Configures integration with specified libraries by ensuring they are available and initialized.

    Args:
        libraries (list of str): Names of libraries to configure.

    Returns:
        None
    """
    for library in libraries:
        try:
            # Attempt to import the library
            lib = importlib.import_module(library)
            logging.info(f"Successfully integrated with {library}.")
            
            # Perform any library-specific configuration if needed
            if library == "example_lib":
                # Example of setting a default configuration for a specific library
                lib.configure_default_settings()
        
        except ImportError:
            logging.error(f"Library {library} is not available. Please install it.")
        except Exception as e:
            logging.error(f"An error occurred while configuring {library}: {e}")
