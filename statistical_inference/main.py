from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class StatisticalTest:
    """
    A base class for performing statistical tests on two datasets. 
    Provides methods for handling zero variance, missing values, and constant values.
    """

    def __init__(self, data1: np.ndarray, data2: np.ndarray, test_params: dict):
        """
        Initialize the StatisticalTest object with datasets and test parameters.

        Args:
            data1 (np.ndarray): The first dataset for comparison.
            data2 (np.ndarray): The second dataset for comparison.
            test_params (dict): Parameters for the statistical test.
        """
        self.data1 = data1
        self.data2 = data2
        self.test_params = test_params

    def check_zero_variance(self) -> bool:
        """
        Check if either dataset has zero variance.

        Returns:
            bool: True if zero variance is detected, False otherwise.
        """
        zero_variance_data1 = np.all(self.data1 == self.data1[0])
        zero_variance_data2 = np.all(self.data2 == self.data2[0])
        
        if zero_variance_data1 or zero_variance_data2:
            print("Warning: Zero variance detected in one of the datasets.")
            return True
        return False

    def handle_missing_values(self):
        """
        Handle missing values in the datasets based on specified parameters.
        Returns the datasets with missing values processed.
        """
        method = self.test_params.get('missing_values_method', 'remove')
        
        if method == 'remove':
            self.data1 = self.data1[~np.isnan(self.data1)]
            self.data2 = self.data2[~np.isnan(self.data2)]
        elif method == 'impute':
            mean1 = np.nanmean(self.data1)
            mean2 = np.nanmean(self.data2)
            self.data1 = np.where(np.isnan(self.data1), mean1, self.data1)
            self.data2 = np.where(np.isnan(self.data2), mean2, self.data2)
        else:
            raise ValueError("Unsupported missing values handling method.")

    def handle_constant_values(self):
        """
        Handle constant values in the datasets by issuing warnings or adjustments.
        """
        if np.all(self.data1 == self.data1[0]):
            print("Warning: Dataset 1 contains constant values.")
        
        if np.all(self.data2 == self.data2[0]):
            print("Warning: Dataset 2 contains constant values.")



class NumericTest(StatisticalTest):
    """
    A class for performing statistical tests on numeric datasets.
    Inherits from the StatisticalTest class to use preprocessing features.
    """

    def perform_t_test(self):
        """
        Perform a t-test on the two datasets.

        Returns:
            t_statistic (float): The calculated t-statistic.
            p_value (float): The p-value corresponding to the t-statistic.
        """
        # Check assumptions and handle data preprocessing using inherited methods
        self.handle_missing_values()
        self.handle_constant_values()
        
        # Performing the t-test
        t_statistic, p_value = stats.ttest_ind(self.data1, self.data2, nan_policy='omit')
        return t_statistic, p_value

    def perform_anova(self):
        """
        Perform a one-way ANOVA test on the two datasets.

        Returns:
            f_statistic (float): The calculated F-statistic.
            p_value (float): The p-value corresponding to the F-statistic.
        """
        # Check assumptions and prepare data using inherited methods
        self.handle_missing_values()
        self.handle_constant_values()
        
        # Performing the ANOVA test
        f_statistic, p_value = stats.f_oneway(self.data1, self.data2)
        return f_statistic, p_value



class CategoricalTest(StatisticalTest):
    """
    A class for performing statistical tests on categorical datasets.
    Inherits from the StatisticalTest class to use preprocessing features.
    """

    def perform_chi_squared_test(self):
        """
        Perform a Chi-squared test on the two categorical datasets.

        Returns:
            chi2_statistic (float): The calculated Chi-squared statistic.
            p_value (float): The p-value corresponding to the Chi-squared statistic.
            degrees_of_freedom (int): The degrees of freedom associated with the test.
            expected_frequencies (np.ndarray): The expected frequencies under the null hypothesis.
        """
        # Preprocess data using inherited methods
        self.handle_missing_values()
        self.handle_constant_values()

        # Construct contingency table
        contingency_table = np.vstack((self.data1, self.data2))
        
        # Perform Chi-squared test
        chi2_statistic, p_value, degrees_of_freedom, expected_frequencies = stats.chi2_contingency(contingency_table)

        return chi2_statistic, p_value, degrees_of_freedom, expected_frequencies



class BooleanTest(StatisticalTest):
    """
    A class for performing statistical tests on boolean datasets.
    Inherits from the StatisticalTest class to use preprocessing features.
    """

    def perform_proportions_test(self):
        """
        Perform a z-test for two proportions on the boolean datasets.

        Returns:
            z_statistic (float): The calculated Z-statistic.
            p_value (float): The p-value corresponding to the Z-statistic.
        """
        # Preprocess data using inherited methods
        self.handle_missing_values()
        self.handle_constant_values()

        # Calculate the number of successes and observations for each dataset
        count1 = np.sum(self.data1)
        count2 = np.sum(self.data2)
        nobs1 = len(self.data1)
        nobs2 = len(self.data2)

        # Perform the proportions z-test
        z_statistic, p_value = proportions_ztest([count1, count2], [nobs1, nobs2])

        return z_statistic, p_value



def compute_descriptive_statistics(data):
    """
    Compute a set of descriptive statistics for the input numeric data.

    Args:
        data (np.ndarray): A NumPy array containing numeric data.

    Returns:
        dict: A dictionary containing descriptive statistics:
            - mean (float)
            - median (float)
            - mode (array or value)
            - variance (float)
            - standard_deviation (float)
            - min (float)
            - max (float)
            - count (int)
    """
    # Remove NaN and infinite values from the data for accurate calculations
    clean_data = data[np.isfinite(data)]

    # Calculate descriptive statistics
    statistics = {
        'mean': np.mean(clean_data),
        'median': np.median(clean_data),
        'mode': stats.mode(clean_data)[0][0],
        'variance': np.var(clean_data, ddof=1),
        'standard_deviation': np.std(clean_data, ddof=1),
        'min': np.min(clean_data),
        'max': np.max(clean_data),
        'count': len(clean_data)
    }

    return statistics



def visualize_results(results):
    """
    Visualize the statistical test results or descriptive statistics.

    Args:
        results (dict): A dictionary containing results to be visualized, possibly including statistical values or descriptive statistics.
    """
    # Example 1: Histogram for distribution visualization
    if 'data' in results:
        plt.figure(figsize=(8, 6))
        sns.histplot(results['data'], bins=10, kde=True)
        plt.title('Data Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.show()

    # Example 2: Bar chart for displaying statistical test metrics
    if 't_statistic' in results and 'p_value' in results:
        metrics = ['T-Statistic', 'P-Value']
        values = [results['t_statistic'], results['p_value']]
        
        plt.figure(figsize=(8, 6))
        sns.barplot(x=metrics, y=values, palette='viridis')
        plt.title('Statistical Test Results')
        plt.ylabel('Value')
        plt.ylim(0, max(values) + 1)
        plt.show()

    # Example 3: Box plot for illustrating spread and outliers
    if 'box_data' in results:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=results['box_data'])
        plt.title('Box Plot of Data')
        plt.xlabel('Category')
        plt.ylabel('Values')
        plt.show()

    # Example 4: Pie chart for category proportions
    if 'category_proportions' in results:
        labels = list(results['category_proportions'].keys())
        sizes = list(results['category_proportions'].values())
        
        plt.figure(figsize=(8, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
        plt.title('Category Proportions')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.show()



def validate_input_data(data1, data2):
    """
    Validate the input datasets to ensure they are suitable for statistical analysis.

    Args:
        data1: The first dataset, expected to be a NumPy array or similar iterable structure.
        data2: The second dataset, expected to be a NumPy array or similar iterable structure.

    Returns:
        bool: True if both datasets are valid, otherwise False.

    Raises:
        ValueError: If the datasets are found to have invalid conditions such as different lengths, types, or contain no data.
    """
    # Validate that inputs are either numpy arrays or lists
    if not isinstance(data1, (np.ndarray, list)):
        raise ValueError("data1 must be a numpy array or a list.")
    if not isinstance(data2, (np.ndarray, list)):
        raise ValueError("data2 must be a numpy array or a list.")

    # Convert lists to numpy arrays for consistency
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    # Check for non-empty datasets
    if data1.size == 0:
        raise ValueError("data1 is empty.")
    if data2.size == 0:
        raise ValueError("data2 is empty.")

    # Check dimensionality consistency
    if data1.ndim != data2.ndim:
        raise ValueError("data1 and data2 must have the same number of dimensions.")

    # Check length if inputs are required to be paired
    if len(data1) != len(data2):
        raise ValueError("data1 and data2 must be of the same length for paired analysis.")

    # Optional: Check for excessive missing values
    missing_threshold = 0.5  # Example threshold
    missing_data1 = np.isnan(data1).sum() / len(data1)
    missing_data2 = np.isnan(data2).sum() / len(data2)
    
    if missing_data1 > missing_threshold:
        raise ValueError("data1 contains too many missing values.")
    if missing_data2 > missing_threshold:
        raise ValueError("data2 contains too many missing values.")

    return True



def impute_missing_values(data, method='mean'):
    """
    Impute missing values in the dataset using the specified method.

    Args:
        data: A NumPy array or list containing numeric data with potential missing values (e.g., NaN).
        method: A string specifying the imputation method. Options include 'mean', 'median', 'mode', and 'zero'.

    Returns:
        np.ndarray: An array with missing values imputed.

    Raises:
        ValueError: If the specified imputation method is not supported.
    """
    # Convert to NumPy array for consistency
    data = np.asarray(data)

    # Identify and impute missing values based on chosen method
    if method == 'mean':
        mean_value = np.nanmean(data)
        imputed_data = np.where(np.isnan(data), mean_value, data)

    elif method == 'median':
        median_value = np.nanmedian(data)
        imputed_data = np.where(np.isnan(data), median_value, data)

    elif method == 'mode':
        mode_result = stats.mode(data[~np.isnan(data)], axis=None)
        if mode_result.count.size > 0:
            mode_value = mode_result.mode[0]
            imputed_data = np.where(np.isnan(data), mode_value, data)
        else:
            raise ValueError("Mode is undefined for empty input with all NaNs")

    elif method == 'zero':
        imputed_data = np.where(np.isnan(data), 0, data)

    else:
        raise ValueError(f"Unsupported imputation method '{method}'. Options are 'mean', 'median', 'mode', or 'zero'.")

    # Log imputation completion
    print(f"Imputation completed using method '{method}'.")

    return imputed_data
