from data_handler import DataHandler
from descriptive_statistics import DescriptiveStatistics
from scipy import stats
from scipy.stats import mode, zscore
from typing import Dict
from typing import Dict, List
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class DescriptiveStatistics:
    """
    A class for calculating descriptive statistics on a numeric column.
    """

    def __init__(self, data: pd.Series):
        """
        Initializes the DescriptiveStatistics object with a pandas Series.

        Args:
            data (pd.Series): The column of numeric data for analysis.
        """
        self.data = data.dropna()  # Remove missing values for initial analysis

    def compute_central_tendency(self) -> Dict[str, float]:
        """
        Computes and returns measures of central tendency: mean, median, and mode.

        Returns:
            dict: A dictionary containing mean, median, and mode.
        """
        mean_value = self.data.mean()
        median_value = self.data.median()
        mode_value = mode(self.data).mode[0]
        return {
            'mean': mean_value,
            'median': median_value,
            'mode': mode_value
        }

    def compute_dispersion(self) -> Dict[str, float]:
        """
        Computes and returns measures of dispersion: variance, standard deviation, range, and IQR.

        Returns:
            dict: A dictionary containing variance, standard deviation, range, and IQR.
        """
        variance_value = self.data.var()
        std_dev_value = self.data.std()
        range_value = self.data.max() - self.data.min()
        iqr_value = np.percentile(self.data, 75) - np.percentile(self.data, 25)
        return {
            'variance': variance_value,
            'std_dev': std_dev_value,
            'range': range_value,
            'IQR': iqr_value
        }

    def detect_outliers(self, method: str = 'z-score') -> List[float]:
        """
        Detects outliers in the data using the specified method.

        Args:
            method (str): The method to use for outlier detection ('z-score', 'IQR'). Default is 'z-score'.

        Returns:
            list: A list of detected outliers.
        """
        if method == 'z-score':
            z_scores = zscore(self.data)
            outliers = self.data[(np.abs(z_scores) > 3)].tolist()
        
        elif method == 'IQR':
            q1 = np.percentile(self.data, 25)
            q3 = np.percentile(self.data, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = self.data[(self.data < lower_bound) | (self.data > upper_bound)].tolist()
        else:
            raise ValueError("Unsupported method provided for outlier detection. Choose 'z-score' or 'IQR'.")

        return outliers

    def estimate_distribution(self) -> str:
        """
        Estimates and displays the likely statistical distribution of the data.

        Returns:
            str: The estimated distribution type ('normal', 'binomial', etc.).
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data, kde=True)
        plt.title('Data Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.show()

        # For simplicity, this example assumes normal distribution estimation
        # In practice, statistical tests might be used to estimate distribution types
        return 'normal'



class DataHandler:
    """
    A class for handling preprocessing and validation of a numeric dataset.
    """

    def __init__(self, data: pd.Series):
        """
        Initializes the DataHandler object with a pandas Series.

        Args:
            data (pd.Series): The column of numeric data for preprocessing and validation.
        """
        self.data = data

    def handle_missing_values(self, strategy: str = 'mean') -> pd.Series:
        """
        Identifies and handles missing values in the data using the specified strategy.

        Args:
            strategy (str): The strategy to handle missing values ('mean', 'median', 'mode', 'drop'). Default is 'mean'.

        Returns:
            pd.Series: A Series with missing values handled according to the specified strategy.
        """
        if strategy == 'mean':
            return self.data.fillna(self.data.mean())
        elif strategy == 'median':
            return self.data.fillna(self.data.median())
        elif strategy == 'mode':
            return self.data.fillna(self.data.mode()[0])
        elif strategy == 'drop':
            return self.data.dropna()
        else:
            raise ValueError("Unsupported strategy. Choose 'mean', 'median', 'mode', or 'drop'.")

    def handle_infinite_values(self, strategy: str = 'mean') -> pd.Series:
        """
        Identifies and processes infinite values in the data, applying the same strategy as for missing values.

        Args:
            strategy (str): The strategy to handle infinite values ('mean', 'median', 'mode', 'drop'). Default is 'mean'.

        Returns:
            pd.Series: A Series with infinite values handled according to the specified strategy.
        """
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        return self.handle_missing_values(strategy)

    def check_null_trivial(self) -> bool:
        """
        Checks if the data is null or trivial (contains only a single unique value).

        Returns:
            bool: True if the data is null or trivial, otherwise False.
        """
        if self.data.isnull().all():
            print("The series is entirely null.")
            return True
        
        unique_values = self.data.dropna().unique()
        if len(unique_values) <= 1:
            print("The series is trivial, containing only one unique value:", unique_values)
            return True
        
        return False



def calculate_statistics(data: pd.Series) -> Dict:
    """
    Calculates comprehensive descriptive statistics for a given pandas Series.

    Args:
        data (pd.Series): A pandas Series containing the numeric column of data for analysis.

    Returns:
        dict: A dictionary containing the computed descriptive statistics.
    """
    # Step 1: Preprocess Data
    handler = DataHandler(data)
    if handler.check_null_trivial():
        return {
            'message': "The data is null or trivial and cannot be processed for statistics.",
            'handled_data': data
        }
    
    cleaned_data = handler.handle_infinite_values(strategy='mean')
    
    # Step 2: Compute Statistics
    stats = DescriptiveStatistics(cleaned_data)
    central_tendency = stats.compute_central_tendency()
    dispersion = stats.compute_dispersion()
    outliers = stats.detect_outliers(method='z-score')
    estimated_distribution = stats.estimate_distribution()
    
    # Step 3: Compile Results
    result = {
        'central_tendency': central_tendency,
        'dispersion': dispersion,
        'outliers': outliers,
        'estimated_distribution': estimated_distribution,
        'handled_data': cleaned_data
    }
    
    return result



def visualize_distribution(data: pd.Series) -> None:
    """
    Visualizes the distribution of data in a pandas Series using a histogram
    and kernel density estimate (KDE).

    Args:
        data (pd.Series): A pandas Series containing the numeric column of data to be visualized.

    Returns:
        None: The function directly displays the plot.
    """
    # Check if the data is empty
    if data.empty:
        print("The provided data series is empty.")
        return

    # Check if the data is numeric
    if not pd.api.types.is_numeric_dtype(data):
        print("The provided data series is not numeric.")
        return

    # Plot settings
    plt.figure(figsize=(10, 6))
    sns.histplot(data, bins=30, kde=True, color='skyblue', edgecolor='black')
    
    # Optional: Add a box plot for additional insight
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=data, color='lightgreen')

    # Title and labels
    plt.title('Data Distribution Visualization')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Display the plot
    plt.show()



def detect_outliers(data: pd.Series, method: str = 'z-score') -> List[float]:
    """
    Detects and returns outliers in a given pandas Series using the specified method.

    Args:
        data (pd.Series): A pandas Series containing the numeric column of data for outlier detection.
        method (str): The method for outlier detection ('z-score'), default is 'z-score'.

    Returns:
        list: A list of detected outliers.
    """
    if data.empty:
        print("The provided data series is empty.")
        return []

    if not pd.api.types.is_numeric_dtype(data):
        print("The provided data series is not numeric.")
        return []

    if method == 'z-score':
        mean = data.mean()
        std_dev = data.std()
        
        # Calculate z-scores
        z_scores = (data - mean) / std_dev
        
        # Define outliers as those with a |z-score| > 3
        outlier_mask = np.abs(z_scores) > 3
        outliers = data[outlier_mask].tolist()
    else:
        raise ValueError("Unsupported method provided for outlier detection. Supported methods: 'z-score'.")

    return outliers



def estimate_likely_distribution(data: pd.Series) -> str:
    """
    Estimates the likely statistical distribution of a given pandas Series.

    Args:
        data (pd.Series): A pandas Series containing the numeric column of data for analysis.

    Returns:
        str: The estimated statistical distribution ('normal', 'exponential', 'uniform', etc.).
    """
    if data.empty:
        print("The provided data series is empty.")
        return "No Data"

    if not pd.api.types.is_numeric_dtype(data):
        print("The provided data series is not numeric.")
        return "Non-numeric Data"

    # Remove NaNs for distribution fitting
    data_cleaned = data.dropna()

    # Performing goodness-of-fit test for normal distribution
    k2, normal_p_value = stats.normaltest(data_cleaned)
    normal_fit = "normal" if normal_p_value > 0.05 else None

    # Exponential distribution test
    _, exp_p_value = stats.kstest(data_cleaned, 'expon', args=(data_cleaned.min(), data_cleaned.std()))
    exp_fit = "exponential" if exp_p_value > 0.05 else None

    # Uniform distribution test
    _, uni_p_value = stats.kstest(data_cleaned, 'uniform', args=(data_cleaned.min(), data_cleaned.max()-data_cleaned.min()))
    uni_fit = "uniform" if uni_p_value > 0.05 else None

    # Compile fits that were successful
    fits = [fit for fit in [normal_fit, exp_fit, uni_fit] if fit is not None]

    # Visual Analysis via Q-Q plot for Normality
    plt.figure(figsize=(6, 4))
    stats.probplot(data_cleaned, dist="norm", plot=plt)
    plt.title('Q-Q Plot for Normality')
    plt.show()

    # Logical conclusion of best fit
    if not fits:
        return "No Likely Distribution Found"
    
    return fits[0]  # Return the most likely fit from the tests
