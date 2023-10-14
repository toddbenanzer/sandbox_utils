andas as pd
import numpy as np
import requests
import pymysql
import matplotlib.pyplot as plt
import schedule
import time
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
import scipy.stats as stats

def read_csv_file(file_path):
    """
    Function to read in a CSV file.
    
    Parameters:
    - file_path: str, the path to the CSV file
    
    Returns:
    - pandas.DataFrame, the data from the CSV file
    """
    df = pd.read_csv(file_path)
    return df

def fetch_data(url):
    response = requests.get(url)
    data = response.text
    return data

def connect_to_database(host, username, password, database):
    try:
        connection = pymysql.connect(
            host=host,
            user=username,
            password=password,
            db=database
        )
        print("Connected to the database")
        return connection
    except pymysql.Error as e:
        print(f"Error connecting to the database: {e}")
        return None

def retrieve_data(connection, query):
    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchall()
            return result
    except pymysql.Error as e:
        print(f"Error retrieving data from the database: {e}")
        return None

def clean_data(data):
    # Remove missing values
    data = data.dropna()

    # Remove outliers using z-score
    z_scores = np.abs((data - data.mean()) / data.std())
    data = data[(z_scores < 3).all(axis=1)]

    return data

def calculate_mean(data):
    """
    Calculate the mean of a given dataset.

    Parameters:
        data (list or numpy.ndarray): The input dataset.

    Returns:
        float: The mean value.
    """
    return np.mean(data)

def calculate_median(data):
    """
    Calculate the median of a given dataset.

    Parameters:
        data (list or numpy.ndarray): The input dataset.

    Returns:
        float: The median value.
    """
    return np.median(data)

def calculate_standard_deviation(data):
    """
    Calculate the standard deviation of a given dataset.

    Parameters:
        data (list or numpy.ndarray): The input dataset.

    Returns:
        float: The standard deviation value.
    """
    return np.std(data)

def visualize_data(data, plot_type):
    """
    Visualizes data using different types of plots such as bar charts, line graphs, and scatter plots.

    Args:
        data (dict): A dictionary containing the data to be visualized.
        plot_type (str): The type of plot to be used ('bar', 'line', or 'scatter').

    Returns:
        None
    """
    x = list(data.keys())
    y = list(data.values())

    if plot_type == 'bar':
        plt.bar(x, y)
    elif plot_type == 'line':
        plt.plot(x, y)
    elif plot_type == 'scatter':
        plt.scatter(x, y)
        
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(f'{plot_type.capitalize()} Plot')
    
    plt.show()

def generate_summary_statistics(dataset):
    # Convert the dataset to a pandas DataFrame (assuming it is in CSV format)
    df = pd.read_csv(dataset)
    
    # Calculate summary statistics using describe() method
    summary_stats = df.describe()
    
    return summary_stats

def create_report(data, columns, format_options):
    # Prepare the report header
    report = "Report\n\n"

    # Add the column headers to the report
    for column in columns:
        report += f"{column}\t"

    report += "\n"

    # Add the data rows to the report
    for row in data:
        for column in columns:
            report += f"{row[column]}\t"
        report += "\n"

    # Apply formatting options to the report
    if "bold" in format_options:
        report = "**" + report.strip() + "**"
    
    if "underline" in format_options:
        report = "__" + report.strip() + "__"
    
    return report

def generate_daily_report():
    # Code to generate daily report
    pass

def generate_weekly_report():
    # Code to generate weekly report
    pass

def generate_monthly_report():
    # Code to generate monthly report
    pass

def schedule_report_generation(report_type, interval):
    if report_type == "daily":
        schedule.every().day.at("09:00").do(generate_daily_report)
    elif report_type == "weekly":
        schedule.every().monday.at("09:00").do(generate_weekly_report)
    elif report_type == "monthly":
        schedule.every().day(1).at("09:00").do(generate_monthly_report)
    
    while True:
        schedule.run_pending()
        time.sleep(1)

def export_report(report_data, output_format):
    if output_format.lower() == 'pdf':
        # Export report as PDF
        # Code to export report_data as PDF
        print("Report exported as PDF")
    elif output_format.lower() == 'excel':
        # Export report as Excel
        # Code to export report_data as Excel
        print("Report exported as Excel")
    elif output_format.lower() == 'html':
        # Export report as HTML
        # Code to export report_data as HTML
        print("Report exported as HTML")
    else:
        print("Invalid output format")

def save_data(data, filename):
    """
    Save intermediate results or processed data for future use or reference.

    Parameters:
        - data: The data to be saved.
        - filename: The name of the file to save the data to.

    Returns:
        None
    """
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def handle_exceptions(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            print(f"ValueError occurred: {e}")
        except FileNotFoundError as e:
            print(f"FileNotFoundError occurred: {e}")
        except KeyError as e:
            print(f"KeyError occurred: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    return wrapper

def remove_missing_values(dataset):
    """
    Function to remove missing values from a dataset.
    
    Parameters:
        - dataset (pandas.DataFrame): The input dataset
        
    Returns:
        - cleaned_dataset (pandas.DataFrame): The dataset with missing values removed
    """
    cleaned_dataset = dataset.dropna()
    return cleaned_dataset

def normalize_data(data):
    """
    Normalize numeric data using min-max normalization technique.

    Args:
        data (list or numpy.ndarray): The numeric data to be normalized.

    Returns:
        list or numpy.ndarray: The normalized data.
    """
    # Calculate the minimum and maximum values in the data
    min_val = min(data)
    max_val = max(data)

    # Normalize each value in the data
    normalized_data = [(val - min_val) / (max_val - min_val) for val in data]

    return normalized_data

def encode_categorical_variables(dataframe, columns):
    """
    Encode categorical variables using one-hot encoding.
    
    Parameters:
        - dataframe (pandas DataFrame): The input dataframe containing the categorical variables.
        - columns (list): A list of column names to encode.
        
    Returns:
        - encoded_dataframe (pandas DataFrame): The dataframe with the encoded categorical variables.
    """
    encoded_dataframe = pd.get_dummies(dataframe, columns=columns)
    
    return encoded_dataframe

def handle_outliers(data, threshold=3):
    """
    Function to handle outliers in the data using the z-score method.
    
    Parameters:
        - data: 1-dimensional numpy array or list of numeric values.
        - threshold: Number of standard deviations to consider as outlier. Default is 3.
        
    Returns:
        - Numpy array with outliers replaced by NaN.
    """
    mean = np.mean(data)
    std = np.std(data)
    z_scores = (data - mean) / std
    outliers = np.abs(z_scores) > threshold
    data[outliers] = np.nan
    
    return data

def scale_data(data, method='min-max'):
    if method == 'min-max':
        # Min-Max Scaling
        min_val = np.min(data)
        max_val = np.max(data)
        scaled_data = (data - min_val) / (max_val - min_val)
    elif method == 'standardization':
        # Standardization (Z-score Scaling)
        mean = np.mean(data)
        std = np.std(data)
        scaled_data = (data - mean) / std
    else:
        raise ValueError("Invalid scaling method. Please choose 'min-max' or 'standardization'.")
    
    return scaled_data

def handle_imbalanced_data(X, y):
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def perform_feature_selection(X, y, k):
    """
    Perform feature selection and extraction using SelectKBest algorithm.
    
    Parameters:
        X (array-like): The input features.
        y (array-like): The target variable.
        k (int): The number of top features to select.
        
    Returns:
        X_selected (array-like): The selected features.
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    
    return X_selected

def split_data(data, test_size=0.2):
    """
    Split data into training and testing sets.
    
    Parameters:
        - data: input data to be split
        - test_size: the proportion of the dataset to include in the testing set (default is 0.2)
    
    Returns:
        - train_data: training data
        - test_data: testing data
    """
    train_data, test_data = train_test_split(data, test_size=test_size)
    return train_data, test_data

def perform_cross_validation(X, y, model):
    """
    Perform cross-validation on the data.

    Parameters:
    - X (array-like): The input features.
    - y (array-like): The target variable.
    - model: The model to use for cross-validation.

    Returns:
    - scores (array): The cross-validation scores.
    """
    scores = cross_val_score(model, X, y, cv=5)  # Assuming 5-fold cross-validation
    return scores

def calculate_correlation(dataframe, var1, var2):
    """
    Calculate the correlation between two variables in a given dataframe.
    
    Parameters:
        - dataframe: pandas.DataFrame
            The dataframe containing the variables.
        - var1: str
            The name of the first variable.
        - var2: str
            The name of the second variable.
    
    Returns:
        float
            The correlation coefficient between the two variables.
    """
    return dataframe[var1].corr(dataframe[var2])

def hypothesis_testing(data, alpha=0.05):
    """
    Perform hypothesis testing on the data.

    Parameters:
    - data: List or numpy array of numerical values.
    - alpha: The significance level (default is 0.05).

    Returns:
    - result: A dictionary containing the test statistic, p-value, and conclusion.
    """

    # Perform the desired hypothesis test
    # For example, let's perform a one-sample t-test against a population mean of 0
    test_statistic, p_value = stats.ttest_1samp(data, 0)

    # Determine the conclusion based on the p-value and significance level
    if p_value < alpha:
        conclusion = "Reject the null hypothesis"
    else:
        conclusion = "Fail to reject the null hypothesis"

    # Create a dictionary to store the result
    result = {
        "Test Statistic": test_statistic,
        "P-value": p_value,
        "Conclusion": conclusion
    }

    return result

def read_data_file(file_path):
    file_extension = file_path.split('.')[-1]
    
    if file_extension == 'csv':
        df = pd.read_csv(file_path)
    elif file_extension == 'xls' or file_extension == 'xlsx':
        df = pd.read_excel(file_path)
    elif file_extension == 'json':
        df = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file type")

    return d