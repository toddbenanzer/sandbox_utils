from sklearn.base import is_classifier, is_regressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns


class PredictiveModelBuilder:
    """
    A class for building, evaluating, and fine-tuning predictive models.
    """

    def __init__(self, model_type: str, hyperparameters: dict):
        """
        Initializes the PredictiveModelBuilder with a specified model type and hyperparameters.

        Args:
            model_type (str): The type of predictive model to be constructed.
            hyperparameters (dict): Hyperparameters specific to the model type.
        """
        self.model_type = model_type
        self.hyperparameters = hyperparameters
        self.model = self.build_model()

    def build_model(self):
        """
        Constructs and initializes a predictive model based on the specified model type.

        Returns:
            model: The constructed predictive model object.
        """
        if self.model_type == 'linear_regression':
            from sklearn.linear_model import LinearRegression
            return LinearRegression(**self.hyperparameters)
        elif self.model_type == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(**self.hyperparameters)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def evaluate_model(self, X, y, metrics: list):
        """
        Evaluates the model's performance using specified metrics.

        Args:
            X: Input features for evaluation.
            y: True labels/target values.
            metrics (list): Names of metrics to assess model performance.

        Returns:
            dict: A dictionary mapping metric names to their computed values.
        """
        predictions = self.model.predict(X)
        results = {}

        if "R_squared" in metrics:
            results["R_squared"] = r2_score(y, predictions)
        if "MSE" in metrics:
            results["MSE"] = mean_squared_error(y, predictions)

        return results

    def fine_tune_parameters(self, X, y, method: str, param_grid: dict):
        """
        Adjusts the model's hyperparameters to improve performance using the specified method.

        Args:
            X: Input features for tuning.
            y: True labels/target values.
            method (str): The tuning strategy ("grid_search" or "random_search").
            param_grid (dict): The parameter grid/distribution used for tuning.

        Returns:
            self: Updates the model with optimized parameters.
        """
        if method == "grid_search":
            search = GridSearchCV(estimator=self.model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
        elif method == "random_search":
            search = RandomizedSearchCV(estimator=self.model, param_distributions=param_grid, cv=5, scoring='neg_mean_squared_error', n_iter=10)
        else:
            raise ValueError(f"Unsupported tuning method: {method}")

        search.fit(X, y)
        self.model = search.best_estimator_
        return self



class CustomerBehaviorModel:
    """
    A class for predicting customer behavior, including churn likelihood and lifetime value estimation.
    """

    def __init__(self):
        """
        Initializes the CustomerBehaviorModel, setting up the random forest classifier for churn prediction
        and linear regression for lifetime value estimation.
        """
        self.churn_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.ltv_model = LinearRegression()

    def predict_churn(self, data: pd.DataFrame):
        """
        Predicts customer churn likelihood based on input data.

        Args:
            data (pd.DataFrame): The dataset containing customer information necessary for churn prediction.

        Returns:
            np.ndarray: An array containing the predicted probabilities of churn for each customer in the dataset.
        """
        features = data.drop(columns=['churn'])
        labels = data['churn']
        self.churn_model.fit(features, labels)
        churn_probabilities = self.churn_model.predict_proba(features)[:, 1]
        return churn_probabilities

    def estimate_lifetime_value(self, data: pd.DataFrame):
        """
        Estimates customer lifetime value based on transaction history and customer information.

        Args:
            data (pd.DataFrame): The dataset containing historical transaction data and customer information.

        Returns:
            np.ndarray: An array of estimated lifetime values for each customer in the dataset.
        """
        features = data.drop(columns=['ltv'])
        ltv_values = data['ltv']
        self.ltv_model.fit(features, ltv_values)
        ltv_estimates = self.ltv_model.predict(features)
        return ltv_estimates



class MarketingTrendModel:
    """
    A class for detecting seasonality and analyzing trends in marketing data.
    """

    def __init__(self):
        """
        Initializes the MarketingTrendModel, preparing necessary components for trend and seasonality analysis.
        """
        pass

    def detect_seasonality(self, data: pd.DataFrame, column: str, freq: str = 'M'):
        """
        Detects and analyzes seasonal patterns within the provided marketing data.

        Args:
            data (pd.DataFrame): The dataset containing time-series marketing data for seasonality detection.
            column (str): The name of the column with time-series data to analyze.
            freq (str): Frequency of the time series. Default is 'M' for monthly data.

        Returns:
            dict: A dictionary detailing identified seasonal patterns and characteristics.
        """
        decomposition = seasonal_decompose(data[column], model='additive', period=12)
        seasonality = decomposition.seasonal
        trend = decomposition.trend

        seasonality_patterns = {
            'seasonal_component': seasonality,
            'average_seasonality': seasonality.mean(),
            'seasonal_strength': seasonality.std() / seasonality.std(ddof=0) if trend is not None else np.nan
        }

        return seasonality_patterns

    def analyze_trends(self, data: pd.DataFrame, column: str):
        """
        Identifies and provides insights into underlying trends in marketing data.

        Args:
            data (pd.DataFrame): The dataset containing time-series marketing data for trend analysis.
            column (str): The name of the column with time-series data to analyze.

        Returns:
            dict: A dictionary containing metrics and insights about trend behavior over time.
        """
        decomposition = seasonal_decompose(data[column], model='additive', period=12)
        trend = decomposition.trend.dropna()

        trend_statistics = {
            'overall_trend': 'increasing' if trend.iloc[-1] > trend.iloc[0] else 'decreasing',
            'trend_strength': np.mean(np.diff(trend)),
            'trend_start_value': trend.iloc[0],
            'trend_end_value': trend.iloc[-1]
        }

        return trend_statistics



class KPIModel:
    """
    A class for predicting key performance indicators (KPIs) such as conversion rates and return on investment (ROI).
    """

    def __init__(self):
        """
        Initializes the KPIModel, setting up logistic regression for conversion rate prediction
        and linear regression for ROI forecasting.
        """
        self.conversion_model = LogisticRegression()
        self.roi_model = LinearRegression()

    def predict_conversion_rate(self, data: pd.DataFrame, target: str):
        """
        Predicts conversion rates based on the input marketing data.

        Args:
            data (pd.DataFrame): The dataset containing features relevant to predicting conversion rates.
            target (str): The column name representing the conversion target variable.

        Returns:
            np.ndarray: An array containing the predicted conversion rates for each data entry.
        """
        features = data.drop(columns=[target])
        labels = data[target]
        self.conversion_model.fit(features, labels)
        conversion_rate_predictions = self.conversion_model.predict_proba(features)[:, 1]
        return conversion_rate_predictions

    def forecast_roi(self, data: pd.DataFrame, target: str):
        """
        Forecasts the return on investment (ROI) for given marketing investment data.

        Args:
            data (pd.DataFrame): The dataset containing financial metrics and data necessary for ROI forecasting.
            target (str): The column name representing the ROI target variable.

        Returns:
            np.ndarray: An array of forecasted ROI values for each data entry.
        """
        features = data.drop(columns=[target])
        roi_values = data[target]
        self.roi_model.fit(features, roi_values)
        roi_forecasts = self.roi_model.predict(features)
        return roi_forecasts



class DataPreprocessing:
    """
    A class for preprocessing marketing data, including handling missing values and normalizing datasets.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initializes the DataPreprocessing class with the provided dataset.

        Args:
            data (pd.DataFrame): The dataset to be preprocessed.
        """
        self.data = data

    def handle_missing_values(self):
        """
        Handles any missing values within the dataset by filling them with the mean of their respective columns.

        Returns:
            pd.DataFrame: The dataset after handling missing values.
        """
        preprocessed_data = self.data.fillna(self.data.mean())
        return preprocessed_data

    def normalize_data(self):
        """
        Normalizes the dataset to ensure all features are on a comparable scale using standardization.

        Returns:
            pd.DataFrame: The dataset after normalization processes have been applied.
        """
        scaler = StandardScaler()
        normalized_data = pd.DataFrame(scaler.fit_transform(self.data), columns=self.data.columns)
        return normalized_data



def load_historical_data(file_path: str) -> pd.DataFrame:
    """
    Loads historical marketing data from a specified file path into a Pandas DataFrame.

    Args:
        file_path (str): The path to the file containing historical marketing data.

    Returns:
        pd.DataFrame: The loaded historical data as a Pandas DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist at the provided path.
        ValueError: If the file format is unsupported.
        Exception: For any other errors encountered during loading.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} does not exist.")

    try:
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            data = pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            data = pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format. Please use CSV, Excel, or JSON files.")
    except Exception as e:
        raise Exception(f"An error occurred while loading the data: {e}")

    return data



def cross_validate_model(model, data: tuple, cv_folds: int, scoring: str = None) -> np.ndarray:
    """
    Performs cross-validation on the specified predictive model.

    Args:
        model: The predictive model instance to be evaluated. It must implement the `fit` and `predict` methods.
        data (tuple): A tuple containing the feature matrix `X` and the target vector `y` in the form `(X, y)`.
        cv_folds (int): The number of folds to use in cross-validation.
        scoring (str, optional): The scoring metric to use for evaluation. Defaults to model type ('accuracy' for classifiers, 'neg_mean_squared_error' for regressors).

    Returns:
        np.ndarray: An array containing the cross-validation scores for each fold.
    """
    X, y = data
    
    if scoring is None:
        # Determine default scoring based on whether the model is a classifier or regressor
        if is_classifier(model):
            scoring = 'accuracy'
        elif is_regressor(model):
            scoring = 'neg_mean_squared_error'
        else:
            raise ValueError("Model type not recognized. Please specify a valid scoring method.")
    
    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
    return cv_scores



def compute_performance_metrics(model, data: tuple, metrics: list = ['R_squared', 'MSE', 'AUC_ROC']) -> dict:
    """
    Evaluates the performance of a predictive model using specified metrics.

    Args:
        model: The predictive model instance that has been trained and will be evaluated.
        data (tuple): A tuple containing the feature matrix `X` and the target vector `y` in the form `(X, y)`.
        metrics (list of str): A list of metric names to compute for performance evaluation.

    Returns:
        dict: A dictionary mapping each metric name to its computed value.
    """
    X, y_true = data
    y_pred = model.predict(X)
    results = {}

    for metric in metrics:
        if metric == 'R_squared' and is_regressor(model):
            results['R_squared'] = r2_score(y_true, y_pred)
        elif metric == 'MSE' and is_regressor(model):
            results['MSE'] = mean_squared_error(y_true, y_pred)
        elif metric == 'AUC_ROC' and is_classifier(model):
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X)[:, 1]
            else:  # Use decision function scores for models without predict_proba
                y_prob = model.decision_function(X)
            results['AUC_ROC'] = roc_auc_score(y_true, y_prob)
        else:
            raise ValueError(f"Unsupported metric '{metric}' for the given model type.")

    return results



def perform_grid_search(model, parameter_grid: dict, X, y, cv: int = 5, scoring: str = None) -> tuple:
    """
    Performs a grid search to find the best hyperparameters for a given model.

    Args:
        model: The predictive model instance for which hyperparameter tuning is to be performed.
        parameter_grid (dict): A dictionary where keys are parameter names and values are lists of parameters to try.
        X: The input features for training the model.
        y: The target values for training the model.
        cv (int): The number of cross-validation folds. Defaults to 5.
        scoring (str, optional): The scoring metric to use for evaluation. If None, defaults to model type.

    Returns:
        tuple: A tuple containing the best model, best parameters, and the grid search results.
    """
    grid_search = GridSearchCV(estimator=model, param_grid=parameter_grid, cv=cv, scoring=scoring, return_train_score=True)
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    grid_search_results = grid_search.cv_results_

    return best_model, best_params, grid_search_results



def perform_random_search(model, parameter_distribution: dict, X, y, n_iter: int = 10, cv: int = 5, scoring: str = None, random_state: int = None) -> tuple:
    """
    Performs a random search to find the best hyperparameters for a given model.

    Args:
        model: The predictive model instance for which hyperparameter tuning is to be performed.
        parameter_distribution (dict): A dictionary where keys are parameter names and values are distributions or lists to sample from.
        X: The input features for training the model.
        y: The target values for training the model.
        n_iter (int): Number of parameter settings that are sampled. Defaults to 10.
        cv (int): The number of cross-validation folds. Defaults to 5.
        scoring (str, optional): The scoring metric to use for evaluation. If None, defaults to model type.
        random_state (int, optional): Controls the randomness of the search. Defaults to None.

    Returns:
        tuple: A tuple containing the best model, best parameters, and the random search results.
    """
    random_search = RandomizedSearchCV(estimator=model, param_distributions=parameter_distribution, n_iter=n_iter, cv=cv, scoring=scoring, random_state=random_state, return_train_score=True)
    random_search.fit(X, y)

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    random_search_results = random_search.cv_results_

    return best_model, best_params, random_search_results



class Utilities:
    """
    A collection of helper functions for data transformations and visualizations.
    """

    @staticmethod
    def one_hot_encode(data: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Apply one-hot encoding to specified categorical columns.

        Args:
            data (pd.DataFrame): The dataset containing categorical features.
            columns (list): List of column names to encode.

        Returns:
            pd.DataFrame: Transformed dataset with one-hot encoded features.
        """
        encoder = OneHotEncoder(sparse=False, drop='first')
        encoded_cols = encoder.fit_transform(data[columns])
        encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(columns))
        data = data.drop(columns, axis=1).reset_index(drop=True)
        return pd.concat([data, encoded_df], axis=1)

    @staticmethod
    def scale_features(data: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Standardize features by removing the mean and scaling to unit variance.

        Args:
            data (pd.DataFrame): The dataset containing features to scale.
            columns (list): List of column names to scale.

        Returns:
            pd.DataFrame: Transformed dataset with scaled features.
        """
        scaler = StandardScaler()
        scaled_cols = scaler.fit_transform(data[columns])
        scaled_df = pd.DataFrame(scaled_cols, columns=columns)
        data.update(scaled_df)
        return data

    @staticmethod
    def plot_distribution(data: pd.Series, title: str = "Distribution Plot", bins: int = 30):
        """
        Plot a histogram to show the distribution of a feature.

        Args:
            data (pd.Series): The data series to plot.
            title (str): The title of the plot.
            bins (int): The number of bins for the histogram.

        Returns:
            None
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(data, bins=bins, kde=True)
        plt.title(title)
        plt.xlabel(data.name)
        plt.ylabel('Frequency')
        plt.show()

    @staticmethod
    def plot_correlation_heatmap(data: pd.DataFrame, title: str = "Correlation Heatmap"):
        """
        Plot a heatmap to show the correlation matrix of the dataset.

        Args:
            data (pd.DataFrame): The dataset to analyze.
            title (str): The title of the heatmap.

        Returns:
            None
        """
        plt.figure(figsize=(12, 8))
        cor_matrix = data.corr()
        sns.heatmap(cor_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_scatter(x: pd.Series, y: pd.Series, hue: pd.Series = None, title: str = "Scatter Plot"):
        """
        Create a scatter plot to visualize the relationship between two variables.

        Args:
            x (pd.Series): Data for the x-axis.
            y (pd.Series): Data for the y-axis.
            hue (pd.Series, optional): Categorical variable for coloring.
            title (str): The title of the scatter plot.

        Returns:
            None
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=x, y=y, hue=hue)
        plt.title(title)
        plt.xlabel(x.name)
        plt.ylabel(y.name)
        plt.show()


def load_documentation():
    """
    Loads the documentation for the module and returns it as a formatted string.

    Returns:
        str: The documentation for the module, including descriptions of classes and functions.
    """
    documentation = """
    Module Documentation:
    
    - **Class: PredictiveModelBuilder**
        - **__init__(self, model_type, hyperparameters)**: Initializes the model builder with a specified type and hyperparameters.
        - **build_model(self)**: Builds the predictive model based on the type and hyperparameters.
        - **evaluate_model(self, metrics)**: Evaluates the model using specified metrics.
        - **fine_tune_parameters(self, method)**: Fine-tunes the model parameters using a specified method (e.g., grid search).

    - **Class: CustomerBehaviorModel**
        - **__init__(self)**: Initializes the customer behavior model.
        - **predict_churn(self, data)**: Predicts the churn rate for the given data.
        - **estimate_lifetime_value(self, data)**: Estimates the customer lifetime value.

    - **Class: MarketingTrendModel**
        - **__init__(self)**: Initializes the marketing trend model.
        - **detect_seasonality(self, data)**: Detects seasonal patterns in the data.
        - **analyze_trends(self, data)**: Analyzes trends in marketing data.

    - **Class: KPIModel**
        - **__init__(self)**: Initializes the KPI model.
        - **predict_conversion_rate(self, data)**: Predicts conversion rates.
        - **forecast_roi(self, data)**: Forecasts return on investment (ROI).

    - **Class: DataPreprocessing**
        - **__init__(self, data)**: Initializes data preprocessing with the given data.
        - **handle_missing_values(self)**: Handles missing values in the dataset.
        - **normalize_data(self)**: Normalizes the data.

    - **Functions:**
        - **load_historical_data(file_path)**: Loads historical data from a file.
        - **cross_validate_model(model, data, cv_folds)**: Cross-validates a model with the given data and number of folds.
        - **compute_performance_metrics(model, data, metrics=['R_squared', 'MSE', 'AUC_ROC'])**: Computes performance metrics for the model.
        - **perform_grid_search(model, parameter_grid)**: Performs a grid search for hyperparameter tuning.
        - **perform_random_search(model, parameter_distribution)**: Performs a random search for hyperparameter tuning.

    - **Module: utilities**
        - Helper functions for data transformations and visualizations.

    - **Module: documentation**
        - **load_documentation()**: Loads module documentation.
        - **show_examples()**: Displays examples of how to use the module.
    """
    return documentation

def show_examples():
    """
    Displays example usages of different classes and functions within the module.

    Returns:
        str: Examples illustrating how to use the module's functionalities.
    """
    examples = """
    Module Usage Examples:
    
    Example 1: Using PredictiveModelBuilder
        cb_model = CustomerBehaviorModel()
    churn_rates = cb_model.predict_churn(data)
    ltv = cb_model.estimate_lifetime_value(data)
        data_preprocessor = DataPreprocessing(data)
    clean_data = data_preprocessor.handle_missing_values()
    normalized_data = data_preprocessor.normalize_data()
        import Utilities
    encoded_data = Utilities.one_hot_encode(data, columns=['category'])
    scaled_data = Utilities.scale_features(data, columns=['value1', 'value2'])
    Utilities.plot_distribution(data['value1'])
    Utilities.plot_correlation_heatmap(data)
    Utilities.plot_scatter(data['feature1'], data['feature2'])
    