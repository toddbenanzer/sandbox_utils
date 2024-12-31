from compute_metrics import compute_performance_metrics  # Adjust import according to your module's structure
from cross_validate import cross_validate_model  # Adjust import according to your module's structure
from customer_behavior_model import CustomerBehaviorModel
from data_preprocessing import DataPreprocessing
from documentation_module import load_documentation, show_examples  # Adjust import according to your module's structure
from kpi_model import KPIModel
from load_data_module import load_historical_data  # Adjust import according to your module's structure
from marketing_trend_model import MarketingTrendModel
from numpy import ndarray
from perform_grid_search import perform_grid_search  # Adjust import according to your module's structure
from perform_random_search import perform_random_search  # Adjust import according to your module's structure
from predictive_model_builder import PredictiveModelBuilder
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification
from sklearn.datasets import make_classification, make_regression
from sklearn.datasets import make_regression
from sklearn.datasets import make_regression, make_classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from unittest.mock import patch, mock_open
from utilities import Utilities  # Adjust import according to your module's structure
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1)
    return X, y

def test_init_linear_regression():
    builder = PredictiveModelBuilder(model_type='linear_regression', hyperparameters={})
    assert isinstance(builder.model, LinearRegression)

def test_init_random_forest():
    builder = PredictiveModelBuilder(model_type='random_forest', hyperparameters={'n_estimators': 10})
    assert isinstance(builder.model, RandomForestRegressor)
    assert builder.model.n_estimators == 10

def test_build_model_invalid_type():
    with pytest.raises(ValueError):
        PredictiveModelBuilder(model_type='unsupported_model', hyperparameters={})

def test_evaluate_model(sample_data):
    X, y = sample_data
    builder = PredictiveModelBuilder(model_type='linear_regression', hyperparameters={})
    builder.model.fit(X, y)
    metrics = builder.evaluate_model(X, y, metrics=['R_squared', 'MSE'])
    assert 'R_squared' in metrics
    assert 'MSE' in metrics
    assert np.isclose(metrics['R_squared'], 1, atol=0.1)

def test_fine_tune_parameters(sample_data):
    X, y = sample_data
    builder = PredictiveModelBuilder(model_type='random_forest', hyperparameters={'n_estimators': 10})
    param_grid = {'n_estimators': [5, 10, 15]}
    builder.fine_tune_parameters(X, y, method='grid_search', param_grid=param_grid)
    assert builder.model.n_estimators in [5, 10, 15]

def test_fine_tune_parameters_invalid_method(sample_data):
    X, y = sample_data
    builder = PredictiveModelBuilder(model_type='linear_regression', hyperparameters={})
    with pytest.raises(ValueError):
        builder.fine_tune_parameters(X, y, method='unsupported_method', param_grid={})



@pytest.fixture
def churn_data():
    """Fixture for creating a sample dataset for churn prediction."""
    data = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'feature3': np.random.rand(100),
        'churn': np.random.randint(0, 2, 100)
    })
    return data

@pytest.fixture
def ltv_data():
    """Fixture for creating a sample dataset for lifetime value estimation."""
    data = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'feature3': np.random.rand(100),
        'ltv': np.random.rand(100) * 1000
    })
    return data

def test_predict_churn(churn_data):
    model = CustomerBehaviorModel()
    churn_probabilities = model.predict_churn(churn_data)
    assert churn_probabilities.shape[0] == churn_data.shape[0]
    assert all(0 <= prob <= 1 for prob in churn_probabilities)

def test_estimate_lifetime_value(ltv_data):
    model = CustomerBehaviorModel()
    ltv_estimates = model.estimate_lifetime_value(ltv_data)
    assert ltv_estimates.shape[0] == ltv_data.shape[0]
    assert isinstance(ltv_estimates, np.ndarray)



@pytest.fixture
def time_series_data():
    """Fixture for creating a sample time-series dataset."""
    # Create a date range
    dates = pd.date_range(start='2020-01-01', periods=24, freq='M')
    
    # Create a time-series data with a seasonal component
    data_values = np.sin(np.linspace(0, 3.14 * 4, len(dates))) + np.random.normal(scale=0.1, size=len(dates))
    
    # Create a DataFrame
    df = pd.DataFrame(data={'date': dates, 'sales': data_values})
    df.set_index('date', inplace=True)
    return df

def test_detect_seasonality(time_series_data):
    model = MarketingTrendModel()
    seasonality_patterns = model.detect_seasonality(time_series_data, 'sales')
    
    assert 'seasonal_component' in seasonality_patterns
    assert 'average_seasonality' in seasonality_patterns
    assert 'seasonal_strength' in seasonality_patterns

def test_analyze_trends(time_series_data):
    model = MarketingTrendModel()
    trend_statistics = model.analyze_trends(time_series_data, 'sales')
    
    assert 'overall_trend' in trend_statistics
    assert trend_statistics['overall_trend'] in ['increasing', 'decreasing']
    assert 'trend_strength' in trend_statistics
    assert 'trend_start_value' in trend_statistics
    assert 'trend_end_value' in trend_statistics



@pytest.fixture
def conversion_data():
    """Fixture for creating a sample dataset for conversion rate prediction."""
    data = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'conversion': np.random.randint(0, 2, 100)
    })
    return data

@pytest.fixture
def roi_data():
    """Fixture for creating a sample dataset for ROI forecasting."""
    data = pd.DataFrame({
        'feature1': np.random.rand(100) * 1000,
        'feature2': np.random.rand(100) * 1000,
        'roi': np.random.rand(100) * 10
    })
    return data

def test_predict_conversion_rate(conversion_data):
    model = KPIModel()
    conversion_rates = model.predict_conversion_rate(conversion_data, target='conversion')
    
    assert conversion_rates.shape[0] == conversion_data.shape[0]
    assert all(0 <= prob <= 1 for prob in conversion_rates)
    assert isinstance(conversion_rates, np.ndarray)

def test_forecast_roi(roi_data):
    model = KPIModel()
    roi_forecasts = model.forecast_roi(roi_data, target='roi')

    assert roi_forecasts.shape[0] == roi_data.shape[0]
    assert isinstance(roi_forecasts, np.ndarray)



@pytest.fixture
def sample_data():
    """Fixture to provide sample data with missing values."""
    data = pd.DataFrame({
        'feature1': [1.0, 2.0, None, 4.0, 5.0],
        'feature2': [2.0, None, 3.0, 4.0, 5.0],
        'feature3': [None, 1.0, 2.0, None, 4.0]
    })
    return data

def test_handle_missing_values(sample_data):
    """Test handling missing values."""
    preprocessing = DataPreprocessing(sample_data)
    processed_data = preprocessing.handle_missing_values()

    assert processed_data.isnull().sum().sum() == 0
    assert processed_data.shape == sample_data.shape

def test_normalize_data(sample_data):
    """Test normalization of data."""
    preprocessing = DataPreprocessing(sample_data.fillna(0))  # Fill NaN to proceed with normalization
    normalized_data = preprocessing.normalize_data()

    assert normalized_data.mean().round(0).equals(pd.Series([0, 0, 0]))  # Mean should be approximately zero
    assert normalized_data.std().round(0).equals(pd.Series([1, 1, 1]))  # Standard deviation should be approximately one



def test_load_csv_file():
    csv_content = "column1,column2\nvalue1,value2\nvalue3,value4"
    with patch("builtins.open", mock_open(read_data=csv_content)):
        with patch("os.path.exists", return_value=True):
            result_df = load_historical_data("test.csv")
            assert isinstance(result_df, pd.DataFrame)
            assert not result_df.empty
            assert list(result_df.columns) == ['column1', 'column2']

def test_load_excel_file():
    excel_content = pd.DataFrame({
        'column1': ['value1', 'value3'],
        'column2': ['value2', 'value4']
    })
    with patch("pandas.read_excel", return_value=excel_content):
        with patch("os.path.exists", return_value=True):
            result_df = load_historical_data("test.xlsx")
            assert isinstance(result_df, pd.DataFrame)
            assert not result_df.empty
            assert list(result_df.columns) == ['column1', 'column2']

def test_load_json_file():
    json_content = pd.DataFrame({
        'column1': ['value1', 'value3'],
        'column2': ['value2', 'value4']
    })
    with patch("pandas.read_json", return_value=json_content):
        with patch("os.path.exists", return_value=True):
            result_df = load_historical_data("test.json")
            assert isinstance(result_df, pd.DataFrame)
            assert not result_df.empty
            assert list(result_df.columns) == ['column1', 'column2']

def test_file_not_found_error():
    with patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            load_historical_data("non_existent_file.csv")

def test_unsupported_file_format():
    with patch("os.path.exists", return_value=True):
        with pytest.raises(ValueError):
            load_historical_data("test.txt")

def test_other_exceptions():
    with patch("os.path.exists", return_value=True):
        with patch("pandas.read_csv", side_effect=Exception("Custom error")):
            with pytest.raises(Exception):
                load_historical_data("test.csv")



@pytest.fixture
def classification_data():
    """Fixture to generate sample classification data."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    return X, y

@pytest.fixture
def regression_data():
    """Fixture to generate sample regression data."""
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    return X, y

def test_cross_validate_model_classifier_default_scoring(classification_data):
    model = LogisticRegression()
    X, y = classification_data
    cv_scores = cross_validate_model(model, (X, y), cv_folds=5)
    
    assert isinstance(cv_scores, ndarray)
    assert len(cv_scores) == 5

def test_cross_validate_model_classifier_custom_scoring(classification_data):
    model = LogisticRegression()
    X, y = classification_data
    cv_scores = cross_validate_model(model, (X, y), cv_folds=5, scoring='f1')
    
    assert isinstance(cv_scores, ndarray)
    assert len(cv_scores) == 5

def test_cross_validate_model_regressor_default_scoring(regression_data):
    model = LinearRegression()
    X, y = regression_data
    cv_scores = cross_validate_model(model, (X, y), cv_folds=5)
    
    assert isinstance(cv_scores, ndarray)
    assert len(cv_scores) == 5

def test_cross_validate_model_regressor_custom_scoring(regression_data):
    model = LinearRegression()
    X, y = regression_data
    cv_scores = cross_validate_model(model, (X, y), cv_folds=5, scoring='r2')
    
    assert isinstance(cv_scores, ndarray)
    assert len(cv_scores) == 5

def test_unsupported_model_type():
    class UnsupportedModel(BaseEstimator):
        def fit(self): pass
        def predict(self): pass
    
    model = UnsupportedModel()
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    
    with pytest.raises(ValueError, match="Model type not recognized"):
        cross_validate_model(model, (X, y), cv_folds=5)



@pytest.fixture
def regression_data():
    """Fixture to generate sample regression data."""
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    return X, y

@pytest.fixture
def classification_data():
    """Fixture to generate sample classification data."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    return X, y

def test_compute_performance_metrics_regressor(regression_data):
    model = LinearRegression()
    X, y = regression_data
    model.fit(X, y)
    expected_metrics = {
        'R_squared': r2_score(y, model.predict(X)),
        'MSE': mean_squared_error(y, model.predict(X))
    }
    result = compute_performance_metrics(model, (X, y), metrics=['R_squared', 'MSE'])
    assert result == pytest.approx(expected_metrics)

def test_compute_performance_metrics_classifier(classification_data):
    model = LogisticRegression(solver='liblinear')
    X, y = classification_data
    model.fit(X, y)
    y_prob = model.predict_proba(X)[:, 1]
    expected_auc = roc_auc_score(y, y_prob)
    result = compute_performance_metrics(model, (X, y), metrics=['AUC_ROC'])
    assert result['AUC_ROC'] == pytest.approx(expected_auc)

def test_unsupported_metric_for_regressor(regression_data):
    model = LinearRegression()
    X, y = regression_data
    model.fit(X, y)
    with pytest.raises(ValueError, match="Unsupported metric 'AUC_ROC' for the given model type."):
        compute_performance_metrics(model, (X, y), metrics=['AUC_ROC'])

def test_unsupported_metric_for_classifier(classification_data):
    model = LogisticRegression(solver='liblinear')
    X, y = classification_data
    model.fit(X, y)
    with pytest.raises(ValueError, match="Unsupported metric 'R_squared' for the given model type."):
        compute_performance_metrics(model, (X, y), metrics=['R_squared'])



@pytest.fixture
def classification_data():
    """Fixture to generate sample classification data."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def test_perform_grid_search(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    model = LogisticRegression(solver='liblinear')
    param_grid = {'C': [0.1, 1, 10]}
    
    best_model, best_params, search_results = perform_grid_search(model, param_grid, X_train, y_train, cv=3, scoring='accuracy')
    
    assert isinstance(best_model, LogisticRegression)
    assert best_params['C'] in [0.1, 1, 10]
    assert 'mean_test_score' in search_results

def test_perform_grid_search_invalid_param(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    model = LogisticRegression(solver='liblinear')
    param_grid = {'invalid_param': [0.1, 1, 10]}
    
    with pytest.raises(ValueError):
        perform_grid_search(model, param_grid, X_train, y_train, cv=3, scoring='accuracy')



@pytest.fixture
def classification_data():
    """Fixture to generate sample classification data."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def test_perform_random_search(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    model = LogisticRegression(solver='liblinear')
    param_dist = {'C': [0.01, 0.1, 1, 10, 100]}
    
    best_model, best_params, search_results = perform_random_search(model, param_dist, X_train, y_train, n_iter=5, scoring='accuracy', random_state=42)
    
    assert isinstance(best_model, LogisticRegression)
    assert best_params['C'] in [0.01, 0.1, 1, 10, 100]
    assert 'mean_test_score' in search_results

def test_perform_random_search_invalid_param(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    model = LogisticRegression(solver='liblinear')
    param_dist = {'invalid_param': [0.1, 1, 10]}
    
    with pytest.raises(ValueError):
        perform_random_search(model, param_dist, X_train, y_train, n_iter=5, scoring='accuracy')



@pytest.fixture
def sample_data():
    """Fixture to create sample data for testing."""
    data = pd.DataFrame({
        'category': ['A', 'B', 'A', 'C'],
        'value1': [1, 2, 3, 4],
        'value2': [10.0, 15.0, 14.0, 16.0]
    })
    return data

def test_one_hot_encode(sample_data):
    encoded_data = Utilities.one_hot_encode(sample_data, ['category'])
    assert 'category_B' in encoded_data.columns
    assert 'category_C' in encoded_data.columns
    assert encoded_data.shape[1] == sample_data.shape[1] + 1  # Since 'A' is dropped

def test_scale_features(sample_data):
    scaled_data = Utilities.scale_features(sample_data.copy(), ['value1', 'value2'])
    assert np.isclose(scaled_data['value1'].mean(), 0, atol=1e-7)
    assert np.isclose(scaled_data['value2'].mean(), 0, atol=1e-7)
    assert np.isclose(scaled_data['value1'].std(), 1, atol=1e-7)
    assert np.isclose(scaled_data['value2'].std(), 1, atol=1e-7)

def test_plot_distribution(sample_data):
    Utilities.plot_distribution(sample_data['value1'])
    # No assertions necessary as plot visual output is not verified via unit testing

def test_plot_correlation_heatmap(sample_data):
    Utilities.plot_correlation_heatmap(sample_data)
    # No assertions necessary as plot visual output is not verified via unit testing

def test_plot_scatter(sample_data):
    Utilities.plot_scatter(sample_data['value1'], sample_data['value2'])
    # No assertions necessary as plot visual output is not verified via unit testing



def test_load_documentation():
    """Test load_documentation function to ensure it returns documentation as a string."""
    documentation = load_documentation()
    assert isinstance(documentation, str), "Documentation should be a string"
    assert "Class: PredictiveModelBuilder" in documentation, "Documentation should include PredictiveModelBuilder details"
    assert "Functions:" in documentation, "Documentation should include Functions section"
    assert "load_historical_data(file_path)" in documentation, "Documentation should include load_historical_data function"

def test_show_examples():
    """Test show_examples function to ensure it returns examples as a string."""
    examples = show_examples()
    assert isinstance(examples, str), "Examples should be a string"
    assert "Example 1: Using PredictiveModelBuilder" in examples, "Examples should include a section on PredictiveModelBuilder"
    assert "cb_model = CustomerBehaviorModel()" in examples, "Examples should demonstrate CustomerBehaviorModel usage"
    assert "Utilities.plot_distribution(data['value1'])" in examples, "Examples should demonstrate Utilities module usage"
