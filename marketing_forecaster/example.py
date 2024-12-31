from compute_metrics import compute_performance_metrics  # Adjust import according to your module's structure
from cross_validate import cross_validate_model  # Adjust import according to your module's structure
from customer_behavior_model import CustomerBehaviorModel
from data_preprocessing import DataPreprocessing
from kpi_model import KPIModel
from load_data_module import load_historical_data  # Adjust import according to your module's structure
from marketing_trend_model import MarketingTrendModel
from perform_grid_search import perform_grid_search  # Adjust import according to your module's structure
from perform_random_search import perform_random_search  # Adjust import according to your module's structure
from predictive_model_builder import PredictiveModelBuilder
from sklearn.datasets import make_classification
from sklearn.datasets import make_classification, make_regression
from sklearn.datasets import make_regression
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from utilities import Utilities  # Adjust import according to your module's structure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Generate sample data
X, y = make_regression(n_samples=100, n_features=10, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Example 1: Build and evaluate a Linear Regression model
linear_builder = PredictiveModelBuilder(model_type='linear_regression', hyperparameters={})
linear_builder.model.fit(X_train, y_train)
metrics = linear_builder.evaluate_model(X_test, y_test, metrics=['R_squared', 'MSE'])
print("Linear Regression Evaluation:", metrics)

# Example 2: Build and evaluate a Random Forest model
rf_builder = PredictiveModelBuilder(model_type='random_forest', hyperparameters={'n_estimators': 100})
rf_builder.model.fit(X_train, y_train)
metrics = rf_builder.evaluate_model(X_test, y_test, metrics=['R_squared', 'MSE'])
print("Random Forest Evaluation:", metrics)

# Example 3: Fine-tune Random Forest model using grid search
param_grid = {'n_estimators': [50, 100, 150]}
rf_builder.fine_tune_parameters(X_train, y_train, method='grid_search', param_grid=param_grid)
optimized_metrics = rf_builder.evaluate_model(X_test, y_test, metrics=['R_squared', 'MSE'])
print("Optimized Random Forest Evaluation:", optimized_metrics)



# Create a sample dataset for churn prediction
churn_data = pd.DataFrame({
    'feature1': np.random.rand(10),
    'feature2': np.random.rand(10),
    'feature3': np.random.rand(10),
    'churn': np.random.randint(0, 2, 10)
})

# Initialize the CustomerBehaviorModel
cb_model = CustomerBehaviorModel()

# Predict churn probabilities
churn_probabilities = cb_model.predict_churn(churn_data)
print("Churn Probabilities:", churn_probabilities)

# Create a sample dataset for lifetime value estimation
ltv_data = pd.DataFrame({
    'feature1': np.random.rand(10),
    'feature2': np.random.rand(10),
    'feature3': np.random.rand(10),
    'ltv': np.random.rand(10) * 1000
})

# Estimate lifetime values
ltv_estimates = cb_model.estimate_lifetime_value(ltv_data)
print("Lifetime Value Estimates:", ltv_estimates)



# Example 1: Detecting Seasonality in Sales Data
dates = pd.date_range(start='2020-01-01', periods=24, freq='M')
sales_data = np.sin(np.linspace(0, 3.14 * 4, len(dates))) + np.random.normal(scale=0.1, size=len(dates))
df_sales = pd.DataFrame(data={'date': dates, 'sales': sales_data})
df_sales.set_index('date', inplace=True)

# Initialize the MarketingTrendModel
model = MarketingTrendModel()

# Detect seasonality
seasonality_info = model.detect_seasonality(df_sales, 'sales')
print("Seasonality Information:", seasonality_info)

# Example 2: Analyzing Trends in Sales Data
trend_info = model.analyze_trends(df_sales, 'sales')
print("Trend Information:", trend_info)

# Example 3: Visualizing Seasonal Component

plt.figure(figsize=(10, 5))
plt.plot(seasonality_info['seasonal_component'], label='Seasonal Component')
plt.title('Seasonal Component')
plt.legend()
plt.show()

# Example 4: Visualizing Trend Component
trend = seasonal_decompose(df_sales['sales'], model='additive', period=12).trend

plt.figure(figsize=(10, 5))
plt.plot(trend, label='Trend Component', color='orange')
plt.title('Trend Component')
plt.legend()
plt.show()



# Create a sample dataset for conversion rate prediction
conversion_data = pd.DataFrame({
    'feature1': np.random.rand(10),
    'feature2': np.random.rand(10),
    'conversion': np.random.randint(0, 2, 10)
})

# Initialize the KPIModel
kpi_model = KPIModel()

# Predict conversion rates
conversion_rate_predictions = kpi_model.predict_conversion_rate(conversion_data, target='conversion')
print("Conversion Rate Predictions:", conversion_rate_predictions)

# Create a sample dataset for ROI forecasting
roi_data = pd.DataFrame({
    'feature1': np.random.rand(10) * 1000,
    'feature2': np.random.rand(10) * 1000,
    'roi': np.random.rand(10) * 10
})

# Forecast ROI
roi_forecasts = kpi_model.forecast_roi(roi_data, target='roi')
print("ROI Forecasts:", roi_forecasts)



# Example 1: Handling Missing Values
data_with_missing_values = pd.DataFrame({
    'feature1': [1.0, 2.0, None, 4.0, 5.0],
    'feature2': [2.0, None, 3.0, 4.0, 5.0],
    'feature3': [None, 1.0, 2.0, None, 4.0]
})

# Initialize the DataPreprocessing class
preprocessor = DataPreprocessing(data_with_missing_values)

# Handle missing values
processed_data = preprocessor.handle_missing_values()
print("Data after handling missing values:")
print(processed_data)

# Example 2: Normalizing Data
# Fill missing data first to demonstrate normalization
filled_data = processed_data  

# Normalize data
normalized_data = preprocessor.normalize_data()
print("\nNormalized Data:")
print(normalized_data)



# Example 1: Load a CSV file
csv_file_path = 'historical_data.csv'
try:
    csv_data = load_historical_data(csv_file_path)
    print("CSV data loaded successfully:")
    print(csv_data.head())
except Exception as e:
    print(f"Failed to load CSV data: {e}")

# Example 2: Load an Excel file
excel_file_path = 'historical_data.xlsx'
try:
    excel_data = load_historical_data(excel_file_path)
    print("\nExcel data loaded successfully:")
    print(excel_data.head())
except Exception as e:
    print(f"Failed to load Excel data: {e}")

# Example 3: Load a JSON file
json_file_path = 'historical_data.json'
try:
    json_data = load_historical_data(json_file_path)
    print("\nJSON data loaded successfully:")
    print(json_data.head())
except Exception as e:
    print(f"Failed to load JSON data: {e}")

# Example 4: Handling a non-existent file
non_existent_file_path = 'non_existent_data.csv'
try:
    non_existent_data = load_historical_data(non_existent_file_path)
except FileNotFoundError as e:
    print(f"File not found error: {e}")

# Example 5: Handling an unsupported file format
unsupported_file_path = 'unsupported_data.txt'
try:
    unsupported_data = load_historical_data(unsupported_file_path)
except ValueError as e:
    print(f"Unsupported file format error: {e}")



# Example 1: Cross-validate a logistic regression model with default scoring (accuracy)
X_class, y_class = make_classification(n_samples=100, n_features=5, random_state=42)
log_reg_model = LogisticRegression()
cv_scores_class = cross_validate_model(log_reg_model, (X_class, y_class), cv_folds=5)
print("Logistic Regression CV Scores (Accuracy):", cv_scores_class)

# Example 2: Cross-validate a logistic regression model with custom scoring (F1 score)
cv_scores_class_f1 = cross_validate_model(log_reg_model, (X_class, y_class), cv_folds=5, scoring='f1')
print("Logistic Regression CV Scores (F1):", cv_scores_class_f1)

# Example 3: Cross-validate a decision tree classifier with default scoring (accuracy)
tree_model = DecisionTreeClassifier()
cv_scores_tree = cross_validate_model(tree_model, (X_class, y_class), cv_folds=5)
print("Decision Tree CV Scores (Accuracy):", cv_scores_tree)

# Example 4: Cross-validate a linear regression model with default scoring (neg_mean_squared_error)
X_reg, y_reg = make_regression(n_samples=100, n_features=5, random_state=42)
lin_reg_model = LinearRegression()
cv_scores_reg = cross_validate_model(lin_reg_model, (X_reg, y_reg), cv_folds=5)
print("Linear Regression CV Scores (Neg MSE):", cv_scores_reg)

# Example 5: Cross-validate a linear regression model with custom scoring (R2 score)
cv_scores_reg_r2 = cross_validate_model(lin_reg_model, (X_reg, y_reg), cv_folds=5, scoring='r2')
print("Linear Regression CV Scores (R2):", cv_scores_reg_r2)



# Example 1: Evaluating a Regression Model
X_reg, y_reg = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_reg, y_reg)

regression_metrics = compute_performance_metrics(lin_reg_model, (X_reg, y_reg), metrics=['R_squared', 'MSE'])
print("Regression Model Metrics:", regression_metrics)

# Example 2: Evaluating a Classification Model with AUC_ROC
X_class, y_class = make_classification(n_samples=100, n_features=5, random_state=42)
log_reg_model = LogisticRegression(solver='liblinear')
log_reg_model.fit(X_class, y_class)

classification_metrics = compute_performance_metrics(log_reg_model, (X_class, y_class), metrics=['AUC_ROC'])
print("Classification Model Metrics (AUC_ROC):", classification_metrics)

# Example 3: Evaluating a Classification Model with multiple metrics
multi_metrics = compute_performance_metrics(log_reg_model, (X_class, y_class), metrics=['AUC_ROC'])
print("Classification Model Metrics:", multi_metrics)



# Example 1: Perform grid search on logistic regression model
X, y = make_classification(n_samples=100, n_features=5, random_state=42)
log_reg_model = LogisticRegression(solver='liblinear')
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

best_model, best_params, search_results = perform_grid_search(log_reg_model, param_grid, X, y, scoring='accuracy')

print("Best Model:", best_model)
print("Best Parameters:", best_params)

# Example 2: Perform grid search with a different scoring metric (F1 score)
best_model_f1, best_params_f1, search_results_f1 = perform_grid_search(log_reg_model, param_grid, X, y, scoring='f1')
print("\nBest Model (F1):", best_model_f1)
print("Best Parameters (F1):", best_params_f1)

# Example 3: Analyzing grid search results
print("\nGrid Search Results Keys:", search_results.keys())
print("Mean Test Scores:", search_results['mean_test_score'])



# Example 1: Perform random search on a logistic regression model
X, y = make_classification(n_samples=100, n_features=5, random_state=42)
log_reg_model = LogisticRegression(solver='liblinear')
param_dist = {'C': [0.01, 0.1, 1, 10, 100]}

best_model, best_params, search_results = perform_random_search(log_reg_model, param_dist, X, y, n_iter=5, scoring='accuracy', random_state=42)

print("Best Model:", best_model)
print("Best Parameters:", best_params)

# Example 2: Perform random search with a different scoring metric (F1 score)
best_model_f1, best_params_f1, search_results_f1 = perform_random_search(log_reg_model, param_dist, X, y, n_iter=5, scoring='f1', random_state=42)
print("\nBest Model (F1):", best_model_f1)
print("Best Parameters (F1):", best_params_f1)

# Example 3: Analyzing random search results
print("\nRandom Search Results Keys:", search_results.keys())
print("Mean Test Scores:", search_results['mean_test_score'])



# Create a sample DataFrame
data = pd.DataFrame({
    'category': ['A', 'B', 'A', 'C'],
    'value1': [1, 2, 3, 4],
    'value2': [10.0, 15.0, 14.0, 16.0]
})

# Example 1: One-hot encode categorical columns
encoded_data = Utilities.one_hot_encode(data, ['category'])
print("One-hot Encoded Data:\n", encoded_data)

# Example 2: Scale numerical features
scaled_data = Utilities.scale_features(data.copy(), ['value1', 'value2'])
print("\nScaled Data:\n", scaled_data)

# Example 3: Plot the distribution of a column
Utilities.plot_distribution(data['value1'], title="Value1 Distribution")

# Example 4: Plot a correlation heatmap
Utilities.plot_correlation_heatmap(data, title="Data Correlation Heatmap")

# Example 5: Create a scatter plot between two columns
Utilities.plot_scatter(data['value1'], data['value2'], title="Value1 vs Value2 Scatter Plot")


# Example 1: Load and print module documentation
module_doc = load_documentation()
print("Module Documentation:")
print(module_doc)

# Example 2: Load and print module usage examples
module_examples = show_examples()
print("\nModule Usage Examples:")
print(module_examples)
