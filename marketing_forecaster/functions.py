
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import boxcox
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, VotingRegressor
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import matplotlib.pyplot as plt

# Preprocess and Clean Marketing Data
def preprocess_marketing_data(data):
    data = data.drop(['id', 'date'], axis=1)
    data = data.fillna(data.mean())
    
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    numeric_cols = data.select_dtypes(include=['float', 'int']).columns.tolist()
    for col in numeric_cols:
        mean = data[col].mean()
        std = data[col].std()
        data[col] = (data[col] - mean) / std
    
    return data

# Perform Exploratory Data Analysis (EDA)
def perform_eda(data):
    print(data.describe())
    
    numerical_cols = data.select_dtypes(include=['float', 'int']).columns
    for column in numerical_cols:
        sns.histplot(data[column])
        plt.title(f'Distribution of {column}')
        plt.show()
    
    categorical_cols = data.select_dtypes(include='object').columns
    for column in categorical_cols:
        sns.countplot(x=data[column])
        plt.title(f'Count of {column}')
        plt.xticks(rotation=90)
        plt.show()

# Generate Descriptive Statistics for Marketing Datasets
def generate_descriptive_statistics(data):
    stats = data.describe()
    stats['median'] = data.median()
    stats['mode'] = data.mode().iloc[0]
    
    correlation = data.corr()
    
    result = pd.concat([stats, correlation], keys=['Statistics', 'Correlation'])
    
    return result

# Split Marketing Data into Training and Testing Sets
def split_data(data, test_size=0.2, random_state=42):
    X = data.drop('target', axis=1)
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

# Train a Regression Model for Forecasting Marketing Trends
def train_regression_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    
    return model

# Train a Classification Model for Predicting Customer Behavior
def train_classification_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    
    accuracy = clf.score(X_test, y_test)
    
    return clf, accuracy

# Train a Time Series Model for Forecasting KPIs Using ARIMA
def train_time_series_model(data, target_column, order):
    model_data = data[[target_column]].copy()
    model_data.columns = ['KPI']
    
    model = ARIMA(model_data, order=order)
    trained_model = model.fit()
    
    return trained_model

# Evaluate the Performance of a Model Using Evaluation Metrics
def evaluate_model(y_true, y_pred):
    evaluation_metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
    
    return evaluation_metrics

# Fine-tune the Hyperparameters of a Model Using Grid Search Cross-validation
def fine_tune_model(model, X, y, param_grid):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(X,y)

   return grid_search.best_estimator_

# Visualize the Predictions of a Model Against Actual Values   
def visualize_predictions(y_true,y_pred): 
     plt.plot(y_true,label='Actual') 
     plt.plot(y_pred,label='Predicted') 
     plt.xlabel('Time') 
     plt.ylabel('Value') 
     plt.legend() 
     plt.show()

# Interpret Feature Importance in Models for Marketing Trends Customer Behavior and KPIs Forecasting   
def calculate_feature_importance(model,X_train): 
      if hasattr(model,'feature_importances_'): 
          feature_importances=model.feature_importances_ elif hasattr(model,'coef_'): feature_importances=np.abs(model.coef_) else: raise ValueError("Model does not have a 'feature_importances_' or 'coef_' attribute.")
          feature_importance_scores={} 
          for i ,feature in enumerate(X_train.columns): feature_importance_scores[feature]=feature_importances[i]

          return feature_importance_scores 

# Save Trained Models for Future Use or Deployment  
 def save_model(model ,filename): with open(filename,'wb') as file: pickle.dump(model,file) 

 # Load Pre-trained Models for Prediction Purposes  
 def load_model(model_path): with open(model_path,'rb') as file: loaded_model=pickle.load(file)

      return loaded_model 

 # Handle Missing Values in Marketing Datasets  
 def handle_missing_values(df ,method='mean' ,axis=0): if method in ['mean','median','mode']: if axis==0: if method=='mean': df=df.fillna(df.mean()) elif method=='median': df=df.fillna(df.median()) elif method=='mode': df=df.fillna(df.mode().iloc[0]) elif axis==1: if method=='mean': df=df.T.fillna(df.mean(axis=1)).T elif method=='median': df=df.T.fillna(df.median(axis=1)).T elif method=='mode':
             mode_values=df.mode(axis=1).iloc[:, 0]
             for col in df.columns:
                 df[col].fillna(mode_values,inplace=True) elif method=='drop_row': df=df.dropna(axis=0) elif method=='drop_column': df=df.dropna(axis=1)

       return df 

 # Handle Categorical Variables in Marketing Datasets Through Encoding or Feature Engineering Techniques   
 def handle_categorical_variables(data,categorical_cols): encoded_data=data.copy() for col in categorical_cols: encoded_cols=pd.get_dummies(data[col],prefix=col ,drop_first=True) encoded_data=pd.concat([encoded_data ,encoded_cols],axis=1) encoded_data.drop(col ,axis=1,inplace=True)

       return encoded_data 

 # Perform Dimensionality Reduction on Marketing Datasets if Necessary  
 def perform_dimensionality_reduction(data,n_components): pca=PCA(n_components=n_components) reduced_data=pca.fit_transform(data)

       return reduced_data 

 # Identify Outliers in Marketing Datasets and Handle Them Accordingly  
 def handle_outliers(data): z_scores=(data -np.mean(data)) /np.std(data) threshold=3 outlier_indices=np.where(np.abs(z_scores)>threshold) median_value=np.median(data) data[outlier_indices]=median_value

       return data 

 # Transform Skewed Variables in Marketing Datasets to Achieve a More Normal Distribution  
 def transform_skewed_variables(data ,variables): transformed_data=data.copy() for var in variables: skewness=data[var].skew() if abs(skewness)>0.5: transformed_data[var],_=boxcox(data[var]+ 1)

       return transformed_data 

 # Handle Imbalanced Classes in Classification Models for Customer Behavior Prediction   
 def handle_imbalanced_classes(X,y): X_train,X_test,y_train,y_test=train_test_split(X,y,test_size= 0.2 ,random_state=42) smote=SMOTE(random_state=42) X_train_resampled,y_train_resampled_=smote.fit_resample(X_train,y_train)

       return X_train_resampled,y_train_resampled_,X_test,y_test 

 # Perform Cross-validation on Models for Reliable Performance Estimation   
 def perform_cross_validation(model,X,y=cv): scores=cross_val_score(model,X,y=cv=cv)

       return scores 

 # Ensemble Multiple Models for Improved Accuracy or KPIs Forecasting Accuracy  
 def ensemble_models(models,X): ensemble_model_=VotingRegressor(estimators=models) ensemble_model_.fit(X,y) predictions_=ensemble_model_.predict(X)

       return predictions_

 # Detect and Address Multicollinearity Issues in Regression Models   
 def detect_multicollinearity(X): vif=pd.DataFrame() vif["Variable"]=X.columns vif["VIF"]=[variance_inflation_factor(X.values,i )for iin range (X.shape[1])]

       return vif

 
   # Implement Feature Selection Techniques in Models for Improved Interpretability and Performance   
   def select_features(X_train,y_train,n_features): estimator_=LinearRegression() selector_=RFE(estimator_, n_features_to_select=n_features) selector_.fit(X_train_,y_train ) selected_features=[ featurefor feature,_maskin zip (X_train.columns_,selector_.support_)if mask]

      return selected_features
   
   # Utility Reading CSV Files Excel Files or Data from SQL Databases   
   def read_data( source,file_path ): if source=='csv ':data=pd.read_csv (file_path )return _:data; elifsource =='excel ':data=pd.read_excel (file_path )return_:data; elif source =='sql '; query="SELECT * FROM table_name";_:data=pd.read_sql_query(query,<connection>)return_:data else: raise ValueError("Invalid source specified.")     

   # Utility Handling Time Series Data Including Lagging Variables and Creating Rolling Windows   

   def create_lagged_variables( variables,lags ):result=data.copy ()for variablein variables:for lagin lags :col_name=f"{variable}_lag{lag} ";result[col_name]=result[variable].shift(lag )

      return result;   

   def create_rolling_windows( window_size ):result=data.copy ()foriin range(window_size ):col_name=f"window{i} ";result[col_name]=result.iloc[:,0 ].shift(i ).rolling(window_size ).mean ()

         return result;   

   # Utility Performing Feature Scaling or Normalization on Marketing Datasets Before Model Training   
   def feature_scaling( scaling ):scaler_=MinMaxScaler ();scaled_data=scaler_.fit_transform(scaling )

         return scaled_data;

