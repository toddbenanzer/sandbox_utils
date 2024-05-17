
import pandas as pd
import requests
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import schedule
import time
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from textblob import TextBlob
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def read_csv_file(file_path):
    """
    Read a CSV file and return the contents as a pandas DataFrame.

    Parameters:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The contents of the CSV file.
    """
    df = pd.read_csv(file_path)
    return df


def fetch_data_from_api(url):
    response = requests.get(url)
    data = response.json()
    return data


def connect_to_database(database_path):
    """
    Connects to a database and returns a connection object.
    
    Parameters:
        database_path (str): The path to the database file.
    
    Returns:
        connection (sqlite3.Connection): The connection object.
    """
    connection = sqlite3.connect(database_path)
    return connection


def clean_and_preprocess_data(data):
    data = data.dropna()
    data = pd.get_dummies(data)
    
    numeric_cols = data.select_dtypes(include=['float', 'int']).columns
    data[numeric_cols] = (data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std()

    return data


def perform_statistical_analysis(data):
    """
    Perform statistical analysis on the data.

    Args:
        data (pd.DataFrame): The input data for analysis.

    Returns:
        dict: A dictionary containing the results of the statistical analysis.
    """
    statistics = data.describe()
    
    correlation_matrix = data.corr()
    
    group1 = data[data['group'] == 1]['value']
    group2 = data[data['group'] == 2]['value']
    
    t_statistic, p_value = stats.ttest_ind(group1, group2)
    
    results = {
        'descriptive_statistics': statistics,
        'correlation_matrix': correlation_matrix,
        't_statistic': t_statistic,
        'p_value': p_value,
    }

    return results


def calculate_descriptive_statistics(data):
    """
    Calculate descriptive statistics for a given dataset.
    
    Parameters:
        - data: list or numpy array of numeric values
    
    Returns:
        - dictionary containing the calculated descriptive statistics
    """
    
    data = np.array(data)
    
    statistics = {
        'mean': np.mean(data),
        'median': np.median(data),
        'minimum': np.min(data),
        'maximum': np.max(data),
        'standard_deviation': np.std(data),
      }
      
      return statistics



def calculate_correlation(dataframe, variable1, variable2):
      """
      Calculate the correlation between two variables in a dataframe.

      Parameters:
          dataframe (pd.DataFrame): The dataframe containing the variables.
          variable1 (str): The name of the first variable.
          variable2 (str): The name of the second variable.

      Returns:
          float: The correlation coefficient between the two variables.
     """
     correlation = dataframe[variable1].corr(dataframe[variable2])
     return correlation


def perform_regression_analysis(X, y):
     """
     Perform regression analysis using the scikit-learn library.

     Parameters:
         X (array-like): The independent variable(s) for regression analysis.
         y (array-like): The dependent variable for regression analysis.

     Returns:
         model (LinearRegression): The trained linear regression model.
     """
     model = LinearRegression()
     model.fit(X, y)

     return model


def create_visualization(x, y, chart_type):
      """
      Create a visualization using matplotlib.

      Parameters:
          x (list): The x-axis values.
          y (list): The y-axis values.
          chart_type (str): The type of chart to create ('scatter', 'bar', etc.).

      Returns:
          None
      """
      if chart_type == 'scatter':
            plt.scatter(x, y)
       elif chart_type == 'bar':
             plt.bar(x, y)

       plt.show()


def generate_summary_report(data):
       summary_stats = data.describe()
       report = f"Summary Report:\n\n{summary_stats}"
       return report



def generate_report():
       print("Report generated!")


def schedule_report_generation():
       schedule.every().day.at("08:00").do(generate_report)

       while True:
            schedule.run_pending()
            time.sleep(1)



def export_data(data: pd.DataFrame, file_path: str, file_format: str):
       """
       Export data to various file formats.

       Parameters:
           - data: The pandas DataFrame containing the data to be exported.
           - file_path: The path where the exported file should be saved.
           - file_format: The desired file format (e.g., 'csv', 'xlsx', etc.).
       """
       if file_format == 'csv':
            data.to_csv(file_path, index=False)
       elif file_format in ['excel', 'xlsx']:
             data.to_excel(file_path, index=False)
       else:
              raise ValueError("Unsupported file format.")


def filter_and_sort_data(data, criteria):
        filtered_data = [item for item in data if item['criteria'] == criteria]
        
         sorted_data = sorted(filtered_data, key=lambda item: item['sort_key'])
         
         return sorted_data



def merge_datasets(datasets, on=None, how='inner'):
         """
         Function to merge or join multiple datasets.

         Parameters:
             - datasets: A list of pandas DataFrames to be merged.
             - on: Column(s) name(s) to merge on. If not specified, all common columns will be used.
             - how: Type of merge to be performed. Possible values are 'inner' (default), 'left', 
               right', and 'outer'.

         Returns:
             - merged_dataset: A pandas DataFrame containing the merged result.
         """

         merged_dataset = datasets[0].copy()
         
         for dataset in datasets[1:]:
               merged_dataset.merge(dataset, on=on, how=how)

         return merged_dataset



def handle_missing_values(data, method='mean'):
          """
          Function to handle missing values in the data.

          Parameters:
              -data: pandas DataFrame, the input value 
              method:str optional(default mean),the method used to replace missing values 
                     Available options are mean and median

                     Returns:

                     -data:pandas DataFrame,the with missing values replaced

                      """

                       if method not in ['mean','median']:
                            raise ValueError("Invalid method.Available options are mean and median")
                       if method=='mean':
                             return  fillna(df.mean())
                         else method=='median':
                             return fillna(df.median())



                      def perform_time_series_analysis(df,date_column,value_column):
                      '''
                      Function performs time series analysis on given dataframe 

                      parameters:

                      df:pandas dataframe containing time series date

                      date column:str column containg dates

                      value_column:str column containg required values

                      returns:

                       new df containg results

                       '''
                       #convert date column 
                        df[date_column]=pd.to_datetime(df[date_column])

                        #set date column as index 
                         df.set_index(date_columns,inplace=True)

                         #return new df 
                         return df 


                         def apply_ml_algorithms(df,target):

                         '''
                          function applies machine learning models 

                          parameters:

                          df:pandas dataframe dataset

                           target :target labels 

                           returns:

                           accuracy score 

                           '''
                          #splitting dataset into training and testing set 

                           X_train,X_test,y_train,y_test=train_test_split(df,target,test_size=0.2,rndom_state=42)

                          #creating instance 

                           model=LogisticRegression()

                          #train model 

                           model.fit(X_train,y_train)

                           #predictions:

                            pred=model.predict(X_test)

                            #calculate accuracy :

                            accuaracy=accuracy_score(y_test,predictions)

                            #return accuracy score 
                             return accuaracy 




                             def perform_sentiment_analysis(text):

                             '''
                              function performs sentiment analysis on text

                              parameters :

                              text:str string required for sentiment analysis 


                              returns:

                              polarity(float)-ranges from (-1 t0 1)(negative positive neutral )

                              subjectivity(float)-ranges from( 0-1)(objective/subjective )

                               '''

                               blob=TextBlob(text)


                                polarity=blob.sentiment.polarity

                                subjectivity=blob.sentiment.subjectivity 

                                #return value :
                                return polarity ,subjectivity 



                                def cluster_data(df,kclusters):

                                '''
                                 function clusters given dataset into k clusters 

                                 parameters:

                                  df:dataframe required dataset  

                                   kclusters:int number of desired clusters

                                   returns:

                                   labels :cluster labels 

                                    '''

                                    kmn=KMeans(n_clusters=kclusters)


                                    labels=kmn.fit_predict(df)


                                     #return cluster labels 
                                     retrun labels 


                                     def detect_outliers(df):

                                      ''''
                                       function detects outliers from given dataset :

                                       parameters:


                                       df:dataframe required dataset  

                                        returns:

                                        indices having outliers 


                                        '''

                                         mean=np.mean(df)


                                         std=np.std(df)


                                          z_scores=(df-mean)/std
    
                                           outlier_index=np.where(np.abs(z_scores)>3)[0]

                                           #return outlier indices :
                                           retrun outlier_indices 




                                           def perform_pca(dataset,n_components):

                                           ''''
                                            function performs pca dimensionality reduction 

                                            parameters:


                                            dataset:dataframe required dataset  


                                            n_components:int no of components 


                                             returns :

                                             transformed matrix with reduced dimension 


                                             '''

                                              pca=PCA(n_components=n_components)


                                              transformed=pca.fit_transform(dataset)


                                              #return value 
                                               retrun transformed 



                                               def feature_selection(X,y,n_features):

                                               ''''
                                                function selects features using RFE:


                                                parameters:


                                                X:dataframe features matrix  


                                                y:dataframe target array


                                                 n_features:int no of required features 


                                                 returns :

                                                 selected_features:list names of selected features  


                                                 '''

                                                  lr=LinearRegression()

                                                   rfe=RFE(lr,n_features_to_select=n_features)


                                                    rfe.fit(X,y)

                                                    selected=[feature for feature ,support in zip(X.columns,rfe.support_)if support]


                                                     #return vales 

                                                     retrun selected 



                                                     def forecast_arima(tsdata,tup):

                                                     ''''
                                                      function forecasts time series using ARIMA models:


                                                      parameters:


                                                      tsdata:dataframe containing time series 


                                                      tup:int tuple order(p,d,q)of arima model


                                                       returns :

                                                       forecasted values 


                                                        '''

                                                        arima_model=ARIMA(tsdata ,order=tup)


                                                         fit_model=model.fit()

                                                         forecasted=pd.Series(fit_model.predict(start=len(tsdata),end=len(tsdata)+n-1),index=pd.date_range(start=data.index[-1],periods=n+1)[1:])


                                                          retrun forecasted




                                                          def forecast_lstm(tsdata,lenback):

                                                          ''''
                                                           function forecasts time series using LSTM models:


                                                           parameters:


                                                           tsdata :numpy array/pandas dataframe containing time series 


                                                           lenback:int number of previous steps considered forecasting future steps


                                                            returns :

                                                            forecasted value 


                                                             '''
                                                              X=[]

                                                               y=[]



                                                                for i in range(len(tsdata)-lenback):

                                                                 X.append(tsdata[i:i+lenback])

                                                                  y.append(tsdata[i+lenback])


                                                                   X=np.array(X)

                                                                    Y=np.array(y)



                                                                      lst_model=LSTM(input_shape=(lenback  , 64))
                                                                     lst_model.add(Dense(1))
                                                                      lst_model.compile(loss='mean_squared_error' ,optimizer='adam')


                                                                       lst_model.fit(X,y_epochs=10,batch_size=32)



                                                                        forecasted=lstm.predict(tsdta[-lookback:].reshape(lookback , 1))


                                                                         retrun forecasted.flatten()






