
import pandas as pd
import numpy as np
from scipy.stats import skew, ttest_ind, ttest_rel, chi2_contingency, ttest_1samp
from statsmodels.stats.anova import AnovaRM
from statsmodels.formula.api import ols
from statsmodels.api import Logit, add_constant
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from factor_analyzer import FactorAnalyzer
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Statistical Functions

def calculate_mean(dataframe):
    return dataframe.mean()

def calculate_median(dataframe):
    return dataframe.median()

def calculate_mode(dataframe):
    return dataframe.mode()

def calculate_variance(dataframe):
    variances = dataframe.var()
    return pd.DataFrame({'Column': variances.index, 'Variance': variances.values})

def calculate_std(dataframe):
    return dataframe.std(axis=0, skipna=True, numeric_only=True)

def calculate_skewness(dataframe):
    skewness_values = dataframe.apply(skew)
    return pd.DataFrame(skewness_values, columns=['Skewness'])

def calculate_kurtosis(dataframe):
    return dataframe.kurtosis()

# Data Handling Functions

def handle_missing_data(df):
    df.replace(np.nan, None, inplace=True)
    df.replace([np.inf, -np.inf], None, inplace=True)
    return df

def handle_infinite_data(dataframe, replace_value=np.nan):
    return dataframe.replace([np.inf, -np.inf], replace_value)

# Column Check and Drop Functions

def check_null_trivial(column):
    if column.isnull().all():
        return True
    
    unique_values = column.dropna().unique()
    
    if len(unique_values) == 1:
        return True
    
    return False

def drop_null_or_trivial_columns(dataframe):
    dataframe = dataframe.dropna(axis=1, how='all')
    dataframe = dataframe.loc[:, dataframe.nunique() > 1]
    
    return dataframe

# Column Operations Functions

def add_column_sum(df, column1, column2, new_column):
    if column1 not in df.columns or column2 not in df.columns:
        raise ValueError("Both columns should exist in the dataframe.")
    
    df[new_column] = df[column1] + df[column2]
    
    return df

def add_column_difference(df, column1, column2, new_column_name):
    df[new_column_name] = df[column1] - df[column2]
    return df

def calculate_product(df, column1, column2, new_column_name):
    df[new_column_name] = df[column1] * df[column2]
    return df

def calculate_quotient(df, column1, column2, new_column_name):
    if column1 not in df.columns or column2 not in df.columns:
        raise ValueError("One or both of the specified columns do not exist in the dataframe.")

    df[new_column_name] = df[column1] / df[column2]

    return df

# Range Calculation Functions

def calculate_range(dataframe):
    ranges = dataframe.max() - dataframe.min()
    
    result = pd.DataFrame(ranges, columns=['Range'])
    
    return result

def calculate_interquartile_range(dataframe):
    iqr_values = dataframe.quantile(0.75) - dataframe.quantile(0.25)
    
    iqr_df = pd.DataFrame(iqr_values, columns=['Interquartile Range'])
    
    return iqr_df

# Coefficient and Z-score Calculation Functions

def calculate_coefficient_of_variation(dataframe):
     result = {}
     
     for column in dataframe.columns:
         mean = dataframe[column].mean()
         std = dataframe[column].std()
         coefficient_of_variation = std / mean
         result[column] = coefficient_of_variation
    
     return result

def calculate_zscore(df, column_name):
     if column_name not in df.columns:
         raise ValueError(f"Column '{column_name}' does not exist in the dataframe.")
     
     values = df[column_name].values
     mean = np.mean(values)
     std = np.std(values)
     
     z_scores = (values - mean) / std
    
     return z_scores


# Hypothesis Testing Functions 

def independent_t_test(df,column1,column2): 
   if column1 not in  df.columns or column2 not  in  df.columns: 
      raise ValueError("Column(s)  not  found in the  dataframe") 
    
   data1=df[column1].replace([np.inf,-np.inf],np.nan).dropna().values 
   data2=df[column2].replace([np.inf,-inf],np.nan).dropna().values 

   t_statistic,p_value=ttest_ind(data1,data2) 
    
   return t_statistic,p_value 

 def paired_t_test(df,column1,column2): 
   if column1 not  in  df.columns or column2 not  in  df.columns: 
     raise ValueError("One or both of the specified columns do not exist in the DataFrame.") 
    
   data1=df[column1] 
   data2=df[column2] 
    
   t_statistic,p_value=ttest_rel(data1,data2) 
    
     return t_statistic,p_value


 def perform_anova(df,categorical_var ,columns): 
   anova_results={} 
  
   for col in columns: 
     groups=df.groupby(categorical_var)[col] 
     f_statistic,p_value=stats.f_oneway(*[group.values for name ,group in groups]) 
     anova_results[col]={'F-Statistic ':f_statistic,'P-Value ':p_value} 
  
   anova_result=pd.DataFrame.from_dict(anova_results ,orient='index') 
  
          return anova_result
  
 def chi_square_test(df,column1,column2): 
     contingency_table=pd.crosstab(df [column1],df [column2]) 
     chi2,p_value,_=chi_square_contingency(contingency_table) 
  
          return chi_square ,p_value
  
 def correlation_analysis (df,column_ ,column_ ,method='pearson'): 
              return  df [column_ ].corr (df [column_ ],method=method )
  

 # Plotting Functions 

 def plot_histogram(df ): 
     for columndf .columns : 
         if pd.api.types.is_numeric_dtype (df [column ]): 
             plt.figure() 
             plt.hist(df [column ]) 
             plt.title(f'Histogram of {column}') 
             plt.xlabel(column ) plt.ylabel('Frequency ') plt.show() 
  
 def plot_boxplot (df ):  
                                                
       data.boxplot(grid=False ) plt.show() 



 def scatter_plot (df,x_col,y_col ):  
            plt.scatter (df[x_col ],df[y_col ]) plt.xlabel(x_col ) plt.ylabel(y_col )
            plt.title("Scatter Plot") plt.show()

  
 def plot_line_graph (data):  
            for columndf .columns :  
                plt.plot(df [column ]) plt.xlabel ("Index ") plt.ylabel (column )                
                title(f"Line Graph-{col}" )                                                
                plt.show ()               
                
              
              
 def plot_bar_chart (data):  
            for col_data .columns :  
                plt.figure () data[col ].value_counts().plot(kind='bar')                                                    
                title(f'Bar Chart for {col}') xlabel(col ) ylabel ('Count ')                                                 show()                                                                                              



# Sample Size and Power Calculations             


 def calculate_sample_size (margin_of_error ,confidence_level ):                            
               
               z_score=get_z_score(confidence_level )              
               sample_size=(z_score** *0.5*0.5)/(margin_of_error** )
               sample_size=math.ceil(sample_size )              
               sample_size                                                                                      


 def get_z_score(confidence_level ):                                                    
                  
             if confidence_level==0.90 :             
                                                                                                                                                                                                                                                                                                                                                                                                                                                                else :                                                                                                                                                                                                                                                                                                                                              
    
    
                    raise valueerror("Invalid confidence level provided.")   
               
                                                                                                                                                                                                                                                                                                                                          

                                                                                                                                                     


                                                                                                                                                     


 from scipy.stats import power 


 def calculate_power(effect_size,sample_size ,significance_level ):                
                     power.solve_power(effect_size,nobs=sample_size ,alpha=significance_level )

                 
 


                                                                                                                                                     


                                                                                                                                                     # One Sample T-Test 


 def one_sample_ttest(data,colum,population_mean):                                                            
                    sample=data[colum].dropna ()                   
                    if sample.empty:                   
                          {"t_statistic ":np.nan ,"p_value ":np.nan ,"reject_null_hypothesis ":None }                   
                    t_statistic,p_value=ttest_ samp(sample,population_mean )                   
                    reject_null_hypothesis=True if p value<0.05 else False                   
                    {"t_statistic ":t_statistic ,"p-value ":p-value ,"reject null hypothesis":reject null hypothesis }

 
 
 
 
# One-Way ANOVA
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
    
    
    
    
    
    
    
    
    
                    
    
    
                    
    
    
                    
    
                    
    
    
                    
    
                    
    
    
                    
    
    
                    
    
    
                    
    
    
                    
    
    
                    
    
    
                                                                                           
  
  

 from statsmodels.stats.anova import AnovaRM 

 def one_way_anova(data,categorical_var ):                      
           grouped_data=data .groupby(categorical_var )                     
           continuous_columns=data.select_dtypes(include='number').columns.tolist ()                      
           results={}                      
           for col_grouped_data :                      
           groups=[group[col].values for _,group grouped_data ]                      
           anova_result=AnovaRM(groups,np.ones(len(groups[0])),group_order-grouped_data.groups.keys()).fit ()                      
              results[col]=anova_result                        
              results                        
            
            
# Two-Way ANOVA
            
            
            

           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
        
        
        

        
        

        

        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                            
                                      
                                          
                                          
                                          
      
      
      
      
      
      
      
      
      
      

      

      











       
       
       
       

       
       
       
       
       
       

       
       

                                               
        
        
        




        
        

        
        
        
        
        
        
        
        
        
       
        
        
        

        
        
from sqlalchemy.orm import aliased                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
        
         
         
         
         
         
         
         
         
         
                                                                                                              
 
 


 



                                                                                                                    
                                                                                                                  













       
        
        
        
        
        
        
       
        
        
        
        
        


     
     
     
     
    
    


import pandas as pd                                                                                                                                                
import statsmodels.api as sm                                                                                                                                    import pandas as pd                                                                                                                                    from sklearn.linear_model import LinearRegression                                                                                                                               from statsmodels.formula.api ols                                                                                                                               from sklearn.decomposition import PCA                                                                                                                                from factor_analyzer import FactorAnalyzer                                                                                                                                from scipy.stats.contingency chisquare_contingency from scipy.stats power solve_power from scipy.stats.ttest_samp from scipy.stats.ttest_rel from scipy.stats.ttest_ind 



one_way_anova results two_way_anova perform_factor_analysis_bootstrap_resampling perform_permutation_test perform_pca perform_logistic_regression linear_regression_result=pd.DataFrame.from_dict(anovafuncs(),orient='index') print(result.head()) two_way_anovafuncs(),result.head()


