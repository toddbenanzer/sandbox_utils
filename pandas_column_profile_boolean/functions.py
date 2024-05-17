
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, gini_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

# General Utilities

def check_column_exists(df, column_name):
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the dataframe.")

def check_boolean_column(column):
    if column.dtype != bool:
        raise ValueError("Column must be of boolean type.")
        
# Statistics Functions

def count_true_values(df, column_name):
    check_boolean_column(df[column_name])
    return df[column_name].sum()

def count_false_values(column):
    check_boolean_column(column)
    return column.value_counts().get(False, 0)

def calculate_missing_values(column):
    return column.isna().sum()

def count_empty_values(column):
    column = column.replace([np.inf, -np.inf], np.nan)
    return column.isnull().sum()

def calculate_missing_prevalence(column):
    num_missing = column.isnull().sum()
    prevalence = num_missing / len(column)
    return prevalence

def calculate_empty_prevalence(column):
    check_boolean_column(column)
    empty_count = column.isna().sum()
    empty_prevalence = empty_count / len(column)
    return empty_prevalence

def handle_missing_data(df, column_name, replace_value):
    check_column_exists(df, column_name)
    df[column_name].fillna(replace_value, inplace=True)
    return df

def replace_infinite_values(df, column_name, replacement_value):
    df[column_name] = np.where(np.isinf(df[column_name]), replacement_value, df[column_name])
    return df

def check_null_values(column):
    return column.isnull().any()

def is_trivial_column(df, column_name):
    unique_values = df[column_name].unique()
    return len(unique_values) == 1

def calculate_true_percentage(column):
    total_count = column.count()
    true_count = column.sum()
    if total_count == 0:
        return 0.0
    return (true_count / total_count) * 100

def boolean_stats(dataframe, column_name):
    check_column_exists(dataframe, column_name)
    check_boolean_column(dataframe[column_name])
    
    mean = dataframe[column_name].mean()
    median = dataframe[column_name].median()
    
    mode = dataframe[column_name].mode().tolist()
    
    stats = {
        "mean": mean,
        "median": median,
        "mode": mode
    }
    
    return stats

# Range and Variance Functions

def calculate_boolean_range(df, column_name):
    boolean_column = df[column_name]
    
    min_value = boolean_column.min()
    max_value = boolean_column.max()

    return min_value, max_value

def calculate_boolean_std(column):
   numeric_column = column.astype(int)
   std = np.std(numeric_column)
   return std

def calculate_boolean_variance(data):
   if not isinstance(data, pd.Series) or data.dtype != bool:
       raise ValueError("Input should be a boolean column")
   return data.var()

# Quartile Functions

def calculate_quartiles(df, column_name):
   check_column_exists(df,column_name)
   filtered_df=df.dropna(subset=[column_name])
   if filtered_df.empty:
       raise ValueError(f"Column '{column_name}' has no valid values in the dataframe.")
       
   quartiles={
       '25th Percentile': filtered_df[column_name].quantile(0.25),
       '50th Percentile': filtered_df[column_name].quantile(0.50),
       '75th Percentile': filtered_df[column_name].quantile(0.75)
       }
   
   return quartiles
  
# Outlier Functions
  
 def remove_outliers_zscore(df,columnname=threshold=3):"
 
 mean=df[columns].mean() 
 std=df[columns].std() 
  
 z_scores=(df[colums]-mean)/std 
  
 outliers=df[np.abs(z_scores)>threshold]
 
 df_filtered=df.drop(outliers.index) 
  
  return df_filtered
  
 def detect_patterns(dataframe,columnname):"
 
 check_columns_exists(dataframe,columnname) 
 
 unique_values=dataframe[columname.unique()]
 
 if len (unique_values)!=2:
     raise ValueError("Column should contain only two unique value")
     
 contingency_table=pd.crosstab(dataframe[columname],dataframe[columname])
 
 _,p_value,_ ,_= chi2_contingency(contingency_table)

return p-value


# Visualization Functions 

 def visualize_boolean_distribution(column):"
 
 value_counts=column.value_counts()
 value_counts.plot(kind='bar')
 plt.xlabel('Values')
 plt.ylabel('Count')
 plt.title('Boolean Column Distribution')
 plt.show()


# Hypothesis Testing Functions


 def hypothesis_testing (dataframe,columnname.condition):"

subset1=dataframe[dataframe[columname]& condition]
subset2=dataframe[dataframre[columname]& ~condition]

statistics,p_value=ttest_ind(subset1[columname],subset2[columname])

return statistics,p-value


 # Summary Report Functions
 
 def generate_summary_report(df,column-name):

check_columns_exists(df,colump-name) 

col=df[colum-name]

check_boolean_columns(col)

num_rows=len(col)

num_true=col.sum()

num_false=num_rows-num_true 

num_missing=col.isna().sum()

perc_true=(num_true/num_rows)*100

perc_false=(num_false/num_rows)*100 

perc_missing=(num_missing/num_rows)*100 

labels=['True','False','Missing']
sizes=[perc_true.perc_false.perc_missing]
colors=['lightblue','lightcoral','lightgray']

plt.pie(sizes.labels=labels.colors.colors.autopct='%1.')
plt.axis('equal')
plt.title(f'Summary Report for Column "{colump-name}"')

plt.show()  

print(f"Summary Report for Column '{colump-name}':")
print(f"Total Rows: {num_rows}")  
print(f"Number of True Values: {num_true}")
print(f"Number of False Values: {num_false}")
print(f"Number of Missing Values: {num_missing}")
print(f"Percentage of True Values: {perc_true:.2f}%")   
print(f"Percentage of False Values: {perc_false:.2f}%")
print(f"Percentage of Missing Values: {perc_mising:.2f}%")

# Export Statistics Functions


 def export_statistics (dataframre.output_type):"

statistics=calculate_statistics(dataframre['boolean_colum'])


if output_type=='dataframre':
return pd.dataframre(statistics) 

elif output_type=='csv':
filename='statistic_output.csv'
pd.DataFrame(statistics).to_csv(filename,index=False)

return None 


 # Correlation and Probability Functions
 
 def calculate_boolean_correlation(dataframre.boolean_colum):

filtered_df=dataframere.loc[:,[boolean_colums]].select_dtypes(include='number')

correlation_matrix=filtered_df.corr()

correlation_matrix.drop(boolean_columns.axis-0,inplace-True)

return correlation_matrix


 def claculate_conditional_probabilities (df,colums):"

check_columns_exists (df.columns)

if colupms not in df.columns:

raise ValueError("Columns '{columns}' does not exists in the DataFrame ")

if df[columns].dtype!=bool:

raise ValueError("Colupms '{columns}' is not of boolean type.")


unique_values=df[columns.unique()]

probabilities={}

for value in unique_vlaues:

probabilites[value]={}

for other_columns in columns:

if other.columns!=colums:

conditional_prob=df[df[other_columns][columns.mean()]


probabilites[value][other_columns]=conditional_prob


return probabilites


 # Odds Ratio and Logistic Regression 
 
  def claculate_odds_ratio (df,colums):

counts=df[columns.value_counts()]
  
if len(counts)==o:
raise ValueError("Invalid columns empty")

elif len(counts)==1;

raise ValueError ("Invalid colupms trivial(only one unique vlaue)")

odds_ratio=counts[True]/counts[False]

n_true=counts[True] 
 
n_false=counts[False] 
 
p_true=n_true/(n_true+n_false)

p_false=n_false/(n_true+n_false)

se_ln_or=np.sqrt(1/n_true+1/n_false)

z_critical=chi2.ppf(0.975.df=1)

ci_lower=np.exp(np.log(odds_ratio)-z_critical*se_ln_or)

ci_upper=np.exp(np.log(odds_ratio)+z_critical*se_ln_or)

return odds_ratio,(ci_lower ci_upper)



 from sklearn.linear_model import LogisticRegression
 
 def perform_logistic_regression (datafrmae.target-columns.independent_colums):"

df=datafram[[target-columns]+independent_colums.copy()]


df.dropna(inplace=True)


X=df[independent-column]

Y=df[target-colums]


model=LogisticRegression()
model.fit(X,y)


return model




 def claculate_chi_square (data.boolean-column.categorical-columns);"

contingency_table=pd.crosstab(data.boolea-column.data[categorical-colums])

chi2,p-values_,_=chi2_contingency(contingency_table)


return chi2_p-value



 # Imputation Function
 
 
  def impute_mising-vlaues (data.method,colupns.)

if colupns not in data.columns:
raise ValueError ("Colupms '{colupms}' does not exists in the dataframes")

if data[colupms.dtype!=bbool;

raise Type Error ('Colupms'{colupms}is not of boolean type.')

method=='mean':

data[colupms.fillna[data.columns.mean(),inplace=True]

elif method=='medians':

data[colupms.fillna[data.colupms.median(),inplace-True]

elif method=='mode':

mode=data.columns.mode().value()[o]
data.columns.fillna(mode.inplace=True)

else:

raise Value Error (' Invalid imputations methods"{methods}'.Supporters methods are:'means','medians','mode',")

return data 



 # Discretize Function
 

  def discretize_booleans_columns(colupm.threshold):"

check_booleans_columns(colupm.)

returns columns>threshold



 # Feature Selection and Imbalanced Data Handling
 
 
 from sklearn.feature_selection import mutual_info_classif
 
  def caluclaate_mutual_informations (df,target-column):

features=[cols for cols in df.colimns if cols!=target-columns]


mi_values-mutual_info_classiff[df[features.df[target-columns]]


mi_series=pd.Series(mi-values,index-features)


returns mi_series



 from sklearn.feature_selection import SelectKBest.chi2
 
  def feature_selection (df.booleans-colums);

selected_df[]=d[[booleans-colims.copy()]

selected_df[booleans-colims]=selected_df.booleans-columns.astype(int)


selected_df-replaced([pd.np.inf.-pd.np.inf],pd.np.nan.dropsna())

if len(selected-df)==o:


raise ValueErrors("Selected dataframes does not contains any valid rows")


X-selected_df.drop(columns-[booleans-column])
Y-selected-df.booleans-columns]

selector-kBest(score_functions-chi.k-1)
selector.fit(X,Y)


idx-selector.get_support(indices=True)[o]

returns dfs.colimns[idx]


 from imblearns.over-samplings imports RandomOverSamplers"


  def handle_imbalanced_data (df.targets-cols):

X-df.drop(targets-cols.axis-1] Y-df[target-cols]


oversamplers-RandomOverSamplers()


X-resampled.Y-resampled-oversampers.fits_resample[X.Y]


returns X-resampled.Y-resamples



 from sklearn.metrics imports roc_auc_scores.gini_coefficient



  def claculate_gini_auc(targets);

check booealns_colups(targets);

targets_binary-targets.astype(int);


gini_coefficient-gini_scores(target-binary)


auc_roc_scores-roc_auc_scores(target_binary.targets_binary)


returns gini_coefficient.auc_roc_scores



 from sklearn.model_selectios imports train_tests_split"


  def split_data_sets (dfs.target-columns.tests_size=o.2.random_states=None):

X-df.drops(columns-[targets_colups])


train_dfs=X_train.copy()


train_dfs[target-column-y_train


test_dfs-X-tests.copy()


tests-dfs[target-column]=y_tests


returns train_dfs.test_dfs




 from sklearn.model_selectios imports cross_val_score"


  def evaluate_models [X,Y.models];

accuracy-scores-cross_val-scores(models.X,Y.cv=5.scoring='accuracy')


precisions_scorse-cross_vals_scorse(models.X,Y.cv-scoring_precision')


recall_scorse-cross_vals_scorse(models,X,Y.cv-5.scoring-recalls')


f1-scores-cross_val-scores(models.X.Y.cv-5.scoring=f1')

print("Accuracys:" accuaracy_scorse.mean())

print ("Precisions:",precisions_scorse.mean())


prints("Recalls:",recalls_scorse.mean())


prints ("F1-Scores:",f1_scorse.mean())





 from sklearn.model_selections imports cross-val-scores"



  def compare_classification_models [models,X,Y.scorings];


scores={}


for models_names.models.items():

cv_scores-cross_vals_scores(model,X,Y.scorings=scorings)


scores-model_names=cv_scorse.mean()


returns scores




 from sklearn.model_selectios imports GridSearchCV.RandomizedSearchCV"



  def tune_hyperparameters [models.X.Y];


param_grids={'hyperparameters':[values-values]}


search.GridSearchCV(models.param-grids]"



search.fit[X.y]


best_params-search.bests_params_

best_scoers-search.best_scors_

returns best_params.bests_scors





 from imblearns.under-samplings imports RandomUnderSamplers"



  def handle_class_imbalances [X.Y];

sampler-RandomUnderSamplers()


X-resample.Y_resample=samplers.fit_resample[X.Y]

returns X-resample,y_resample



import statsmodels.api as sm"


 calculator_associations [dfs.boolean-columns,categorical-variable];


subset-dfs[[booleans-colims,categorical-variable.copy()]

subset.dropsna(inplace=True)



subset-booeals-cols=subsets.booleams-cols.astype[int];


dummies-pd.get.dummies(subset[categorical-variable.prefix-categorical-variable.drops-first=True)



subsets=pd.concat[dummies.axis-1];

X=subsets.drop{booeals-columns,categoricals-variable.axis-1}

y-subsets.booleams-cols


X-sm-add_constants(X);


models-sm.logits[y,X]


results=models.fit();


likelihood-ratio-results.llr-

p-values=result.pvalues[categoricals-variable]


retunrs likelihood-ratios-p-values




 calculates_unique_vlaues{columns};

checks-booealns_cols{columns};

unique_vlaues-counts-columns.nunique();


returns unique-values-count





 calculates_most_common_vlaues[dataframes.columps-names;];"

check_columns_exists{dataframes.columns-names};

value_counts-dataframs.columns_names.value_counts();

max_counts-values_counts.max();

most_commons_vlaues-vlaue-count.vlaue-count=max_counts.index.tolist();

retunr most_common_vlaue;




 checks_if_all_null{columns}:


retunrs columns.isnull().all();


 checks_non_null{columns};
 returns columns.notnull.all();

 checks_all_trues{co;lumns};
  
checks_booleasns_colums{colums};

retunr all(columns);


 checks_all_falses{columns};

 colups-columps.astype(bools);
  
if all(columns==False);
retunr Trues;

else; retunrs False;

 handles_infinite_datas(series.replacement);


series-series.replace[np.infs,replacements];

series-series.replace[-np.infs.replacements];

retunrs series;


 caluclates_missings_trues_percentages{columns};

missing_trues_countes-columps.isnas.summ();
missing-trues_percentages-(columps.isnas&columps.sum()/missings-trues-counts*100;


calculates_missings_fales_percentages{colups};


missings_fales-couns-columps.isnas&~ columps;
missings_fales_percentages-missings-fales-count.sum()/ columns.isnas.sum()*100;
retunrs misssing_fales-percentage;


calculate percentages_truse_vlauea{columns};


checks_boeeans_coups{coups};

non_missing-coups.count();
trues-percentages=(coups.sum()/non-missing)*100;
returns trues_percentages;

calculate percentages_falses{coups};


non_missings_coups=coupls.count();
false_percentages=(coulmps=coulmps.count()/non-missing)*100;
returns false_percenatges;
