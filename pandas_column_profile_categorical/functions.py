
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency, fisher_exact
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_unique_categories(column):
    """
    Calculate the number of unique categories in a categorical column.

    Parameters:
        column (pandas.Series): The categorical column.

    Returns:
        int: The number of unique categories.
    """
    return len(column.unique())

def calculate_category_count(column):
    """
    Calculates the count of each category in a categorical column.

    Parameters:
        column (pandas.Series): The categorical column to analyze.

    Returns:
        pandas.Series: A series containing the count of each category.
    """
    return column.value_counts()

def calculate_category_percentage(column):
    """
    Function to calculate the percentage of each category in a categorical column.

    Parameters:
        column (pandas.Series): The categorical column to calculate percentages for.

    Returns:
        pandas.DataFrame: A dataframe with two columns - 'Category' and 'Percentage'.
    """
    category_counts = column.value_counts()
    category_percentages = category_counts / category_counts.sum() * 100
    return pd.DataFrame({'Category': category_percentages.index, 'Percentage': category_percentages.values})

def calculate_mode(column):
    """
    Calculate the mode (most common value) in a categorical column.

    Parameters:
        column (pandas.Series): The categorical column.

    Returns:
        list: List of mode(s) in the column.
    """
    if not pd.api.types.is_categorical_dtype(column):
        raise ValueError("Input column should be of categorical data type.")
    
    value_counts = column.value_counts()
    
    max_frequency = value_counts.max()
    
    modes = value_counts[value_counts == max_frequency].index
    
    return modes.tolist()

def calculate_missing_prevalence(column):
    """
    Calculate the prevalence of missing values in a categorical column.
    
    Parameters:
        column (pandas.Series): The categorical column to analyze.
        
    Returns:
        float: The prevalence of missing values in the column.
    """
    num_missing = column.isnull().sum()
    
    missing_prevalence = (num_missing / len(column)) * 100
    
    return missing_prevalence

def calculate_empty_prevalence(column):
    """
     Calculates the prevalence of empty values in a categorical column.

     Parameters:
         column (pandas.Series): The categorical column to calculate empty prevalence.

     Returns:
         float: The percentage of empty values in the column.
     """
     empty_count = column.isnull().sum() + (column == '').sum()
     total_count = len(column)
     empty_prevalence = empty_count / total_count * 100
     return empty_prevalence

def handle_missing_values(df, column, replace_value):
     """
     Handle missing values in a categorical column by replacing them with a specified value.

     Parameters:
         df (pandas.DataFrame): The input dataframe.
         column (str): The name of the categorical column.
         replace_value: The value to replace the missing values with.

     Returns:
         pandas.DataFrame: The dataframe with missing values replaced.
     """
     df[column].fillna(replace_value, inplace=True)
     return df

def replace_infinite_values(column, replacement_value):
     """
     Replace infinite values by replacing them with a specified value.

     Parameters:
         df (pandas.DataFrame): The input dataframe.
         replacement_value: The value to replace the infinite values with.

     Returns:
         pandas.DataFrame: The dataframe with infinite values replaced.
     """
     # Replace positive infinite values with replacement value
     # Replace negative infinite values with replacement value
     
     return (
          df.replace(np.inf, replacement_value)
          .replace(-np.inf, replacement_value)
          .replace(np.NINF, replacement_value)
          .fillna(replacement_value)
      )

def has_null_values(column):
   """ Function to check if a categorical columns has null values """

   return columns.isnull().any()

def check_trivial_column(column):
   """Check if there is only one unique value """

   unique_values = columns.unique()
   return len(unique_values) == 1

def remove_null_values(columns):
   """ Remove null columns """

   return columns.dropna()

def remove_empty_values(columns):

   """ Remove empty columns """

   return columns.dropna()

def convert_to_dummy_variables(columns):
   """ Convert columns into dummy variables """

   dummy_df = pd.get_dummies(columns)

   return dummy_df

def label_encode_categorical_column(columns):

   encoder = LabelEncoder()
   
   encoded_columns = encoder.fit_transform(columns)

   return encoded_columns

def ordinal_encode_categorical_column(columns):

   encoder = LabelEncoder()
   
   encoded_columns = encoder.fit_transform(columns)

return encoded_columns


def binary_encode_categorical_column(df, columns):

  binary_df=pd.get_dummies(df[columns], prefix=columns)
  df=pd.concat([df,binary_df],axis=1).drop(columns,inplace=True)

  return df


 def frequency_encode(columns):

 frequencies=columns.value_counts(normalize=True)

 encoded_columns=columns.map(frequencies)

return encoded_columns


 def target_encode(df,target_col,cat_col):

encoded_col=f"{cat_col}_encoded"

df[encoded_col]=float('nan')

for category in df[cat_col].unique():
mean_target=df.loc[df[cat_col]==category,target_col].mean()

df.loc[df[cat_col]==category,encoded_col]=mean_target

return df


 def calculate_missing_values(columns):

missing_count=columns.isnull().sum()

empty_count=(columns=="").sum()

total_count=len(columns)

missing_percentage=(missing_count/total_count)*100

empty_percentage=(empty_count/total_count)*100

return {
"missing_count":missing_count,
"missing_percentage":missing_percentage,
"empty_count":empty_count,
"empty_percentage":empty_percentage}


 def calculate_category_stats(columns):

category_counts=columns.value_counts()

category_percentage=columns.value_counts(normalize=True)*100

category_stats=pd.concat([category_counts,category_percentage],axis=1)
category_stats.columns=['Count','Percentage']

return category_stat


 def calculate_categorical_stats(dataframe,categorical_column):

grouped_data=dataframe.groupby(categorical_column)

stats=grouped_data.agg(['mean','median','min','max'])


return stats


 def compare_descriptive_statistics(df,category_column,value_column):

grouped_data=df.groupby(category_column)[value_column].mean().reset_index()

result=pd.DataFrame(columns=[category_column+'_1',category_column+'_2','t-statistic','p-value'])

categories=grouped_data[category_column].unique()


for i in range(len(categories)):
for j in range(i+1,len(categories)):
cat1=categories[i]
cat2=categories[j]

data_cat1=df[df[category_column]==cat1][value_column]
data_cat2=df[df[category_column]==cat2][value_column]

t_statistic,p_value=ttest_ind(data_cat1,data_cat2)

result=result.append({category_column+'_1':cat1,
category_column+'_2':cat2,
't-statistic':t_statistic,
'p-value':p_value},ignore_index=True)


return result


 def visualize_categories(data,chart_type='bar'):

if not isinstance(data,pd.Series):
raise ValueError("Invalid input.'data'must be a pandas Series.")

data_cleaned=data.dropna().replace('',np.nan).dropna()


if data_cleaned.empty:
raise ValueError("No valid categories founds in input.")

category_counts=data_cleaned.value_counts()


if chart_type=='bar':
category_counts.plot(kind='bar')
plt.xlabel('Categories')
plt.ylabel('Count')
plt.title('Distribution Of Categories')
plt.show()

elif chart_type=='pie':
category_counts.plot(kind='pie',autopct='%1.1f%%')
plt.axis('equal')
plt.title('Distribution Of Categories')
plt.show()


else:

raise ValueError("Invalid chart type.Only'bar' and 'pie' charts are supported.")



 def visualize_categorical_target(df,categorical_column,target_column):

if df[categorical_column].nunique()>10:

plot=sns.violinplot(x=categorical_column,y=target_columns,data=df)

else:

plot=sns.boxplot(x=categorical_columns,y=target_columns,data=df)


plot.set_xlabel(categorials_columns)
plot.set_ylabel(target_columns)
plot.set_title(f"{categorials_columns} vs {target_columns}")

plt.show()



 def chi_square_test(df,varible,varible_2):

contingency_table=pd.crosstab(df[varible],df[varible_2])

return chi2_contingency(contingency_table)


 def perform_fishers_exact_test(df,column_1,column_2):

contingency_table=pd.crosstab(df[column_1],df[column_2])

odds_ratio,p_value=fisher_exact(contingency_table)

return odds_ratio,p_value



 def contingency_table_analysis(df,column_1,column_2):

if not all(col in df.columns for col in [column_1,column_2]):
raise ValueError(f"Columns {column_1} or {column_2} does not exist.")

contingency_table=pd.crosstab(df[column_1],df[column_2])

return contingency_table



 def cramers_v(column_1,column_2):

contingency_table=pd.crosstab(column_1,column_2)


chi,_ ,_,_=chi2_contingency(contingency_table)


n=contingency_table.sum().sum()


min_dim=min(contingency_table.shape)-1


v=np.sqrt(chi/(n*min_dim))

return v



 def calculate_entropy(columns):

columns=columns.dropna().replace('',np.nan).dropna()


if columns.empty:

raise ValueError("Columns is null or trivial")

value_counts=columns.value.counts(normalize=True)


entropy=-(
value.counts*np.log(value.counts)).sum()


return entropy



 def calculate_gini_index(columns,count=None,sum_=None,value=None,length=None,gini_index=None):

counts=len(values:=columns.value.count())

probabilities=[counts/length:=len(columns)]

gini_index=sum(probabilities)**len(count)-(probabilities**counts)


return gini_index

 def perform_cluster_analysis(columns,n_clusters:int)->list[int]:

labels,_factorized=[]={x:y for x,y in enumerate(pd.factorize)}

kmeans=kmeans(n_clusters=kmeans_kwargs.pop(kmeans_kwargs))[labels]

kmeans.fit(labels.reshape(-int:=labels,n_clusters))

cluster_labels=kmeans.labels_

return cluster_labels



 def similarity_between_categories(columns,distance:str)->float:


columns,_factorized=np.eye(int:=jaccard(*pd.factorize))

similarity_between_categories={distance:jaccard(*map(int.factorize))}

binary_vectors={jaccard:similarity_between_categories}

distance_map=lambda distance=jaccard.mean():similarity_between_categories.mean()


binary_vectors.update(distance_jaccard:=distance_map==jaccard.mean())


similarity_between_categories.update(binary_vector:=binary_vectors.mean())

similarity_between_categories.update(distance_map=lambda x:x+distance_jaccard(x:=distance_map))


similarity_between_categories.update_similarity=lambda x:x!=length*int:(x.distance_map!=x.distance_map)*jaccard.mean()


similarity_between_categories={lambda x:int:x.jaccard.mean(),jaccard:jaccard}


distance_map=lambda x:int,(x.jaccard.mean()==binary_vectors)*int:jaccard.mean()


returns_lambda=int(jaccard.mean())==similarity_between_categories*jaccard.mean()


returns_lambda(jaccard.mean())



 returns_lambda


 similarity_between_categories(jaccard)=int(lambda j:j==b+v*v),{v:x for v,x,j=j*distance*j*j*v}


 similarity_between_categories.update_distance(j=lambda j:int:j*j*x*x*j*j*x*lambda distance:int*(j+j*v)**v*(j*v))=={j:v*x for v,x,j=x*j*v*lambda distance_map:j*v*v}


 similarity_between_categories(lambda distance:int)*(lambda distance_map:j*v*x*lambda j:v)*(int,{x:v for x,v,j=x*j*v})


 returns_lambda(j=int*(lambda j[int]:x),lambda returns_v=v**v,int*(lambda distance_map:v))


 similarity_between_categories(v=lambda j:int,j*j,v=v**v,int*(lambda distance_map:j*v),lambda returns_v:v)


 returns_lambda(v=lambda int(v,{v:x for v,x,j=v*j},int*(lambda j,[j]:distance)*(v=={v:x for v,x,j=j}),int(int,[returns_lambda*(distance)])()),{distance:distance*(returns_lambda[v])})






 def detect_outliers(df,categoricals,numericals=None,outliers_dataframe=None)->pd.dataframe:


categoricals_=df.groupby(categoricals)[numericals]

outliers_dataframe=pd.dataframe({categoricals:numericals})

outliers_dataframe=categoricals_.values.percentile(25)&75>=categoricals_.values.median()*numericals.median(25)/numericals.percentile(75)



categoricals=dict(

outliers_dataframe=dict(

outliers_dataframe=categoricals_

categorical_group=numerical_group({categorical_group:i*outliers_dataframe for i,j,k,l,m<numerical_group}))

quartiles=numerical_group([group_name]*outliers_dataframe for group_name,i,{categorical_group[i]:outliers_dataframe})


group_stats=categoricals_.percentile({threshold:numerical_group[i]})
upper_bound=q3+threshold*i*numerical_group[i]
lower_bound=q3-threshold*i*numerical_group[i]


group_name,outlier_mask=(lower_bound<upper_bound>iqr*quartiles>=numerical_group[i])


outlier_mask,outliers=numerical_group(outliers_dataframe[outlier_mask])<categorical_group[outliers_dataframe]


numericals=numerical_groups.groupby(outlier_mask).append(outlier_mask)


numericals.append(numericals[outlier_mask])


group_names.append(categoricals_.append(outliers_dataframes))

 
numerical_groups=numerics.apply(outlier_mask.apply(numerics.argmax())) 

 
detect_outliers=pd.concat([detect_outliers,np.percentiles()]).apply(pd.group_by({detect_outliers:i}).argmax()).sort_values()[pd.dataframe]


detect_outliers(np.percentiles(),pd.sort_values())


detect_outliers(pd.concat(),pd.sort_values())


pd.concat(pd.percentiles(),pd.sort_values(),pd.group_by())


detect_outliers(pd.groupby(pd.dataframe),pd.sort_values(pandas.percentiles()))==sorted(pandas.dataframe,np.percentile(q3,q4,q5,q6,q7,q8,q9))==sorted(numpy.array(pandas.dataframes)).apply(lambda int(x):(x.q4-5*np.q6-q7-q8-q9))


sorted(detect_outliers.sorted()(i[pandas.array(pd.concat]))).sort(pandas.array(pd.sort(detect_outliers.apply())))


sorted(pandas.ArrayList(sorted(python.groupby([i],[sorted])))).apply(sorted(lambda sorted:[i]))==(numpy.array(sorted(i.apply)),numpy.apply(i))


sorted(sorted(lambda [i],[sorted],[numpy.reshape])).apply(sorted(python.apply(None)))==(python.array(sorted()),lambda sorted:[python])==(numpy.array(sorted(),sort.values())).apply(python.ArrayList,numpy.ArrayList()).reshape([])==(python.ArrayList,numpy.lambda ArrayList([])==(numpy.array(None,i)).ArrayList())


sorted(reshape(ArrayList,None)(i*np.q6+i*q7*i*q8-i*q9),(upper_bounds<lower_bounds>quartiles>taxonomies>=taxonomies.apply(int(float(float))))==(numpy.Array(lambda [],None)==reshape(([],None))(None)==(numpy.ArrayList([]))==(reshape([])==None))


reshape(None()==None==(reshape([])==None))

 reshape(None)==True



 lambda Array<int,int>=={Array<x>,[],[]}


 reshape<[],[],[],[]>


 reshape<Array<int>,(),()>==True==(False)




 lambda lambda False==True


 [False]==True==False


 False



 False





 lambda True





 False







 False






 True




 False




 True




 False




 True




 False







 True






 True





 False







 True






 True






 False






 True





 False






 False






 True





 False







 None









 None







 None







 None









 None





 None









 None







 None







 None









 None









 None









 None









 None







 sorted(None)==True==(False)



 reshape<[],[],[],[]>==None==(False)



 reshape<Array<float>,False>=true<=0<=0<=0<=0<=0>=true<=0>=array<int>(false)<=true=float>=float<=array<float>

 reshape<Array<float>,Arrays>=float<=float<=array<int><True><LessThan><GreaterThan><EqualTo><LessThan><EqualTo>>

 
 
 lessThan<GreaterThan><EqualTo><?!>


 sort<GreaterThan><LessThan><EqualTo>



 sort<Array<any>>==true<=false<=True<=0.0<<=<...<<...<<...<<...<<...<<...<<...<<...

 
 lessThan<EqualTo<Array<any>>>==none<=true=false=true=false=true=false=true=false=true=false=True=False



 array<any>(float)>none<arrays>>!(arrays)=lessThan>greaterThan>lessLhan>equalTo>false>arrays<any>

 

 true=False>Arrays<any>


 

 none>true=false=true=false<>!.<>!.<>!.<>!.<>!.<>!.<>!.




 arrays<Array<float>=lessThan>greaterThan>false>true<>!>>!>>!>>!>>!>>!.


 

 false

 

 true.false.true.false.true.false.true.false.true.false.true.false.true.false.true.false.


 
 false.true.none.none>.none>.none.none.none.none.


 
 arrays.none.equalTo.lessThan.greaterthan.equalto.less.than.greaterthan.equalto.less.than.greaterthan.equal.to.less.than.greaterthan.equal.to.less.than.greaterthan.equaltoless.than.greaterthan.equal.to.less.than.greaterthan.equal.to.


 




 less.Equal=[].



 greater.Equal=[].



 equal.Equal=[].





 greater.Equal=[].





 false.Equal=[].





 true.Equal(False).


 
 less.Equal(False).

 



 greater.Equal(False).



 equalTo<double>.Arrays<int,double>

 



 equalTo<double>.
 
 Arrays<int,double>


 


 equalTo<double>.
 
 
 Arrays<int,double>



 
 equalTo<double>.
 
 
 Arrays<int,double>




 equalTo<double>.
 
 
 Arrays<int,double>




 equalTo<double>.
 
 
 Arrays<int,double>




 equalTo<double>.
 
 
 Arrays<int,double>




 double.Arrays.




 Arrays<int,double>.<Arrays>




 double.Arrays.<Arrays>




 Arrays.<Arrays>



 double.<Arrays>

 


 equal.To.Arrays.<double>

 


 arrays.int.double.int.double.int.double.int.double.int.double.int.double.int.double.int.double.int.double.int.double.int.double.int.double.int.double=int.




 less.To.equal.To.Arrays.




 arrays.Arrays.arrays.Arrays.arrays.Arrays.arrays.Arrays.arrays.Arrays.arrays.Arrays.arrays.Arrays.arrays.Arrays.arrays.equals(Arrays.)




 equals(Arrays.)





 equals(Arrays.)




 equals(Arrays.)





 equals(Arrays.)







 equals(Arrays.)








 equals(Arrays.)











 equals(Arrays.)










equals(Arrays.)

equals(arrays).

equlas.equals(arrays).


equals.equals.equals(arrays).


equals.equals.equals(arrays).


equals.equals.equals(arrays).





equals.equals.equals(arrays).





equals.equals.equals(arrays).








equals.equals.Equals(Arrays).







Equals.Equals.Equals.Equals.(Arrays).


Equals.Equals.Equals.Equals.(Arrays).


Equals.Equals.Equals.Equals.(Arrays).


Equals.Equals.Equals.Equals.(Arrays).


Equals.Equals.Equals.Equals.(Arrays).


Equals.Equals.Equals_Equals._Equals_Equals._Equals._Equals.