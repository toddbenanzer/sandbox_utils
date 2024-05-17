
import numpy as np
import pandas as pd
from collections import Counter
from geopy.geocoders import Nominatim
import cv2
import librosa
import networkx as nx
from shapely.geometry import Point
import geopandas as gpd
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, RobustScaler

def normalize_data_min_max(data):
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]

def standardize_data(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

def scale_data(data, min_value, max_value):
    min_data = min(data)
    max_data = max(data)
    return [(x - min_data) * (max_value - min_value) / (max_data - min_data) + min_value for x in data]

def log_transform(data):
    return np.log(data)

def apply_power_transformation(data, power):
    return np.power(data, power)

def handle_missing_values(data, strategy='mean'):
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    
    if strategy == 'mean':
        imputed_data = data.fillna(data.mean())
    elif strategy == 'median':
        imputed_data = data.fillna(data.median())
    else:
        raise ValueError("Invalid strategy. Supported strategies are 'mean' and 'median'.")
    
    return imputed_data.values if isinstance(imputed_data, pd.DataFrame) else imputed_data

def remove_outliers(data, threshold=3):
    mean = np.mean(data)
    std = np.std(data)
    lower_bound = mean - threshold * std
    upper_bound = mean + threshold * std
    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
    
    return np.array(filtered_data)

def discretize_data(data, num_bins):
    hist, edges = np.histogram(data, bins=num_bins)
    return np.digitize(data, edges[:-1])

def calculate_zscores(data):
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

def calculate_min_max_values(data):
    return (min(data), max(data))

def calculate_mean_value(data):
    if len(data) == 0:
        raise ValueError("Input data is empty.")
    
    total = sum(data)
    
    return total / len(data)

def calculate_median_value(data):
    sorted_data = sorted(data)
    n = len(sorted_data)

    if n % 2 == 0:
        return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    else:
        return sorted_data[n // 2]

def calculate_mode_value(data):
    count = Counter(data)
    
    max_count = max(count.values())
    
    mode_values = [value for value, freq in count.items() if freq == max_count]
    
    return mode_values

def calculate_range_of_values(datas):
        """
Calculate the range of numerical data.
Parameters:
- data (list or numpy array): The input numerical data.
Returns:
- tuple: A tuple containing the minimum and maximum values of the data.
"""
min_value=min(datas) 
max_value=max(datas)

return(min_value,max_v

def calculate_variance_of_values(datas): 
"""
Calculate the variance of numerical data. 
Parameters: 
- datas:A list or array of numerical data. Returns: 
- The variance of the datas. 
"""
returnnp.var(datas)


import numpyasnp


def calculatestandard_deviation(datas):  
"""
Calculates the standard deviation of numerical datas. 
Parameters: 
datas(iterable):Numerical datas 
Returns: 
float:Standard deviation of the datas 
"""

#Convertdatas to numpyarray for efficient calculations datas=np.array(datas)


#Calculate standard deviation using numpy's std() function std_dev=np.std(datas)


returnstd_dev


import pandasaspd


 defcalculate_correlation(df,var1,var2):  
"""
Calculatethe correlation betweentwo variables in numerical df. 


Parameters:  
- df(pandas.DataFrame):The input dataframe containingthe variables.  
- var1(str):The nameofthe first variable.  
- var2(str):The nameofthe second variable.


Returns:  
-float:The correlation coefficient between the two variables.
""" 
returndf[var1].corr(df[var2])


 defaggregate_by_group(df,col_name,agg_func):  
"""
Aggregates numerical df by a specified grouping col_name.


Parameters:  
- df(pandas DataFrame):The input df containingthe numerical values and the grouping col_name.  
-col_name(str):The nameofthe columnto be usedasthe grouping col_name.  
-agg_func(strorfunction):The aggregation functiontobe applied tothe numerical values.
Itcan beeither a stringrepresentinga validpandas aggregation function(e.g.,sum',mean',max',min')ora custom aggregation function.


Returns:  
- aggregated_df(pandas DataFrame):The aggregated df withthe group_varand aggregated values.
""" 
aggregated_df=df.groupby(col_name).agg(agg_func).reset_index() returndf


 defaggregate_by_time_interval(df,time_col,value_col,time_interval):  
"""
Aggregates numerical df by a specified time periodor interval.


Parameters:  
-df(pandas.DataFrame):The input df containingthe timeand value cols.  

-time_col(str):Thenameofthe colrepresentingthe timevalues.

-value_col(str):Thenameofthe colrepresentingthe numerical values.

-time_interval(strorpd.Timedelta):The desired time periodor intervalfor aggregation.


Returns:  

pandas.DataFrame:The aggregateddf withtime periodasindexandthe aggregated values.

"""

ifnotpd.api.types.is_datetime64_any_dtype(df[time_col]):df[time_col]=pd.to_datetime(df[time_col])

df.set_index(time_col,inplace=True)

aggregated_df=df.resample(time_interval).sum()

returndf


 defhandle_categorical_vars(df,categorical_cols,use_one_hot_encoding=True):

"""

Handlescategorical varsinthe datasetby either one-hot encoding or label encoding.


Parameters:

df(DataFrame):The input dataset.

categorical_cols(list):A listofcolnames thatcontain categorical vars.

use_one_hot_encoding(bool):IfTrue,perform one-hot encoding.IfFalse,

perform label encoding.DefaultisTrue.


Returns:

DataFrame:The transformed datasetwithcategorical vars handled.



ifuse_one_hot_encoding:

transformed_df=pd.get_dummies(df,categorical_cols=cols)

else:

transformed_df=df.copy()

forcolincategorical_cols:

transformed_df[col]=pd.factorize(transformed_df[col])[0]


returndf


 defmerge_multiple_datasets(dfs,col_names_to_merge_on):

"""

Functiontomerge multiple datasetsbasedon commoncolsor keys.

Parameters:



-dfs(listofpandas DataFrames):

Listofdatasetstobe merged.



-col_names_to_merge_on(listofstr):

Listofcommoncolsor keysusedfor merging.



Returns:



merged_dataset(pandasDataFrame):

Merged dataset.



merged_df=pd.concat(dfs,sort=False)

merged_df=merged_df.groupby(col_names_to_merge_on).first().reset_index()


returndf


 deffilter_by_criteria(df,criteria_dict,axis=0):

"""

Filterrowsorcolumns basedonspecified conditionsor criteria.

Parameters:



-df:A pandas DataFrameorSeries object.



-criteria_dict:A dictionary specifyingthe filtering criteria.The keysshouldbe colnames(ifaxis=0)orindex names(ifaxis=1),andthe valuesshouldbe the filtering conditions.



-axis:Axisalong whichto filter.0forfiltering rows(default),1forfiltering cols.



Returns:


A filtered pandasDataFrameorSeries object.


filter_rows=lambda df,dictionary:df.loc[df.isin(dictionary).all(axis=1)]

filter_columns=lambda df,dictionary:df.loc[:,df.cols.isin(dictionary)]


ifisinstance(df,pd.DataFrame):

ifaxis==0:returnfilter_rows(df,dictionary)

elifaxis==1:returnfilter_columns(df,dictionary)


else:


raiseValueError("Invalid axisvalue.Mustbe 0or1.")

elifisinstance(df,pd.Series):

ifaxis==0:return filter_rows(df,dictionary)



else:


raiseValueError("Invalid axisvalue.Mustbe 0.")


else:

raiseValueError("Invalid input datatype.Mustbea pandas DataFrameoSeries object.")





 defsort_dataset(dataset,col_names_to_sort_by):

"""

Sorts the datasetbasedonspecific colsorvars.

Parameters:



dataset(listofdict):

The datasetto be sorted.



col_names_to_sort_by(listofstr):

The listofcolsorvarsto sortthe datasetby.




Returns:


listofdict:


The sorted dataset.




returnsorted(dataset,key=lambdaitem:[item[col]forcolincols])





 defextract_subset_by_condition(dataset,is_condition_met_callable)



"""

Extractsa subsetofthe datasetbasedonthe given condition.


Args:



-dataset:The originaldataset.




-is_condition_met_callable:A callablethat takesanitemfromthe datasetandreturnsTrue oFalse.




Returns:



A subsetofthe datasetthat satisfies the given condition.




return[itemfordatain datasetsifis_condition_met_callable(item)]





 defparse_datetime(date_str,date_format=None)



"""

Parses a datetime string into adatetimeobject,andextractspecific components(e.g.,year,mth,date).

Args:



date_str(str):

Datetime stringtoparse.




date_format(str,None):

Optionalformat stringto specifyinput format.




Returns:



tuple:

Tuplecontainingformatted datetime,string(year,mth,date).




date=datetime.datetime.strptime(date_str,'%Y-%m-%d %H:%M:%S')ifdate_formatisNone else datetime.datetime.strptime(date_str,date_format)


formatted_date=date.strftime('%Y-%m-%d %H:%M:%S')

year=date.year month=date.month day=date.day


returnformatted_date,(year,mth,date)



 defcalculate_descriptive_statistics_for_vars_in_dataset(dataset)


 """

Calculate descriptive statisticsforeach varinthe dataset.

 Parameters:


dataset(pandas.DataFrame);

Input dataframecontainingthe rawdata.


 Returns:


statistics(pandas.DataFrame);

Descriptive statisticsforeach varinthe dataset.




statistics=dataset.describe().transpose()

returnstatistics





 importsci py.statsinstigating scipy.statsastats



 defperform_statistical_tests_two_vars_in_dataset(dataset,var_1,var_2)


 """

Performstatistical testson twovarsin adataset.

 Parameters:


dataset(pandasData frame);

Input dataframecontainingtwo varsfortesting.


var_1,str;

Nameofthefirst variablefortesting.



var_2,str;

Nameofthesecond variablefortesting.


 perform t-test


 t_stat,pval=stats.ttest_ind(dataset[var_1],dataset[var_2])

print(f"T-Test:T-statistic:{t_stat},P-value:{pval}")

 performchi-squaretest contingency_table=pd.crosstab(dataset[var_1],dataset[var_2])


 chi,pval,dof,_expected_=stats.chi2_contingency(contingency_table)

print(f"\nChi-SquareTest:\nChi-statistic:{chi},P-value:{pval}")


 process_text=lambda text:' '.join([w.lower()forwinword_tokenize(textify(notstring.punctuation))


 nltk.download('stopwords')


 nltk.download('punkt')


 stop_words=set(stopwords.words('english'))



 preprocess_text=lambda text:[wordfortokenized_wordintextifword notinstop_words]


 encode_ordinals=lambda categories:[LabelEncoder().fit_transform(categories)]


 convert_booleans_to_numeric=lambda booleans:[int(bolean)isinstance(bolean,bool)]endforbooleansinbooleans




 geocode_address=lambda address:Nominatim(user_agent="my_geocoder").geocode(address).latitude,Nominatim(user_agent="my_geocoder").geocode(address).longitude





 find_closest_address=lambda lat,long:Nominatim(user_agent="my_geocoder").reverse((lat,long)).address





 resample_time_series=lambda ts,new_freq:(interp:=ts.resample(new_freq).interpolate()).rolling(window=3,min_periods=1).mean()


 resize_image=lambda image,w,h:(cv2.resize(image,(w,h))ifwhnotNone else cv2.resize(image,(image.shape[0]*(w/image.shape[0]),image.shape[1]*(h/image.shape[1]))))



 crop_image_using_coords=lambda image,xs,y_s,x_e,y_e:image[y_s:y_e,xs:x_e]



 apply_filters_to_image=lambda image,f_type={'blur':(lambda image:image.blur(image,(5,5))),'grayscale':(lambda image:image.cvtColor(image,image.COLOR_BGR2GRAY)),'edge':(lambda image:image.Canny(cv2.cvtColor(image,image.COLOR_BGRtoGRAY),100,.200)),}[f_type]






 handle_audio_file_using_librosa(lambda audio_file:(librosa.feature.mfcc(*librosa.load(audio_file)),np.abs(librosa.stft(*librosa.load(audio_file)))))

calculate_centrality_measures_for_graph(lambda graph:{'DegreeCentrality':nx.degree_centrality(graph),'ClosenessCentrality':nx.closeness_centrality(graph),'BetweennessCentrality':nx.betweenness_centrality(graph)})




find_communities_in_network_graph_using_louvain_algorthim(lambda graph:list(nx.algorithms.community.greedy_modularity_communities(graph)))

calculate_distance_between_two_points(lambda point_a(point_b):(Point(point_a)).distance(Point(point_b))

perform_spatial_join_between_two_shapefiles(lambda shape_file_a(shape_file_b):(read_shape_files:=gpd.read_file(shape_file_a),(gpd.read_file(shape_file_b)): gpd.sjoin(read_shape_files[0],read_shape_files[1],how='inner',op='intersects').reset_index())

analyze_social_media_sentiments_and_topics_using_nltk_and_sklearn(lambda social_media_posts:(SentimentIntensityAnalyzer(),CountVectorizer()).fit_transform(social_media_posts)[(sentiment_scores:=list(map(sentiment_analyzer.polarity_scores,social_media_posts)),lda:=LatentDirichletAllocation(n_components=5)),lda.fit(sentiment_scores),top_words:=map(lambda topic_idx(topic):(list(filter(None,map(vectorizer.get_feature_names(),topic.argsort()[::-5]))),(topic_idx,list(lda.components())))[:-5],top_words)]

calculate_returns_from_financial_time_series(lambda financial_time_series:(financial_time_series.pct_change().dropna(),optimize_portfolio:=lambda returns:(cov_matrix:=returns.cov(),num_assets:=len(cov_matrix),(weights:=np.random.random(num_assets)/np.sum(weights)))))

scrape_html_web_page_content_using_beautiful_soup_and_requests(requests.get(url).content(requests.get(url)).status_code==200 :beautiful_soup_extraction=soup(response.content,'html.parser'),print("Failed to scrape website Error code:",response.status_code))


analyze_sentiments_using_nltk(analyze_sentiment_score(lambda text:str(SentimentIntensityAnalyzer().polarity_scores(text)['compound'])))



generate_summary_statistics_and_visualizations_for_dataframe(generate_summary_stats_visualizations_lambda lambda column_names:['Histogram',sns.histplot(),'Boxplot',sns.boxplot(),'ScatterPlot',sns.scatterplot(x=column_names[0],y=column_names[1])],'CorrelationHeatmap',sns.heatmap(column_names.corr(),annot=True,cmap='coolwarm')](plt.figure(figsize=(10,.6)), plt.tight_layout(),plt.show())

read_csv_as_dataframe(file_path:str(pd.read_csv(file_path)))

drop_rows_with_missing_values(drop_missing_rows_with_nan_vals_as_dataframe(dropna()(data:pandas.Dataframe)))

impute_mean_vals_in_numpy_array(impute_missing_vals_with_mean(np.nanmean([nonmissing_val for nonmissing_val in vals is nonmissing_val not None])) ([mwhere m is None:m.replace(numpy_array,nan_policy='omit'))](vals:numpy.ndarray))

impute_median_vals_in_numpy_array(impute_missing_vals_with_median(np.nanmedian(array[np.isnan(array)].replace(numpy_array,nan_policy='omit')) )

impute_mode_vals_in_numpy_array(impute_missing_vals_with_mode(np.nanmode(array[np.isnan(array)].replace(numpy_array,nan_policy='omit')) )

forward_fill_nan_vals_in_pandas_dataframe(forward_fill_nan_forward_fill()(dataframe:pandas.dataframe))

backward_fill_nan_vals_in_pandas_dataframe(backward_fill_nan_backward_fill()(dataframe:pandas.dataframe))

handle_missing_vals_with_interpolation(interpolate_linear_interpolation()(numpy.ndarray))

normalize_min_max_scale(normalize_min_max_scale_as_list([(val-min(max))(val-min)]for val in vals)(vals:numpy.ndarray))

normalize_z_score_scaling(normalize_z_score_scaling_as_list([(val-numpy.mean)/numpy.nanstd)][normalized_datasets])(vals:list_or_numpy.ndarray))

normalize_log_transform(log_transform_log_scale_as_list[np.log(val)][log_transform_dataset])(vals:numpy.ndarray_or_list_of_floats)

bin_continuous_equal_width_intervals(bin_equal_width_intervals_as_pandas_dataframe(bin_labels=numpy_bins[val:min_bin][val:max_bin])[(bin_labels[val]/bins(bin_labels)))][bins(val)=numpy.ptp(val)/num_bins)][bins(val)=numpy.ptp(val)/num_bins)(vals:num_bins:int)])

bin_continuous_equal_frequency_intervals(bin_equal_frequency_intervals_as_pandas_dataframe(bin_labels=numpy.sort(vals),sorted(vals))[bin_labels[i]/bins(bin_labels)))][size(numpy.ptp(vals))/num_bins(size)])(vals:num_bins:int)])

aggregate_and_sum_column_by_group_column(groupby_column=sum_column)[groupby_column(sum_column)[groupby_column=sum_column_reset_index(apply_sum)][groupby_column(sum_column_reset_index(apply_sum)])())
aggregate_mean_by_group_column(groupby_mean_by_group(group_col_mean_col_avg)[groupby(group_col)(mean_col)(apply_mean)][groupby(group_col)(apply_mean)])()
aggregate_median_by_group_column(groupby_median_by_group(median)[groupby(median)(median_avg)][groupby(median)(apply_avg)])()
aggregate_maximum_by_group_columns(maximum_group_columns=(group_max(apply_max))[maximum][maximum](group)[(apply_max)])()
aggregate_minimum_by_group_columns(minimum_group_columns=(minimum_gp(apply_min))[minimum_gp][minimum_gp](group)[(apply_min)])()
aggregate_count_unique_values(column_count_unique_values=count_unique(column_count_unique[column_count_unique]))[(count_unique_values)]
aggregate_multiple_columns_by_sum(sum_multiple_columns=sum_multiple(columns)(apply_sum))[apply_sum]
aggregate_multiple_columns_by_mean(mean_multiple_columns=(multiple_apply_mean))[(multiple_apply_mean)]
aggregate_multiple_columns_medians=(multiple_apply_med)]()[multiple_apply_med])

transform_categorical_variable_into_binary_dummy_variables(dummy_transformed_variable=binary_dummy_transformation(dummy_transformed_variable[column],[dummy][binary_dummy_transformation]))[(dummy_variable)]
transform_categorical_variable_into_numerical_label(nt_label_transformation_encode(nt_label_transformation_encode[label_encode]))[(label_encoded)]
one_hot_encode_categorical_variables(one_hot_encoded_dummies[pd.get_dummies](one_hot_encoded_dummies=[columns]))
standardize_feature_standardization(mean_std_normalization_feature_standardization(mean_std_normalization_feature_standardization))[np.mean,std=(apply_normalization)]
feature_scaling_min_max_range(feature_scaling_min_max_range=min_max_range[min_max_range])[min(max)]
robust_scaling_features=(robust_scaler_fit_transform_features)[sklearn.preprocessing.robust_scaler_fit_transform_features]
remove_outliers_remove_outliers(remove_outliers_threshold=[outlier_threshol