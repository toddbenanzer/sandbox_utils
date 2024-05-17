
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import matplotlib.pyplot as plt


def get_spark_session():
    return SparkSession.builder.getOrCreate()


def read_csv_to_dataframe(file_path: str) -> DataFrame:
    spark = get_spark_session()
    return spark.read.csv(file_path, header=True, inferSchema=True)


def read_json_to_dataframe(file_path: str) -> DataFrame:
    spark = get_spark_session()
    return spark.read.json(file_path)


def read_parquet_to_dataframe(file_path: str) -> DataFrame:
    spark = get_spark_session()
    return spark.read.parquet(file_path)


def read_table_to_dataframe(url: str, table: str, properties: dict) -> DataFrame:
    spark = get_spark_session()
    return spark.read.jdbc(url=url, table=table, properties=properties)


def filter_dataframe(df: DataFrame, condition: str) -> DataFrame:
    return df.filter(condition)


def select_columns(df: DataFrame, columns: list) -> DataFrame:
    return df.select(columns)


def join_dataframes(df1: DataFrame, df2: DataFrame, column: str) -> DataFrame:
    return df1.join(df2, column)


def group_by_columns(df: DataFrame, columns: list) -> DataFrame:
    return df.groupBy(*columns)


def calculate_row_count(df: DataFrame) -> int:
    return df.count()


def calculate_sum(df: DataFrame, column_name: str) -> float:
    return df.agg({column_name: 'sum'}).collect()[0][0]


def calculate_mean(df: DataFrame, column_name: str) -> float:
    return df.select(F.mean(column_name)).collect()[0][0]


def calculate_minimum(df: DataFrame, column_name: str) -> float:
    return df.select(F.min(column_name)).first()[0]


def calculate_maximum(df: DataFrame, column_name: str) -> float:
    return df.select(F.max(column_name)).first()[0]


def calculate_stddev(df: DataFrame, column_name: str) -> float:
    return df.select(F.stddev(column_name)).first()[0]


def calculate_correlation(df: DataFrame, col1: str, col2: str) -> float:
    return float(df.corr(col1, col2))


def calculate_covariance(df: DataFrame, col1: str, col2: str) -> float:
    covariance = (
        df.select(((F.col(col1) - F.mean(col1)) * (F.col(col2) - F.mean(col2))).alias('cov'))
        .agg({'cov': 'sum'})
        .collect()[0][0]
        / df.count()
    )
    return covariance


def calculate_percentile(df: DataFrame, column_name:str , percentile :float ) -> float :
  window = Window.orderBy(F.col(column_name))
  ranked_df =df.withColumn("rank",F.percent_rank().over(window)).filter(F.col("rank")>=percentile)
  min_value=ranked_df.select(F.min(column_name)).first()[0]
  return min_value


def calculate_frequency_distribution(df :Dataframe ,column_name :str)->Dataframe :
   freq_dist=df.groupby(column_name).count().withColumnRenamed(column_name ,"value")
   return freq_dist


from pyspark.sql import functions as F

def calculate_cross_tabulation(
  df :Dataframe ,col1:str,col2:str)->Dataframe :
  
  distinct_col1_values=[row[col1 ] for row in df.select (col(col1 )).distinct().collect()]
  distinct_col2_values=[row[col2 ] for row in df.select (col(col2 )).distinct().collect()]
  
 cross_tabulation=df.crosstab (col1 ,col2)
 
 for value in distinct_col1_values :
   if value not in cross_tabulation.columns :
      cross_tabulation=cross_tabulation.withColumn(value,F.lit(0))
 
 for value in distinct_col2_values :
   if value not in cross_tabulation.columns :
       cross_tabulation=cross_tabulation.withColumn(value,F.lit(0))
       
return cross_tabulation

#Example usage

data=[("A","X"),("A","X"),("A","Y"),("B","X"),("B","Y")]
df=spark.createDataframe (data ,["col1","col2"])
cross_tabulation=calculate_cross_tabulation (df,"col1" ,"col2")
cross_tabulation.show() 


from pyspark.sql import SparkSession

 def calculate_overlap(population_query_1:str,population_query_2:str)->int :

   #Create a sparksession
   
   spark=sparkSession.builder.getOrCreate()

   #create dataframe using SQL queries
   population_df_1=spark.sql(population_query_1)
   population_df_2=spark.sql(population_query_2)

 overlap_count=population_df_1.intersect (population_df_2).count()
 
return overlap_count


from pyspark.sql import SparkSession

 def calculate_population_union(population_query_1:str,population_query_2:str)->Dataframe :

   #Create a sparksession
   
   spark=sparkSession.builder.getOrCreate()

   #create dataframe using SQL queries
   population_df_1=spark.sql(population_query_1)
   population_df_2=spark.sql(population_query_2)

 union_df=population_df_1.union(population_df_2)
 
return union_df


from pyspark.sql import SparkSession

 def calculate_intersection(pop_query_1:str,pop_query_2:str)->Dataframe :

   #Create a sparksession
   
   spark=sparksession.builder.getOrCreate()

#Load dataframe based on SQL queries
pop_df_1=sparksql(pop_query _  )
pop_df _  =spark .sql(potquery _ )

#Calculate intersection 
intersection=pop _df _ intersects (pop _df _ )

return intersection 


from pyspark .sql import SparkSession

 def compute_profile_statistics(dataframe :Dataframe ,population_sql:str)->dict:

#Creat a sparksession 
spark=SparkSession.builder.getorCreate ()

#register dataframe as a tempview 

dataframe.createorReplaceTempView ("data")

population_dataframe=sparksql (population_sql )

profile_stats={}
for colname in population_dataframe.columns :

count_val=population_dataframe.select (F.count (F.col("column"))).first()[0]
sum_val=population_dataframe.select (F.sum(F.col("column"))).first[0]

mean_val=population_dataframe.select (F.mean(F.col("column"))).first ()[0]

profile_stats[colname]={"count":count_val,"sum":sum_val,"mean":mean_val}

return profile_statistics 



  

import matplotlib.pyplot as plt 

 def visualize_column_distribution(dataframe :Dataframe,column_name :str )->None:

 data=dataframe.select(columnname ).toPandas()

plt.hist(data[columnname ])

plt.xlabel(columnname )
plt.ylabel ("Frequency ")
plt.title ("Distribution of values in {column}".format(column="columnname "))

plt.show()


import matplotlib.pyplot as plt 

 def scatter_plot(dataframe :Dataframe,x_col:str,y_col:str )->None:

 data=dataframe.select(x_col,y_col ).toPandas()

plt.scatter(data[x_column],data[y_column])
plt.xlabel(x_column)
plt.ylabel(y_column)
plt.title(f"Scatter plot:{x_column}vs {y_column}")

plt.show()


import matplotlib.pyplot as plt 

 def visualize_relationship(dataframe :Dataframe,x_column:str,y_column:str )->None:

 data=dataframe.select(x_column,y_column ).toPandas()

plt.plot(data[x_column],data[y_column])
plt.xlabel(x_column)
plt.ylabel(y_column)
plt.title(f"Scatter plot:{x_column}vs {y.column}")

plt.show()



import matplotlib.pyplot as plt 

 def visualize_relationship_with_barplot(data_frame :Dataframe,column_x:str,column_y:str )->None:

 counts=data_frame.groupby (column_x ).count().orderBy(column_x )
 x_values=[row[column_x] for row in counts.collect()]
 y_values=[row["count"] for row in counts.collect()]

 plt.bar(x_values,y_values )
 plt.xlabel(x_label)
 plt.ylabel(y_label)
 plt.title(f"Relationship between {x_label} and {y_label}")

 plt.show()

  
import matplotlib.pyplot as plt 

 def visualize_histogram(data_frame :Dataframe,column_x:str,column_y:str )->None:


 data_frame_x=data_frame .select(column_x ).rdd.flatMap(lambda x:x).collect()
 data_frame_y=data_frame .select(column_y ).rdd.flatMap(lambda x:x).collect()

 plt.hist(data_frame_x ,alpha=5,label="{}".format( x))
 plt.hist(data_frame_y ,alpha=5,label="{}".format(y))


 #Add labels and title to the plot
 
 plt.xlabel(x_label)
 plt.ylabel(y_label)

 plt.legend ()
 
 plt.show()


  
from pysark.sql import SparkSession

 def export_to_database(table_url,str,data_frame :str,url,str,user,password ):->None:

 data_frame.write.format('jdbc' ).option('url',url).option ('db_table',table_url ).option ('user',user ).option ('password',password ).mode ('overwrite').save()



 def export_to_csv_file( data_frameto_export,file_path)->None:

 pandas_data_frame=data_frameto_export.toPandas()
 pandas_data_frame.to_csv(file_path,index=False)



 from pysark.sql import SparkSession

  
 def export_to_parquet_file(saving_path,data_frameto_export )->None:

data_frameto_export.write.parquet(saving_path )

  
 from pysark.sql import SparkSession

  
 def save_overlap_analysis(result_output_path,populationsql_query,populationsqlquery,saving_metrics_sqlquery )->None:


spark=sparksession.builder.getorcreate ()

result_output_populationdf=sparksql(result_output_populationquery )
saving_metrics_populationdf=sparksql(saving_metrics_populationquery )


results=result_output_populationdf.join(saving_metrics_populationdf,on=["commonmetrics"],how='inner' )

results.write.parquet(result_outputpath )


 from pysark.sql import SparkSession

  
 def save_profile_results(result_output_dict,data_frameto_calculate_profiles )->dict:


spark=sparksession.builder.getorcreate ()

resulting_profiles={}

for query_key,result_output_metric_sqlquery in result_output_dict.items():

resulting_profiles[query_key]=sparksql(result_output_metric_sqlquery)

return resulting_profiles 



 from pysark.sql import SparkSession

  
 def save_visualized_data_as_imagefile(pandas_data_frameto_visualize_dataset,path_to_save_imagefile ):

pandas_data_frameto_visualize_dataset.plot->(visualization_plot )

fig.savefig(path_to_save_imagefile )


 from pysark .sql import dataframe

  
 def cache_or_persist_df(caching_type,data_frameto_cache_and_persist,caching_type_choice ):


if caching_type_choice=='memory_only':

return data_frameto_cache_and_persist.cache ()

if caching_type_choice=='disk_only':

return data_frameto_cache_and_persist.persist(storage_level=disk_only)

return None

