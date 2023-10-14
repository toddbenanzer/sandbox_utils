yspark.sql.functions as F
from pyspark.sql import SparkSession

def read_csv_file(file_path):
    """
    Read a CSV file using pyspark.

    :param file_path: The path of the CSV file to be read.
    :return: The DataFrame containing the CSV data.
    """
    # Create a SparkSession
    spark = SparkSession.builder.getOrCreate()

    # Read the CSV file into a DataFrame
    df = spark.read.csv(file_path, header=True, inferSchema=True)

    return df


def read_data_from_database(database_url, table_name):
    """
    Read data from a database using pyspark.

    :param database_url: The URL or connection string of the target database.
    :param table_name: The name of the table to read from.
    :return: The DataFrame containing the data from the database.
    """
    # Create a SparkSession
    spark = SparkSession.builder \
        .appName("Read Data from Database") \
        .getOrCreate()
    
    # Read data from the database
    df = spark.read \
        .format("jdbc") \
        .option("url", database_url) \
        .option("dbtable", table_name) \
        .load()
    
    # Return the DataFrame
    return df


def define_population(sql_query):
    """
    Define populations based on SQL queries.

    :param sql_query: The SQL query defining the population.
    :return: DataFrame representing the defined population.
    """
    # Create a SparkSession
    spark = SparkSession.builder.getOrCreate()
    
    # Execute the SQL query and return the result as a DataFrame
    population = spark.sql(sql_query)
    
    return population


def overlap_analysis(population1, population2):
  
  """
  Perform overlap analysis between two populations.

  :param population1: DataFrame representing the first population.
  :param population2: DataFrame representing the second population.
  :return: DataFrame containing the overlap analysis results.
  """
  
  # Calculate the count of distinct individuals in each population
  count_population1 = population1.select("individual_id").distinct().count()
  count_population2 = population2.select("individual_id").distinct().count()

  # Calculate the count of distinct individuals in the overlapping region
  count_overlap = population1.intersect(population2).select("individual_id").distinct().count()

  # Calculate the percentage overlap for each population
  percentage_overlap_population1 = (count_overlap / count_population1) * 100
  percentage_overlap_population2 = (count_overlap / count_population2) * 100

  # Create a DataFrame to store the overlap analysis results
  spark = SparkSession.builder.getOrCreate()
  results_df = spark.createDataFrame([(count_population1, count_population2, count_overlap,
                                       percentage_overlap_population1, percentage_overlap_population2)],
                                     ["population1_count", "population2_count", "overlap_count",
                                      "percentage_overlap_population1", "percentage_overlap_population2"])

  return results_df


def calculate_profiles(population_query, metric_queries):
    """
    Calculate profiles of populations across a set of metrics defined by SQL queries.

    :param population_query: SQL query defining the population.
    :param metric_queries: Dictionary of metric names and their corresponding SQL queries.
    :return: Dictionary containing the calculated profiles for each metric.
    """
    # Create a SparkSession
    spark = SparkSession.builder.getOrCreate()

    # Read population data using the provided query
    population_data = spark.sql(population_query)

    profiles = {}

    # Calculate profile for each metric query
    for metric_name, metric_query in metric_queries.items():
        # Read metric data using the provided query
        metric_data = spark.sql(metric_query)

        # Perform overlap analysis between population and metric data
        overlap_data = population_data.join(metric_data, on='common_column', how='inner')

        # Calculate profile metrics (e.g. count, sum, average) for each population group
        profile_metrics = overlap_data.groupby('population_group').agg(
            {'metric_column': 'count', 'metric_column': 'sum', 'metric_column': 'avg'}
        )

        # Store profile metrics in dictionary with metric name as key
        profiles[metric_name] = profile_metrics

    return profiles


def export_to_csv(dataset, file_path):
    """
    Export a PySpark dataset to a CSV file.

    :param dataset: The dataset to be exported.
    :param file_path: The path of the CSV file to be saved.
    """
    # Create a SparkSession
    spark = SparkSession.builder.getOrCreate()

    # Write the dataset to the CSV file
    dataset.write.csv(file_path, header=True, mode="overwrite")


def export_to_database(dataset, database_url, table_name):
    """
    Export a dataset to a database.

    :param dataset: The dataset to be exported.
    :param database_url: The URL or connection string of the target database.
    :param table_name: The name of the table in the database.
    """
    # Write the dataset to the database
    dataset.write.format("jdbc").options(
        url=database_url,
        dbtable=table_name,
        driver="com.mysql.jdbc.Driver"
        # Add any other required options for the specific database
    ).mode("overwrite").save()


def filter_dataset(dataset, conditions):
    """
    Filter a dataset based on specified conditions.

    :param dataset: The dataset to be filtered.
    :param conditions: A list of conditions to filter the dataset.
    :return: The filtered dataset.
    """
    
   # Apply each condition to the dataset
    for condition in conditions:
        dataset = dataset.filter(condition)

    return dataset


def group_datasets(df, group_columns):
    """
    Group datasets based on specified columns.

    :param df: Input DataFrame to be grouped.
    :param group_columns: List of column names to group by.
    :return: Grouped DataFrame.
    """
    # Create a SparkSession if not already exists
    spark = SparkSession.builder.getOrCreate()

    # Group the DataFrame by specified columns
    grouped_df = df.groupby(group_columns)

    # Return the grouped DataFrame
    return grouped_df


def aggregate_data(df, group_col, metric_cols):
    """
    Aggregate data within groups based on specified metrics.

    :param df: Input DataFrame.
    :param group_col: Column(s) used for grouping.
    :param metric_cols: List of columns to aggregate.
    :return: Aggregated DataFrame.
    """
    
   # Group by the specified column(s)
  grouped_df = df.groupBy(group_col)

  # Aggregate the specified metrics
  aggregated_df = grouped_df.agg(*[F.col(metric_col).alias(f"agg_{metric_col}") for metric_col in metric_cols])

  return aggregated_df


def merge_datasets(dataset1, dataset2, common_columns):
  
  """
  Merge datasets based on common columns.

  :param dataset1: The first dataset to be merged.
  :param dataset2: The second dataset to be merged.
  :param common_columns: The common columns used for the merge.
  :return: The merged dataset.
  """
  
  spark = SparkSession.builder.getOrCreate()

  # Convert datasets to DataFrames if they are not already
  if not isinstance(dataset1, spark.sql.DataFrame):
      dataset1 = spark.createDataFrame(dataset1)
  if not isinstance(dataset2, spark.sql.DataFrame):
      dataset2 = spark.createDataFrame(dataset2)

  # Perform the merge
  merged_dataset = dataset1.join(dataset2, on=common_columns, how='inner')

  return merged_dataset


def sort_dataset(dataset, columns):
    """
    Sorts a dataset based on specified columns.

    :param dataset: The dataset to be sorted.
    :param columns: The names of the columns to sort by.
    :return: The sorted dataset.
    """
    return dataset.sort(*columns)


def combine_populations(populations):
    """
    Combine multiple populations into one.

    :param populations: List of population DataFrames.
    :return: Combined population DataFrame.
    """
    spark = SparkSession.builder.getOrCreate()

    combined_population = spark.createDataFrame([], populations[0].schema)

    for population in populations:
        combined_population = combined_population.unionAll(population)

    return combined_population


def calculate_overlap(population1, population2):
  
  """
  Calculate the overlap between two populations.

  :param population1: DataFrame representing the first population.
  :param population2: DataFrame representing the second population.
  :return: DataFrame containing the overlap between the two populations.
  """
  
  # Calculate the overlap between two populations
  overlap = population1.intersect(population2)

  return overlap


def filter_population(population, criteria):
  
   """
   Filter a population based on specific criteria.

   :param population: DataFrame representing the population.
   :param criteria: SQL query string representing the filtering criteria.
   :return: A new DataFrame with the filtered population.
   """
   
   # Create a Spark session
   spark = SparkSession.builder.getOrCreate()

   # Register the population DataFrame as a temporary view
   population.createOrReplaceTempView("population")

   # Apply the filtering criteria using SQL query
   filtered_population = spark.sql(f"SELECT * FROM population WHERE {criteria}")

   return filtered_population


def aggregate_profiles(populations, metrics):
    """
    Aggregate the profiles of multiple populations into one summary profile.

    :param populations: List of population DataFrames.
    :param metrics: List of metrics to be included in the summary profile.
    :return: DataFrame representing the aggregated summary profile.
    """
    spark = SparkSession.builder.getOrCreate()

    # Create an empty DataFrame to store the summary profile
    summary_profile = spark.createDataFrame([], metrics)

    # Iterate over each population and calculate its profile
    for population in populations:
        # Execute the SQL query to get the profile of the population
        population_profile = spark.sql(population.sql_query)

        # Add the population profile to the summary profile
        summary_profile = summary_profile.unionAll(population_profile)

    # Calculate the aggregated summary profile by grouping by all metrics columns and calculating their average
    aggregated_summary_profile = summary_profile.groupBy(metrics).avg()

    return aggregated_summary_profile


import matplotlib.pyplot as plt
import pandas as pd

def visualize_profiles(population1_df, population2_df, metrics):
    
   """
   Visualize the profiles of different populations.

   :param population1_df: DataFrame representing the first population.
   :param population2_df: DataFrame representing the second population.
   :param metrics: List of metrics to be analyzed.
   """

  # Convert pyspark DataFrames to pandas DataFrames for easier visualization
  population1_pd = population1_df.toPandas()
  population2_pd = population2_df.toPandas()

  # Create a figure and axis objects
  fig, ax = plt.subplots(len(metrics), 1, figsize=(8, 6))

  # Iterate over each metric
  for i, metric in enumerate(metrics):
      # Get the values for the metric from both populations
      metric_values_population1 = population1_pd[metric]
      metric_values_population2 = population2_pd[metric]

      # Plot histogram for each population's metric values
      ax[i].hist(metric_values_population1, alpha=0.5, label='Population 1')
      ax[i].hist(metric_values_population2, alpha=0.5, label='Population 2')

      # Set plot title and labels
      ax[i].set_title(f'Profile of "{metric}"')
      ax[i].set_xlabel('Metric Value')
      ax[i].set_ylabel('Frequency')

      # Add legend to differentiate populations
      ax[i].legend()

  # Adjust spacing between subplots
  fig.tight_layout()

  # Show the plot
  plt.show()


def load_data(df):
    """
    Load data from a PySpark DataFrame.

    :param df: PySpark DataFrame object.
    :return: List of dictionaries representing the data rows.
    """
    data = []
    columns = df.columns
    rows = df.collect()

    for row in rows:
        data.append({column: row[column] for column in columns})

    return dat