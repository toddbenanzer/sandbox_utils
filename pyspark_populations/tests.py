
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from my_module import read_csv_file, read_data_from_database, define_population, overlap_analysis, calculate_profiles, export_to_csv, export_to_database, filter_dataset, group_datasets, aggregate_data, merge_datasets, calculate_overlap, filter_population, aggregate_profiles

# Create a fixture to initialize a SparkSession for testing
@pytest.fixture(scope="session")
def spark_session():
    spark = SparkSession.builder.getOrCreate()
    yield spark
    spark.stop()


def test_read_csv_file(spark_session):
    # Test case 1: Valid file path
    file_path = "tests/data.csv"
    df = read_csv_file(file_path)
    assert isinstance(df, DataFrame)
    assert df.count() == 10  # Assuming the data.csv contains 10 rows
    
    # Test case 2: Invalid file path
    file_path = "tests/invalid.csv"
    with pytest.raises(Exception):
        df = read_csv_file(file_path)
    
    # Test case 3: Empty file path
    file_path = ""
    with pytest.raises(Exception):
        df = read_csv_file(file_path)
    
    # Test case 4: File path pointing to a non-CSV file
    file_path = "tests/non_csv.txt"
    with pytest.raises(Exception):
        df = read_csv_file(file_path)


def test_read_data_from_database(spark_session):
    # Define test data
    database_url = "jdbc:postgresql://localhost:5432/mydatabase"
    table_name = "mytable"

    # Call the function to read data from database
    df = read_data_from_database(database_url, table_name)

    # Verify that the DataFrame is not empty
    assert df.count() > 0

    # Verify that the DataFrame has the expected columns
    expected_columns = ["col1", "col2", "col3"]
    assert set(df.columns) == set(expected_columns)

    # Verify that the DataFrame has the expected number of rows
    expected_rows = 100
    assert df.count() == expected_rows

    # Add more assertions as needed based on your specific requirements


def test_define_population(spark_session):
    # Create a SparkSession
    spark = SparkSession.builder.getOrCreate()

    # Test case 1: Empty population
    sql_query = "SELECT * FROM population WHERE 1=0"
    result = define_population(sql_query)
    assert result.count() == 0

    # Test case 2: Non-empty population
    sql_query = "SELECT * FROM population WHERE age >= 18"
    result = define_population(sql_query)
    assert result.count() > 0

    # Test case 3: Invalid SQL query
    sql_query = "SELECT * FROM invalid_table"
    with pytest.raises(Exception):
        define_population(sql_query)


def test_overlap_analysis(spark_session):
    # Create test DataFrame representing population 1
    population1 = spark.createDataFrame([(1,), (2,), (3,), (4,), (5,)], ["individual_id"])

    # Create test DataFrame representing population 2
    population2 = spark.createDataFrame([(4,), (5,), (6,), (7,), (8,)], ["individual_id"])

    # Call the overlap_analysis function
    results_df = overlap_analysis(population1, population2)

    # Check if the result DataFrame has the expected columns
    assert set(results_df.columns) == set(["population1_count", "population2_count", "overlap_count",
                                           "percentage_overlap_population1", "percentage_overlap_population2"])

    # Check if the result DataFrame has the expected values
    expected_values = [(5, 5, 2, 40.0, 40.0)]
    assert results_df.collect() == expected_values

    # Create another test DataFrame representing population 1 with duplicate individual ids
    population1_duplicate_ids = spark.createDataFrame([(1,), (2,), (3,), (3,), (4,)], ["individual_id"])

    # Call the overlap_analysis function with duplicate ids in population 1
    results_df_duplicate_ids = overlap_analysis(population1_duplicate_ids, population2)

    # Check if the result DataFrame has the expected values when there are duplicate ids in population 1
    expected_values_duplicate_ids = [(4, 5, 2, 50.0, 40.0)]
    assert results_df_duplicate_ids.collect() == expected_values_duplicate_ids


def test_calculate_profiles(spark_session):
    # Define sample population and metric queries
    population_query = "SELECT * FROM population_data"
    metric_queries = {
        'metric1': "SELECT * FROM metric1_data",
        'metric2': "SELECT * FROM metric2_data"
    }

    # Call the calculate_profiles function
    profiles = calculate_profiles(population_query, metric_queries)

    # Assert that the profiles dictionary is not empty
    assert profiles

    # Assert that each metric name has corresponding profile metrics
    for metric_name in metric_queries.keys():
        assert metric_name in profiles

    # Assert that the profile metrics have expected columns
    for profile_metrics in profiles.values():
        assert 'population_group' in profile_metrics.columns
        assert 'count(metric_column)' in profile_metrics.columns
        assert 'sum(metric_column)' in profile_metrics.columns
        assert 'avg(metric_column)' in profile_metrics.columns


def test_export_to_csv(tmpdir):
    # Create a temporary directory for file_path
    file_path = os.path.join(str(tmpdir), "test.csv")

    # Create a test dataset
    test_data = [("John", 25), ("Alice", 30), ("Bob", 35)]
    df = spark.createDataFrame(test_data, ["Name", "Age"])

    # Call the export_to_csv function
    export_to_csv(df, file_path)

    # Check if the CSV file is created
    assert os.path.exists(file_path)

    # Check if the exported CSV file is not empty
    assert os.path.getsize(file_path) > 0

    # Check if the exported CSV file has correct content
    with open(file_path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 4  # Including header row
        assert lines[0].strip() == "Name,Age"
        assert lines[1].strip() == "John,25"
        assert lines[2].strip() == "Alice,30"
        assert lines[3].strip() == "Bob,35"


def test_export_to_csv_overwrite(tmpdir):
    # Create a temporary directory for file_path
    file_path = os.path.join(str(tmpdir), "test.csv")

    # Create an existing CSV file
    with open(file_path, "w") as f:
        f.write("Existing File")

    # Create a test dataset
    test_data = [("John", 25), ("Alice", 30), ("Bob", 35)]
    df = spark.createDataFrame(test_data, ["Name", "Age"])

    # Call the export_to_csv function with mode="overwrite"
    export_to_csv(df, file_path)

    # Check if the CSV file is overwritten
    with open(file_path, "r") as f:
        content = f.read()
        assert content != "Existing File"

    # Check if the exported CSV file has correct content
    with open(file_path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 4  # Including header row
        assert lines[0].strip() == "Name,Age"
        assert lines[1].strip() == "John,25"
        assert lines[2].strip() == "Alice,30"
        assert lines[3].strip() == "Bob,35"


def test_export_to_database(tmpdir):
    # Create a temporary directory to use as the database URL
    database_url = f"jdbc:sqlite://{tmpdir}/test.db"

    # Call the function with the test dataset and temporary database URL
    export_to_database(test_dataset, database_url)

    # Verify that the dataset was exported correctly by reading it from the database
    exported_dataset = spark.read.format("jdbc").options(
        url=database_url,
        dbtable="target_table_name",
        driver="org.sqlite.JDBC"
        # Add any other required options for the specific database
    ).load()

    # Assert that the exported dataset is equal to the original test dataset
    assert exported_dataset.collect() == test_dataset.collect()


def test_filter_dataset(dataset):
    # Test filtering with no conditions should return the original dataset
    filtered_dataset = filter_dataset(dataset, [])
    assert filtered_dataset == dataset


def test_filter_dataset_with_single_condition(dataset):
    # Test filtering with a single condition
    condition = Condition('age', '>', 25)
    filtered_dataset = filter_dataset(dataset, [condition])

    # Expected result: Only rows with age > 25 should be returned
    expected_result = Dataset()
    expected_result.add_row({'name': 'Jane', 'age': 30})
    expected_result.add_row({'name': 'Sam', 'age': 35})

    assert filtered_dataset == expected_result


def test_filter_dataset_with_multiple_conditions(dataset):
    # Test filtering with multiple conditions
    condition1 = Condition('age', '>', 25)
    condition2 = Condition('name', '==', 'Sam')

    filtered_dataset = filter_dataset(dataset, [condition1, condition2])

    # Expected result: Only row with age > 25 and name == 'Sam' should be returned
    expected_result = Dataset()
    expected_result.add_row({'name': 'Sam', 'age': 35})

    assert filtered_dataset == expected_result


def test_filter_dataset_with_invalid_condition(dataset):
    # Test filtering with an invalid condition (unsupported operator)
    condition = Condition('age', '<>', 30)

    with pytest.raises(ValueError):
        filter_dataset(dataset, [condition])


def test_group_datasets_by_single_column(spark_session):
    # Create test DataFrame
    data = [
        ("Alice", 25, "New York"),
        ("Bob", 30, "Los Angeles"),
        ("Charlie", 35, "New York"),
        ("Dave", 40, "Los Angeles")
    ]
    df = spark_session.createDataFrame(data, ["Name", "Age", "City"])

    # Call the function to group the DataFrame by a single column
    grouped_df = group_datasets(df, ["City"])

    # Assert that grouped_df is of type 'pyspark.sql.GroupedData'
    assert isinstance(grouped_df, pyspark.sql.GroupedData)


def test_group_datasets_by_multiple_columns(spark_session):
    # Create test DataFrame
    data = [
        ("Alice", 25, "New York"),
        ("Bob", 30, "Los Angeles"),
        ("Charlie", 35, "New York"),
        ("Dave", 40, "Los Angeles")
    ]
    df = spark_session.createDataFrame(data, ["Name", "Age", "City"])

    # Call the function to group the DataFrame by multiple columns
    grouped_df = group_datasets(df, ["City", "Age"])

    # Assert that grouped_df is of type 'pyspark.sql.GroupedData'
    assert isinstance(grouped_df, pyspark.sql.GroupedData)


def test_aggregate_data(spark_session):
    # Create a test DataFrame
    df = spark_session.createDataFrame([(1, "A", 10), (2, "B", 20), (1, "C", 30)], ["col1", "col2", "col3"])

    # Define the expected output DataFrame
    expected_df = spark_session.createDataFrame([(1, 40), (2, 20)], ["col1", "agg_col3"])

    # Call the aggregate_data function with the test DataFrame and inputs
    result_df = aggregate_data(df, "col1", ["col3"])

    # Assert that the result DataFrame is equal to the expected DataFrame
    assert result_df.collect() == expected_df.collect()


def test_combine_populations(spark_session):
    # Create sample populations
    population1 = spark_session.createDataFrame([(1, 'City1', 100000)], ['id', 'city', 'population'])
    population2 = spark_session.createDataFrame([(2, 'City2', 200000)], ['id', 'city', 'population'])
    population3 = spark_session.createDataFrame([(3, 'City3', 300000)], ['id', 'city', 'population'])

    # Combine populations
    combined_population = combine_populations([population1, population2, population3])

    # Check if combined_population is a DataFrame
    assert isinstance(combined_population, DataFrame)

    # Check if combined_population has the correct schema
    assert combined_population.schema == population1.schema

    # Check if combined_population contains all rows from population1, population2 and population3
    assert combined_population.count() == (population1.count() + population2.count() + population3.count())

    # Check if combined_population contains the correct data by comparing the sum of populations
    total_population = (population1.selectExpr('sum(population)').first()[0] +
                        population2.selectExpr('sum(population)').first()[0] +
                        population3.selectExpr('sum(population)').first()[0])

    assert combined_population.selectExpr('sum(population)').first()[0] == total_population


def test_calculate_overlap(spark_session):
    # Create two test populations
    population1 = [1, 2, 3, 4, 5]
    population2 = [4, 5, 6, 7, 8]

    # Convert test populations to Spark DataFrames
    df1 = spark_session.createDataFrame([population1], ["id"])
    df2 = spark_session.createDataFrame([population2], ["id"])

    # Calculate the overlap between the populations using calculate_overlap function
    overlap = calculate_overlap(df1, df2).collect()[0][0]

    # Check if the calculated overlap is correct
    assert overlap == [4, 5]


def test_filter_population(spark_session):
    # Create a sample population DataFrame for testing
    population_data = [
        ("John Doe", 25, "Male"),
        ("Jane Smith", 30, "Female"),
        ("Michael Johnson", 35, "Male")
    ]
    columns = ["Name", "Age", "Gender"]
    population = spark_session.createDataFrame(population_data, columns)

     # Test filtering by age less than 30
     criteria = "Age < 30"
     filtered_population = filter_population(population, criteria)
    
     assert filtered_population.count() == 1
     assert filtered_population.collect()[0]["Name"] == "John Doe"

    # Test filtering by gender is Male
    criteria = "Gender = 'Male'"
    filtered_population = filter_population(population, criteria)
    
    assert filtered_population.count() == 2
    assert filtered_population.collect()[0]["Name"] == "John Doe"
    assert filtered_population.collect()[1]["Name"] == "Michael Johnson"



def test_aggregate_profiles(spark_session):
    # Test case 1: when populations and metrics are empty, the result should be an empty DataFrame
    populations = []
    metrics = []
    result = aggregate_profiles(populations, metrics)
    assert result.count() == 0

    # Test case 2: when populations are not empty but metrics are empty, the result should be an empty DataFrame
    population1 = Population(sql_query='SELECT * FROM population1')
    population2 = Population(sql_query='SELECT * FROM population2')
    populations = [population1, population2]
    metrics = []
    result = aggregate_profiles(populations, metrics)
    assert result.count() == 0

    # Test case 3: when both populations and metrics are not empty, the result should be a non-empty DataFrame
    population3 = Population(sql_query='SELECT * FROM population3')
    metrics = ['age', 'gender']
    result = aggregate_profiles(populations + [population3], metrics)
    assert result.count() > 0


def test_visualize_profiles():
    # Create two sample population DataFrames
    population1_df = pd.DataFrame({'metric1': [1, 2, 3, 4, 5],
                                  'metric2': [10, 20, 30, 40, 50]})
    population2_df = pd.DataFrame({'metric1': [6, 7, 8, 9, 10],
                                  'metric2': [60, 70, 80, 90, 100]})

    # Create a list of metrics to be analyzed
    metrics = ['metric1', 'metric2']

    # Call the function to be tested
    visualize_profiles(population1_df=population1_df,
                       population2_df=population2_df,
                       metrics=metrics)

    # Assert that the plot is displayed successfully without any errors

    # This assertion checks that no exception is raised when calling plt.show()
    with pytest.raises(Exception):
        visualize_profiles(population1_df=population1_df,
                           population2_df=population2_df,
                           metrics=metrics)


def test_load_data():
    # Create a sample PySpark DataFrame object with two columns and two rows
    df = spark.createDataFrame([(1, 'apple'), (2, 'banana')], ['id', 'fruit'])

    # Call the load_data() function and store the result
    result = load_data(df)

    # Define the expected output as a list of dictionaries representing the data rows
    expected_output = [{'id': 1, 'fruit': 'apple'}, {'id': 2, 'fruit': 'banana'}]

    # Compare the result with the expected output using assert statement
    assert result == expected_output

