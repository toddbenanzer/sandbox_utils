from io import StringIO
from load_data_module import load_data  # Assuming the function is in a module named `load_data_module`
from overlap_analyzer import OverlapAnalyzer  # Assuming this module path
from population_manager import PopulationManager  # Assume this is the import path
from population_manager import PopulationManager  # Assuming this module path
from profile_calculator import ProfileCalculator  # Assuming this module path
from pyspark.sql import SparkSession
from pyspark.sql.utils import AnalysisException
from save_results_module import save_results  # Assuming the function is in a module named `save_results_module`
from setup_logging_module import setup_logging  # Assuming the function is in a module named `setup_logging_module`
from validate_sql_query_module import validate_sql_query  # Assuming the function is in a module named `validate_sql_query_module`
import logging
import os
import pytest


@pytest.fixture(scope="module")
def spark():
    spark_session = SparkSession.builder.master("local").appName("Test").getOrCreate()
    data = [("Alice", 1), ("Bob", 2), ("Cathy", 3)]
    schema = ["name", "id"]
    df = spark_session.createDataFrame(data, schema)
    df.createOrReplaceTempView("people")
    yield spark_session
    spark_session.stop()

@pytest.fixture
def pop_manager(spark):
    return PopulationManager(spark)

def test_define_population_success(pop_manager):
    pop_manager.define_population("test_pop", "SELECT * FROM people")
    result = pop_manager.get_population("test_pop")
    assert result.count() == 3
    assert set(result.columns) == {"name", "id"}

def test_define_population_duplicate(pop_manager):
    pop_manager.define_population("duplicate_pop", "SELECT * FROM people")
    with pytest.raises(ValueError, match="Population 'duplicate_pop' is already defined."):
        pop_manager.define_population("duplicate_pop", "SELECT * FROM people")

def test_define_population_invalid_sql(pop_manager):
    with pytest.raises(ValueError, match="Failed to define population 'invalid_pop'"):
        pop_manager.define_population("invalid_pop", "SELECT * FROM non_existing_table")

def test_get_population_not_found(pop_manager):
    with pytest.raises(ValueError, match="Population 'not_defined' not found."):
        pop_manager.get_population("not_defined")

def test_get_population_success(pop_manager):
    pop_manager.define_population("valid_pop", "SELECT name FROM people")
    result = pop_manager.get_population("valid_pop")
    assert result.count() == 3
    assert "name" in result.columns
    assert "id" not in result.columns



@pytest.fixture(scope="module")
def spark():
    spark_session = SparkSession.builder.master("local").appName("Test").getOrCreate()
    yield spark_session
    spark_session.stop()

@pytest.fixture
def pop_manager(spark):
    manager = PopulationManager(spark)
    data1 = [("Alice", 1), ("Bob", 2), ("Cathy", 3)]
    data2 = [("Alice", 1), ("David", 4), ("Eve", 5)]
    schema = ["name", "id"]
    df1 = spark.createDataFrame(data1, schema)
    df2 = spark.createDataFrame(data2, schema)
    df1.createOrReplaceTempView("people1")
    df2.createOrReplaceTempView("people2")
    manager.define_population("pop1", "SELECT * FROM people1")
    manager.define_population("pop2", "SELECT * FROM people2")
    return manager

@pytest.fixture
def overlap_analyzer(pop_manager):
    return OverlapAnalyzer(pop_manager)

def test_analyze_overlap_success(overlap_analyzer):
    result_df = overlap_analyzer.analyze_overlap(["pop1", "pop2"])
    assert result_df.count() == 1
    assert set(result_df.columns) == {"name", "id"}

def test_analyze_overlap_insufficient_populations(overlap_analyzer):
    with pytest.raises(ValueError, match="At least two populations must be specified for overlap analysis."):
        overlap_analyzer.analyze_overlap(["pop1"])

def test_analyze_overlap_nonexistent_population(overlap_analyzer):
    with pytest.raises(ValueError, match="Error retrieving populations: Population 'pop3' not found."):
        overlap_analyzer.analyze_overlap(["pop1", "pop3"])



@pytest.fixture(scope="module")
def spark():
    spark_session = SparkSession.builder.master("local").appName("Test").getOrCreate()
    yield spark_session
    spark_session.stop()

@pytest.fixture
def profile_calculator(spark):
    calculator = ProfileCalculator(spark)
    # Define a mock population
    data = [("Alice", 1), ("Bob", 2), ("Cathy", 3)]
    schema = ["name", "id"]
    df = spark.createDataFrame(data, schema)
    df.createOrReplaceTempView("population")

    # Define some metrics
    metric_data_1 = [(1, 50), (2, 60), (3, 70)]
    metric_data_2 = [(1, 85), (2, 95), (3, 75)]
    metric_schema = ["id", "metric_value"]
    metric_df1 = spark.createDataFrame(metric_data_1, metric_schema)
    metric_df2 = spark.createDataFrame(metric_data_2, metric_schema)
    metric_df1.createOrReplaceTempView("metric1")
    metric_df2.createOrReplaceTempView("metric2")

    calculator.define_metrics("metric1", "SELECT * FROM metric1")
    calculator.define_metrics("metric2", "SELECT * FROM metric2")
    return calculator

def test_define_metrics_duplicate(profile_calculator):
    with pytest.raises(ValueError, match="Metric 'metric1' is already defined."):
        profile_calculator.define_metrics("metric1", "SELECT * FROM metric1")

def test_calculate_profiles_success(profile_calculator):
    result_df = profile_calculator.calculate_profiles("population", ["metric1", "metric2"])
    assert result_df.count() == 3
    assert set(result_df.columns) == {"name", "id", "metric_value"}

def test_calculate_profiles_missing_population(profile_calculator):
    with pytest.raises(ValueError, match="Population 'unknown_population' not found or could not be accessed."):
        profile_calculator.calculate_profiles("unknown_population", ["metric1", "metric2"])

def test_calculate_profiles_missing_metric(profile_calculator):
    with pytest.raises(ValueError, match="Metric 'nonexistent_metric' not found."):
        profile_calculator.calculate_profiles("population", ["metric1", "nonexistent_metric"])



@pytest.fixture(scope="module")
def spark():
    spark_session = SparkSession.builder.master("local").appName("TestLoadData").getOrCreate()
    yield spark_session
    spark_session.stop()

def test_load_data_csv(spark, tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("name,age\nAlice,30\nBob,25\n")
    
    df = load_data(spark, str(csv_file), "csv")
    assert df.count() == 2
    assert "name" in df.columns
    assert "age" in df.columns

def test_load_data_json(spark, tmp_path):
    json_file = tmp_path / "data.json"
    json_file.write_text('[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]')
    
    df = load_data(spark, str(json_file), "json")
    assert df.count() == 2

def test_load_data_parquet(spark, tmp_path):
    parquet_file = tmp_path / "data.parquet"
    data = [("Alice", 30), ("Bob", 25)]
    schema = ["name", "age"]
    df = spark.createDataFrame(data, schema)
    df.write.parquet(str(parquet_file))
    
    loaded_df = load_data(spark, str(parquet_file), "parquet")
    assert loaded_df.count() == 2

def test_load_data_unsupported_format(spark):
    with pytest.raises(ValueError, match="Unsupported file format: txt"):
        load_data(spark, "path/to/data.txt", "txt")

def test_load_data_file_not_found(spark):
    with pytest.raises(IOError, match="Failed to load data from"):
        load_data(spark, "path/to/nonexistent.csv", "csv")



@pytest.fixture(scope="module")
def spark():
    spark_session = SparkSession.builder.master("local").appName("TestSaveResults").getOrCreate()
    yield spark_session
    spark_session.stop()

@pytest.fixture
def sample_dataframe(spark):
    data = [("Alice", 1), ("Bob", 2)]
    schema = ["name", "id"]
    return spark.createDataFrame(data, schema)

def test_save_results_csv(sample_dataframe, tmp_path):
    csv_path = tmp_path / "output.csv"
    save_results(sample_dataframe, str(csv_path), "csv")
    assert csv_path.exists()

def test_save_results_json(sample_dataframe, tmp_path):
    json_path = tmp_path / "output.json"
    save_results(sample_dataframe, str(json_path), "json")
    assert json_path.exists()

def test_save_results_parquet(sample_dataframe, tmp_path):
    parquet_path = tmp_path / "output.parquet"
    save_results(sample_dataframe, str(parquet_path), "parquet")
    assert parquet_path.exists()

def test_save_results_unsupported_format(sample_dataframe):
    with pytest.raises(ValueError, match="Unsupported file format: txt"):
        save_results(sample_dataframe, "path/to/output.txt", "txt")

def test_save_results_invalid_path(sample_dataframe):
    with pytest.raises(IOError, match="Failed to save DataFrame to"):
        save_results(sample_dataframe, "/nonexistent_path/output.csv", "csv")



@pytest.fixture(scope="module")
def spark():
    spark_session = SparkSession.builder.master("local").appName("TestValidateSQL").getOrCreate()
    yield spark_session
    spark_session.stop()

def test_validate_sql_query_valid(spark):
    valid_query = "SELECT * FROM range(10)"
    assert validate_sql_query(spark, valid_query) is True

def test_validate_sql_query_invalid(spark):
    invalid_query = "SELECTE * FROM range(10)"  # Typo in SELECT keyword
    assert validate_sql_query(spark, invalid_query) is False

def test_validate_sql_query_empty_string(spark):
    with pytest.raises(ValueError, match="The SQL query must be a non-empty string."):
        validate_sql_query(spark, "")

def test_validate_sql_query_null_string(spark):
    with pytest.raises(ValueError, match="The SQL query must be a non-empty string."):
        validate_sql_query(spark, None)

def test_validate_sql_query_syntax_error(spark):
    syntax_error_query = "SELECT * FROM"
    assert validate_sql_query(spark, syntax_error_query) is False



@pytest.fixture
def log_capture():
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    yield log_stream
    root_logger.removeHandler(handler)

def test_setup_logging_debug(log_capture):
    setup_logging("DEBUG")
    logging.debug("Test debug message")
    log_contents = log_capture.getvalue()
    assert "DEBUG" in log_contents
    assert "Test debug message" in log_contents

def test_setup_logging_info(log_capture):
    setup_logging("INFO")
    logging.debug("This should not appear")
    logging.info("Test info message")
    log_contents = log_capture.getvalue()
    assert "This should not appear" not in log_contents
    assert "INFO" in log_contents
    assert "Test info message" in log_contents

def test_setup_logging_invalid():
    with pytest.raises(ValueError, match="Invalid logging level: INVALID"):
        setup_logging("INVALID")
