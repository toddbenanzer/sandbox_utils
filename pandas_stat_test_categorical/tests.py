from .data_frame_handler import DataFrameHandler
from .descriptive_statistics import DescriptiveStatistics
from .exceptions import DataFrameValidationError
from .statistical_tests import StatisticalTests
from mypackage.data_loader import load_dataframe
from mypackage.logging_config import configure_logging
from mypackage.output_manager import OutputManager
from mypackage.output_manager import save_summary
from pandas.testing import assert_frame_equal
from pandas.testing import assert_series_equal, assert_frame_equal
import logging
import pandas as pd
import pytest


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'Category1': pd.Categorical(['A', 'B', 'A', None, 'B', 'A']),
        'Category2': pd.Categorical(['X', 'Y', 'X', 'Y', None, float('inf')]),
        'NonCategorical': [1, 2, 3, 4, 5, 6],
    })

def test_init_with_valid_dataframe(sample_dataframe):
    handler = DataFrameHandler(sample_dataframe, ['Category1', 'Category2'])
    assert handler.dataframe is sample_dataframe
    assert handler.columns == ['Category1', 'Category2']

def test_init_raises_error_for_missing_column(sample_dataframe):
    with pytest.raises(DataFrameValidationError, match="Column 'Missing' not found in DataFrame."):
        DataFrameHandler(sample_dataframe, ['Category1', 'Missing'])

def test_init_raises_error_for_empty_dataframe():
    empty_df = pd.DataFrame()
    with pytest.raises(DataFrameValidationError, match="DataFrame is empty."):
        DataFrameHandler(empty_df, [])

def test_init_raises_error_for_non_categorical_column(sample_dataframe):
    with pytest.raises(DataFrameValidationError, match="Column 'NonCategorical' is not of categorical type."):
        DataFrameHandler(sample_dataframe, ['NonCategorical'])

def test_handle_missing_infinite_values(sample_dataframe):
    expected_data = {
        'Category1': pd.Categorical(['A', 'B', 'A', 'A', 'B']),
        'Category2': pd.Categorical(['X', 'Y', 'X', 'Y', 'X']),
    }
    expected_df = pd.DataFrame(expected_data)
    
    handler = DataFrameHandler(sample_dataframe, ['Category1', 'Category2'])
    result_df = handler.handle_missing_infinite_values()

    assert_frame_equal(result_df, expected_df)



@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'Category1': pd.Categorical(['A', 'B', 'A', 'C', 'B']),
        'Category2': pd.Categorical(['X', 'Y', 'X', 'Y', 'Z']),
        'NonCategorical': [1, 2, 3, 4, 5],
    })

def test_init_with_valid_dataframe(sample_dataframe):
    stats = DescriptiveStatistics(sample_dataframe, ['Category1', 'Category2'])
    assert stats.dataframe is sample_dataframe
    assert stats.columns == ['Category1', 'Category2']

def test_init_raises_error_for_missing_column(sample_dataframe):
    with pytest.raises(ValueError, match="Column 'Missing' not found in DataFrame."):
        DescriptiveStatistics(sample_dataframe, ['Category1', 'Missing'])

def test_init_raises_error_for_non_categorical_column(sample_dataframe):
    with pytest.raises(ValueError, match="Column 'NonCategorical' is not of categorical type."):
        DescriptiveStatistics(sample_dataframe, ['NonCategorical'])

def test_calculate_frequencies(sample_dataframe):
    stats = DescriptiveStatistics(sample_dataframe, ['Category1', 'Category2'])
    frequencies = stats.calculate_frequencies()
    expected_frequencies_cat1 = pd.Series([2, 2, 1], index=['A', 'B', 'C'])
    expected_frequencies_cat2 = pd.Series([2, 2, 1], index=['X', 'Y', 'Z'])
    
    assert_series_equal(frequencies['Category1'], expected_frequencies_cat1, check_dtype=False)
    assert_series_equal(frequencies['Category2'], expected_frequencies_cat2, check_dtype=False)

def test_compute_mode(sample_dataframe):
    stats = DescriptiveStatistics(sample_dataframe, ['Category1', 'Category2'])
    modes = stats.compute_mode()
    expected_mode_cat1 = pd.Series(['A', 'B'], index=[0, 1])  # Two modes: A, B
    expected_mode_cat2 = pd.Series(['X', 'Y'], index=[0, 1])  # Two modes: X, Y
    
    assert_series_equal(modes['Category1'], expected_mode_cat1)
    assert_series_equal(modes['Category2'], expected_mode_cat2)

def test_generate_contingency_table(sample_dataframe):
    stats = DescriptiveStatistics(sample_dataframe, ['Category1', 'Category2'])
    contingency_table = stats.generate_contingency_table('Category1', 'Category2')
    expected_table = pd.DataFrame({
        'X': [2, 0, 0],
        'Y': [0, 1, 1],
        'Z': [0, 1, 0]
    }, index=['A', 'B', 'C'])
    expected_table.index.name = 'Category1'
    expected_table.columns.name = 'Category2'
    
    assert_frame_equal(contingency_table, expected_table)



@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'Category1': pd.Categorical(['A', 'A', 'B', 'B', 'C', 'C']),
        'Category2': pd.Categorical(['X', 'Y', 'X', 'Y', 'X', 'Z']),
    })

def test_init_with_valid_dataframe(sample_dataframe):
    tests = StatisticalTests(sample_dataframe, ['Category1', 'Category2'])
    assert tests.dataframe is sample_dataframe
    assert tests.columns == ['Category1', 'Category2']

def test_init_raises_error_for_missing_column(sample_dataframe):
    with pytest.raises(ValueError, match="Column 'Missing' not found in DataFrame."):
        StatisticalTests(sample_dataframe, ['Category1', 'Missing'])

def test_init_raises_error_for_non_categorical_column(sample_dataframe):
    sample_dataframe['Numeric'] = [1, 2, 3, 4, 5, 6]
    with pytest.raises(ValueError, match="Column 'Numeric' is not of categorical type."):
        StatisticalTests(sample_dataframe, ['Category1', 'Numeric'])

def test_perform_chi_squared_test(sample_dataframe):
    tests = StatisticalTests(sample_dataframe, ['Category1', 'Category2'])
    chi2, p, dof, expected = tests.perform_chi_squared_test('Category1', 'Category2')

    assert isinstance(chi2, float)
    assert isinstance(p, float)
    assert isinstance(dof, int)
    assert_frame_equal(expected, pd.DataFrame({
        'X': [1.5, 1.0, 1.5],
        'Y': [1.5, 1.0, 1.5],
        'Z': [0.0, 0.0, 1.0]
    }, index=['A', 'B', 'C']).rename_axis('Category1').rename_axis('Category2', axis=1))

def test_perform_fishers_exact_test():
    table = pd.DataFrame([[8, 2], [1, 5]], index=['Row1', 'Row2'], columns=['Col1', 'Col2'])
    tests = StatisticalTests(pd.DataFrame(), [])

    oddsratio, p_value = tests.perform_fishers_exact_test(table)

    assert isinstance(oddsratio, float)
    assert isinstance(p_value, float)

def test_fishers_exact_test_invalid_table():
    table = pd.DataFrame([[8, 2, 3], [1, 5, 2]])
    tests = StatisticalTests(pd.DataFrame(), [])

    with pytest.raises(ValueError, match="Fisher's exact test is only applicable to 2x2 contingency tables."):
        tests.perform_fishers_exact_test(table)



@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'Category': ['A', 'B', 'C'],
        'Values': [10, 20, 30]
    })

def test_export_to_csv_valid(sample_dataframe, tmp_path):
    output_manager = OutputManager(sample_dataframe)
    file_path = tmp_path / "output.csv"
    
    output_manager.export_to_csv(file_path)
    result_df = pd.read_csv(file_path)

    assert result_df.equals(sample_dataframe)

def test_export_to_csv_invalid_results():
    output_manager = OutputManager("Invalid Data")
    with pytest.raises(ValueError, match="Results must be a DataFrame to export to CSV."):
        output_manager.export_to_csv("output.csv")

def test_export_to_excel_valid(sample_dataframe, tmp_path):
    output_manager = OutputManager(sample_dataframe)
    file_path = tmp_path / "output.xlsx"
    
    output_manager.export_to_excel(file_path)
    result_df = pd.read_excel(file_path)

    assert result_df.equals(sample_dataframe)

def test_export_to_excel_invalid_results():
    output_manager = OutputManager("Invalid Data")
    with pytest.raises(ValueError, match="Results must be a DataFrame to export to Excel."):
        output_manager.export_to_excel("output.xlsx")

def test_generate_visualization_valid_chart(sample_dataframe, mocker):
    output_manager = OutputManager(sample_dataframe)
    mocker.patch("matplotlib.pyplot.show")

    output_manager.generate_visualization('bar')  # Should not raise an exception
    output_manager.generate_visualization('pie')  # Should not raise an exception

def test_generate_visualization_invalid_results():
    output_manager = OutputManager("Invalid Data")
    with pytest.raises(ValueError, match="Results must be a DataFrame to generate visualizations."):
        output_manager.generate_visualization('bar')

def test_generate_visualization_invalid_chart_type(sample_dataframe):
    output_manager = OutputManager(sample_dataframe)
    with pytest.raises(ValueError, match="Unsupported chart type: line"):
        output_manager.generate_visualization('line')



def test_load_csv_file(tmp_path):
    # Create a sample CSV file
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("A,B,C\n1,2,3\n4,5,6")

    # Load the DataFrame
    df = load_dataframe(str(csv_file))

    # Expected DataFrame
    expected_df = pd.DataFrame({
        "A": [1, 4],
        "B": [2, 5],
        "C": [3, 6]
    })

    # Assert the DataFrame is loaded correctly
    pd.testing.assert_frame_equal(df, expected_df)

def test_load_excel_file(tmp_path):
    # Create a sample Excel file
    excel_file = tmp_path / "data.xlsx"
    sample_df = pd.DataFrame({
        "A": [1, 4],
        "B": [2, 5],
        "C": [3, 6]
    })
    sample_df.to_excel(excel_file, index=False)

    # Load the DataFrame
    df = load_dataframe(str(excel_file))

    # Assert the DataFrame is loaded correctly
    pd.testing.assert_frame_equal(df, sample_df)

def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_dataframe("non_existent_file.csv")

def test_unsupported_file_format(tmp_path):
    unsupported_file = tmp_path / "data.txt"
    unsupported_file.write_text("Some content")

    with pytest.raises(ValueError, match="Unsupported file format: txt"):
        load_dataframe(str(unsupported_file))

def test_invalid_csv_content(tmp_path):
    invalid_csv_file = tmp_path / "invalid_data.csv"
    invalid_csv_file.write_text("Some text that is not CSV format")

    with pytest.raises(ValueError):
        load_dataframe(str(invalid_csv_file))



@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        "Column1": [1, 2, 3],
        "Column2": [4, 5, 6]
    })

def test_save_summary_csv(tmp_path, sample_dataframe):
    file_path = tmp_path / "summary.csv"
    
    save_summary(sample_dataframe, str(file_path), 'csv')
    
    # Load the CSV file back to a DataFrame and compare
    loaded_df = pd.read_csv(file_path)
    pd.testing.assert_frame_equal(loaded_df, sample_dataframe)

def test_save_summary_excel(tmp_path, sample_dataframe):
    file_path = tmp_path / "summary.xlsx"
    
    save_summary(sample_dataframe, str(file_path), 'excel')
    
    # Load the Excel file back to a DataFrame and compare
    loaded_df = pd.read_excel(file_path)
    pd.testing.assert_frame_equal(loaded_df, sample_dataframe)

def test_save_summary_unsupported_format(sample_dataframe, tmp_path):
    file_path = tmp_path / "summary.txt"
    
    with pytest.raises(ValueError, match="Unsupported format: txt"):
        save_summary(sample_dataframe, str(file_path), 'txt')

def test_save_summary_invalid_results(tmp_path):
    file_path = tmp_path / "summary.csv"
    
    with pytest.raises(ValueError, match="Results are not a DataFrame and cannot be saved as CSV."):
        save_summary({"not": "a dataframe"}, str(file_path), 'csv')

def test_save_summary_case_insensitive_format(tmp_path, sample_dataframe):
    file_path = tmp_path / "summary.csv"
    
    # Test with uppercase format
    save_summary(sample_dataframe, str(file_path), 'CSV')
    
    # Load the CSV file back to a DataFrame and compare
    loaded_df = pd.read_csv(file_path)
    pd.testing.assert_frame_equal(loaded_df, sample_dataframe)



def test_configure_logging_valid_levels(mocker):
    # Mock logging.basicConfig to prevent actual configuration
    mock_basic_config = mocker.patch('logging.basicConfig')

    # Test valid log levels
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    for level in valid_levels:
        configure_logging(level)
        mock_basic_config.assert_called_with(
            level=getattr(logging, level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

def test_configure_logging_invalid_level():
    with pytest.raises(ValueError, match="Invalid log level: 'INVALID'. Must be one of "):
        configure_logging('INVALID')

def test_configure_logging_case_insensitivity(mocker):
    # Mock logging.basicConfig to prevent actual configuration
    mock_basic_config = mocker.patch('logging.basicConfig')

    # Test with lowercase valid level
    configure_logging('debug')
    mock_basic_config.assert_called_with(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
