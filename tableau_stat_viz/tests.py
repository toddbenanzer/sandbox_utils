from correlation_visualizer import CorrelationVisualizer
from data_handler import DataHandler
from distribution_visualizer import DistributionVisualizer
from matplotlib.figure import Figure
from pandas.util.testing import assert_frame_equal
from setup_logging import setup_logging
from setup_visualization_style import setup_visualization_style
from statistical_analyzer import StatisticalAnalyzer
from tableau_exporter import TableauExporter
import logging
import matplotlib.pyplot as plt
import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [5, 4, 3, 2, 1],
        'C': [2, 3, 4, 5, 6]
    })

def test_init_invalid_data():
    with pytest.raises(ValueError):
        DistributionVisualizer(data=[1, 2, 3])

def test_create_histogram_valid_column(sample_data):
    visualizer = DistributionVisualizer(data=sample_data)
    fig = visualizer.create_histogram(column='A', bins=5)
    assert isinstance(fig, Figure)

def test_create_histogram_invalid_column(sample_data):
    visualizer = DistributionVisualizer(data=sample_data)
    with pytest.raises(ValueError):
        visualizer.create_histogram(column='D')

def test_create_box_plot_valid_columns(sample_data):
    visualizer = DistributionVisualizer(data=sample_data)
    fig = visualizer.create_box_plot(columns=['A', 'B'])
    assert isinstance(fig, Figure)

def test_create_box_plot_invalid_columns(sample_data):
    visualizer = DistributionVisualizer(data=sample_data)
    with pytest.raises(ValueError):
        visualizer.create_box_plot(columns=['D', 'E'])

def test_create_box_plot_missing_columns_arg(sample_data):
    visualizer = DistributionVisualizer(data=sample_data)
    with pytest.raises(ValueError):
        visualizer.create_box_plot()



@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [5, 4, 3, 2, 1],
        'C': [2, 3, 4, 5, 6]
    })

def test_init_invalid_data():
    with pytest.raises(ValueError):
        CorrelationVisualizer(data=[1, 2, 3])

def test_create_correlation_matrix(sample_data):
    visualizer = CorrelationVisualizer(data=sample_data)
    fig = visualizer.create_correlation_matrix()
    assert isinstance(fig, Figure)

def test_create_scatter_plot_valid_columns(sample_data):
    visualizer = CorrelationVisualizer(data=sample_data)
    fig = visualizer.create_scatter_plot(x='A', y='B')
    assert isinstance(fig, Figure)

def test_create_scatter_plot_with_trendline(sample_data):
    visualizer = CorrelationVisualizer(data=sample_data)
    fig = visualizer.create_scatter_plot(x='A', y='B', with_trendline=True)
    assert isinstance(fig, Figure)

def test_create_scatter_plot_invalid_columns(sample_data):
    visualizer = CorrelationVisualizer(data=sample_data)
    with pytest.raises(ValueError):
        visualizer.create_scatter_plot(x='A', y='D')



@pytest.fixture
def csv_file(tmp_path):
    file = tmp_path / "test.csv"
    df = pd.DataFrame({
        'A': [1, 2],
        'B': [3, 4]
    })
    df.to_csv(file, index=False)
    return file

@pytest.fixture
def excel_file(tmp_path):
    file = tmp_path / "test.xlsx"
    df = pd.DataFrame({
        'A': [1, 2],
        'B': [3, 4]
    })
    df.to_excel(file, index=False)
    return file

@pytest.fixture
def json_file(tmp_path):
    file = tmp_path / "test.json"
    df = pd.DataFrame({
        'A': [1, 2],
        'B': [3, 4]
    })
    df.to_json(file)
    return file

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'A': [1, None],
        'B': [None, 4]
    })

def test_import_data_csv(csv_file):
    handler = DataHandler()
    data = handler.import_data(str(csv_file), 'csv')
    expected = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    assert_frame_equal(data, expected)

def test_import_data_excel(excel_file):
    handler = DataHandler()
    data = handler.import_data(str(excel_file), 'xlsx')
    expected = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    assert_frame_equal(data, expected)

def test_import_data_json(json_file):
    handler = DataHandler()
    data = handler.import_data(str(json_file), 'json')
    expected = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    assert_frame_equal(data, expected)

def test_import_data_unsupported_type():
    handler = DataHandler()
    with pytest.raises(ValueError):
        handler.import_data("unsupported.file", 'txt')

def test_preprocess_data_dropna(sample_data):
    handler = DataHandler()
    handler.data = sample_data
    processed = handler.preprocess_data({'dropna': {}})
    expected = pd.DataFrame(columns=['A', 'B'])
    assert_frame_equal(processed, expected)

def test_preprocess_data_fillna(sample_data):
    handler = DataHandler()
    handler.data = sample_data
    processed = handler.preprocess_data({'fillna': {'value': 0}})
    expected = pd.DataFrame({'A': [1, 0], 'B': [0, 4]})
    assert_frame_equal(processed, expected)

def test_preprocess_data_unavailable_strategy(sample_data):
    handler = DataHandler()
    handler.data = sample_data
    with pytest.raises(ValueError):
        handler.preprocess_data({'unknown_strategy': {}})



@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'numeric1': [1, 2, 3, 4, 5],
        'numeric2': [2, 3, 4, 5, 6],
        'category1': ['A', 'A', 'B', 'B', 'C'],
        'category2': ['X', 'Y', 'X', 'Y', 'X']
    })

def test_init_invalid_data():
    with pytest.raises(ValueError):
        StatisticalAnalyzer(data=[1, 2, 3])

def test_compute_basic_statistics(sample_data):
    analyzer = StatisticalAnalyzer(data=sample_data)
    stats = analyzer.compute_basic_statistics(['numeric1', 'numeric2'])
    assert 'numeric1' in stats
    assert 'numeric2' in stats
    assert 'mean' in stats['numeric1']
    assert stats['numeric1']['mean'] == 3.0  # Mean of [1, 2, 3, 4, 5]
    assert stats['numeric1']['mode'] == 1  # Single mode of [1, 2, 3, 4, 5]

def test_compute_basic_statistics_invalid_column(sample_data):
    analyzer = StatisticalAnalyzer(data=sample_data)
    with pytest.raises(ValueError):
        analyzer.compute_basic_statistics(['nonexistent_column'])

def test_perform_t_test(sample_data):
    analyzer = StatisticalAnalyzer(data=sample_data)
    result = analyzer.perform_hypothesis_tests(test_type='t-test', columns=['numeric1', 'numeric2'])
    assert 't_statistic' in result
    assert 'p_value' in result

def test_perform_t_test_invalid_columns(sample_data):
    analyzer = StatisticalAnalyzer(data=sample_data)
    with pytest.raises(ValueError):
        analyzer.perform_hypothesis_tests(test_type='t-test', columns=['numeric1'])

def test_perform_chi_squared_test(sample_data):
    analyzer = StatisticalAnalyzer(data=sample_data)
    result = analyzer.perform_hypothesis_tests(test_type='chi-squared', columns=['category1', 'category2'])
    assert 'chi2_statistic' in result
    assert 'p_value' in result

def test_perform_invalid_test_type(sample_data):
    analyzer = StatisticalAnalyzer(data=sample_data)
    with pytest.raises(ValueError):
        analyzer.perform_hypothesis_tests(test_type='invalid_test', columns=['numeric1', 'numeric2'])



@pytest.fixture
def sample_figure():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    return fig

def test_init_with_valid_figure(sample_figure):
    exporter = TableauExporter(sample_figure)
    assert exporter.visualization is sample_figure

def test_init_with_invalid_figure():
    with pytest.raises(ValueError):
        TableauExporter(visualization="Not a Figure")

def test_export_to_tableau_format_valid_path(sample_figure, tmp_path):
    exporter = TableauExporter(sample_figure)
    output_file = tmp_path / "test_export.png"
    exporter.export_to_tableau_format(str(output_file))
    assert output_file.exists()

def test_export_to_tableau_format_invalid_path(sample_figure):
    exporter = TableauExporter(sample_figure)
    with pytest.raises(OSError):
        exporter.export_to_tableau_format("/invalid_path/test_export.png")



def test_setup_color_palette():
    setup_visualization_style({'color_palette': 'seaborn-darkgrid'})
    assert plt.rcParams['axes.prop_cycle'].by_key()['color'] is not None

def test_setup_font_size():
    setup_visualization_style({'font_size': 14})
    assert plt.rcParams['font.size'] == 14

def test_setup_line_style():
    setup_visualization_style({'line_style': '--'})
    assert plt.rcParams['lines.linestyle'] == '--'

def test_setup_figure_size():
    setup_visualization_style({'figure_size': (8, 6)})
    assert plt.rcParams['figure.figsize'] == [8, 6]

def test_setup_background_color():
    setup_visualization_style({'background_color': 'lightgrey'})
    assert plt.rcParams['axes.facecolor'] == 'lightgrey'

def test_invalid_style_option():
    with pytest.raises(ValueError):
        setup_visualization_style({'invalid_option': 'some_value'})



def test_setup_logging_valid_levels():
    for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        setup_logging(level)
        assert logging.getLogger().level == logging.getLevelName(level)

def test_setup_logging_invalid_level():
    with pytest.raises(ValueError):
        setup_logging('INVALID_LEVEL')

def test_logging_message(capsys):
    setup_logging('INFO')
    logging.info("This is an info message.")
    captured = capsys.readouterr()
    assert "INFO - This is an info message." in captured.out

    setup_logging('ERROR')
    logging.info("This message should not appear.")
    logging.error("This is an error message.")
    captured = capsys.readouterr()
    assert "INFO - This message should not appear." not in captured.out
    assert "ERROR - This is an error message." in captured.out
