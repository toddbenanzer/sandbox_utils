from .cohort_analysis import CohortAnalysis
from .dashboard_creator import DashboardCreator
from .funnel_plot import FunnelPlot
from .plotly_template import PlotlyTemplate
from .time_series_plot import TimeSeriesPlot
from unittest.mock import mock_open, patch
from your_module_path import export_dashboard
from your_module_path import import_data
from your_module_path import setup_config
from your_module_path import setup_logging
from your_module_path.data_handler import DataHandler
import json
import logging
import os
import pandas as pd
import plotly.graph_objs as go
import pytest


def test_init_with_valid_dataframe():
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    dashboard_creator = DashboardCreator(df)
    assert dashboard_creator.data.equals(df)

def test_init_with_valid_dict():
    data_dict = {'x': [1, 2, 3], 'y': [4, 5, 6]}
    dashboard_creator = DashboardCreator(data_dict)
    assert dashboard_creator.data == data_dict

def test_init_with_valid_list():
    data_list = [[1, 2, 3], [4, 5, 6]]
    dashboard_creator = DashboardCreator(data_list)
    assert dashboard_creator.data == data_list

def test_init_with_invalid_data():
    with pytest.raises(TypeError):
        DashboardCreator("invalid data type")

def test_create_dashboard_without_layout():
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    dashboard_creator = DashboardCreator(df)
    component = go.Scatter(x=df['x'], y=df['y'])
    dashboard = dashboard_creator.create_dashboard(scatter=component)
    assert len(dashboard.data) == 1

def test_create_dashboard_with_valid_layout():
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    dashboard_creator = DashboardCreator(df)
    component = go.Scatter(x=df['x'], y=df['y'])
    layout = [{'name': 'scatter', 'props': {}}]
    dashboard = dashboard_creator.create_dashboard(layout=layout, scatter=component)
    assert len(dashboard.data) == 1

def test_create_dashboard_with_invalid_component_name():
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    dashboard_creator = DashboardCreator(df)
    component = go.Scatter(x=df['x'], y=df['y'])
    layout = [{'name': 'invalid_component', 'props': {}}]
    with pytest.raises(ValueError):
        dashboard_creator.create_dashboard(layout=layout, scatter=component)

def test_preview_dashboard_without_components():
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    dashboard_creator = DashboardCreator(df)
    with pytest.raises(ValueError):
        dashboard_creator.preview_dashboard()



def test_init_valid_template():
    template = PlotlyTemplate('funnel')
    assert template.template_type == 'funnel'

def test_init_invalid_template():
    with pytest.raises(ValueError):
        PlotlyTemplate('invalid_template')

def test_load_template_funnel():
    template = PlotlyTemplate('funnel')
    figure = template.load_template()
    assert isinstance(figure, go.Figure)
    assert figure.data[0].type == 'funnel'

def test_load_template_cohort_analysis():
    template = PlotlyTemplate('cohort_analysis')
    figure = template.load_template()
    assert isinstance(figure, go.Figure)

def test_load_template_time_series():
    template = PlotlyTemplate('time_series')
    figure = template.load_template()
    assert isinstance(figure, go.Figure)

def test_load_template_invalid():
    template = PlotlyTemplate('funnel')
    with pytest.raises(ValueError):
        template.template_type = 'invalid'
        template.load_template()

def test_customize_template_without_load():
    template = PlotlyTemplate('funnel')
    with pytest.raises(ValueError):
        template.customize_template(title='Custom Title')

def test_customize_template():
    template = PlotlyTemplate('funnel')
    figure = template.load_template()
    customized_figure = template.customize_template(title='Custom Title')
    assert customized_figure.layout.title.text == 'Custom Title'



def test_init_with_dataframe():
    data = pd.DataFrame({'x': [1, 2, 3], 'y': [100, 80, 60]})
    funnel_plot = FunnelPlot(data)
    assert funnel_plot.data.equals(data)

def test_init_with_dict():
    data = {'x': [1, 2, 3], 'y': [100, 80, 60]}
    funnel_plot = FunnelPlot(data)
    assert funnel_plot.data == data

def test_init_with_list():
    data = [[1, 2, 3], [100, 80, 60]]
    funnel_plot = FunnelPlot(data)
    assert funnel_plot.data == data

def test_init_with_invalid_data():
    with pytest.raises(TypeError):
        FunnelPlot("invalid data")

def test_generate_plot_with_dataframe():
    data = pd.DataFrame({'x': [1, 2, 3], 'y': [100, 80, 60]})
    funnel_plot = FunnelPlot(data)
    plot = funnel_plot.generate_plot(x='x', y='y')
    assert isinstance(plot, go.Figure)
    assert len(plot.data) == 1
    assert plot.data[0].type == 'funnel'

def test_generate_plot_with_dict():
    data = {'x': [1, 2, 3], 'y': [100, 80, 60]}
    funnel_plot = FunnelPlot(data)
    plot = funnel_plot.generate_plot(x=data['x'], y=data['y'])
    assert isinstance(plot, go.Figure)
    assert len(plot.data) == 1
    assert plot.data[0].type == 'funnel'

def test_generate_plot_without_x_or_y():
    data = pd.DataFrame({'x': [1, 2, 3], 'y': [100, 80, 60]})
    funnel_plot = FunnelPlot(data)
    with pytest.raises(ValueError):
        funnel_plot.generate_plot(x='x')

def test_generate_plot_with_customization():
    data = pd.DataFrame({'x': [1, 2, 3], 'y': [100, 80, 60]})
    funnel_plot = FunnelPlot(data)
    plot = funnel_plot.generate_plot(x='x', y='y', title='Custom Title')
    assert plot.layout.title.text == 'Custom Title'



def test_init_with_dataframe():
    data = pd.DataFrame({'cohort': ['A', 'B'], 'value': [10, 20]})
    cohort_analysis = CohortAnalysis(data)
    assert cohort_analysis.data.equals(data)

def test_init_with_invalid_data():
    with pytest.raises(TypeError):
        CohortAnalysis("invalid data type")

def test_perform_analysis_with_valid_data():
    data = pd.DataFrame({'cohort': ['A', 'A', 'B'], 'value': [10, 15, 20]})
    cohort_analysis = CohortAnalysis(data)
    results = cohort_analysis.perform_analysis()
    assert 'cohort' in results.columns
    assert 'size' in results.columns
    assert len(results) == 2  # Two unique cohorts

def test_perform_analysis_without_cohort_column():
    data = pd.DataFrame({'value': [10, 20]})
    cohort_analysis = CohortAnalysis(data)
    with pytest.raises(ValueError):
        cohort_analysis.perform_analysis()

def test_generate_plot_without_analysis():
    data = pd.DataFrame({'cohort': ['A', 'B'], 'value': [10, 20]})
    cohort_analysis = CohortAnalysis(data)
    with pytest.raises(ValueError):
        cohort_analysis.generate_plot()

def test_generate_plot_with_analysis():
    data = pd.DataFrame({'cohort': ['A', 'A', 'B'], 'value': [10, 15, 20]})
    cohort_analysis = CohortAnalysis(data)
    cohort_analysis.perform_analysis()
    plot = cohort_analysis.generate_plot(title="Cohort Analysis Plot")
    assert isinstance(plot, go.Figure)
    assert len(plot.data) == 1
    assert plot.layout.title.text == "Cohort Analysis Plot"



def test_init_with_dataframe():
    dates = pd.date_range(start='2021-01-01', periods=4, freq='D')
    data = pd.DataFrame({'value': [100, 200, 300, 400]}, index=dates)
    tsp = TimeSeriesPlot(data)
    assert tsp.data.equals(data)

def test_init_with_invalid_data_type():
    with pytest.raises(TypeError):
        TimeSeriesPlot("invalid data type")

def test_init_with_non_datetime_index():
    data = pd.DataFrame({'value': [100, 200, 300, 400]}, index=[1, 2, 3, 4])
    with pytest.raises(ValueError):
        TimeSeriesPlot(data)

def test_generate_plot_daily():
    dates = pd.date_range(start='2021-01-01', periods=4, freq='D')
    data = pd.DataFrame({'value': [100, 200, 300, 400]}, index=dates)
    tsp = TimeSeriesPlot(data)
    plot = tsp.generate_plot('D')
    assert isinstance(plot, go.Figure)
    assert len(plot.data) == 1
    assert plot.data[0].x.tolist() == dates.tolist()

def test_generate_plot_with_customization():
    dates = pd.date_range(start='2021-01-01', periods=4, freq='D')
    data = pd.DataFrame({'value': [100, 200, 300, 400]}, index=dates)
    tsp = TimeSeriesPlot(data)
    plot = tsp.generate_plot('D', title="Custom Time Series Plot")
    assert plot.layout.title.text == "Custom Time Series Plot"



def test_init_data_handler():
    handler = DataHandler("dummy_path.csv")
    assert handler.file_path == "dummy_path.csv"
    assert handler.data is None

def test_load_data_csv(mocker):
    mock_data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    mocker.patch('pandas.read_csv', return_value=mock_data)
    handler = DataHandler("dummy_path.csv")
    data = handler.load_data()
    assert data.equals(mock_data)

def test_load_data_excel(mocker):
    mock_data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    mocker.patch('pandas.read_excel', return_value=mock_data)
    handler = DataHandler("dummy_path.xlsx")
    data = handler.load_data()
    assert data.equals(mock_data)

def test_load_data_unsupported_format():
    handler = DataHandler("dummy_path.txt")
    with pytest.raises(ValueError):
        handler.load_data()

def test_load_data_file_not_found():
    handler = DataHandler("nonexistent.csv")
    with pytest.raises(FileNotFoundError):
        handler.load_data()

def test_transform_data_single_function(mocker):
    mock_data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    mocker.patch('pandas.read_csv', return_value=mock_data)
    handler = DataHandler("dummy_path.csv")
    handler.load_data()
    transformed_data = handler.transform_data(lambda df: df + 1)
    expected_data = mock_data + 1
    assert transformed_data.equals(expected_data)

def test_transform_data_multiple_functions(mocker):
    mock_data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    mocker.patch('pandas.read_csv', return_value=mock_data)
    handler = DataHandler("dummy_path.csv")
    handler.load_data()
    transformations = [lambda df: df + 1, lambda df: df * 2]
    transformed_data = handler.transform_data(transformations)
    expected_data = (mock_data + 1) * 2
    assert transformed_data.equals(expected_data)

def test_transform_data_without_loading():
    handler = DataHandler("dummy_path.csv")
    with pytest.raises(ValueError):
        handler.transform_data(lambda df: df + 1)



def test_import_data_csv(mocker):
    mock_data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    mocker.patch('pandas.read_csv', return_value=mock_data)
    mocker.patch('os.path.exists', return_value=True)
    data = import_data("dummy_path.csv", "csv")
    assert data.equals(mock_data)

def test_import_data_xlsx(mocker):
    mock_data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    mocker.patch('pandas.read_excel', return_value=mock_data)
    mocker.patch('os.path.exists', return_value=True)
    data = import_data("dummy_path.xlsx", "xlsx")
    assert data.equals(mock_data)

def test_import_data_json(mocker):
    mock_data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    mocker.patch('pandas.read_json', return_value=mock_data)
    mocker.patch('os.path.exists', return_value=True)
    data = import_data("dummy_path.json", "json")
    assert data.equals(mock_data)

def test_import_data_unsupported_type():
    with pytest.raises(ValueError):
        import_data("dummy_path.txt", "txt")

def test_import_data_file_not_found(mocker):
    mocker.patch('os.path.exists', return_value=False)
    with pytest.raises(FileNotFoundError):
        import_data("nonexistent.csv", "csv")

def test_import_data_exception(mocker):
    mocker.patch('pandas.read_csv', side_effect=Exception('Unexpected error'))
    mocker.patch('os.path.exists', return_value=True)
    with pytest.raises(Exception):
        import_data("dummy_path.csv", "csv")



@pytest.fixture
def sample_dashboard():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
    return fig

def test_export_dashboard_html(mocker, sample_dashboard):
    mocker.patch('os.path.exists', return_value=True)
    mocker.patch.object(sample_dashboard, 'write_html')
    export_dashboard(sample_dashboard, 'html', 'output/test_dashboard.html')
    sample_dashboard.write_html.assert_called_once_with('output/test_dashboard.html')

def test_export_dashboard_png(mocker, sample_dashboard):
    mocker.patch('os.path.exists', return_value=True)
    mocker.patch.object(sample_dashboard, 'write_image')
    export_dashboard(sample_dashboard, 'png', 'output/test_dashboard.png')
    sample_dashboard.write_image.assert_called_once_with('output/test_dashboard.png')

def test_export_dashboard_pdf(mocker, sample_dashboard):
    mocker.patch('os.path.exists', return_value=True)
    mocker.patch.object(sample_dashboard, 'write_image')
    export_dashboard(sample_dashboard, 'pdf', 'output/test_dashboard.pdf')
    sample_dashboard.write_image.assert_called_once_with('output/test_dashboard.pdf', format='pdf')

def test_export_dashboard_unsupported_format(sample_dashboard):
    mocker.patch('os.path.exists', return_value=True)
    with pytest.raises(ValueError):
        export_dashboard(sample_dashboard, 'txt', 'output/test_dashboard.txt')

def test_export_dashboard_nonexistent_directory(sample_dashboard):
    mocker.patch('os.path.exists', return_value=False)
    with pytest.raises(FileNotFoundError):
        export_dashboard(sample_dashboard, 'html', 'nonexistent_dir/test_dashboard.html')

def test_export_dashboard_invalid_dashboard():
    with pytest.raises(TypeError):
        export_dashboard("invalid_dashboard_object", 'html', 'output/test_dashboard.html')



def test_setup_logging_valid_string_levels(caplog):
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    
    for level in valid_levels:
        setup_logging(level)
        assert logging.getLogger().level == getattr(logging, level)
        with caplog.at_level(getattr(logging, level)):
            logging.info(f"Testing {level} level")
            assert f"Testing {level} level" in caplog.text

def test_setup_logging_valid_integer_levels(caplog):
    level_to_str = {10: 'DEBUG', 20: 'INFO', 30: 'WARNING', 40: 'ERROR', 50: 'CRITICAL'}
    
    for level in level_to_str.keys():
        setup_logging(level)
        assert logging.getLogger().level == level
        with caplog.at_level(level):
            logging.info(f"Testing {level_to_str[level]} level")
            assert f"Testing {level_to_str[level]} level" in caplog.text

def test_setup_logging_invalid_string_level():
    with pytest.raises(ValueError) as excinfo:
        setup_logging('INVALID')
    assert "Invalid logging level" in str(excinfo.value)

def test_setup_logging_invalid_integer_level():
    with pytest.raises(ValueError) as excinfo:
        setup_logging(99)
    assert "Invalid logging level" in str(excinfo.value)

def test_setup_logging_invalid_type_level():
    with pytest.raises(TypeError) as excinfo:
        setup_logging(3.5)
    assert "Logging level must be a string or an integer" in str(excinfo.value)



def test_setup_config_valid_file():
    mock_config = '{"setting1": "value1", "setting2": 20}'
    with patch("builtins.open", mock_open(read_data=mock_config)):
        with patch("os.path.exists", return_value=True):
            config = setup_config("dummy_config.json")
            assert config == json.loads(mock_config)

def test_setup_config_file_not_found():
    with patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            setup_config("nonexistent_config.json")

def test_setup_config_invalid_json():
    mock_invalid_json = '{"setting1": "value1", "setting2": 20'  # Missing closing brace
    with patch("builtins.open", mock_open(read_data=mock_invalid_json)):
        with patch("os.path.exists", return_value=True):
            with pytest.raises(ValueError):
                setup_config("invalid_config.json")

def test_setup_config_unexpected_error():
    with patch("builtins.open", side_effect=Exception("Unexpected error")):
        with patch("os.path.exists", return_value=True):
            with pytest.raises(Exception):
                setup_config("unexpected_error_config.json")
