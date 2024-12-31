from CustomizerModule import Customizer  # Replace CustomizerModule with the actual module name
from ExporterModule import Exporter  # Replace ExporterModule with the actual module name
from InteractionModule import Interaction  # Replace InteractionModule with the actual module name
from io import StringIO
from plotly.graph_objects import Figure
from plotly.graph_objects import Figure, Scatter
from your_module import Animator  # assuming the Animator class is in a module named your_module
from your_module import example_script  # Replace your_module with the actual module name
from your_module import load_data  # Replace your_module with the actual module name
from your_module import save_animation  # Replace your_module with the actual module name
import os
import pandas as pd
import plotly.express as px
import pytest


def test_animator_init():
    data = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    animator = Animator(data, 'scatter')
    assert isinstance(animator, Animator)
    assert animator.data.equals(data)
    assert animator.plot_type == 'scatter'
    assert animator.animation is None

def test_generate_animation_scatter():
    data = pd.DataFrame({'id': [1, 1, 2, 2], 'x': [1, 2, 1, 2], 'y': [3, 4, 5, 6], 'frame': [1, 2, 1, 2]})
    animator = Animator(data, 'scatter')
    animation = animator.generate_animation(animation_frame='frame', animation_group='id')
    assert isinstance(animation, Figure)
    assert animator.animation == animation

def test_generate_animation_line():
    data = pd.DataFrame({'id': [1, 1, 2, 2], 'x': [1, 2, 1, 2], 'y': [3, 4, 5, 6], 'frame': [1, 2, 1, 2]})
    animator = Animator(data, 'line')
    animation = animator.generate_animation(animation_frame='frame', animation_group='id')
    assert isinstance(animation, Figure)
    assert animator.animation == animation

def test_generate_animation_invalid_type():
    data = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    animator = Animator(data, 'invalid_type')
    with pytest.raises(ValueError, match="Unsupported plot_type 'invalid_type'"):
        animator.generate_animation()

def test_set_animation_properties():
    data = pd.DataFrame({'frame': [1, 2], 'x': [1, 2], 'y': [3, 4]})
    animator = Animator(data, 'scatter')
    animator.generate_animation(animation_frame='frame')
    animator.set_animation_properties(duration=1000, easing='linear')
    assert animator.animation.layout.updatemenus[0]['buttons'][0]['args'][1]['frame']['duration'] == 1000
    # Cannot directly test easing as Plotly does not expose this setting directly



def test_customizer_init():
    fig = Figure()  # Create an empty Plotly Figure
    customizer = Customizer(fig)
    assert customizer.animation == fig

def test_customizer_init_invalid():
    with pytest.raises(ValueError, match="Expected a Plotly Figure object for animation."):
        Customizer(animation="Not a Figure")

def test_set_color_palette():
    fig = Figure()
    customizer = Customizer(fig)
    palette = ['red', 'green', 'blue']
    customizer.set_color_palette(palette)
    for trace in customizer.animation['data']:
        assert trace['marker']['color'] == palette

def test_set_marker_style():
    fig = Figure()
    customizer = Customizer(fig)
    style = {'size': 10, 'symbol': 'circle'}
    customizer.set_marker_style(style)
    for trace in customizer.animation['data']:
        assert trace['marker']['size'] == style['size']
        assert trace['marker']['symbol'] == style['symbol']

def test_customize_layout():
    fig = Figure()
    customizer = Customizer(fig)
    title = "Sample Animation"
    labels = {'x': 'X-axis', 'y': 'Y-axis'}
    legends = {'orientation': 'h', 'x': 0.5, 'y': -0.1}
    customizer.customize_layout(title, labels, legends)
    layout = customizer.animation.layout
    assert layout.title.text == title
    assert layout.xaxis.title.text == labels['x']
    assert layout.yaxis.title.text == labels['y']
    assert layout.legend.orientation == legends['orientation']
    assert layout.legend.x == legends['x']
    assert layout.legend.y == legends['y']



def test_interaction_init():
    fig = Figure(data=[Scatter(x=[1, 2, 3], y=[3, 1, 6])])
    interaction = Interaction(fig)
    assert interaction.animation == fig

def test_interaction_init_invalid():
    with pytest.raises(ValueError, match="Expected a Plotly Figure object for animation."):
        Interaction(animation="Not a Figure")

def test_add_hover_info():
    fig = Figure(data=[Scatter(x=[1, 2, 3], y=[3, 1, 6])])
    interaction = Interaction(fig)
    hover_info = ['Point A', 'Point B', 'Point C']
    interaction.add_hover_info(hover_info)
    for trace in interaction.animation['data']:
        assert trace['hoverinfo'] == 'text'
        assert trace['text'] == hover_info

def test_set_clickable_elements():
    fig = Figure(data=[Scatter(x=[1, 2, 3], y=[3, 1, 6])])
    interaction = Interaction(fig)
    with pytest.raises(NotImplementedError, match="Clickable elements require JavaScript/HTML integration in the output environment."):
        interaction.set_clickable_elements({'element_id': 'action'})



def test_exporter_init():
    fig = Figure(data=[Scatter(x=[1, 2, 3], y=[3, 1, 6])])
    exporter = Exporter(fig)
    assert exporter.animation == fig

def test_exporter_init_invalid():
    with pytest.raises(ValueError, match="Expected a Plotly Figure object for animation."):
        Exporter(animation="Not a Figure")

def test_export_to_html(tmpdir):
    fig = Figure(data=[Scatter(x=[1, 2, 3], y=[3, 1, 6])])
    exporter = Exporter(fig)
    file_name = os.path.join(tmpdir, "test_animation.html")
    exporter.export_to_html(file_name)
    assert os.path.exists(file_name)

def test_export_to_gif():
    fig = Figure(data=[Scatter(x=[1, 2, 3], y=[3, 1, 6])])
    exporter = Exporter(fig)
    with pytest.raises(NotImplementedError, match="Plotly does not natively support exporting animations to GIF directly."):
        exporter.export_to_gif("test_animation.gif")

def test_save_as_image(tmpdir):
    fig = Figure(data=[Scatter(x=[1, 2, 3], y=[3, 1, 6])])
    exporter = Exporter(fig)
    file_name = os.path.join(tmpdir, "test_animation.png")
    exporter.save_as_image(file_name, "png")
    assert os.path.exists(file_name)



def test_load_data_csv(monkeypatch):
    csv_content = "col1,col2\n1,a\n2,b"
    monkeypatch.setattr('builtins.open', lambda f, mode='r': StringIO(csv_content))
    data = load_data("dummy.csv", "csv")
    expected_data = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    pd.testing.assert_frame_equal(data, expected_data)

def test_load_data_json(monkeypatch):
    json_content = '[{"col1": 1, "col2": "a"}, {"col1": 2, "col2": "b"}]'
    monkeypatch.setattr('builtins.open', lambda f, mode='r': StringIO(json_content))
    data = load_data("dummy.json", "json")
    expected_data = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    pd.testing.assert_frame_equal(data, expected_data)

def test_load_data_unsupported_type():
    with pytest.raises(ValueError, match="Unsupported file type 'xml'. Supported types are 'csv', 'json', 'excel'."):
        load_data("dummy.xml", "xml")

def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError, match="The file 'not_existing.csv' was not found"):
        load_data("not_existing.csv", "csv")



def test_save_animation_html(tmpdir):
    fig = Figure(data=[Scatter(x=[1, 2, 3], y=[3, 1, 6])])
    file_path = os.path.join(tmpdir, "test_animation.html")
    save_animation(fig, file_path)
    assert os.path.exists(file_path)

def test_save_animation_json(tmpdir):
    fig = Figure(data=[Scatter(x=[1, 2, 3], y=[3, 1, 6])])
    file_path = os.path.join(tmpdir, "test_animation.json")
    save_animation(fig, file_path)
    assert os.path.exists(file_path)

def test_save_animation_unsupported_extension():
    fig = Figure(data=[Scatter(x=[1, 2, 3], y=[3, 1, 6])])
    with pytest.raises(ValueError, match="Unsupported file extension '.txt'. Supported extensions are '.html' and '.json'."):
        save_animation(fig, "test_animation.txt")

def test_save_animation_invalid_figure():
    with pytest.raises(ValueError, match="Expected a Plotly Figure object for animation."):
        save_animation("Not a Figure", "test_animation.html")



def test_example_script_basic_animation():
    result = example_script("basic_animation")
    assert "import pandas as pd" in result
    assert "animator = Animator(data, 'scatter')" in result

def test_example_script_custom_interactivity():
    result = example_script("custom_interactivity")
    assert "import pandas as pd" in result
    assert "animator = Animator(data, 'line')" in result
    assert "interaction = Interaction(animation)" in result

def test_example_script_invalid_name():
    with pytest.raises(ValueError, match="Unknown script_name 'non_existent_script'. Available scripts: basic_animation, custom_interactivity."):
        example_script("non_existent_script")
