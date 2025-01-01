from color_palette_module import set_color_palette  # Replace with the correct import path
from custom_theme import CustomTheme  # Replace with the correct import path
from font_module import set_fonts  # Replace with the correct import path
from legend_module import position_legend  # Replace with the correct import path
from matplotlib import pyplot as plt
from matplotlib import rc
from reference_lines_module import configure_reference_lines  # Replace with the correct import path
from seaborn.utils import get_color_cycle
import matplotlib.pyplot as plt
import pytest
import seaborn as sns



def test_custom_theme_initialization():
    colors = ['#FF5733', '#33FF57', '#3357FF']
    theme = CustomTheme(colors, title_font_size=16, label_font_size=12)
    
    assert theme.colors == colors
    assert theme.title_font_size == 16
    assert theme.label_font_size == 12

def test_custom_theme_apply():
    colors = ['#FF5733', '#33FF57', '#3357FF']
    theme = CustomTheme(colors)
    theme.apply_theme()

    # Test color palette
    color_cycle = get_color_cycle()
    assert list(color_cycle)[:len(colors)] == sns.color_palette(colors)

    # Test context settings
    context = sns.plotting_context('notebook')
    assert context["axes.titlesize"] == theme.title_font_size
    assert context["axes.labelsize"] == theme.label_font_size
    assert rc['font.family'] == ['serif']
    assert rc['font.serif'] == ['Georgia']

    # Test style settings
    style = sns.axes_style('whitegrid')
    assert style['axes.edgecolor'] == 'lightgray'
    assert style['grid.color'] == 'lightgray'
    assert style['grid.linestyle'] == '--'

    # Test legend positioning
    assert rc['legend.loc'] == 'lower center'




def test_set_color_palette_valid_input():
    colors = ['#FF5733', '#33FF57', '#3357FF']
    set_color_palette(colors)
    
    # Check if the color palette has been updated
    current_palette = sns.color_palette()
    assert list(current_palette)[:len(colors)] == sns.color_palette(colors)

def test_set_color_palette_invalid_input():
    with pytest.raises(ValueError, match="Colors must be a list of string color codes."):
        set_color_palette('#FF5733')  # Not a list

    with pytest.raises(ValueError, match="Colors must be a list of string color codes."):
        set_color_palette([123, 456])  # List does not contain all strings

def test_set_color_palette_effect_on_plot():
    colors = ['#9467bd', '#8c564b', '#e377c2']
    set_color_palette(colors)
    
    # Generate a simple plot to apply the color palette
    sns.lineplot(x=[1, 2, 3], y=[2, 3, 5], color=colors[0])
    
    # Display the plot (to manually verify, if desired)
    plt.draw()




def test_set_fonts_valid_input():
    title_font_size = 18
    label_font_size = 12
    font_face = 'Arial'
    set_fonts(title_font_size, label_font_size, font_face)
    
    # Check if font settings have been updated
    context = sns.plotting_context('notebook')
    assert context["axes.titlesize"] == title_font_size
    assert context["axes.labelsize"] == label_font_size
    assert context["font.family"] == ['serif']
    assert context["font.serif"] == [font_face]

def test_set_fonts_invalid_title_font_size():
    with pytest.raises(ValueError, match="Font sizes must be integers."):
        set_fonts('large', 12, 'Arial')  # Non-integer title font size

def test_set_fonts_invalid_label_font_size():
    with pytest.raises(ValueError, match="Font sizes must be integers."):
        set_fonts(18, 'small', 'Arial')  # Non-integer label font size

def test_set_fonts_invalid_font_face():
    with pytest.raises(ValueError, match="Font face must be a string."):
        set_fonts(18, 12, 123)  # Non-string font face




def test_position_legend_valid_input():
    positions = ['top', 'bottom', 'left', 'right', 'upper right', 'lower left']
    for pos in positions:
        position_legend(at=pos)
        assert plt.rcParams['legend.loc'] == pos, f"Legend position should be {pos}"

def test_position_legend_invalid_input():
    with pytest.raises(ValueError, match="Legend position must be specified as a string."):
        position_legend(at=123)  # Non-string input

    with pytest.raises(ValueError, match="Legend position must be specified as a string."):
        position_legend(at=None)  # None as input




def test_configure_reference_lines_valid_input():
    # Test with default parameters
    configure_reference_lines()
    style = sns.axes_style('whitegrid')
    assert style['grid.color'] == 'lightgray'
    assert style['grid.linestyle'] == '--'

    # Test with custom color and dotted style
    configure_reference_lines(color='blue', grid_style='dotted')
    style = sns.axes_style('whitegrid')
    assert style['grid.color'] == 'blue'
    assert style['grid.linestyle'] == ':'

    # Test with solid style
    configure_reference_lines(color='black', grid_style='solid')
    style = sns.axes_style('whitegrid')
    assert style['grid.color'] == 'black'
    assert style['grid.linestyle'] == '-'

def test_configure_reference_lines_invalid_color():
    with pytest.raises(ValueError, match="Color must be specified as a string."):
        configure_reference_lines(color=123)

def test_configure_reference_lines_invalid_grid_style():
    with pytest.raises(ValueError, match="Grid style must be specified as a string."):
        configure_reference_lines(grid_style=123)

    with pytest.raises(ValueError, match="Grid style must be 'solid', 'dashed', 'dotted', or 'minimal'."):
        configure_reference_lines(grid_style='bold')
