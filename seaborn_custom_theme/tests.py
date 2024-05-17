
import seaborn as sns
import pytest
import matplotlib.pyplot as plt

def set_theme_style():
    sns.set_theme(style="whitegrid", font_scale=1.4, rc={"axes.labelsize": 10, "axes.titlesize": 14})
    plt.rcParams.update({'font.family': 'Georgia', 'axes.grid': True, 'grid.linestyle': '--', 'grid.color': 'lightgray'})


# Test if set_theme_style() sets the correct seaborn style
def test_set_theme_style():
    # Call the function to set the theme style
    set_theme_style()
    
    # Check if the style has been set correctly
    assert sns.plotting_context().get('font_scale') == 1.4
    assert sns.plotting_context().get('context') == 'notebook'
    assert sns.axes_style().get('axes.grid') == True
    assert plt.rcParams['font.family'] == ['Georgia']

# Test if the function is not throwing any exceptions
def test_set_theme_style_no_exceptions():
    # Call the function without any assertions, just check if it throws any exceptions
    set_theme_style()


# Function to test
def set_font_face_georgia():
    plt.rcParams['font.family'] = 'Georgia'

# Test case for setting font face to Georgia
def test_set_font_face_georgia():
    # Set the font family to Georgia
    set_font_face_georgia()
    
    # Check if the font family has been set correctly
    assert plt.rcParams['font.family'] == ['Georgia']


def set_title_font_size():
    plt.title('Test Title', fontsize=14)

# Test case for setting title font size to 14
def test_set_title_font_size():
    # Create a test plot
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    plt.plot(x, y)

    # Call the function to set the title font size
    set_title_font_size()

    ax = plt.gca()

    # Assert that the title font size is set to 14
    assert ax.title.get_fontsize() == 14


def disable_grid_lines():
    sns.set(style="ticks", rc={
        "axes.grid": False,
        "axes.edgecolor": ".8",
        "grid.linestyle": "",
        "xtick.bottom": True,
        "ytick.left": True
    })

# Test to check if 'axes.grid' is set to False after calling disable_grid_lines()
def test_disable_grid_lines_axes_grid():
    disable_grid_lines()
    assert sns.axes_style()["axes.grid"] == False

# Test to check if 'axes.edgecolor' is set to '.8' after calling disable_grid_lines()
def test_disable_grid_lines_axes_edgecolor():
    disable_grid_lines()
    assert sns.axes_style()["axes.edgecolor"] == ".8"

# Test to check if 'grid.linestyle' is an empty string after calling disable_grid_lines()
def test_disable_grid_lines_grid_linestyle():
    disable_grid_lines()
    assert sns.axes_style()["grid.linestyle"] == ""

# Test to check if 'xtick.bottom' is set to True after calling disable_grid_lines()
def test_disable_grid_lines_xtick_bottom():
    disable_grid_lines()
    assert sns.axes_style()["xtick.bottom"] == True

# Test to check if 'ytick.left' is set to True after calling disable_grid_lines()
def test_disable_grid_lines_ytick_left():
    disable_grid_lines()
    assert sns.axes_style()["ytick.left"] == True


from your_module import create_stacked_bar_chart

@pytest.fixture(scope='function')
def sample_data():
   return pd.DataFrame({'A': [1, 2, 3], 'B': [4, -5, 6]}), ['red', 'blue']

@pytest.mark.parametrize("data", [
   (pd.DataFrame({'A': [1, 2, 3], 'B': [4, -5, 6]}), ['red', 'blue']),
])
def test_create_stacked_bar_chart(data):
   df_data, colors = data
  
   create_stacked_bar_chart(df_data, colors)
  
   fig = plt.gcf()

   assert len(fig.get_axes()[0].texts) > 0


from your_module import validate_color_list

@pytest.mark.parametrize("input_colors", [
   (['#123456', '#abcdef', '#fff']),
])
@pytest.mark.parametrize("expected", [
   (['#123456', '#abcdef', '#fff']),
])
def test_validate_color_list(input_colors: list[str], expected: list[str]):
   result = validate_color_list(input_colors)
  
   assert result == expected


@pytest.fixture(scope='function')
def sample_plot_data():
   return pd.DataFrame({'x': [1,2], 'y': [3,-4]}), 

@pytest.mark.parametrize("axis_labels", [
   ('X Axis', "Y Axis"),
])
@pytest.mark.parametrize("tick_labels", [
   (['A','B'], ['C','D']),
])
@pytest.mark.parametrize("title", [
   ("Test Title")
])
@pytest.mark.parametrize("fontsize", [
   (10),
])
@pytest.mark.parametrize("fontname", [
   ("Georgia"),
])

from your_module import add_axis_labels

@pytest.fixture(scope='function')
def add_axis_labels_fixture(sample_plot_data):
  
 @pytest.mark.parametrize('x_label,y_label,font_size,font_name','tick_labels',add_axis_labels_fixture)
  
 def add_axis_labels(x_label,y_label,font_size,font_name,tick_labels):
      x,y = sample_plot_data[0]

      fig ,ax=plt.subplots(figsize=(1.6))
  
      ax.plot(x,y,label=y_label)
 
      add_axis_labels(ax,x_label,y_label,font_size,font_name) 

      fig.legend(loc='lower center')

      fig.savefig(f'{x}_test.png')

      img=plt.imread(f'{x}_test.png')

      os.remove(f'{x}_test.png')

      labels = []

      for tick in tick_labels:
          labels.append(tick.get_text())
   
     assert all(items in labels for items in tick_labels)


if __name__== "__main__":
  
 pytest.main()

