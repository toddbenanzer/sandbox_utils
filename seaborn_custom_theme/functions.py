eaborn as sns
import matplotlib.pyplot as plt


def set_bar_chart_palette(colors):
    """
    Function to set the color palette for stacked bar charts.
    
    Parameters:
        colors (list): List of colors to use in the palette.
    """
    sns.set_palette(colors)


def set_line_chart_palette(colors):
    """
    Function to set the color palette for line charts.
    
    Parameters:
        colors (list): List of colors to use in the palette.
    """
    sns.set_palette(colors)


def set_georgia_font():
    """
    Function to set the font style as Georgia.
    """
    plt.rcParams['font.family'] = 'Georgia'


def set_title_font_size(font_size):
    """
    Function to set the title font size.
    
    Parameters:
        font_size (int): Font size for the title.
    """
    sns.set_theme(style="ticks", font="Georgia")
    sns.set_context("notebook", font_scale=font_size/14)


def set_label_font_size():
    """
    Function to set the label font size as 10 point.
    """
    plt.rc('font', size=10)


def set_legend_bottom(plot):
    """
    Function to place the legend at the bottom of the chart.

    Parameters:
        plot: Seaborn plot object

    Returns:
        None
    """
    legend = plot.axes.get_legend()
    legend.set_bbox_to_anchor((0.5, -0.1))
    legend.set_title(None)


def set_reference_lines_color():
    """
    Function to set reference lines color as light gray.
    """
    sns.set(style="ticks", rc={"axes.grid": False})
    sns.set_context(rc={"lines.linewidth": 0.8})
    sns.set_palette("pastel")


def remove_grid_lines():
   """
   Function to remove grid lines around bounding boxes.
   """
   sns.set_style('ticks')


def set_minimal_theme():
    """
    Function to set the default seaborn theme to minimal style.
    """
    sns.set_theme(style="ticks", font="Georgia")
    sns.set_context("notebook", font_scale=1.4, rc={"axes.labelsize": 10})
    sns.set(rc={"legend.loc": "lower center"})
    sns.set(rc={"axes.grid": True, "grid.color": "lightgray"})


def set_chart_colors(colors):
    """
    Function to set the list of colors for stacked bar charts and line charts.
    
    Parameters:
        colors (list): List of colors to use in the palette.
    """
    sns.set_palette(colors)


def position_legend_bottom():
    """
    Function to position the legend at the bottom.
    """
    sns.set()
    
    data = [[3, 4, 2], [5, 2, 1], [2, 3, 6]]
    x_labels = ['A', 'B', 'C']
    colors = ['red', 'green', 'blue']
    
    fig, ax = plt.subplots()
    
    for i in range(len(data)):
        ax.bar(x_labels, data[i], bottom=sum(data[:i]), color=colors[i])
    
    ax.legend(labels=['Category 1', 'Category 2', 'Category 3'], loc='lower center')
    
    plt.show()


def set_custom_palette(colors):
    """
    Function to change the color palette used in stacked bar charts and line charts.
    
    Parameters:
        colors (list): List of colors to use in the palette.
    """
    sns.set(style="ticks", font="Georgia", rc={"axes.labelsize": 10, "legend.fontsize": 10})
    sns.set_palette(colors)


def set_reference_lines_light_gray():
   """
   Function to set reference lines to light gray.
   """
   plt.rcParams['axes.edgecolor'] = 'lightgray'
   plt.rcParams['axes.grid'] = False
   plt.rcParams['grid.color'] = 'lightgray'
   plt.rcParams['axes.linewidth'] = 0.5


def set_minimal_style():
    """
    Function to set the minimal style for the charts.
    """
    sns.set_style("whitegrid")
    
    sns.set(font="Georgia", font_scale=1.4)
    
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2)
    
    plt.axhline(color='lightgray')
    plt.axvline(color='lightgray')


def create_stacked_bar_chart(data, colors):
    """
    Function to create a stacked bar chart with specified colors.
    
    Parameters:
        data (DataFrame): Data for the chart.
        colors (list): List of colors to use in the chart.
    """
    sns.set(style="ticks")
    
    plt.rcParams["font.family"] = "Georgia"
    
    plt.rcParams["axes.titlesize"] = 14
    
    plt.rcParams["axes.labelsize"] = 10
    
    ax = sns.barplot(data=data, palette=colors)
    
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=len(data.columns))
    
    plt.show()


def create_line_chart(data, x, y, colors):
    """
    Function to create a line chart with specified colors.
    
    Parameters:
        data (DataFrame): Data for the chart.
        x (str): X-axis column name.
        y (str): Y-axis column name.
        colors (list): List of colors to use in the chart.
    """
    sns.set_style("white")
    
    sns.set(font='Georgia', font_scale=1.2)
    
    sns.lineplot(x=x, y=y, data=data, palette=colors)
    
    plt.title("Line Chart", fontsize=14)
    
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=len(colors))
    
    plt.show(