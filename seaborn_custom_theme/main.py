import matplotlib.pyplot as plt
import seaborn as sns


class CustomTheme:
    """
    A class to encapsulate a custom theme for seaborn charts.

    Attributes:
        colors (list): List of colors to be used in charts.
        title_font_size (int): Font size for chart titles.
        label_font_size (int): Font size for other chart labels.
    """

    def __init__(self, colors, title_font_size=14, label_font_size=10):
        """
        Initializes the CustomTheme with specific style settings.

        Args:
            colors (list): A list of color codes for the charts.
            title_font_size (int, optional): Font size for the titles. Default is 14.
            label_font_size (int, optional): Font size for labels. Default is 10.
        """
        self.colors = colors
        self.title_font_size = title_font_size
        self.label_font_size = label_font_size

    def apply_theme(self):
        """
        Applies the custom theme settings to seaborn plots.
        """
        # Set color palette
        sns.set_palette(sns.color_palette(self.colors))

        # Configure the aesthetics for the plots
        sns.set_context("notebook", rc={
            'axes.titlesize': self.title_font_size,
            'axes.labelsize': self.label_font_size,
            'font.family': 'serif',
            'font.serif': 'Georgia'
        })

        sns.set_style("whitegrid", {
            'axes.edgecolor': 'lightgray',
            'grid.color': 'lightgray',
            'grid.linestyle': '--'
        })

        # Configure the legend position
        plt.rc('legend', loc='lower center')



def set_color_palette(colors):
    """
    Sets and applies a specific color palette for Seaborn plots.

    Args:
        colors (list): A list of color codes (e.g., hex codes or color names) to be used for the charts.

    Returns:
        None: Updates the Seaborn color palette without returning a value.
    """
    if not isinstance(colors, list) or not all(isinstance(color, str) for color in colors):
        raise ValueError("Colors must be a list of string color codes.")
    
    sns.set_palette(sns.color_palette(colors))



def set_fonts(title_font_size, label_font_size, font_face='Georgia'):
    """
    Configures the font settings for Seaborn plots.

    Args:
        title_font_size (int): The font size for chart titles.
        label_font_size (int): The font size for other chart labels.
        font_face (str, optional): The font face to be used for text elements. Default is 'Georgia'.

    Returns:
        None: The function modifies the font settings for plots without returning a value.
    """
    if not isinstance(title_font_size, int) or not isinstance(label_font_size, int):
        raise ValueError("Font sizes must be integers.")
    if not isinstance(font_face, str):
        raise ValueError("Font face must be a string.")
    
    # Configure font settings for Seaborn plots
    sns.set_context("notebook", rc={
        'axes.titlesize': title_font_size,
        'axes.labelsize': label_font_size,
        'font.family': 'serif',
        'font.serif': font_face
    })



def position_legend(at='bottom'):
    """
    Configures the position of the legend in Seaborn plots.

    Args:
        at (str, optional): Specifies the location of the legend. Accepted values include 'top',
        'bottom', 'left', 'right', and compound positions like 'upper right'. Default is 'bottom'.

    Returns:
        None: The function modifies the legend position without returning a value.
    """
    if not isinstance(at, str):
        raise ValueError("Legend position must be specified as a string.")

    # Set legend location using the `rc` parameters
    plt.rc('legend', loc=at)



def configure_reference_lines(color='lightgray', grid_style='minimal'):
    """
    Configures the appearance of reference lines in Seaborn plots.

    Args:
        color (str, optional): Specifies the color of the reference lines. Default is 'lightgray'.
        grid_style (str, optional): Specifies the style of the grid lines. Accepted values can include 
        'solid', 'dashed', 'dotted', or 'minimal' for minimalistic appearance. Default is 'minimal'.

    Returns:
        None: The function modifies the reference lines' appearance without returning a value.
    """
    if not isinstance(color, str):
        raise ValueError("Color must be specified as a string.")
    if not isinstance(grid_style, str):
        raise ValueError("Grid style must be specified as a string.")

    # Set grid line properties based on specified style
    if grid_style == 'minimal':
        line_style = '--'
    elif grid_style == 'dotted':
        line_style = ':'
    elif grid_style == 'dashed':
        line_style = '--'
    elif grid_style == 'solid':
        line_style = '-'
    else:
        raise ValueError("Grid style must be 'solid', 'dashed', 'dotted', or 'minimal'.")

    # Apply settings to seaborn's style
    sns.set_style("whitegrid", {
        'grid.color': color,
        'grid.linestyle': line_style,
        'axes.edgecolor': color
    })
