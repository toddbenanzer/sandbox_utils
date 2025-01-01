from color_palette_module import set_color_palette  # Replace with the correct import path
from custom_theme import CustomTheme  # Replace with the correct import path
from font_module import set_fonts  # Replace with the correct import path
from legend_module import position_legend  # Replace with the correct import path
from reference_lines_module import configure_reference_lines  # Replace with the correct import path
import matplotlib.pyplot as plt
import seaborn as sns


# Example 1: Simple Line Plot with Custom Theme
def line_plot_example():
    # Initialize the custom theme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    theme = CustomTheme(colors, title_font_size=16, label_font_size=12)
    
    # Apply the custom theme
    theme.apply_theme()
    
    # Generate sample data
    data = sns.load_dataset('flights')
    data_pivot = data.pivot("month", "year", "passengers")
    
    # Create a heatmap using seaborn with the custom theme
    sns.heatmap(data_pivot, cmap='coolwarm')

    # Display the plot
    plt.title('Monthly Flights over the Years')
    plt.show()

# Example 2: Stacked Bar Plot with Custom Theme
def stacked_bar_example():
    # Initialize the custom theme
    colors = ['#9467bd', '#8c564b', '#e377c2']
    theme = CustomTheme(colors, title_font_size=14, label_font_size=10)
    
    # Apply the custom theme
    theme.apply_theme()
    
    # Generate sample data
    tips = sns.load_dataset('tips')
    
    # Create a stacked bar plot
    sns.histplot(data=tips, x="day", hue="sex", multiple="stack")

    # Display the plot
    plt.title('Tip Count by Day and Sex')
    plt.show()

# Run examples
line_plot_example()
stacked_bar_example()



# Example 1: Basic Line Plot with Custom Colors
def line_plot_example():
    colors = ['#FF5733', '#33FF57', '#3357FF']
    set_color_palette(colors)
    
    # Create a lineplot using the custom palette
    sns.lineplot(x=[0, 1, 2], y=[3, 1, 4])
    plt.title('Line Plot with Custom Color Palette')
    plt.show()

# Example 2: Histogram with Custom Colors
def histogram_example():
    colors = ['#F39C12', '#D35400', '#8E44AD']
    set_color_palette(colors)
    
    # Load example dataset
    tips = sns.load_dataset('tips')
    
    # Create a histogram using the custom palette
    sns.histplot(tips['total_bill'], binwidth=2)
    plt.title('Histogram with Custom Color Palette')
    plt.show()

# Example 3: Bar Plot with Custom Colors
def bar_plot_example():
    colors = ['#16A085', '#2ECC71', '#D4AC0D']
    set_color_palette(colors)
    
    # Load example dataset
    tips = sns.load_dataset('tips')
    
    # Create a bar plot using the custom palette
    sns.barplot(x='day', y='total_bill', data=tips)
    plt.title('Bar Plot with Custom Color Palette')
    plt.show()

# Run examples
line_plot_example()
histogram_example()
bar_plot_example()



# Example 1: Line Plot with Custom Fonts
def line_plot_example():
    # Set custom font settings
    set_fonts(title_font_size=16, label_font_size=12, font_face='Times New Roman')
    
    # Create a line plot
    sns.lineplot(x=[1, 2, 3], y=[3, 7, 5])
    plt.title('Line Plot with Custom Fonts')
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    plt.show()

# Example 2: Bar Plot with Different Font Sizes
def bar_plot_example():
    # Set custom font settings
    set_fonts(title_font_size=20, label_font_size=14)
    
    # Load example dataset
    tips = sns.load_dataset('tips')
    
    # Create a bar plot
    sns.barplot(x='day', y='total_bill', data=tips)
    plt.title('Bar Plot with Different Font Sizes')
    plt.xlabel('Day of the Week')
    plt.ylabel('Total Bill')
    plt.show()

# Example 3: Histogram with Custom Font Face
def histogram_example():
    # Set custom font face
    set_fonts(title_font_size=14, label_font_size=10, font_face='Comic Sans MS')
    
    # Load example dataset
    tips = sns.load_dataset('tips')
    
    # Create a histogram
    sns.histplot(tips['total_bill'], binwidth=5)
    plt.title('Histogram with Custom Font Face')
    plt.xlabel('Total Bill')
    plt.ylabel('Frequency')
    plt.show()

# Run examples
line_plot_example()
bar_plot_example()
histogram_example()



# Example 1: Position Legend at the Top
def legend_top_example():
    position_legend(at='top')
    
    # Create a plot with a legend
    sns.lineplot(x=[1, 2, 3], y=[3, 2, 5], label='Line 1')
    sns.lineplot(x=[1, 2, 3], y=[2, 3, 4], label='Line 2')
    plt.title('Legend at the Top')
    plt.legend()
    plt.show()

# Example 2: Position Legend at Lower Right
def legend_lower_right_example():
    position_legend(at='lower right')
    
    # Create a plot with a legend
    sns.lineplot(x=[1, 2, 3], y=[1, 4, 6], label='Curve 1')
    sns.lineplot(x=[1, 2, 3], y=[3, 5, 7], label='Curve 2')
    plt.title('Legend at Lower Right')
    plt.legend()
    plt.show()

# Example 3: Position Legend on the Left
def legend_left_example():
    position_legend(at='left')
    
    # Create a plot with a legend
    sns.lineplot(x=[1, 2, 3], y=[4, 3, 8], label='Graph A')
    sns.lineplot(x=[1, 2, 3], y=[7, 6, 5], label='Graph B')
    plt.title('Legend on the Left')
    plt.legend()
    plt.show()

# Run examples
legend_top_example()
legend_lower_right_example()
legend_left_example()



# Example 1: Default Reference Lines
def default_reference_lines_example():
    configure_reference_lines()  # Use default lightgray and minimal style

    # Create a simple plot
    sns.lineplot(x=[1, 2, 3], y=[3, 2, 5])
    plt.title('Default Reference Lines')
    plt.show()

# Example 2: Custom Blue Dotted Reference Lines
def custom_blue_dotted_example():
    configure_reference_lines(color='blue', grid_style='dotted')

    # Create a simple plot
    sns.lineplot(x=[1, 2, 3], y=[2, 3, 4])
    plt.title('Blue Dotted Reference Lines')
    plt.show()

# Example 3: Black Solid Reference Lines
def black_solid_example():
    configure_reference_lines(color='black', grid_style='solid')

    # Create a simple plot
    sns.lineplot(x=[1, 2, 3], y=[4, 3, 6])
    plt.title('Black Solid Reference Lines')
    plt.show()

# Run examples
default_reference_lines_example()
custom_blue_dotted_example()
black_solid_example()
