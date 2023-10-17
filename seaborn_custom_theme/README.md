# Overview

This python script provides various functions to customize the appearance of seaborn and matplotlib charts. It includes functions to set color palettes, font styles, font sizes, legend positions, reference line colors, grid lines, and chart themes. Additionally, it provides functions to create stacked bar charts and line charts with customizable colors.

# Usage

To use this script, you need to have the seaborn and matplotlib packages installed. You can install them using pip:

```
pip install seaborn matplotlib
```

After installing the required packages, you can import the script into your python code using the following line:

```python
import chart_customizer
```

Once imported, you can use any of the provided functions by calling them directly. For example, to set the color palette for stacked bar charts:

```python
chart_customizer.set_bar_chart_palette(['red', 'green', 'blue'])
```

# Examples

Here are some examples demonstrating the usage of the functions in this script:

1. **Set Bar Chart Palette**

   ```python
   import chart_customizer
   
   chart_customizer.set_bar_chart_palette(['red', 'green', 'blue'])
   ```

   This function sets the color palette for stacked bar charts. The `colors` parameter is a list of colors to be used in the palette.

2. **Set Line Chart Palette**

   ```python
   import chart_customizer
   
   chart_customizer.set_line_chart_palette(['orange', 'purple', 'yellow'])
   ```

   This function sets the color palette for line charts. The `colors` parameter is a list of colors to be used in the palette.

3. **Set Georgia Font**

   ```python
   import chart_customizer
   
   chart_customizer.set_georgia_font()
   ```

   This function sets the font style as Georgia for all chart elements.

4. **Set Title Font Size**

   ```python
   import chart_customizer
   
   chart_customizer.set_title_font_size(16)
   ```

   This function sets the font size for chart titles. The `font_size` parameter is an integer representing the desired font size.

5. **Set Label Font Size**

   ```python
   import chart_customizer
   
   chart_customizer.set_label_font_size()
   ```

   This function sets the font size for chart labels to 10 point.

6. **Set Legend Bottom**

   ```python
   import chart_customizer
   
   # Create a seaborn plot object
   plot = sns.barplot(data=data)
   
   chart_customizer.set_legend_bottom(plot)
   ```

   This function places the legend at the bottom of the chart. The `plot` parameter should be a seaborn plot object.

7. **Set Reference Lines Color**

   ```python
   import chart_customizer
   
   chart_customizer.set_reference_lines_color()
   ```

   This function sets the color of reference lines in charts as light gray.

8. **Remove Grid Lines**

   ```python
   import chart_customizer
   
   chart_customizer.remove_grid_lines()
   ```

   This function removes grid lines around bounding boxes in charts.

9. **Set Minimal Theme**

    ```python
    import chart_customizer
    
    chart_customizer.set_minimal_theme()
    ```

    This function sets the default seaborn theme to a minimal style, including font styles, font scales, label sizes, legend location, and grid lines.

10. **Set Chart Colors**

    ```python
    import chart_customizer
    
    chart_customizer.set_chart_colors(['purple', 'yellow'])
    ```

    This function sets the list of colors for stacked bar charts and line charts. The `colors` parameter is a list of colors to be used in the palette.

11. **Position Legend Bottom**

    ```python
    import chart_customizer
    
    chart_customizer.position_legend_bottom()
    ```

    This function positions the legend at the bottom of the chart. It demonstrates how to create a stacked bar chart with custom colors and a legend at the bottom.

12. **Set Custom Palette**

    ```python
    import chart_customizer
    
    chart_customizer.set_custom_palette(['pink', 'gray'])
    ```

    This function changes the color palette used in stacked bar charts and line charts. The `colors` parameter is a list of colors to be used in the palette.

13. **Set Reference Lines Light Gray**

    ```python
    import chart_customizer
    
    chart_customizer.set_reference_lines_light_gray()
    ```

    This function sets reference lines in charts to light gray.

14. **Set Minimal Style**

    ```python
    import chart_customizer
    
    chart_customizer.set_minimal_style()
    ```

    This function sets a minimal style for charts, including font styles, font scales, tick label sizes, legend location, and light gray reference lines.

15. **Create Stacked Bar Chart**

    ```python
    import pandas as pd
    import chart_customizer
    
    data = pd.DataFrame({'Category 1': [3, 4, 2], 'Category 2': [5, 2, 1], 'Category 3': [2, 3, 6]})
    
    chart_customizer.create_stacked_bar_chart(data, ['red', 'green', 'blue'])
   ```

   This function creates a stacked bar chart with specified colors. The `data` parameter is a pandas DataFrame containing the data for the chart, and the `colors` parameter is a list of colors to be used in the palette.

16. **Create Line Chart**

   ```python
   import pandas as pd
   import chart_customizer
   
   data = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [5, 4, 3, 2, 1]})
   
   chart_customizer.create_line_chart(data, 'x', 'y', ['orange', 'purple'])
   ```

   This function creates a line chart with specified colors. The `data` parameter is a pandas DataFrame containing the data for the chart, the `x` parameter is a string representing the column name for the x-axis, the `y` parameter is a string representing the column name for the y-axis, and the `colors` parameter is a list of colors to be used in the palette.