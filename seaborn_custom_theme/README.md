# CustomTheme Class Documentation

## Overview
The `CustomTheme` class encapsulates a custom theme for Seaborn charts, allowing users to specify color palettes, font sizes, and other styling options to enhance visualizations.

## Attributes
- **colors (list)**: 
  - A list of colors to be used in charts, specified as color codes (e.g., hex values).

- **title_font_size (int)**: 
  - The font size for chart titles. Default is 14.

- **label_font_size (int)**: 
  - The font size for other chart labels. Default is 10.

## Methods

### `__init__(self, colors, title_font_size=14, label_font_size=10)`
Initializes the `CustomTheme` with specific style settings.

#### Parameters
- **colors (list)**: 
  - A list of color codes to be applied to the charts.

- **title_font_size (int, optional)**: 
  - Font size for the titles. Default is 14.

- **label_font_size (int, optional)**: 
  - Font size for labels. Default is 10.

### `apply_theme(self)`
Applies the custom theme settings to Seaborn plots.

#### Functionality
- Sets the color palette based on the user's input.
- Configures the aesthetics of the plots, including title and label font sizes, and font family settings.
- Establishes the style used for the plots, including axis colors and grid line styles.
- Positions the legend at the bottom of the charts.

## Example Usage


# set_color_palette Function Documentation

## Overview
The `set_color_palette` function sets and applies a specific color palette for Seaborn plots, allowing users to customize the colors used in their visualizations.

## Parameters
- **colors (list)**: 
  - A list of color codes (e.g., hex codes or color names) that specifies the colors to be used in the charts.

## Returns
- **None**: 
  - This function updates the Seaborn color palette without returning any value.

## Exceptions
- Raises a **ValueError** if the provided `colors` argument is not a list or if any of the elements in the list are not string color codes.

## Example Usage


# set_fonts Function Documentation

## Overview
The `set_fonts` function configures the font settings for Seaborn plots, allowing users to customize the font sizes for titles and labels, as well as the font face used in the visualizations.

## Parameters
- **title_font_size (int)**: 
  - Specifies the font size for chart titles. This should be an integer value indicating the desired size.

- **label_font_size (int)**: 
  - Specifies the font size for other chart labels. This should also be an integer value indicating the desired size.

- **font_face (str, optional)**: 
  - Specifies the font face to be used for all text elements in the plots. The default value is 'Georgia'. This should be a string representing the font family.

## Returns
- **None**: 
  - This function modifies the font settings for Seaborn plots and does not return any value.

## Exceptions
- Raises a **ValueError** if either `title_font_size` or `label_font_size` is not an integer.
- Raises a **ValueError** if `font_face` is not a string.

## Example Usage


# position_legend Function Documentation

## Overview
The `position_legend` function configures the position of the legend in Seaborn plots, allowing users to easily specify where the legend should appear relative to the plot area.

## Parameters
- **at (str, optional)**: 
  - Specifies the location of the legend. Accepted values include:
    - `'top'` - Position the legend at the top of the plot.
    - `'bottom'` - Position the legend at the bottom of the plot.
    - `'left'` - Position the legend on the left side of the plot.
    - `'right'` - Position the legend on the right side of the plot.
    - Compound positions like `'upper right'`, which place the legend in a specific section of the plot.
  - Default value is `'bottom'`.

## Returns
- **None**: 
  - This function modifies the legend position directly in the plot, without returning a value.

## Exceptions
- Raises a **ValueError** if the `at` parameter is not specified as a string.

## Example Usage


# configure_reference_lines Function Documentation

## Overview
The `configure_reference_lines` function configures the appearance of reference lines in Seaborn plots, allowing users to specify the color and style of the grid lines for enhanced visual clarity and aesthetics.

## Parameters
- **color (str, optional)**: 
  - Specifies the color of the reference lines. The default value is `'lightgray'`. This can be any valid color representation, such as a color name or a hex code.

- **grid_style (str, optional)**: 
  - Specifies the style of the grid lines. Accepted values include:
    - `'solid'` for solid lines
    - `'dashed'` for dashed lines
    - `'dotted'` for dotted lines
    - `'minimal'` for a minimalistic appearance 
  - The default value is `'minimal'`.

## Returns
- **None**: 
  - This function modifies the reference lines' appearance in the plots directly, without returning a value.

## Exceptions
- Raises a **ValueError** if the `color` parameter is not specified as a string.
- Raises a **ValueError** if the `grid_style` parameter is not specified as a string.
- Raises a **ValueError** if the provided `grid_style` is not one of the accepted values.

## Example Usage
