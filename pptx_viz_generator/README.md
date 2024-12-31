# PresentationBuilder Class Documentation

## Overview
The `PresentationBuilder` class provides a convenient way to create and manage PowerPoint presentations using the `pptx-python` library. It allows users to add slides with specified layouts and save the presentation to a specified file.

## Class Definition

### `PresentationBuilder`

#### Methods

- **`__init__(self)`**
  - **Description:** Initializes a new `PresentationBuilder` object, setting up an empty presentation.
  - **Returns:** None

- **`add_slide(self, layout)`**
  - **Description:** Adds a new slide to the presentation using the specified `SlideLayout`.
  - **Args:**
    - `layout` (`SlideLayout`): A `SlideLayout` object that defines the layout for the new slide.
  - **Returns:**
    - `slide` (`Slide`): The newly added slide object.

- **`save(self, file_path)`**
  - **Description:** Saves the current state of the presentation to a specified file path.
  - **Args:**
    - `file_path` (`str`): The path where the presentation file will be saved.
  - **Returns:** None


# SlideLayout Class Documentation

## Overview
The `SlideLayout` class provides a mechanism to define and apply layouts to slides in a PowerPoint presentation. It allows users to specify the visual elements such as font style and color schemes that will be applied to a slide, ensuring consistent branding and styling.

## Class Definition

### `SlideLayout`

#### Attributes

- **layout_name** (`str`): 
  - The name of the layout for identification and reference.
  
- **font** (`str`): 
  - The font style to be used in the slide layout.

- **color_scheme** (`str`): 
  - The color scheme to be applied to the slide, guiding the visual appearance.

#### Methods

- **`__init__(self, layout_name, font, color_scheme)`**
  - **Description:** Initializes a new `SlideLayout` object with the specified layout name, font, and color scheme.
  - **Parameters:**
    - `layout_name` (`str`): The name of the layout for identification.
    - `font` (`str`): The font style to be applied in the slide.
    - `color_scheme` (`str`): The color scheme to apply to the slide.
  - **Returns:** None

- **`apply_layout(self, slide)`**
  - **Description:** Applies the defined layout, font, and color scheme to the specified slide.
  - **Parameters:**
    - `slide` (`Slide`): The slide object to which the layout will be applied.
  - **Returns:**
    - `Slide`: The modified slide object with the updated layout and styling.


# PlotlyChartEmbedder Class Documentation

## Overview
The `PlotlyChartEmbedder` class is designed to facilitate the embedding of Plotly charts into PowerPoint slides using the `python-pptx` library. It provides an easy way to include dynamic visualizations within presentations.

## Class Definition

### `PlotlyChartEmbedder`

#### Attributes

- **chart_data** (`plotly.graph_objects.Figure`): 
  - Stores the data necessary for creating the Plotly chart. Typically a Plotly figure object.

#### Methods

- **`__init__(self, chart_data)`**
  - **Description:** Initializes a new `PlotlyChartEmbedder` object with the specified chart data.
  - **Parameters:**
    - `chart_data`: The data used to create a Plotly chart, typically a Plotly figure object.
  - **Returns:** None

- **`embed_chart(self, slide, position)`**
  - **Description:** Embeds a Plotly chart into the specified slide at a given position.
  - **Parameters:**
    - `slide` (`Slide`): The slide object where the chart will be embedded.
    - `position` (`tuple`): A tuple of (x, y, width, height) defining the chart's position and dimensions on the slide, where x and y are the coordinates, and width and height define the chart's size.
  - **Returns:**
    - `Slide`: The modified slide object with the embedded chart.


# BrandingFormatter Class Documentation

## Overview
The `BrandingFormatter` class is designed to apply branding guidelines to a PowerPoint presentation. This includes setting consistent fonts, colors, and other styling attributes across all slides to align with corporate branding standards.

## Class Definition

### `BrandingFormatter`

#### Attributes

- **branding_guidelines** (`dict`): 
  - A dictionary that stores the branding elements such as fonts, colors, and other style attributes to be applied to the presentation.

#### Methods

- **`__init__(self, branding_guidelines)`**
  - **Description:** Initializes a new `BrandingFormatter` object with the specified branding guidelines.
  - **Parameters:**
    - `branding_guidelines` (`dict`): A dictionary containing branding elements like fonts and colors. Example keys might include 'font' and 'color_scheme'.
  - **Returns:** None

- **`apply_branding(self, presentation)`**
  - **Description:** Iterates through each slide in the PowerPoint presentation and applies the branding guidelines to the text elements.
  - **Parameters:**
    - `presentation` (`Presentation`): The PowerPoint presentation object to which the branding will be applied.
  - **Returns:**
    - `Presentation`: The modified presentation object with the branding applied to all relevant slides and shapes.


# load_branding_guidelines Function Documentation

## Overview
The `load_branding_guidelines` function is used to load and parse branding guidelines from a specified JSON file. It provides a structured representation of branding elements that can be used throughout an application, such as for styling presentations or documents.

## Parameters

- **guidelines_file** (`str`): 
  - The file path to the branding guidelines in JSON format.

## Returns

- **dict**: 
  - A dictionary containing the parsed branding elements, such as font styles, color schemes, and other relevant branding information.

## Raises

- **FileNotFoundError**: 
  - If the specified guidelines file does not exist.

- **ValueError**: 
  - If there is an error parsing the contents of the file, indicating that the file's content is not valid JSON.


# create_custom_layout Function Documentation

## Overview
The `create_custom_layout` function creates a custom slide layout for use in PowerPoint presentations based on specified layout details. This function returns a `SlideLayout` object that defines how the title and content of a slide should be arranged and styled.

## Parameters

- **layout_details** (`dict`): 
  - A dictionary containing the following layout configurations:
    - `title_position` (`tuple`): A tuple representing the position (x, y) and size (width, height) of the title box on the slide, specified in inches. Defaults to `(1, 1, 5, 1)`.
    - `content_position` (`tuple`): A tuple representing the position (x, y) and size (width, height) of the content area on the slide, specified in inches. Defaults to `(1, 2, 5, 4)`.
    - `background_color` (`str`): A hexadecimal string representing the background color of the slide. Defaults to `'FFFFFF'` (white).
    - `font_style` (`dict`): A dictionary containing font style attributes:
      - `family` (`str`): The font family to be used for the title. Default is `'Arial'`.
      - `size` (`int`): The font size for the title text. Default is `12`.
      - `color` (`str`): A hexadecimal string for the title font color. Default is `'000000'` (black).

## Returns

- **SlideLayout**: 
  - A custom `SlideLayout` object configured according to the provided layout details, including title and content positioning, background color, and font styles.

## Example Usage



# generate_interactive_chart Function Documentation

## Overview
The `generate_interactive_chart` function creates an interactive chart using Plotly based on the provided data and specified chart type. This function facilitates easy visualization of data structures in various formats.

## Parameters

- **data**: 
  - The input data structure for the chart, which may be a:
    - **pandas DataFrame**: A table-like data structure that allows complex data manipulation.
    - **dictionary**: A mapping of labels to values.
    - **list**: A sequence of values.

- **chart_type** (`str`): 
  - Specifies the type of chart to generate. Supported types include:
    - `'bar'`: Generates a bar chart.
    - `'line'`: Generates a line chart.
    - `'scatter'`: Generates a scatter plot.
    - `'pie'`: Generates a pie chart.

## Returns

- **go.Figure**: 
  - A Plotly Figure object representing the generated interactive chart.

## Raises

- **ValueError**: 
  - If the `chart_type` is unsupported or if the data format is incompatible with the specified chart type.

## Example Usage

