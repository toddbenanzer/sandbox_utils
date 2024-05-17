# PowerPoint Presentation Generator

This package provides a set of functions for generating PowerPoint presentations using the python-pptx library and the Plotly library. The package allows users to easily create slides, add text and images, and insert Plotly charts into their presentations.

## Installation

To install the package, simply run:

```bash
pip install pptx plotly
```

## Usage

To use the package, first import the necessary modules:

```python
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.chart import XL_LABEL_POSITION, XL_TICK_LABEL_POSITION, XL_LEGEND_POSITION, XL_CHART_TYPE
import plotly.graph_objects as go
import os
```

### Creating a Presentation

To create a new PowerPoint presentation object, use the `create_presentation()` function:

```python
prs = create_presentation()
```

### Adding Slides

You can add slides to the presentation using different slide layouts. The available layouts are represented by numbers starting from 0. Use the `add_slide()` function to add a slide with a specific layout:

```python
slide_layout = 1  # Use slide layout number 1
slide = add_slide(prs, slide_layout)
```
### Adding Text

To add text to a slide, use the `add_text_to_slide()` function:

```python
slide_index = 0  # Add text to slide with index 0 (the first slide)
text = "Hello World!"
add_text_to_slide(prs, slide_index, text)
```

### Adding Images

To add an image to a slide, use the `add_image_to_slide()` function:

```python
slide_index = 0  # Add image to slide with index 0 (the first slide)
image_path = "image.png"  # Specify the path to the image file
left = 1  # Specify the position of the image on the slide
top = 1
width = 6  # Specify the width and height of the image
height = 4.5
add_image_to_slide(prs.slides[slide_index], image_path, left, top, width, height)
```

### Adding Plotly Charts

To add a Plotly chart to a slide, use the `add_plotly_chart_to_slide()` function:

```python
slide_index = 0  # Add chart to slide with index 0 (the first slide)
chart_data = [...]  # Specify the data for the chart
chart = go.Figure(data=chart_data)  # Create a Plotly chart object
add_plotly_chart_to_slide(prs, slide_index, chart)
```

### Formatting Text

You can format the font of text on a slide using the `format_text_font()` function:

```python
text_frame = textbox.text_frame  # Get the text frame of a shape containing text
format_text_font(text_frame.text, font_name="Arial", font_size=12, bold=True, italic=False, underline=True, align="center")
```

### Formatting Background Color

You can format the background color of a slide using the `format_background_color()` function:

```python
slide = prs.slides[slide_index]  # Get a specific slide
color = (255, 255, 255)  # Specify the RGB color values (e.g., white)
format_background_color(slide, color)
```

### Exporting Presentation

To export your presentation to a PowerPoint file or a PDF file, use the `export_presentation()` and `save_ppt_as_pdf()` functions respectively:

```python
output_file_pptx = "presentation.pptx"
output_file_pdf = "presentation.pdf"
export_presentation(prs, output_file_pptx)
save_ppt_as_pdf(output_file_pptx, output_file_pdf)
```

## Examples

Here are some examples demonstrating how to use the package:

### Example 1: Creating a Simple Presentation

```python
# Create a new presentation object
prs = create_presentation()

# Add a title slide
title = "My Presentation"
add_title_slide(prs, title)

# Add a slide with text
slide_layout = 1
slide = add_slide(prs, slide_layout)
text = "This is a sample text."
add_text_to_slide(prs, 1, text)

# Export the presentation to a PowerPoint file
output_file = "presentation.pptx"
export_presentation(prs, output_file)
```

### Example 2: Adding an Image and Formatting Text

```python
# Create a new presentation object
prs = create_presentation()

# Add a title slide
title = "My Presentation"
add_title_slide(prs, title)

# Add a slide with an image and formatted text
slide_layout = 1
slide = add_slide(prs, slide_layout)
image_path = "image.png"
left = 1
top = 1
width = 6
height = 4.5
add_image_to_slide(slide, image_path, left, top, width, height)
text = "This is a sample text."
add_text_to_slide(prs, 1, text)

# Format the font of the text on the slide
format_text_font(slide.shapes[1].text_frame.text,
                 font_name="Arial",
                 font_size=12,
                 bold=True,
                 italic=False,
                 underline=True,
                 align="center")

# Export the presentation to a PowerPoint file
output_file = "presentation.pptx"
export_presentation(prs, output_file)
```

### Example 3: Adding a Plotly Chart

```python
# Create a new presentation object
prs = create_presentation()

# Add a title slide
title = "My Presentation"
add_title_slide(prs, title)

# Add a slide with a Plotly chart
slide_layout = 1
slide = add_slide(prs, slide_layout)
chart_data = [...]  # Specify the data for the chart
chart = go.Figure(data=chart_data)  # Create a Plotly chart object
add_plotly_chart_to_slide(prs, 1, chart)

# Export the presentation to a PowerPoint file
output_file = "presentation.pptx"
export_presentation(prs, output_file)
```

## License

This package is licensed under the MIT License.