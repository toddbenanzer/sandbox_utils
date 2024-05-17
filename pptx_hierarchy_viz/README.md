## Overview

This python script provides a set of functions for creating and manipulating PowerPoint presentations using the `pptx` library. It includes functionality for creating blank slides, adding text, images, charts, tables, shapes, hyperlinks, and more. The script also includes functions for setting font styles and sizes, slide background colors, text alignments, border styles, and saving/opening presentations. Additionally, there are functions for creating tree visualizations, organizational charts, hierarchical visualizations, and exporting slides as images or csv files.

## Usage

To use this script, you will need to have the `pptx` and `pandas` libraries installed. You can install them using pip:

```
pip install python-pptx pandas
```

Once you have the required dependencies installed, you can import the script into your Python project using:

```python
from pptx_functions import *
```

Then you can use any of the provided functions by calling them with the required arguments. For example:

```python
prs = Presentation()
slide = create_blank_slide(prs)
add_text_to_slide(slide, "Hello World")
save_presentation_as_file(prs, "presentation.pptx")
```

## Examples

### Creating a Blank Slide and Adding Text

```python
prs = Presentation()
slide = create_blank_slide(prs)
add_text_to_slide(slide, "Hello World")
save_presentation_as_file(prs, "presentation.pptx")
```

In this example, we create a new presentation object `prs`, then create a blank slide using the `create_blank_slide` function and assign it to the variable `slide`. We then add text to the slide using the `add_text_to_slide` function. Finally, we save the presentation as a file named "presentation.pptx" using the `save_presentation_as_file` function.

### Adding an Image to a Slide

```python
prs = Presentation()
slide = create_blank_slide(prs)
add_image_to_slide(slide, "image.jpg", left=Inches(1), top=Inches(1), width=Inches(4), height=Inches(3))
save_presentation_as_file(prs, "presentation.pptx")
```

In this example, we create a new presentation object `prs`, then create a blank slide using the `create_blank_slide` function and assign it to the variable `slide`. We then add an image to the slide using the `add_image_to_slide` function. The image is specified by the file path "image.jpg". We also specify the position and size of the image on the slide. Finally, we save the presentation as a file named "presentation.pptx" using the `save_presentation_as_file` function.

### Adding a Chart to a Slide

```python
prs = Presentation()
slide = create_blank_slide(prs)
data = [
    ["Category 1", "Category 2", "Category 3"],
    [10, 20, 30],
    [40, 50, 60]
]
chart = add_chart(slide, data, chart_type=XL_CHART_TYPE.COLUMN_CLUSTERED)
save_presentation_as_file(prs, "presentation.pptx")
```

In this example, we create a new presentation object `prs`, then create a blank slide using the `create_blank_slide` function and assign it to the variable `slide`. We then create some data for the chart in the form of a nested list. We pass this data along with the chart type (in this case XL_CHART_TYPE.COLUMN_CLUSTERED) to the `add_chart` function to add a chart to the slide. Finally, we save the presentation as a file named "presentation.pptx" using the `save_presentation_as_file` function.

### Adding a Table to a Slide

```python
prs = Presentation()
slide = create_blank_slide(prs)
table = add_table_to_slide(slide, rows=3, columns=3)
table.cell(0, 0).text = "Header 1"
table.cell(0, 1).text = "Header 2"
table.cell(0, 2).text = "Header 3"
table.cell(1, 0).text = "Data 1"
table.cell(1, 1).text = "Data 2"
table.cell(1, 2).text = "Data 3"
save_presentation_as_file(prs, "presentation.pptx")
```

In this example, we create a new presentation object `prs`, then create a blank slide using the `create_blank_slide` function and assign it to the variable `slide`. We then add a table to the slide using the `add_table_to_slide` function and assign it to the variable `table`. We can access individual cells in the table using the `cell` method of the table object. Finally, we save the presentation as a file named "presentation.pptx" using the `save_presentation_as_file` function.

These are just a few examples of what you can do with this script. Please refer to the function definitions and their respective comments for more information on how to use each function.