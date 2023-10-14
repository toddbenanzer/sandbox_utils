## Overview

This python script provides functions to create, modify, and format PowerPoint presentations using the `pptx` library. The script allows users to create empty presentations, define slide layouts based on branding guidelines, embed Plotly charts, format presentations according to branding guidelines, add titles, subtitles, text boxes, images, tables, bullet points to slides, set text formats and slide backgrounds, save presentations as files, open existing presentations, insert text into placeholders on slides, and apply slide layouts and formatting guidelines.

## Usage

To use this script, you need to have the `pptx` library installed. You can install it using pip:

```bash
pip install python-pptx
```

Once you have the `pptx` library installed, you can import the necessary functions from the script and start using them in your own Python code.

## Examples

Here are some examples of how the functions in this script can be used:

1. Create an empty presentation:

```python
from ppt_functions import create_empty_presentation

presentation = create_empty_presentation()
```

2. Define slide layouts based on branding guidelines:

```python
from ppt_functions import define_slide_layouts

branding_guidelines = {
    'slide_layout': 'Title Slide'
}

define_slide_layouts(presentation, branding_guidelines)
```

3. Embed a Plotly chart into a slide:

```python
from ppt_functions import embed_plotly_chart

slide_index = 0
chart_data = [
    ['Category', 'Value'],
    ['A', 10],
    ['B', 20],
    ['C', 15]
]

presentation = embed_plotly_chart(presentation, slide_index, chart_data)
```

4. Format the presentation according to branding guidelines:

```python
from ppt_functions import format_presentation

format_presentation(presentation)
```

5. Add a title and subtitle to a slide:

```python
from ppt_functions import add_title_and_subtitle

slide = presentation.slides[0]
title = 'Title'
subtitle = 'Subtitle'

add_title_and_subtitle(slide, title, subtitle)
```

6. Add a text box to a slide:

```python
from ppt_functions import add_text_box

slide = presentation.slides[0]
content = 'This is some text.'

add_text_box(slide, content)
```

7. Add an image to a slide:

```python
from ppt_functions import add_image_to_slide

slide = presentation.slides[0]
image_path = 'path/to/image.png'
left = 1  # inches
top = 1  # inches
width = 3  # inches
height = 3  # inches

add_image_to_slide(slide, image_path, left, top, width, height)
```

8. Add a table to a slide:

```python
from ppt_functions import add_table_to_slide

slide_index = 0
data = [
    ['Name', 'Age'],
    ['John', 25],
    ['Jane', 30]
]
headers = ['Name', 'Age']

presentation = add_table_to_slide(presentation, slide_index, data, headers)
```

9. Add bullet points to a slide:

```python
from ppt_functions import add_bullet_points

slide = presentation.slides[0]
content = ['Item 1', 'Item 2', 'Item 3']

add_bullet_points(slide, content)
```

10. Set the font style, size, and color for text in slides:

```python
from ppt_functions import set_text_format

slide_index = 0
text_index = 0
font_name = 'Arial'
font_size = 12
font_color = (255, 0, 0)  # Red

set_text_format(presentation, slide_index, text_index, font_name, font_size, font_color)
```

11. Set the background color or image for slides:

```python
from ppt_functions import set_slide_background

slide_index = 0
background = 'path/to/background.png'

presentation = set_slide_background(presentation, slide_index, background)
```

12. Save the generated presentation as a file:

```python
from ppt_functions import save_presentation

filename = 'output.pptx'

save_presentation(presentation, filename)
```

13. Create a new presentation:

```python
from ppt_functions import create_presentation

presentation = create_presentation()
```

14. Open an existing presentation:

```python
from ppt_functions import open_presentation

file_path = 'path/to/presentation.pptx'

presentation = open_presentation(file_path)
```

15. Insert text into placeholders on slides:

```python
from ppt_functions import insert_text

slide_index = 0
placeholder_id = 1  # Placeholder ID for title
text = 'This is the title'

insert_text(presentation, slide_index, placeholder_id, text)
```

16. Apply slide layouts and formatting guidelines:

```python
from ppt_functions import apply_slide_layout

slide_index = 0
layout_name = 'Title Slide'

apply_slide_layout(presentation, slide_index, layout_name)
```