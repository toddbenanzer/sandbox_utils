# README

## Overview

This python script is used to create tables in PowerPoint presentations (.pptx) using the `pptx` library. The `create_table` function takes in a `Presentation` object, slide index, data, and column names as arguments, and generates a table on the specified slide with the given data.

## Usage

To use this script, you will need to have the `pptx` library installed. You can install it using pip:

```bash
pip install python-pptx
```

Once you have the library installed, you can import it into your python script:

```python
from pptx import Presentation
```

Next, you can define your data and column names for the table:

```python
data = [
    ['John', 'Doe', 25],
    ['Jane', 'Smith', 30],
]
column_names = ['First Name', 'Last Name', 'Age']
```

To create a new PowerPoint presentation object and add a table to a slide, use the `create_table` function:

```python
prs = Presentation()
slide_idx = 0

create_table(prs, slide_idx, data, column_names)
```

Finally, save the presentation as a .pptx file:

```python
prs.save('table.pptx')
```

## Examples

Here is an example of how to create a table in a PowerPoint presentation using this script:

```python
from pptx import Presentation

# Create a new presentation object
prs = Presentation()

# Specify the index of the slide where the table will be added
slide_idx = 0

# Define the data for the table
data = [
    ['John', 'Doe', 25],
    ['Jane', 'Smith', 30],
]

# Define the column names for the table
column_names = ['First Name', 'Last Name', 'Age']

# Create the table on the specified slide
create_table(prs, slide_idx, data, column_names)

# Save the presentation as a .pptx file
prs.save('table.pptx')
```

This will create a PowerPoint presentation with a table on the first slide, displaying the provided data and column names.