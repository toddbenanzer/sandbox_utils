# Overview

This python script provides functions for converting pandas DataFrames into PowerPoint tables and manipulating the formatting of the tables. It utilizes the `pandas` and `pptx` libraries to achieve this functionality.

# Usage

To use this script, you need to have `pandas` and `pptx` installed in your Python environment. You can install them using pip:

```
pip install pandas
pip install python-pptx
```

The script contains the following main functions:

1. `convert_dataframe_to_ppt_table(df)`: This function takes a pandas DataFrame as input and converts it into a PowerPoint table. It returns a `Presentation` object containing the table.

2. `add_table_to_slide(prs_path, slide_index, df)`: This function adds a table to a specific slide in an existing PowerPoint presentation. It takes the path of the presentation file, the index of the slide, and the DataFrame as input. It modifies the presentation file in-place.

3. `add_currency_table(presentation, df)`: This function adds a currency-formatted table to a new slide in an existing PowerPoint presentation. It takes the presentation object and the DataFrame as input. It returns the modified presentation object.

4. `format_column_as_percentage(table, column_index)`: This function formats a specific column in a PowerPoint table as a percentage. It takes the table object and the index of the column as input.

5. Utility functions: The script also includes several utility functions for setting various formatting properties of PowerPoint tables.

To use any of these functions, simply import them from the script and call them with appropriate parameters.

# Examples

Here are some examples demonstrating how to use these functions:

```python
import pandas as pd
from pptx import Presentation
from pptx.util import Inches

# Example 1: Convert DataFrame to PowerPoint table
df = pd.DataFrame({'Name': ['John', 'Alice', 'Bob'], 'Age': [25, 30, 35]})
presentation = convert_dataframe_to_ppt_table(df)
presentation.save('table_example.pptx')

# Example 2: Add table to existing slide
existing_presentation_path = 'existing_presentation.pptx'
slide_index = 2
df = pd.DataFrame({'Country': ['USA', 'Canada', 'Mexico'], 'Population': [330, 37, 129]})
add_table_to_slide(existing_presentation_path, slide_index, df)

# Example 3: Add currency-formatted table to new slide
existing_presentation_path = 'existing_presentation.pptx'
df = pd.DataFrame({'Product': ['Apple', 'Banana', 'Orange'], 'Price': [0.99, 0.50, 0.75]})
presentation = Presentation(existing_presentation_path)
presentation = add_currency_table(presentation, df)
presentation.save('currency_table.pptx')

# Example 4: Format a column as percentage
existing_presentation_path = 'existing_presentation.pptx'
slide_index = 3
column_index = 1
prs = Presentation(existing_presentation_path)
slide = prs.slides[slide_index]
table_shape = slide.shapes[0]
format_column_as_percentage(table_shape.table, column_index)
prs.save('formatted_table.pptx')
```

These examples demonstrate the basic usage of the script's functions. Feel free to explore and customize these functions to suit your specific needs.