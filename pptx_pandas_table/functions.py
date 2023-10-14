
from pptx import Presentation

def create_table(prs, slide_idx, data, column_names):
    slide = prs.slides[slide_idx]
    shapes = slide.shapes
    
    # Define table dimensions
    rows = len(data) + 1  # Include header row
    cols = len(column_names)
    
    # Calculate cell width based on slide width and number of columns
    slide_width = prs.slide_width.pt
    cell_width = slide_width / cols
    
    # Calculate cell height based on slide height and number of rows
    slide_height = prs.slide_height.pt
    cell_height = slide_height / rows
    
    # Add table shape to the slide
    left = 0  # Left coordinate of the table shape
    top = 0   # Top coordinate of the table shape
    width = slide_width  # Width of the table shape
    height = slide_height  # Height of the table shape
    table_shape = shapes.add_table(rows, cols, left, top, width, height).table
    
    # Set column names as header row
    for idx, column_name in enumerate(column_names):
        cell = table_shape.cell(0, idx)
        cell.text = column_name
    
    # Set data in the remaining cells
    for row_idx, row_data in enumerate(data):
        for col_idx, value in enumerate(row_data):
            cell = table_shape.cell(row_idx + 1, col_idx)
            cell.text = str(value)

# Example usage:
prs = Presentation()
slide_idx = 0
data = [
    ['John', 'Doe', 25],
    ['Jane', 'Smith', 30],
]
column_names = ['First Name', 'Last Name', 'Age']

create_table(prs, slide_idx, data, column_names)

prs.save('table.pptx')
