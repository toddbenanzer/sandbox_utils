e_cells(presentation, slide_num, start_row, end_row, start_col, end_col):
    """
    Merge cells in a table on a specific slide of a PowerPoint presentation.

    Args:
        presentation (Presentation): The PowerPoint presentation object.
        slide_num (int): The index of the slide containing the table.
        start_row (int): The index of the starting row to merge.
        end_row (int): The index of the ending row to merge.
        start_col (int): The index of the starting column to merge.
        end_col (int): The index of the ending column to merge.

    Raises:
        IndexError: If any of the provided row or column indices are out of range.
    """
    try:
        # Get the specified slide
        slide = presentation.slides[slide_num]
        
        # Get the table shape on the slide
        table_shape = slide.shapes[0]
        
        # Get the table object from the shape
        table = table_shape.table
        
        # Merge cells in the specified range
        for row in range(start_row, end_row + 1):
            for col in range(start_col, end_col + 1):
                cell = table.cell(row, col)
                if row == start_row and col == start_col:
                    cell.merge(table.cell(end_row, end_col))
                else:
                    cell._element.getparent().remove(cell._element)
    
    except IndexError:
        raise IndexError("Invalid row or column indices."