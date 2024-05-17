
import os
import pytest
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.table import Table

@pytest.fixture
def presentation():
    return Presentation()

@pytest.fixture
def data_frame():
    data = {'Column 1': [1, 2, 3], 'Column 2': ['A', 'B', 'C']}
    return pd.DataFrame(data)

def test_add_table_to_slide(presentation, data_frame):
    add_table_to_slide(presentation, 0, data_frame)
    assert len(presentation.slides[0].shapes) == 1

def test_add_table_to_slide_table_dimensions(presentation, data_frame):
    add_table_to_slide(presentation, 0, data_frame)
    slide_width = presentation.slide_width.inches - Inches(2)
    slide_height = presentation.slide_height.inches - Inches(2)
    expected_width = slide_width / data_frame.shape[1]
    expected_height = slide_height / (data_frame.shape[0] + 1)
    table = presentation.slides[0].shapes[0].table

    assert table.width == expected_width
    assert table.height == expected_height

def test_add_table_to_slide_column_widths(presentation, data_frame):
    add_table_to_slide(presentation, 0, data_frame)
    slide_width = presentation.slide_width.inches - Inches(2)
    expected_col_width = slide_width / data_frame.shape[1]
    table = presentation.slides[0].shapes[0].table

    for i in range(data_frame.shape[1]):
        assert table.columns[i].width == expected_col_width

def test_add_table_to_slide_cell_text_and_format(presentation, data_frame):
    add_table_to_slide(presentation, 0, data_frame)
    table = presentation.slides[0].shapes[0].table

    for i, col in enumerate(data_frame.columns):
        header_cell = table.cell(0, i)
        assert header_cell.text == col
        assert header_cell.fill.fore_color.rgb == RGBColor(0, 0, 0)
        assert header_cell.text_frame.paragraphs[0].alignment == PP_ALIGN.CENTER

        for j, value in enumerate(data_frame[col]):
            data_cell = table.cell(j + 1, i)
            assert data_cell.text == str(value)

            if isinstance(value, int) or isinstance(value, float):
                assert data_cell.text_frame.paragraphs[0].alignment == PP_ALIGN.RIGHT
                assert data_cell.text_frame.paragraphs[0].font.size == Pt(12)
            elif isinstance(value, str):
                assert data_cell.text_frame.paragraphs[0].alignment == PP_ALIGN.LEFT
                assert data_cell.text_frame.paragraphs[0].font.size == Pt(10)

def test_add_table_to_slide_saves_updated_presentation(presentation, data_frame):
    filename = "updated_presentation.pptx"
    add_table_to_slide(presentation, 0, data_frame)
    
    presentation.save(filename)
    
    assert os.path.isfile(filename)

@pytest.fixture
def sample_presentation():
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    shapes = slide.shapes.add_table(rows=2, cols=2).table
    return prs

def test_set_table_style(sample_presentation):
    table_shape = sample_presentation.slides[0].shapes[-1]
    
    set_table_style(table_shape.table, 'Table Grid')

def test_set_column_widths_sets_correct_widths(sample_presentation):
   widths=[Inches(1.5),Inches(2.0),Inches(1.5)]
   set_column_widths(sample_presentation,widths)

# More tests go here...

# Run the tests
if __name__ == '__main__':
   pytest.main()
