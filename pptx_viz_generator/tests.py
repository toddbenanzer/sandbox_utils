ytest
from pptx import Presentation
from pptx.enum.chart import XL_CHART_TYPE

@pytest.fixture
def empty_presentation():
    # Create a new presentation object
    presentation = Presentation()
    
    # Return the presentation object
    return presentation

def test_create_empty_presentation(empty_presentation):
    assert isinstance(empty_presentation, Presentation)
    assert len(empty_presentation.slides) == 1


@pytest.fixture
def presentation():
    return Presentation()

def test_embed_plotly_chart(presentation):
    chart_data = [
        ['Category', 'Value'],
        ['A', 10],
        ['B', 20],
        ['C', 30]
    ]
    
    # Call the function
    result = embed_plotly_chart(presentation, 0, chart_data)
    
    # Assert that the result is a Presentation object
    assert isinstance(result, Presentation)
    
    # Assert that the slide was correctly created and chart was added
    slide = result.slides[0]
    assert len(slide.shapes) == 1
    assert slide.shapes[0].has_chart
    
    # Assert that the chart type is correct
    chart = slide.shapes[0].chart
    assert chart.chart_type == XL_CHART_TYPE.COLUMN_CLUSTERED
    
    # Assert that the chart data is correctly set
    chart_data_excel = chart.chart_data.workbook.sheets[0]
    
    assert chart_data_excel.cell(1, 1).value == 'Category'
    assert chart_data_excel.cell(2, 1).value == 'A'
    assert chart_data_excel.cell(3, 1).value == 'B'
    assert chart_data_excel.cell(4, 1).value == 'C'
    
    assert chart_data_excel.cell(1, 2).value == 'Value'
    assert chart_data_excel.cell(2, 2).value == 10
    assert chart_data_excel.cell(3, 2).value == 20
    assert chart_data_excel.cell(4, 2).value == 30


import pytest
from pptx import Presentation
from pptx.util import Pt
from pptx.dml.color import RGBColor

@pytest.fixture
def presentation():
    return Presentation('presentation.pptx')

def test_format_presentation_title(presentation):
    # Call the function to format the presentation
    format_presentation(presentation)

    # Assert that the title font, size, color are correctly applied
    assert presentation.slides[0].shapes.title.text_frame.paragraphs[0].font.name == "Arial"
    assert presentation.slides[0].shapes.title.text_frame.paragraphs[0].font.size == Pt(24)
    assert presentation.slides[0].shapes.title.text_frame.paragraphs[0].font.color.rgb == RGBColor.from_string("000000")

def test_format_presentation_content(presentation):
    # Call the function to format the presentation
    format_presentation(presentation)

    # Assert that the content font, size, color are correctly applied
    for shape in presentation.slides[0].shapes:
        if shape.has_text_frame:
            assert shape.text_frame.paragraphs[0].font.name == "Calibri"
            assert shape.text_frame.paragraphs[0].font.size == Pt(12)
            assert shape.text_frame.paragraphs[0].font.color.rgb == RGBColor.from_string("333333")

def test_format_presentation_text(presentation):
    # Call the function to format the presentation
    format_presentation(presentation)

    # Assert that the title and content text is correctly set
    assert presentation.slides[0].shapes.title.text_frame.paragraphs[0].text == "Title"
    
    for shape in presentation.slides[0].shapes:
        if shape.has_text_frame:
            assert shape.text_frame.paragraphs[0].text == "Content"


from pptx import Presentation
import pytest

# Test add_title_and_subtitle function
def test_add_title_and_subtitle():
    # Create a presentation object
    presentation = Presentation()
    
    # Add a slide with a title and subtitle
    slide_layout = presentation.slide_layouts[0]
    slide = presentation.slides.add_slide(slide_layout)
    
    # Add title and subtitle using the function
    add_title_and_subtitle(slide, "Title", "Subtitle")
    
    # Check if the title and subtitle have been added correctly
    assert slide.shapes.title.text == "Title"
    assert slide.placeholders[1].text == "Subtitle"

# Test saving the presentation
def test_save_presentation():
    # Create a presentation object
    presentation = Presentation()
    
    # Add a slide with a title and subtitle
    slide_layout = presentation.slide_layouts[0]
    slide = presentation.slides.add_slide(slide_layout)
    
    # Add title and subtitle using the function
    add_title_and_subtitle(slide, "Title", "Subtitle")
    
    # Save the presentation to a file
    filename = "presentation.pptx"
    presentation.save(filename)
    
    # Check if the file has been saved successfully
    assert os.path.exists(filename)


import pytest
from pptx import Presentation

# Test if a text box is added to the slide
def test_add_text_box():
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_text_box(slide, "Hello World!")
    
    # Check if the slide has a shape of type 'text box'
    assert any(shape.shape_type == 17 for shape in slide.shapes)

# Test if the text frame is added to the text box shape
def test_add_text_frame():
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_text_box(slide, "Hello World!")

    # Get the first shape of type 'text box'
    txBox = next(shape for shape in slide.shapes if shape.shape_type == 17)
    
    # Check if the text frame exists in the text box shape
    assert txBox.has_text_frame

# Test if the text is added to the text frame
def test_add_text():
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_text_box(slide, "Hello World!")

    # Get the first shape of type 'text box'
    txBox = next(shape for shape in slide.shapes if shape.shape_type == 17)
    
    # Check if the text frame exists in the text box shape
    assert txBox.has_text_frame
    
    # Get the text frame of the text box shape
    tf = txBox.text_frame
    
    # Check if a paragraph with content 'Hello World!' exists in the text frame
    assert any(p.text == "Hello World!" for p in tf.paragraphs)


import pytest
from pptx import Presentation

# Import the function to be tested
from ptx import add_image_to_slide

# Define a fixture to create a blank presentation and slide before each test
@pytest.fixture
def setup_presentation():
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    return prs, slide

# Test case 1: Add image to slide with valid parameters
def test_add_image_to_slide_success(setup_presentation):
    prs, slide = setup_presentation
    image_path = "image.png"
    left = 100
    top = 100
    width = 200
    height = 200

    add_image_to_slide(slide, image_path, left, top, width, height)

    # Assert that the image has been added to the slide
    assert len(slide.shapes) == 1

# Test case 2: Add image to slide with invalid image path
def test_add_image_to_slide_invalid_image_path(setup_presentation):
    prs, slide = setup_presentation
    image_path = "invalid.png"
    left = 100
    top = 100
    width = 200
    height = 200

    # Assert that an exception is raised when trying to add the image to the slide
    with pytest.raises(Exception):
        add_image_to_slide(slide, image_path, left, top, width, height)

# Test case 3: Add image to slide with negative dimensions
def test_add_image_to_slide_negative_dimensions(setup_presentation):
    prs, slide = setup_presentation
    image_path = "image.png"
    left = 100
    top = 100
    width = -200
    height = -200

    # Assert that an exception is raised when trying to add the image to the slide
    with pytest.raises(Exception):
        add_image_to_slide(slide, image_path, left, top, width, height)

# Test case 4: Add image to slide with zero dimensions
def test_add_image_to_slide_zero_dimensions(setup_presentation):
    prs, slide = setup_presentation
    image_path = "image.png"
    left = 100
    top = 100
    width = 0
    height = 0

    # Assert that an exception is raised when trying to add the image to the slide
    with pytest.raises(Exception):
        add_image_to_slide(slide, image_path, left, top, width, height)


from pptx import Presentation
from pptx.util import Inches
import pytest

# Test if table is added to the slide correctly
def test_add_table_to_slide():
    # Create a presentation object
    prs = Presentation()

    # Create some sample data and headers
    data = [
        [1, 'John', 'Doe'],
        [2, 'Jane', 'Smith'],
    ]
    headers = ['ID', 'First Name', 'Last Name']

    # Add a new slide to the presentation
    slide_layout_index = 1  # Choose appropriate layout index (e.g., 1 is Title and Content layout)
    slide = prs.slides.add_slide(prs.slide_layouts[slide_layout_index])

    # Add the table to the slide
    add_table_to_slide(prs, 0, data, headers)

    # Get the added table shape from the slide
    table_shape = slide.shapes[0]

    # Assert that the shape is a table shape
    assert isinstance(table_shape, SlideShape)

    # Assert that the number of rows and columns in the table is correct
    assert table_shape.table.rows.count == len(data) + 1  # Including header row
    assert table_shape.table.columns.count == len(headers)

# Test if headers are added correctly to the first row of the table
def test_add_table_headers():
    # Create a presentation object
    prs = Presentation()

    # Create some sample data and headers
    data = [
        [1, 'John', 'Doe'],
        [2, 'Jane', 'Smith'],
    ]
    headers = ['ID', 'First Name', 'Last Name']

    # Add a new slide to the presentation
    slide_layout_index = 1  # Choose appropriate layout index (e.g., 1 is Title and Content layout)
    slide = prs.slides.add_slide(prs.slide_layouts[slide_layout_index])

    # Add the table to the slide
    add_table_to_slide(prs, 0, data, headers)

    # Get the added table shape from the slide
    table_shape = slide.shapes[0]

    # Check if the headers in the first row of the table are correct
    for i, header in enumerate(headers):
        cell = table_shape.table.cell(0, i)
        assert cell.text == header

# Test if data is added correctly to the remaining cells in the table
def test_add_table_data():
    # Create a presentation object
    prs = Presentation()

    # Create some sample data and headers
    data = [
        [1, 'John', 'Doe'],
        [2, 'Jane', 'Smith'],
    ]
    headers = ['ID', 'First Name', 'Last Name']

    # Add a new slide to the presentation
    slide_layout_index = 1  # Choose appropriate layout index (e.g., 1 is Title and Content layout)
    slide = prs.slides.add_slide(prs.slide_layouts[slide_layout_index])

    # Add the table to the slide
    add_table_to_slide(prs, 0, data, headers)

    # Get the added table shape from the slide
    table_shape = slide.shapes[0]

    # Check if the data in each cell of the table is correct
    for row_idx, row_data in enumerate(data):
        for col_idx, cell_data in enumerate(row_data):
            cell = table_shape.table.cell(row_idx + 1, col_idx)
            assert cell.text == str(cell_data)

# Test if table dimensions are calculated correctly based on number of rows and columns
def test_calculate_table_dimensions():
    # Create some sample data and headers
    data = [
        [1, 'John', 'Doe'],
        [2, 'Jane', 'Smith'],
    ]
    headers = ['ID', 'First Name', 'Last Name']

    # Calculate expected table dimensions
    num_rows = len(data) + 1  # Including header row
    num_cols = len(headers)
    table_width = Inches(6)
    col_width = table_width / num_cols
    table_height = Inches(0.8 * num_rows)

    # Call the function to calculate the actual table dimensions
    actual_table_width, actual_table_height, actual_col_width = calculate_table_dimensions(data, headers)

    # Compare the actual and expected dimensions
    assert actual_table_width == table_width
    assert actual_table_height == table_height
    assert actual_col_width == col_width


from pptx import Presentation
import pytest

@pytest.fixture
def presentation():
    return Presentation('presentation.pptx')

def test_set_text_format_changes_font_name(presentation):
    set_text_format(presentation, 0, 0, 'Arial', 12, (255, 0, 0))
    slide = presentation.slides[0]
    text_frame = slide.shapes[0].text_frame
    for paragraph in text_frame.paragraphs:
        assert paragraph.font.name == 'Arial'

def test_set_text_format_changes_font_size(presentation):
    set_text_format(presentation, 0, 0, 'Arial', 12, (255, 0, 0))
    slide = presentation.slides[0]
    text_frame = slide.shapes[0].text_frame
    for paragraph in text_frame.paragraphs:
        assert paragraph.font.size == Pt(12)

def test_set_text_format_changes_font_color(presentation):
    set_text_format(presentation, 0, 0, 'Arial', 12, (255, 0, 0))
    slide = presentation.slides[0]
    text_frame = slide.shapes[0].text_frame
    for paragraph in text_frame.paragraphs:
        assert paragraph.font.color.rgb == RGBColor(255, 0, 0)


import os
from pptx import Presentation

def test_save_presentation():
    # Create a test presentation
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    title.text = "Test Slide"
    subtitle.text = "This is a test slide"

    # Save the presentation using the function being tested
    save_presentation(prs, 'test.pptx')

    # Check if the file exists
    assert os.path.exists('test.pptx')

    # Check if the saved presentation is valid and contains the expected content
    saved_prs = Presentation('test.pptx')
    assert len(saved_prs.slides) == 1
    saved_slide = saved_prs.slides[0]
    assert saved_slide.shapes.title.text == "Test Slide"
    assert saved_slide.placeholders[1].text == "This is a test slide"

    # Clean up the test file
    os.remove('test.pptx')


import pptx

def test_create_presentation():
    # Arrange
    expected_slide_count = 1
    
    # Act
    presentation = create_presentation()
    
    # Assert
    assert isinstance(presentation, pptx.Presentation)  # Ensure the return type is correct
    assert len(presentation.slides) == expected_slide_count  # Ensure the initial slide count is correct


import pytest
from pptx import Presentation

# Test cases
def test_open_presentation_valid():
    # Arrange
    file_path = "test.pptx"

    # Act
    presentation = open_presentation(file_path)

    # Assert
    assert isinstance(presentation, Presentation)

def test_open_presentation_invalid():
    # Arrange
    file_path = "invalid.pptx"

    # Act and Assert
    with pytest.raises(Exception):
        open_presentation(file_path)


import os
from pptx import Presentation

# Define a fixture that loads the presentation file
@pytest.fixture
def presentation():
    # Load the presentation
    presentation = Presentation('path_to_presentation.pptx')
    yield presentation
    
    # Clean up - delete the output file if it exists
    output_file = 'output.pptx'
    if os.path.exists(output_file):
        os.remove(output_file)

# Test case 1: Verify that the text is correctly inserted into the specified placeholder on the slide
def test_insert_text(presentation):
    # Call the insert_text function with test data
    insert_text(presentation, 0, 1, "Hello World")

    # Get the slide and find the placeholder by its id
    slide = presentation.slides[0]
    for shape in slide.shapes:
        if shape.has_text_frame and shape.placeholder == 1:
            # Access the text frame and check if the text is correctly set
            text_frame = shape.text_frame
            assert text_frame.text == "Hello World"

# Test case 2: Verify that the modified presentation is saved to the specified output file
def test_save_modified_presentation(presentation):
    # Call the insert_text function with test data
    insert_text(presentation, 0, 1, "Hello World")

    # Save the modified presentation to an output file
    output_file = 'output.pptx'
    presentation.save(output_file)

    # Check if the output file exists
    assert os.path.exists(output_file)


from pptx import Presentation
import pytest

@pytest.fixture
def presentation():
    return Presentation('path_to_presentation.pptx')

def test_apply_slide_layout(presentation):
    # Apply the 'Title Slide' layout to slide at index 0
    apply_slide_layout(presentation, 0, 'Title Slide')

    # Check if the layout is applied correctly
    assert presentation.slides[0].layout.name == 'Title Slide'

# Test case for invalid slide index
def test_apply_slide_layout_invalid_index(presentation):
    # Try to apply layout to an invalid slide index (out of range)
    with pytest.raises(IndexError):
        apply_slide_layout(presentation, 100, 'Title Slide')

# Test case for invalid layout name
def test_apply_slide_layout_invalid_layout(presentation):
    # Try to apply an invalid layout name
    with pytest.raises(KeyError):
        apply_slide_layout(presentation, 0, 'Invalid Layout'