
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.chart import XL_CHART_TYPE
from pptx.dml.color import RGBColor


def create_empty_presentation():
    """
    Create an empty PowerPoint presentation.
    
    Returns:
        Presentation: The empty PowerPoint presentation object.
    """
    presentation = Presentation()
    return presentation


def define_slide_layouts(presentation, branding):
    """
    Define slide layouts based on branding guidelines.
    
    Args:
        presentation (Presentation): The PowerPoint presentation object.
        branding (dict): The branding guidelines.
    """
    for slide in presentation.slides:
        layout_name = branding.get('slide_layout')
        if layout_name:
            slide_layout = presentation.slide_layouts.get_by_name(layout_name)
            slide.layout = slide_layout


def embed_plotly_chart(presentation, slide_index, chart_data):
    """
    Embed Plotly charts into slides.
    
    Args:
        presentation (Presentation): The PowerPoint presentation object.
        slide_index (int): The index of the slide to embed the chart.
        chart_data (list): The data for the chart.
        
    Returns:
        Presentation: The modified PowerPoint presentation object.
    """
    slide = presentation.slides[slide_index]
    
    left = Inches(1)
    top = Inches(2)
    width = Inches(6)
    height = Inches(4)
    
    chart_placeholder = slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED, left, top, width, height).chart
    
    chart_data_excel = chart_placeholder.chart_data.workbook
    sheet = chart_data_excel.sheets[0]
    
    for i in range(len(chart_data)):
        for j in range(len(chart_data[i])):
            sheet.cell(i + 1, j + 1).value = chart_data[i][j]
    
    chart_placeholder.chart_data.categories = sheet.range('A2:A{}'.format(len(chart_data) + 1))
    
    for i in range(1, len(chart_data[0])):
        chart_placeholder.chart_data.add_series(sheet.range((1, i+1), (len(chart_data), i+1)))
    
    return presentation


def format_presentation(presentation):
    """
    Format the presentation according to branding guidelines.
    
    Args:
        presentation (Presentation): The PowerPoint presentation object.
    """
    title_font = "Arial"
    title_font_size = 24
    title_color = "000000"  # Black

    content_font = "Calibri"
    content_font_size = 12
    content_color = "333333"  # Dark gray

    for slide in presentation.slides:
        slide.shapes.title.text_frame.clear()
        title_text_frame = slide.shapes.title.text_frame
        title_text_frame.add_paragraph().text = "Title"
        title_text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
        title_text_frame.paragraphs[0].font.name = title_font
        title_text_frame.paragraphs[0].font.size = Pt(title_font_size)
        title_text_frame.paragraphs[0].font.color.rgb = RGBColor.from_string(title_color)

        for shape in slide.shapes:
            if shape.has_text_frame:
                text_frame = shape.text_frame
                text_frame.clear()
                text_frame.add_paragraph().text = "Content"
                text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT
                text_frame.paragraphs[0].font.name = content_font
                text_frame.paragraphs[0].font.size = Pt(content_font_size)
                text_frame.paragraphs[0].font.color.rgb = RGBColor.from_string(content_color)


def add_title_and_subtitle(slide, title, subtitle):
    """
    Add titles and subtitles to slides.
    
    Args:
        slide (Slide): The slide object.
        title (str): The title text.
        subtitle (str): The subtitle text.
    """
    slide.shapes.title.text = title
    slide.placeholders[1].text = subtitle


def add_text_box(slide, content):
    """
    Add text boxes with custom content to slides.
    
    Args:
        slide (Slide): The slide object.
        content (str): The content of the text box.
    """
    left = Inches(1)
    top = Inches(1)
    width = Inches(3)
    height = Inches(3)
    
    txBox = slide.shapes.add_textbox(left, top, width, height)
    
    tf = txBox.text_frame
    p = tf.add_paragraph()
    p.text = content


def add_image_to_slide(slide, image_path, left, top, width, height):
    """
    Add images to slides.
    
    Args:
        slide (Slide): The slide object.
        image_path (str): The path to the image file.
        left (float): The left position of the image in inches.
        top (float): The top position of the image in inches.
        width (float): The width of the image in inches.
        height (float): The height of the image in inches.
    """
    slide.shapes.add_picture(image_path, left, top, width, height)


def add_table_to_slide(presentation, slide_index, data, headers):
    """
    Add tables with custom data to slides.
    
    Args:
        presentation (Presentation): The PowerPoint presentation object.
        slide_index (int): The index of the slide to add the table.
        data (list[list]): The table data.
        headers (list[str]): The table headers.
        
   Returns:
        Presentation: The modified PowerPoint presentation object.
    """
    slide = presentation.slides[slide_index]
    
    num_rows = len(data) + 1
    num_cols = len(headers)
    table_width = Inches(6)
    col_width = table_width / num_cols
    table_height = Inches(0.8 * num_rows)
    
    left = Inches(1)
    top = Inches(2)
    
    table = slide.shapes.add_table(num_rows, num_cols, left, top, table_width, table_height).table
    
    for i, header in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = header
    
    for row_idx, row_data in enumerate(data):
        for col_idx, cell_data in enumerate(row_data):
            cell = table.cell(row_idx + 1, col_idx)
            cell.text = str(cell_data)
    
    return presentation


def add_bullet_points(slide, content):
    """
    Add bullet point lists with custom content to slides.
    
    Args:
        slide (Slide): The slide object.
        content (list[str]): The content of the bullet points.
    """
    slide_layout = presentation.slide_layouts[1]
    slide = presentation.slides.add_slide(slide_layout)

    placeholder = None
    for shape in slide.shapes:
        if shape.placeholder is not None and shape.placeholder.has_text_frame:
            placeholder = shape.placeholder
            break

    if not placeholder:
        raise ValueError("Slide layout does not have a bullet point placeholder")

    text_frame = placeholder.text_frame
    for item in content:
        p = text_frame.add_paragraph()
        p.text = item
        p.level = 0


def set_text_format(presentation, slide_index, text_index, font_name, font_size, font_color):
    """
    Set the font style, size, and color for text in slides.
    
    Args:
        presentation (Presentation): The PowerPoint presentation object.
        slide_index (int): The index of the slide containing the text.
        text_index (int): The index of the text shape in the slide.
        font_name (str): The name of the font.
        font_size (float): The size of the font in points.
        font_color (tuple[int]): The RGB color of the font.
    """
    slide = presentation.slides[slide_index]
    text_frame = slide.shapes[text_index].text_frame
    
    for paragraph in text_frame.paragraphs:
        paragraph.font.name = font_name
        paragraph.font.size = Pt(font_size)
        paragraph.font.color.rgb = RGBColor(*font_color)


def set_slide_background(presentation, slide_index, background):
    """
    Set the background color or image for slides.
    
    Args:
        presentation (Presentation): The PowerPoint presentation object.
        slide_index (int): The index of the slide to set the background.
        background (str): The path to an image file or a RGB color code.
        
   Returns:
        Presentation: The modified PowerPoint presentation object.
    """
    slide = presentation.slides[slide_index]
    
    if background.endswith('.png') or background.endswith('.jpg'):
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = None
        slide.background.fill._fill_element.xpath('./p:blipFill/a:blip')[0].attrib[
            '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed'] = 'rId1'
        
        slide.background.fill._fill_element.xpath(
            './p:blipFill/a:blip/p:nvPicPr/cNvPr')[0].attrib['name'] = 'Background Picture'
        
        slide.background.fill._fill_element.xpath(
            './p:blipFill/a:blip/p:nvPicPr/cNvPr')[0].attrib['id'] = '2'
        
        rId = slide.rels.get_or_add_image(background)
        slide.background.fill._fill_element.xpath('./p:blipFill/a:blip')[0].attrib[
            '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed'] = rId
    else:
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = background
    
    return presentation


def save_presentation(presentation, filename):
    """
    Save the generated PowerPoint presentation as a file.
    
    Args:
        presentation (Presentation): The PowerPoint presentation object.
        filename (str): The name of the output file.
    """
    presentation.save(filename)


def create_presentation():
    """
    Create a new PowerPoint presentation.
    
    Returns:
        Presentation: The newly created PowerPoint presentation object.
    """
    presentation = Presentation()
    return presentation


def open_presentation(file_path):
    """
    Open an existing PowerPoint presentation.
    
    Args:
        file_path (str): The path to the PowerPoint presentation file.
        
   Returns:
        Presentation: The opened PowerPoint presentation object.
    """
    presentation = Presentation(file_path)
    
    return presentation


def insert_text(presentation, slide_index, placeholder_id, text):
    """
    Insert text into placeholders on slides.
    
    Args:
        presentation (Presentation): The PowerPoint presentation object.
        slide_index (int): The index of the slide containing the placeholder.
        placeholder_id (int): The id of the placeholder shape.
        text (str): The text to insert into the placeholder.
    """
    slide = presentation.slides[slide_index]
    
    for shape in slide.shapes:
        if shape.has_text_frame and shape.placeholder == placeholder_id:
            text_frame = shape.text_frame
            text_frame.text = text
            break


def apply_slide_layout(presentation, slide_index, layout_name):
    """
    Apply slide layouts and formatting guidelines.
    
    Args:
        presentation (Presentation): The PowerPoint presentation object.
        slide_index (int): The index of the slide to apply the layout.
        layout_name (str): The name of the slide layout.
    """
    slide = presentation.slides[slide_index]
    layout = presentation.slide_layouts[layout_name]
    slide.layout = layout
