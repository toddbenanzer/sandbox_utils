
import os
import pytest
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.chart import XL_LABEL_POSITION, XL_TICK_LABEL_POSITION, XL_LEGEND_POSITION, XL_CHART_TYPE


@pytest.fixture
def presentation():
    return Presentation()


@pytest.fixture
def sample_slide(presentation):
    slide = presentation.slides.add_slide(presentation.slide_layouts[0])
    title = slide.shapes.title.text_frame.add_paragraph()
    subtitle = slide.shapes.placeholders[1].text_frame.add_paragraph()
    return slide, title, subtitle


def test_create_presentation():
    presentation = create_presentation()
    assert isinstance(presentation, Presentation)


def test_add_title_slide(presentation):
    title = "Test Title"
    add_title_slide(presentation, title)
    assert len(presentation.slides) == 1
    slide = presentation.slides[0]
    assert slide.slide_layout.name == "Title Slide"
    assert slide.shapes.title.text == title


def test_add_slide():
    presentation = Presentation()
    slide = add_slide(presentation, 1)
    assert len(presentation.slides) == 1
    assert slide.slide_layout.name == "Title Slide"
    
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    
    assert title is not None
    assert subtitle is not None
    
    title.text = "My Slide Title"
    subtitle.text = "Subtitle"
    
    assert title.text == "My Slide Title"
    assert subtitle.text == "Subtitle"
    
    presentation.save("my_presentation.pptx")
    
    assert os.path.exists("my_presentation.pptx")


@pytest.fixture
def sample_chart():
    prs = Presentation()
    slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(slide_layout)
    
    chart_data = CategoryChartData()
    chart_data.categories = ['Category 1', 'Category 2', 'Category 3']
    
    chart_data.add_series('Series 1', (1, 2, 3))
    
    chart = slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED,
        x=0,
        y=0,
        cx=100,
        cy=100,
        chart_data=chart_data,
     ).chart
    
     return chart


def test_format_text_font_sets_font_name(sample_slide):
     slide, title, _ = sample_slide
     format_text_font(slide, title, 'Arial', 24)
     assert title.font.name == 'Arial'


def test_format_text_font_sets_font_size(sample_slide):
     slide, title, _ = sample_slide
    
     format_text_font(slide, title, 'Arial', 24)
     assert title.font.size == Pt(24)


def test_format_text_font_sets_bold(sample_slide):
     slide, title, _ = sample_slide
    
     format_text_font(slide, title, 'Arial', 24,bold=True)
     assert title.font.bold is True


def test_format_text_font_sets_italic(sample_slide):
     slide, _, subtitle = sample_slide
    
     format_text_font(slide, subtitle,'Calibri',18 ,italic=True)
     assert subtitle.font.italic is True


def test_format_text_font_sets_underline(sample_slide):
      slide , _, subtitle=sample_slide
    
      format_text_font(slide ,subtitle,'Calibri',18 ,underline=True )
      assert subtitle.font.underline is True
    

def test_format_text_font_sets_alignment_left(sample_slide):
      slide ,title,_=sample_slide
    
      format_text_font(slide,title,'Arial' ,24 ,align='left')
      assert title.alignment==PP_ALIGN.LEFT


def test_format_text_font_sets_alignment_center(sample_slide):
      slide ,_,subtitle=sample_slide
    
      format_text_font(slide ,subtitle,'Calibri' ,18 ,align='center' )
      assert subtitle.alignment==PP_ALIGN.CENTER


def test_format_text_font_sets_alignment_right(sample_slide):
       slide ,_,subtitle=sample_slide
    
       format_text_font(slide ,subtitle ,'Calibri' ,18 ,align='right')
       assert subtitle.alignment==PP_ALIGN.RIGHT
    

def test_format_background_color_sets_background_fill(slide):
       color =(255 ,0 ,0)
    
       format_background_color(slide,color)
       assert slide.background.fill is not None
    

def test_format_background_color_sets_fore_color_rgb(slide):
       color =(255 ,0 ,0)
    
       format_background_color(slide,color )
       assert slide.background.fill.fore_color.rgb==RGBColor(color[0],color[1],color[2])


def test_format_background_color_with_invalid_slide_raises_exception(presentation):
       invalid_slide=None 
       color=(255 ,0 ,0)
      
       with pytest.raises(Exception ):
            format_background_color(invalid_slide,color )


@pytest.fixture 
def presentation_with_chart(presentation ):
        chart_data=CategoryChartData()
        
        chart_data.categories=['Category 1','Category 2','Category 3']
        
        chart_data.add_series('Series 1',(1 ,2 ,3 ))
        
        x,y,cx,cy=Inches(2),Inches(2),Inches(4 ),Inches(3)
        
        # Add a line chart to the first side
        
        chart=presentation.slides[0].shapes.add_chart(
            XL_CHART_TYPE.LINE ,
            x,y,cx ,
            cy ,
            chart_data ,
           ).chart
        
         return presentation


 def test_set_default_line_style(presentation_with_chart ):
         set_default_line_style (presentation_with_chart )
         
         for series in presentation_with_chart.slides[0].shapes[0].chart.series :
              for pt in series.points :
                    pt.format.line.color.rgb==RGBColor (255 ,
                     0 ,
                     0 )


 def test_set_default_line_color_single_series (sample_chart ):
         line_color ='0000FF'
         set_default_line_color (sample_chart,line_color )
         
         for series in sample_chart.series:
             for pt in series.points :
                pt.format.line.color.rgb==RGBColor(255 ,
                  255,
                  255 )


 def check_marker_color (presentation,line_style,color ):
          for shape in presentation .slides [0].shapes:
                  if shape.has_chart:
                      chart=shape.chart 
                      
                      for series in chart.series:
                          for point in series.points :
                                point.marker.style==line_style 
                                point.marker.fill.fore_color.rgb=color

                        
 def set_default_marker_style (presentation,line_style,color ):
           for shape in presentation .slides [0].shapes:
                 if shape.has_chart:
                    chart=shape.chart
                    
                    for series in chart.series:
                          for point in series.points :
                             point.marker.style=line_style 
                             point.marker.fill.solid ()
                             point.marker.fill.fore_color.rgb=color



 def check_marker_size (presentation,sizes ):
           for shape in presentation .slides [0].shapes:
                if shape.has_chart:
                     chart=shape.chart
                    
                     for actual_size in sizes :
                           expected_size=sizes.pop(0 )
                           actual_size==expected_size

                           
 def set_default_marker_size (presentation,sizes ):
           for shape in presentation .slides [0].shapes:
                 if shape.has_chart:
                       chart=shape.chart
                    
                       for actual_size in sizes :
                             expected_size=sizes.pop(0 )
                             actual_size==expected_size

                             

 def set_branding_guidelines(font,color,background ):
          return font,color ,
          background==font,color,background

          
 @pytest.mark.parametrize (
   "slide_width,slide_height,size",(1024 ,
   768,(1024 ,
   768)),
   
   (800 ,
   600,(800,
   600)),
   
   (400,
  300,(400,
  300)),
 )


 def adjust_chart_size_test (
slide_width,
slide_height,size):
         adjusted_width,
         adjusted_height=size 
         
         width,height=size 
         width,height==adjusted_width,
         adjusted_height
 
 