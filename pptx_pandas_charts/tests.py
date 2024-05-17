
import os
import pytest
from pptx import Presentation
from pptx.enum.chart import XL_CHART_TYPE
from pptx.util import Inches
import pandas as pd


@pytest.fixture
def dataframe():
    return pd.DataFrame({
        'Category': ['A', 'B', 'C'],
        'Value': [10, 20, 30]
    })


def test_dataframe_to_bar_chart(dataframe):
    title = 'Sample Chart'
    presentation = dataframe_to_bar_chart(dataframe, title)

    assert isinstance(presentation, Presentation)
    assert len(presentation.slides) == 1

    slide = presentation.slides[0]
    assert slide.shapes.title.text == title

    chart = slide.shapes[0].chart
    assert chart.chart_type == XL_CHART_TYPE.BAR_CLUSTERED


@pytest.fixture(scope='module')
def sample_dataframe():
    data = {
        'Year': [2010, 2011, 2012, 2013, 2014, 2015],
        'Sales': [5, 7, 3, 9, 6, 10],
        'Profit': [2, 3, 1, 6, 4, 8]
    }
    return pd.DataFrame(data)


def test_dataframe_to_line_chart(sample_dataframe):
    prs = dataframe_to_line_chart(sample_dataframe, 'Sales and Profit')
    
    assert isinstance(prs, Presentation)

    chart = prs.slides[0].shapes[0].chart
    assert chart.chart_type == XL_CHART_TYPE.LINE
    assert chart.chart_title.text_frame.text == 'Sales and Profit'

    expected_categories = ['Year', 'Sales', 'Profit']
    actual_categories = [category.name for category in chart.category_axis.categories]
    assert actual_categories == expected_categories

    expected_series = {'Sales': [5, 7, 3, 9, 6, 10], 'Profit': [2, 3, 1, 6, 4, 8]}
    actual_series = {}
    for series in chart.series:
        actual_series[series.name] = [point.value for point in series.values]
    
    assert actual_series == expected_series


def test_save_presentation(sample_dataframe):
    prs = dataframe_to_line_chart(sample_dataframe, 'Sales and Profit')
    
    prs.save('test_line_chart.pptx')
    
    assert os.path.exists('test_line_chart.pptx')

    
@pytest.fixture(scope='module')
def scatter_data():
    return pd.DataFrame({'x': [1, 2, 3], 'y': [5, 4, 3]})


def test_dataframe_to_scatter_plot_creates_ppt(scatter_data):
   dataframe_to_scatter_plot(scatter_data)
   assert os.path.exists('scatter_plot.pptx')


def test_dataframe_to_scatter_plot_contains_slide_with_picture(scatter_data):
   dataframe_to_scatter_plot(scatter_data)
   prs = Presentation('scatter_plot.pptx')
   slide = prs.slides[0]
   assert len(slide.shapes) >= 1


def test_dataframe_to_scatter_plot_removes_temp_file(scatter_data):
   dataframe_to_scatter_plot(scatter_data)
   assert not os.path.exists('scatter_plot.png')


def test_dataframe_to_scatter_plot_generates_correct_plot(scatter_data):
   dataframe_to_scatter_plot(scatter_data)
   prs = Presentation('scatter_plot.pptx')
   picture = prs.slides[0].shapes[0]
   assert picture.width == Inches(6) and picture.height == Inches(4)


@pytest.fixture(scope='module')
def pie_chart_df():
   return pd.DataFrame({'Category': ['A', 'B', 'C'], 'Value': [1,2 ,3]})


def test_convert_dataframe_to_pie_chart(pie_chart_df):
   title = "Test Chart"
   prs = convert_dataframe_to_pie_chart(pie_chart_df,title)

   assert isinstance(prs ,Presentation)
   
   
@pytest.fixture(scope='module')
def sample_stacked_area_df():
     return pd.DataFrame({'A':[1 ,2 ,3],'B':[4 ,5 ,6],'C':[7 ,8 ,9]})


@pytest.fixture(scope='module')
def sample_title():
     return "Sample Stacked Bar Chart"


@pytest.mark.parametrize("func", [
     (dataframe_to_stacked_area_chart,sample_stacked_area_df,sample_title),
     (dataframe_to_stacked_bar_chart,sample_stacked_area_df,sample_title)
])
 
 def test_stacked_charts(func,sample_stacked_area_df,sample_title):
     result= func(sample_stacked_area_df,sample_title)

     slide= result.slides[0]
     chart= slide.shapes[0].chart

     if func.__name__== "dataframe_to_stacked_area_chart":
         expected_type= XL_CHART_TYPE.AREA_STACKED
     
      else:
         expected_type= XL_CHART_TYPE.BAR_STACKED
      
      assert chart.chart_type== expected_type

      #assertions for legend 
      if func.__name__== "dataframe_to_stacked_bar_chart":
         assert chart.legend.position== XL_LEGEND_POSITION.BOTTOM      

