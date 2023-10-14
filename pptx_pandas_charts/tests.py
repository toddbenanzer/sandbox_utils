
import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE


def test_create_bar_chart():
    # Create a test dataframe
    data = {
        'Category': ['A', 'B', 'C'],
        'Value': [1, 2, 3]
    }
    df = pd.DataFrame(data)

    # Call the function to be tested
    create_bar_chart(df, "Test Chart")

    # Load the saved presentation
    prs = Presentation("bar_chart.pptx")

    # Check if the presentation has exactly one slide
    assert len(prs.slides) == 1

    # Get the shapes in the slide
    slide = prs.slides[0]
    shapes = slide.shapes

    # Check if the title is set correctly
    title_shape = shapes.title
    assert title_shape.text == "Test Chart"

    # Check if the chart data is set correctly
    chart = shapes[1].chart  # Assuming chart shape is at index 1 in shapes list

    # Get the categories from the chart data
    categories = [category.label for category in chart.plots[0].categories]

    # Check if the categories are set correctly
    assert categories == ['Category']

    # Get the series values from the chart data
    series_values = []
    for series in chart.series:
        series_values.append([point.value for point in series.points])

    # Check if the series values are set correctly
    assert series_values == [[1], [2], [3]]


def test_convert_df_to_line_chart():
    # Create a dummy DataFrame for testing
    data = {
        'X': [1, 2, 3, 4, 5],
        'Y': [10, 20, 30, 40, 50]
    }
    df = pd.DataFrame(data)

    # Call the function with the dummy DataFrame and chart title
    chart_title = "Test Line Chart"
    convert_df_to_line_chart(df, chart_title)

    # Verify that the file has been created
    assert os.path.exists("line_chart.pptx")

    # Verify that the chart title is set correctly
    presentation = Presentation("line_chart.pptx")
    slide = presentation.slides[0]
    chart = slide.shapes[0].chart

    assert chart.has_title
    assert chart.chart_title.text_frame.text == chart_title

    # Clean up: delete the created file
    os.remove("line_chart.pptx")
