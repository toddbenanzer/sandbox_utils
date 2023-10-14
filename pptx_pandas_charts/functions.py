andas as pd
from pptx import Presentation
from pptx.util import Inches
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE

def create_bar_chart(dataframe, title):
    # Create a blank presentation
    prs = Presentation()

    # Create a slide with a title and content layout
    slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(slide_layout)
    shapes = slide.shapes

    # Add the title to the slide
    title_shape = shapes.title
    title_shape.text = title

    # Convert dataframe to CategoryChartData
    chart_data = CategoryChartData()
    chart_data.categories = list(dataframe.columns)

    # Add series data to the chart
    for row in dataframe.itertuples(index=False):
        chart_data.add_series(row._fields, list(row))

    # Add a bar chart to the slide
    x, y, cx, cy = 0.1, 0.2, 6, 4.5  # Set the position and size of the chart
    chart = shapes.add_chart(
        XL_CHART_TYPE.BAR_CLUSTERED, x, y, cx, cy, chart_data
    ).chart

    # Customize the look and feel of the chart if desired

    # Save the presentation as a PowerPoint file
    prs.save("bar_chart.pptx"