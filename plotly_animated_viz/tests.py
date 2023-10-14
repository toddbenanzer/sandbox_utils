ytest
import plotly.graph_objects as go
import numpy as np

from my_module import create_animated_line_chart


# Test case 1: Check if the returned object is an instance of go.Figure
def test_create_animated_line_chart_returns_figure():
    x = np.linspace(0, 2 * np.pi, 100)
    y = [np.sin(x + i / 10) for i in range(len(x))]
    result = create_animated_line_chart(x, y)
    assert isinstance(result, go.Figure)


# Test case 2: Check if the number of frames in the returned figure is equal to the specified frames argument
def test_create_animated_line_chart_frames():
    x = np.linspace(0, 2 * np.pi, 100)
    y = [np.sin(x + i / 10) for i in range(len(x))]
    frames = 5
    result = create_animated_line_chart(x, y, frames=frames)
    assert len(result.frames) == frames


# Test case 3: Check if the x and y coordinates of the line trace in each frame are correctly updated
def test_create_animated_line_chart_trace_update():
    x = np.linspace(0, 2 * np.pi, 100)
    y = [np.sin(x + i / 10) for i in range(len(x))]
    result = create_animated_line_chart(x, y)

    # Get initial x and y coordinates of the line trace
    initial_x = result.data[0].x
    initial_y = result.data[0].y

    # Update trace to a different frame
    update_trace(1)

    # Get updated x and y coordinates of the line trace
    updated_x = result.data[0].x
    updated_y = result.data[0].y

    # Check if x and y coordinates are different
    assert initial_x != updated_x
    assert initial_y != updated_y


# Test case 4: Check if the returned figure has the expected layout properties
def test_create_animated_line_chart_layout():
    x = np.linspace(0, 2 * np.pi, 100)
    y = [np.sin(x + i / 10) for i in range(len(x))]
    result = create_animated_line_chart(x, y)

    # Check if the layout has the expected title text
    assert result.layout.title_text == 'Frame 0'

    # Check if showlegend is False
    assert result.layout.showlegend == Fals