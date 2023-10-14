lotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import plotly.io as pio
import imageio
import time

def create_animated_line_chart(x, y, frames=None):
    """
    Function to create an animated line chart using Plotly.

    Parameters:
        - x (list): The x-axis data points.
        - y (list): The y-axis data points.
        - frames (int): The number of frames for the animation. Defaults to None.

    Returns:
        - fig (plotly.graph_objects.Figure): The animated line chart figure.
    """
    # Create figure
    fig = go.Figure()

    # Add initial line trace
    fig.add_trace(go.Scatter(x=x[0], y=y[0], mode='lines', name='line'))

    # Set up animation frames
    if frames is None:
        frames = len(x)

    # Define animation update function
    def update_trace(frame):
        fig.data[0].x = x[frame]
        fig.data[0].y = y[frame]

    # Create animation
    fig.update_layout(updatemenus=[dict(type="buttons", buttons=[dict(label="Play",
                                                                      method="animate",
                                                                      args=[None, {"frame": {"duration": 500, "redraw": False},
                                                                                   "fromcurrent": True,
                                                                                   "transition": {"duration": 0}}])])])

    fig.frames = [go.Frame(data=[go.Scatter(x=x[i], y=y[i], mode='lines', name='line')],
                          layout=go.Layout(title_text=f'Frame {i}', showlegend=False)) for i in range(frames)]

    fig.update(frames=fig.frames)

    return fig

# Example usage
x = np.linspace(0, 2 * np.pi, 100)
y = [np.sin(x + i / 10) for i in range(len(x))]

fig = create_animated_line_chart(x, y)
fig.show()


def create_animated_bar_chart(data, labels, title):
    """
    Function to create an animated bar chart using Plotly.

    Parameters:
        - data (list): A list of lists representing the data for each frame. Each sublist should contain the values for each bar in the chart.
        - labels (list): A list of labels for each bar in the chart.
        - title (str): The title of the chart.

    Returns:
        - fig (plotly.graph_objects.Figure): A Plotly figure object representing the animated bar chart.
    """
    frames = []
    for i in range(len(data)):
        frame = go.Frame(data=[go.Bar(x=labels, y=data[i])])
        frames.append(frame)

    fig = go.Figure(
        data=[go.Bar(x=labels, y=data[0])],
        frames=frames,
        layout=go.Layout(
            title=title,
            xaxis=dict(title='X-axis'),
            yaxis=dict(title='Y-axis')
        )
    )

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    dict(
                        args=[None, {"frame": {"duration": 500, "redraw": False}}],
                        label="Play",
                        method="animate"
                    ),
                    dict(
                        args=[[None], {"frame": {"duration": 0, "redraw": False}}],
                        label="Pause",
                        method="animate"
                    )
                ],
                direction="left",
                pad={"r": 10, "t": 87},
                showactive=False,
                type="buttons"
            )
        ]
    )

    return fig


def create_animated_scatter_plot(data, x, y, animation_frame, animation_group=None, color=None, size=None):
    """
    Create an animated scatter plot using Plotly.

    Parameters:
        - data (DataFrame or array-like object): DataFrame or array-like object containing the data.
        - x (str or array-like object): Column name or array-like object representing the x-axis values.
        - y (str or array-like object): Column name or array-like object representing the y-axis values.
        - animation_frame (str or array-like object): Column name or array-like object representing the values used for animation frames.
        - animation_group (str or array-like object, optional): Column name or array-like object representing the values used for grouping in animations. Defaults to None.
        - color (str or array-like object, optional): Column name or array-like object representing the values used for coloring the points. Defaults to None.
        - size (str or array-like object, optional): Column name or array-like object representing the values used for sizing the points. Defaults to None.

    Returns:
        - fig (plotly.graph_objects.Figure): The animated scatter plot figure.

    """
    fig = px.scatter(data_frame=data, x=x, y=y, animation_frame=animation_frame,
                     animation_group=animation_group, color=color, size=size)

    return fig


def create_animated_bubble_chart(x_data, y_data, size_data, text_data, animation_frame):
    """
    Create an animated bubble chart using Plotly.

    Parameters:
        - x_data (list): The x-axis data points for each frame.
        - y_data (list): The y-axis data points for each frame.
        - size_data (list): The sizes of the bubbles for each frame.
        - text_data (list): The text labels for each bubble in each frame.
        - animation_frame (list): The animation frames.

    Returns:
        - fig (plotly.graph_objects.Figure): The animated bubble chart figure.
    """
    fig = go.Figure(data=go.Scatter(
        x=x_data[0],
        y=y_data[0],
        mode='markers',
        marker=dict(
            size=size_data[0],
            sizemode='diameter',
            sizeref=2 * np.max(size_data[0]) / (10 ** 2),
            sizemin=4,
            color=size_data[0],
            colorscale='Viridis',
            showscale=True
        ),
        text=text_data[0]
    ))

    fig.update_layout(
        title='Animated Bubble Chart',
        xaxis=dict(title='X Axis'),
        yaxis=dict(title='Y Axis'),
        hovermode='closest'
    )

    frames = []
    for i in range(1, len(x_data)):
        frame = go.Frame(
            data=go.Scatter(
                x=x_data[i],
                y=y_data[i],
                mode='markers',
                marker=dict(
                    size=size_data[i],
                    sizemode='diameter',
                    sizeref=2 * np.max(size_data[i]) / (10 ** 2),
                    sizemin=4,
                    color=size_data[i],
                    colorscale='Viridis',
                    showscale=True
                ),
                text=text_data[i]
            )
        )
        frames.append(frame)

    fig.frames = frames
    fig.update_layout(updatemenus=[dict(type='buttons', showactive=False,
                                        buttons=[dict(label='Play',
                                                      method='animate',
                                                      args=[None, {'frame': {'duration': 500, 'redraw': True},
                                                                   'fromcurrent': True,
                                                                   'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}])])])

    fig.update_layout(height=600, width=800)

    return fig


def create_animated_area_chart(data, x, y, animation_frame, title):
    """
    Create an animated area chart using Plotly.

    Parameters:
        - data (DataFrame or array-like object): DataFrame or array-like object containing the data.
        - x (str or array-like object): Column name or array-like object representing the x-axis values.
        - y (str or array-like object): Column name or array-like object representing the y-axis values.
        - animation_frame (str or array-like object): Column name or array-like object representing the values used for animation frames.
        - title (str): The title of the chart.

    Returns:
        - fig (plotly.graph_objects.Figure): The animated area chart figure.
    """
    fig = go.Figure(data=[
        go.Scatter(x=data[x], y=data[y], fill='tozeroy', line=dict(color='rgb(0,120,200)', width=2))
    ])

    fig.update_layout(
        title=title,
        xaxis=dict(title=x),
        yaxis=dict(title=y),
        hovermode='x',
        updatemenus=[dict(type='buttons',
                          buttons=[dict(label='Play',
                                        method='animate',
                                        args=[None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}])])])

    fig.frames = [go.Frame(data=[go.Scatter(x=data[x][:i+1], y=data[y][:i+1])],
                           layout=go.Layout(title_text=title)) for i in range(len(data))]

    return fig


def create_animated_heatmap(data, labels):
    """
    Create an animated heatmap using Plotly.

    Parameters:
        - data (numpy.ndarray): A 3D numpy array representing the data for each frame.
            The shape of the array should be (num_frames, num_rows, num_columns).
        - labels (list): A list of labels for each frame.

    Returns:
        - fig (plotly.graph_objects.Figure): The animated heatmap figure.
    """
    # Get the number of frames, rows, and columns from the data array
    num_frames, num_rows, num_columns = data.shape

    # Create a list of frames for the animation
    frames = []
    for i in range(num_frames):
        frames.append(go.Frame(data=go.Heatmap(z=data[i], colorscale='Viridis'),
                              layout=go.Layout(title_text=labels[i])))

    # Create the initial heatmap figure
    fig = go.Figure(data=go.Heatmap(z=data[0], colorscale='Viridis'),
                    layout=go.Layout(title_text=labels[0]))

    # Add frames to the figure
    fig.frames = frames

    # Add animation settings to the figure
    fig.update_layout(updatemenus=[dict(type='buttons', showactive=False,
                                        buttons=[dict(label='Play',
                                                      method='animate',
                                                      args=[None,
                                                            dict(frame=dict(duration=500,
                                                                            redraw=True),
                                                                 fromcurrent=True,
                                                                 transition=dict(duration=300,
                                                                                 easing='quadratic-in-out'))])])])

    return fig


def create_animated_pie_chart(labels, values, duration=1000):
    """
    Function to create an animated pie chart using Plotly.

    Parameters:
        - labels (list): A list of labels for each slice of the pie chart.
        - values (list): A list of values for each slice of the pie chart.
        - duration (int): The duration of the animation in milliseconds. Defaults to 1000.

    Returns:
        - fig (plotly.graph_objects.Figure): The animated pie chart figure.
    """
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

    # Set up animation frames
    frames = []
    for i in range(len(values)):
        frame = go.Frame(data=[go.Pie(values=values[:i+1])])
        frames.append(frame)

    # Configure animation settings
    animation = go.animation.Animation(
        frame=frames,
        transition=dict(duration=duration),
        fromcurrent=True,
        mode='immediate'
    )

    # Update layout to enable animation
    fig.update_layout(updatemenus=[go.layout.Updatemenu(buttons=[animation])])

    # Show the animated pie chart
    fig.show()


def create_animated_histogram(data, frames, title):
    """
    Function to create an animated histogram using Plotly.

    Parameters:
        - data (list): A list of lists representing the data for each frame.
            Each sublist should contain the values for each bin in the histogram.
        - frames (int): The number of frames for the animation.
        - title (str): The title of the histogram.

    Returns:
        - fig (plotly.graph_objects.Figure): The animated histogram figure.
    """
    fig = make_subplots()

    # Create initial histogram
    fig.add_trace(go.Histogram(x=data[0], nbinsx=30), 1, 1)

    # Update function to be called for each frame
    def update_hist(frame):
        fig.data[0].x = data[frame]

    # Create frames for animation
    frames = [go.Frame(data=[go.Histogram(x=data[frame], nbinsx=30)]) for frame in range(frames)]

    # Configure animation settings
    animation_settings = {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}

    # Add frames to figure and configure animation settings
    fig.frames = frames
    fig.layout.updatemenus = [
        {
            'buttons': [
                {
                    'args': [None, animation_settings],
                    'label': '&raquo; Play',
                    'method': 'animate'
                }
            ]
        }
    ]

    # Set layout and title of the figure
    fig.update_layout(title_text=title)

    # Show the figure
    fig.show()


def set_animation_speed(speed):
    """
    Function to customize the animation speed.

    Parameters:
        - speed (int or float): The desired animation speed in seconds.

    Returns:
        - None
    """
    # Set the animation speed
    pio.templates.default['layout']['transition']['duration'] = int(speed * 1000)


def add_labels_and_titles(fig, labels=None, title=None):
    """
    Function to add labels and titles to an animation.

    Parameters:
        - fig (plotly.graph_objects.Figure): The figure object representing the animation.
        - labels (list, optional): A list of labels for each frame of the animation. If provided,
                                   it should have the same length as the number of frames.
        - title (str, optional): The title for the animation.

    Returns:
        - fig (plotly.graph_objects.Figure): The updated figure object with labels and title added.
    """

    # Add labels to each frame if provided
    if labels:
        for i, label in enumerate(labels):
            fig.frames[i].data[0].update(text=label)

    # Add title if provided
    if title:
        fig.update_layout(title=title)

    return fig


def control_animation_playback(playback_time):
    
  while True:
      print("Playing animation...")
      time.sleep(playback_time)
      print("Pausing animation...")
      time.sleep(playback_time)
      print("Continuing animation...")
    

def export_animation_as_video(animation, filename, fps=10):
    """
    Export an animation as a video file.

    Parameters:
        - animation (plotly.graph_objects.Figure): The animated plotly figure object.
        - filename (str): The name of the output video file.
        - fps (int, optional): The frames per second of the output video. Defaults to 10.

    Returns:
        - None
    """
    # Save the animation frames as images
    frames = pio.to_image(animation)

    # Define the output file format based on the filename extension
    output_format = 'mp4' if filename.endswith('.mp4') else 'avi'

    # Convert the frames to a video file using imageio
    with imageio.get_writer(filename, format=output_format, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)


def save_animation_as_html(animation, filename):
    """
    Save an animation as an HTML file for web embedding.

    Parameters:
        - animation (plotly.graph_objects.Figure): The animated plotly figure object.
        - filename (str): The name of the output HTML file.

    Returns:
        - None
    """
    fig = go.Figure(animation)
    pio.write_html(fig, file=filename)


def initialize_figure():
    
  fig = go.Figure()
  
  return fig


def add_line_plot(fig, x_data, y_data, name=None):
    
  fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name=name))
  return fig


def update_line_plot(fig, x_data, y_data):
    
  fig.data[0].x = x_data
  fig.data[0].y = y_data
  
  return fig


def set_animation_properties(duration, frame_rate):
    
  pio.templates.default['layout']['transition']['duration'] = int(duration * 1000)
  pio.templates.default['layout']['transition']['frame_rate'] = frame_rate


def add_annotations(plot, x, y, text):
    
    for i in range(len(x)):
        plot.add_annotation(
            x=x[i],
            y=y[i],
            text=text[i],
            showarrow=True,
            arrowhead=1
        )

    return plot


def toggle_elements(fig, show_axes=True, show_gridlines=True, show_legend=True):
    
    # Update x and y axes visibility
    for axis in fig.layout.xaxis:
        axis.visible = show_axes
    for axis in fig.layout.yaxis:
        axis.visible = show_axes

    # Update gridlines visibility
    fig.update_xaxes(showgrid=show_gridlines)
    fig.update_yaxes(showgrid=show_gridlines)

    # Update legend visibility
    fig.update_layout(showlegend=show_legend)

    return fig


def display_animation(fig):
  
  pio.show(fig