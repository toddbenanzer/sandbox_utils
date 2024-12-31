from plotly.graph_objects import Figure
from plotly.graph_objects import Scatter
from your_module import Animator
from your_module import Animator, Interaction
import pandas as pd
import plotly.express as px
import plotly.io as pio


class Animator:
    """
    A class for creating animations with Plotly based on data and plot type.
    """

    def __init__(self, data: pd.DataFrame, plot_type: str):
        """
        Initializes the Animator with data and plot type.

        Args:
            data (pd.DataFrame): The data to be used for the animation.
            plot_type (str): The type of plot for the animation (e.g., 'scatter', 'line').
        """
        self.data = data
        self.plot_type = plot_type
        self.animation = None

    def generate_animation(self, **kwargs) -> Figure:
        """
        Generates an animation using the specified plot type and data.

        Args:
            **kwargs: Additional arguments to customize the animation (e.g., frame rate, loop settings).

        Returns:
            plotly.graph_objects.Figure: The generated animation ready for customization or export.
        """
        if self.plot_type == 'scatter':
            self.animation = px.scatter(self.data, animation_frame=kwargs.get('animation_frame'),
                                        animation_group=kwargs.get('animation_group'))
        elif self.plot_type == 'line':
            self.animation = px.line(self.data, animation_frame=kwargs.get('animation_frame'),
                                     animation_group=kwargs.get('animation_group'))
        else:
            raise ValueError(f"Unsupported plot_type '{self.plot_type}'. Supported types are 'scatter', 'line'.")

        return self.animation

    def set_animation_properties(self, duration: int, easing: str):
        """
        Sets the animation properties such as duration and easing.

        Args:
            duration (int): Duration of the animation in milliseconds.
            easing (str): Easing function applied to the animation (e.g., 'ease-in-out', 'linear').
        """
        if not self.animation:
            raise ValueError("Animation has not been generated yet. Call generate_animation first.")

        self.animation.update_layout(
            updatemenus=[{
                'buttons': [{
                    'args': [None, {'frame': {'duration': duration, 'redraw': True}, 'fromcurrent': True}],
                    'label': 'Play',
                    'method': 'animate'
                }],
                'direction': 'left',
                'pad': {'r': 10, 't': 87},
                'showactive': False,
                'type': 'buttons',
                'x': 0.1,
                'xanchor': 'right',
                'y': 0,
                'yanchor': 'top'
            }]
        )
        self.animation.update_traces(marker={'ease': easing})



class Customizer:
    """
    A class for customizing the visual aesthetics of Plotly animations.
    """

    def __init__(self, animation: Figure):
        """
        Initializes the Customizer with an existing Plotly animation.

        Args:
            animation (Figure): The animation to be customized.
        """
        if not isinstance(animation, Figure):
            raise ValueError("Expected a Plotly Figure object for animation.")
        self.animation = animation

    def set_color_palette(self, palette):
        """
        Applies a color palette to the animation.

        Args:
            palette (list or dict): Colors to be used for different data groups/categories.
        """
        self.animation.update_traces(marker=dict(color=palette))

    def set_marker_style(self, style):
        """
        Sets marker styles in the animation to differentiate data points.

        Args:
            style (dict): Marker properties like size, symbol, or line width (e.g., {'size': 10, 'symbol': 'circle'}).
        """
        self.animation.update_traces(marker=style)

    def customize_layout(self, title: str, labels: dict, legends: dict):
        """
        Modifies the layout of the animation, including title, axis labels, and legend position.

        Args:
            title (str): Title of the animation.
            labels (dict): Axis labels specified as {'x': 'X-axis Label', 'y': 'Y-axis Label'}.
            legends (dict): Settings for legend appearance and position (e.g., {'orientation': 'h', 'x': 0.5, 'y': -0.1}).
        """
        self.animation.update_layout(
            title=title,
            xaxis_title=labels.get('x'),
            yaxis_title=labels.get('y'),
            legend=legends
        )



class Interaction:
    """
    A class to add interactive features to Plotly animations, allowing for hover details and clickable elements.
    """

    def __init__(self, animation: Figure):
        """
        Initializes the Interaction object with an existing Plotly animation.

        Args:
            animation (Figure): The animation to be enhanced with interactivity.
        """
        if not isinstance(animation, Figure):
            raise ValueError("Expected a Plotly Figure object for animation.")
        self.animation = animation

    def add_hover_info(self, info):
        """
        Adds hover information to elements in the animation.

        Args:
            info (str or dict): Content to be displayed on hover (e.g., a list of strings or a data column name).
        """
        self.animation.update_traces(hoverinfo='text', text=info)

    def set_clickable_elements(self, elements):
        """
        Sets elements within the animation to be clickable, triggering actions.

        Args:
            elements (dict): A dictionary specifying elements and actions to be executed on click events.
                             Format: {'element_id': 'action'}, where 'action' could be a JavaScript function or URL.
        """
        # Example implementation for clickable elements would require custom JavaScript and HTML setup.
        # This method serves as a placeholder to indicate functionality and would need further dev in a real project.
        # Plotly natively does not support complex element click event handling within the Python library alone.
        raise NotImplementedError("Clickable elements require JavaScript/HTML integration in the output environment.")



class Exporter:
    """
    A class for exporting Plotly animations to various file formats.
    """

    def __init__(self, animation: Figure):
        """
        Initializes the Exporter with an existing Plotly animation.

        Args:
            animation (Figure): The animation to be exported.
        """
        if not isinstance(animation, Figure):
            raise ValueError("Expected a Plotly Figure object for animation.")
        self.animation = animation

    def export_to_html(self, file_name: str):
        """
        Exports the animation as an HTML file.

        Args:
            file_name (str): The name of the HTML file to which the animation will be exported.
        """
        pio.write_html(self.animation, file=file_name)

    def export_to_gif(self, file_name: str):
        """
        Converts and exports the animation as a GIF file.

        Args:
            file_name (str): The name of the GIF file to which the animation will be exported.
        """
        raise NotImplementedError("Plotly does not natively support exporting animations to GIF directly. "
                                  "Consider using external tools like imageio or moviepy for this purpose.")

    def save_as_image(self, file_name: str, image_type: str):
        """
        Saves the animation or a snapshot of it as a static image.

        Args:
            file_name (str): The name of the image file to save the snapshot.
            image_type (str): The format of the image file (e.g., 'png', 'jpeg').
        """
        pio.write_image(self.animation, file=file_name, format=image_type)



def load_data(file_path: str, file_type: str):
    """
    Loads data from a specified file into a pandas DataFrame.

    Args:
        file_path (str): The path to the data file.
        file_type (str): The format of the file ('csv', 'json', 'excel').

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.

    Raises:
        ValueError: If an unsupported file type is provided.
        FileNotFoundError: If the file at the given path does not exist.
        Exception: For any other issues encountered during file loading.
    """
    try:
        if file_type == 'csv':
            data = pd.read_csv(file_path)
        elif file_type == 'json':
            data = pd.read_json(file_path)
        elif file_type in ['xls', 'xlsx', 'excel']:
            data = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type '{file_type}'. Supported types are 'csv', 'json', 'excel'.")
        return data
    except FileNotFoundError as fnf_error:
        raise FileNotFoundError(f"The file '{file_path}' was not found: {fnf_error}")
    except Exception as e:
        raise Exception(f"An error occurred while loading the file: {e}")



def save_animation(animation: Figure, file_path: str):
    """
    Saves a Plotly animation to a specified file path.

    Args:
        animation (Figure): The Plotly Figure object representing the animation to be saved.
        file_path (str): The destination path to save the animation, including the filename and extension.

    Raises:
        ValueError: If an unsupported file extension is provided.
        Exception: If an error occurs while saving the file.
    """
    if not isinstance(animation, Figure):
        raise ValueError("Expected a Plotly Figure object for animation.")

    # Determine file format based on file extension
    file_extension = file_path.split('.')[-1].lower()

    try:
        if file_extension == 'html':
            pio.write_html(animation, file=file_path)
        elif file_extension == 'json':
            pio.write_json(animation, file=file_path)
        else:
            raise ValueError(f"Unsupported file extension '.{file_extension}'. Supported extensions are '.html' and '.json'.")
    except Exception as e:
        raise Exception(f"An error occurred while saving the animation: {e}")


def example_script(script_name: str) -> str:
    """
    Retrieves a pre-defined example script illustrating the use of the Python package.

    Args:
        script_name (str): The name of the example script to retrieve.

    Returns:
        str: A string containing the Python code of the example script.

    Raises:
        ValueError: If an unknown script_name is provided.
    """
    scripts = {
        'basic_animation': """

# Dummy data
data = pd.DataFrame({'x': [1, 2, 3], 'y': [3, 1, 6]})

# Initialize and generate an animation
animator = Animator(data, 'scatter')
animation = animator.generate_animation(animation_frame='x')

# Display the animation
animation.show()
""",
        'custom_interactivity': """

# Dummy data
data = pd.DataFrame({'x': [1, 2, 3], 'y': [3, 1, 6]})

# Initialize and generate an animation
animator = Animator(data, 'line')
animation = animator.generate_animation(animation_frame='x')

# Add custom interactivity
interaction = Interaction(animation)
interaction.add_hover_info(['point 1', 'point 2', 'point 3'])

# Display the animation
animation.show()
"""
    }
    
    if script_name in scripts:
        return scripts[script_name]
    else:
        raise ValueError(f"Unknown script_name '{script_name}'. Available scripts: {', '.join(scripts.keys())}.")
