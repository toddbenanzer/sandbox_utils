from AnimatorModule import Animator  # Replace AnimatorModule with the actual module name
from CustomizerModule import Customizer  # Replace CustomizerModule with the actual module name
from ExporterModule import Exporter  # Replace ExporterModule with the actual module name
from InteractionModule import Interaction  # Replace InteractionModule with the actual module name
from plotly.graph_objects import Figure, Scatter
import pandas as pd


# Example 1: Simple Scatter Animation
data_scatter = pd.DataFrame({
    'x': [1, 2, 3, 4],
    'y': [4, 3, 2, 1],
    'frame': [0, 0, 1, 1]
})
animator_scatter = Animator(data_scatter, 'scatter')
animation_scatter = animator_scatter.generate_animation(animation_frame='frame')
animator_scatter.set_animation_properties(duration=500, easing='ease-out')
animation_scatter.show()

# Example 2: Simple Line Animation
data_line = pd.DataFrame({
    'x': [1, 2, 3, 4],
    'y': [1, 3, 2, 4],
    'frame': [0, 1, 1, 2]
})
animator_line = Animator(data_line, 'line')
animation_line = animator_line.generate_animation(animation_frame='frame')
animator_line.set_animation_properties(duration=1000, easing='linear')
animation_line.show()

# Example 3: Unsupported Plot Type
try:
    data_invalid = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
    animator_invalid = Animator(data_invalid, 'bar')
    animator_invalid.generate_animation()
except ValueError as e:
    print(e)  # Expected output: Unsupported plot_type 'bar'. Supported types are 'scatter', 'line'.



# Example 1: Simple Color Customization
fig = Figure(data=[Scatter(x=[1, 2, 3], y=[3, 1, 6])])
customizer = Customizer(fig)
customizer.set_color_palette(['red', 'green', 'blue'])
fig.show()

# Example 2: Marker Style Customization
fig = Figure(data=[Scatter(x=[1, 2, 3], y=[4, 3, 2])])
customizer = Customizer(fig)
customizer.set_marker_style({'size': 12, 'symbol': 'triangle-up'})
fig.show()

# Example 3: Layout Customization
fig = Figure(data=[Scatter(x=[1, 2, 3], y=[2, 5, 3])])
customizer = Customizer(fig)
customizer.customize_layout(
    title="Customized Animation",
    labels={'x': 'Time (s)', 'y': 'Value'},
    legends={'orientation': 'v', 'x': 1, 'y': 1}
)
fig.show()



# Example 1: Adding Hover Information
fig = Figure(data=[Scatter(x=[1, 2, 3], y=[10, 11, 12])])
interaction = Interaction(fig)
hover_info = ["Data point 1", "Data point 2", "Data point 3"]
interaction.add_hover_info(hover_info)
fig.show()

# Example 2: Attempting to Set Clickable Elements
# This will raise NotImplementedError, demonstrating the limitation of the current placeholder method.
try:
    interaction.set_clickable_elements({'element_1': 'open_url'})
except NotImplementedError as e:
    print(e)  # Expected output: Clickable elements require JavaScript/HTML integration in the output environment.



# Example 1: Exporting to HTML
fig = Figure(data=[Scatter(x=[1, 2, 3], y=[3, 1, 6])])
exporter = Exporter(fig)
exporter.export_to_html("animation.html")

# Example 2: Attempting to Export to GIF
# This will raise NotImplementedError, demonstrating the limitation of the current method.
try:
    exporter.export_to_gif("animation.gif")
except NotImplementedError as e:
    print(e)  # Expected output: Plotly does not natively support exporting animations to GIF directly.

# Example 3: Saving as PNG Image
exporter.save_as_image("animation.png", "png")


# Example 1: Loading a CSV file
data_csv = load_data("data.csv", "csv")
print(data_csv.head())  # Display the first few rows of the CSV data

# Example 2: Loading a JSON file
data_json = load_data("data.json", "json")
print(data_json.head())  # Display the first few rows of the JSON data

# Example 3: Loading an Excel file
data_excel = load_data("data.xlsx", "excel")
print(data_excel.head())  # Display the first few rows of the Excel data

# Example 4: Attempting to load an unsupported file type
try:
    data_xml = load_data("data.xml", "xml")
except ValueError as e:
    print(e)  # Expected output: Unsupported file type 'xml'. Supported types are 'csv', 'json', 'excel'.



# Example 1: Save animation as HTML file
fig = Figure(data=[Scatter(x=[1, 2, 3], y=[3, 1, 6])])
save_animation(fig, "animation.html")

# Example 2: Save animation as JSON file
fig = Figure(data=[Scatter(x=[4, 5, 6], y=[6, 5, 4])])
save_animation(fig, "animation.json")

# Example 3: Attempting to save with unsupported extension
try:
    save_animation(fig, "animation.txt")
except ValueError as e:
    print(e)  # Expected output: Unsupported file extension '.txt'. Supported extensions are '.html' and '.json'.


# Example 1: Retrieve the basic animation script
basic_animation_script = example_script('basic_animation')
print(basic_animation_script)

# Example 2: Retrieve the custom interactivity script
custom_interactivity_script = example_script('custom_interactivity')
print(custom_interactivity_script)

# Example 3: Attempt to retrieve a non-existent script
try:
    non_existent_script = example_script('advanced_interactivity')
except ValueError as e:
    print(e)  # Expected output: Unknown script_name 'advanced_interactivity'. Available scripts: basic_animation, custom_interactivity.
