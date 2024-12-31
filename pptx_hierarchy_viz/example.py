from hierarchical_visualizer import HierarchicalVisualizer
from powerpoint_exporter import PowerPointExporter


# Initialize the visualizer
visualizer = HierarchicalVisualizer()

# Example 1: Create a tree structure visualization
tree_data = {
    'root': {
        'child1': {
            'grandchild1': {}
        },
        'child2': {}
    }
}
tree_vis = visualizer.create_tree_structure(tree_data)
print(tree_vis)  # Output: Tree visualization object

# Example 2: Create an organizational chart
org_chart_data = {
    'CEO': {
        'CTO': {
            'Engineer1': {},
            'Engineer2': {}
        },
        'CFO': {
            'Accountant1': {}
        }
    }
}
org_chart_vis = visualizer.create_org_chart(org_chart_data)
print(org_chart_vis)  # Output: Org chart visualization object

# Example 3: Export a visualization to PowerPoint
output_file = "hierarchical_visualization.pptx"
visualizer.export_to_powerpoint(output_file)
print(f"PPT presentation created: {output_file}")


# Create a root node
root_node = Node(id=1, label="Root")

# Create child nodes
child1 = Node(id=2, label="Child 1")
child2 = Node(id=3, label="Child 2")

# Add children to the root node
root_node.add_child(child1)
root_node.add_child(child2)

# Customize the style of a child node
child1.set_style(color="green", shape="ellipse", line_thickness=2)

# Display the tree structure
print(f"Root Node: {root_node.label}, Children: {[child.label for child in root_node.children]}")
# Output: Root Node: Root, Children: ['Child 1', 'Child 2']

# Display style of a child node
print(f"Child 1 Style: {child1.style}")
# Output: Child 1 Style: {'color': 'green', 'shape': 'ellipse', 'line_thickness': 2}

# Create a deeper hierarchy
grandchild1 = Node(id=4, label="Grandchild 1")
child1.add_child(grandchild1)

# Display the parent of a grandchild
print(f"Grandchild 1 Parent: {grandchild1.parent.label}")
# Output: Grandchild 1 Parent: Child 1



# Initialize PowerPointExporter with a template
exporter = PowerPointExporter(slide_template="path/to/template.pptx")
exporter.init_powerpoint()

# Example 1: Add a slide with some visual data
visual_data = "Introduction to Hierarchical Structures"
exporter.add_slide(visual_data, layout=0)  # Use layout 0, typically a title slide layout

# Example 2: Add another slide with different visual data
another_visual_data = "Details about Node Relationships"
exporter.add_slide(another_visual_data, layout=5)  # Use layout 5, a content layout

# Example 3: Save the presentation to a file
output_file = "hierarchical_visualizations.pptx"
exporter.save_presentation(output_file)
print(f"Presentation saved as {output_file}")


# Example 1: Load data from a valid CSV file

# Assuming 'hierarchy.csv' contains:
# id,parent_id,name
# 1,,Root
# 2,1,Child A
# 3,1,Child B
# 4,2,Child A1

try:
    hierarchy_data = load_data_from_csv("hierarchy.csv")
    print(hierarchy_data)
except Exception as e:
    print(f"An error occurred: {e}")


# Example 2: Handle non-existing CSV file

try:
    hierarchy_data = load_data_from_csv("non_existing_file.csv")
except FileNotFoundError as e:
    print(e)  # Output: The file non_existing_file.csv was not found.


# Example 3: Handle invalid CSV content

# Assuming 'invalid.csv' contains:
# This is not a CSV formatted file

try:
    hierarchy_data = load_data_from_csv("invalid.csv")
except csv.Error as e:
    print(e)  # Output: CSV parsing error...


# Example 1: Load hierarchical data from a valid JSON file

# Assuming 'data.json' contains:
# {
#     "root": {
#         "child1": {
#             "grandchild1": "value1"
#         },
#         "child2": "value2"
#     }
# }

try:
    data = load_data_from_json("data.json")
    print(data)  # Output: {'root': {'child1': {'grandchild1': 'value1'}, 'child2': 'value2'}}
except Exception as e:
    print(f"An error occurred: {e}")


# Example 2: Handle non-existing JSON file

try:
    data = load_data_from_json("nonexistent_file.json")
except FileNotFoundError as e:
    print(e)  # Output: The file nonexistent_file.json was not found.


# Example 3: Handle invalid JSON content

# Assuming 'invalid.json' contains malformed JSON:
# {root: {child: value}

try:
    data = load_data_from_json("invalid.json")
except json.JSONDecodeError as e:
    print(e)  # Output: JSON decoding error: ...


# Example 1: Load hierarchical data from a valid XML file

# Assuming 'data.xml' contains:
# <root>
#     <child1 attr1="value1">
#         <grandchild attr2="value2"/>
#     </child1>
#     <child2 attr3="value3"/>
# </root>

try:
    data = load_data_from_xml("data.xml")
    print(data)
    # Output:
    # {'root': {
    #     'attributes': {},
    #     'children': [
    #         {'child1': {
    #             'attributes': {'attr1': 'value1'},
    #             'children': [
    #                 {'grandchild': {
    #                     'attributes': {'attr2': 'value2'},
    #                     'children': []
    #                 }}
    #             ]
    #         }},
    #         {'child2': {
    #             'attributes': {'attr3': 'value3'},
    #             'children': []
    #         }}
    #     ]
    # }}

except Exception as e:
    print(f"An error occurred: {e}")


# Example 2: Handle non-existing XML file

try:
    data = load_data_from_xml("nonexistent_file.xml")
except FileNotFoundError as e:
    print(e)  # Output: The file nonexistent_file.xml was not found.


# Example 3: Handle invalid XML content

# Assuming 'invalid.xml' contains:
# <root><unclosed></root>

try:
    data = load_data_from_xml("invalid.xml")
except ET.ParseError as e:
    print(e)  # Output: XML parsing error: ...


# Example 1: Fetch hierarchical data from a valid API endpoint

try:
    api_url = "https://api.example.com/hierarchy"
    data = fetch_data_from_api(api_url)
    print(data)  # Assuming the response is a dictionary representing the hierarchy
except Exception as e:
    print(f"An error occurred: {e}")


# Example 2: Handle HTTP error from an API endpoint

try:
    invalid_api_url = "https://api.example.com/invalid_endpoint"
    data = fetch_data_from_api(invalid_api_url)
except requests.exceptions.RequestException as e:
    print(e)  # Output: An error occurred during the network request: HTTP Error


# Example 3: Handle invalid JSON response from an API endpoint

try:
    invalid_json_api_url = "https://api.example.com/invalid_json"
    data = fetch_data_from_api(invalid_json_api_url)
except json.JSONDecodeError as e:
    print(e)  # Output: JSON decoding error...


# Example 4: Handle a valid response with unexpected (non-dict) JSON format

try:
    non_dict_api_url = "https://api.example.com/non_dict_response"
    data = fetch_data_from_api(non_dict_api_url)
except ValueError as e:
    print(e)  # Output: JSON content is not in the expected hierarchical format.


# Example 1: Validate a well-formed hierarchical data structure

valid_hierarchy_data = {
    'root': {
        'attributes': {'type': 'company'},
        'children': [
            {
                'department1': {
                    'attributes': {'head': 'John Smith'},
                    'children': []
                }
            },
            {
                'department2': {
                    'attributes': {'head': 'Jane Doe'},
                    'children': []
                }
            }
        ]
    }
}
print(validate_hierarchy_data(valid_hierarchy_data))  # Output: True


# Example 2: Validate a data structure with missing 'children' key

missing_children_data = {
    'root': {
        'attributes': {'type': 'company'}
        # Missing 'children' key
    }
}
print(validate_hierarchy_data(missing_children_data))  # Output: False


# Example 3: Validate a data structure where 'children' is not a list

invalid_children_data = {
    'root': {
        'attributes': {'type': 'company'},
        'children': {}  # 'children' should be a list, not a dictionary
    }
}
print(validate_hierarchy_data(invalid_children_data))  # Output: False


# Example 4: Validate a hierarchical structure with circular references

circular_node = {
    'attributes': {},
    'children': []
}
circular_node['children'].append({'child': circular_node})  # Adding self-reference

circular_hierarchy_data = {
    'root': circular_node
}
print(validate_hierarchy_data(circular_hierarchy_data))  # Output: False


# Example 1: Correctly format visual elements with a valid style configuration

style_config = {
    'node_color': 'red',
    'node_shape': 'ellipse',
    'line_thickness': 2,
    'font_size': 12,
    'font_color': 'white'
}

formatted_elements = format_visual_elements(style_config)
print(formatted_elements)
# Output: {'node_color': 'red', 'node_shape': 'ellipse', 'line_thickness': 2, 'font_size': 12, 'font_color': 'white'}


# Example 2: Attempt to format with a missing attribute

try:
    incomplete_style_config = {
        'node_color': 'green',
        'node_shape': 'rectangle',
        'line_thickness': 1,
        'font_size': 10
        # Missing 'font_color'
    }
    format_visual_elements(incomplete_style_config)
except KeyError as e:
    print(e)  # Output: Configuration error: 'Missing required style attribute: font_color'


# Example 3: Attempt to format with an incorrect type for 'line_thickness'

try:
    invalid_type_config = {
        'node_color': 'blue',
        'node_shape': 'circle',
        'line_thickness': 'thick',  # Should be a number
        'font_size': 14,
        'font_color': 'black'
    }
    format_visual_elements(invalid_type_config)
except TypeError as e:
    print(e)  # Output: Type error in style configuration: Expected a number for 'line_thickness'.
