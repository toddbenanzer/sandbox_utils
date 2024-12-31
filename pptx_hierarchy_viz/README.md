# HierarchicalVisualizer Documentation

## Class: HierarchicalVisualizer

### Description
The `HierarchicalVisualizer` class provides methods to create hierarchical visualizations, such as tree structures and organizational charts, and export them to PowerPoint slides.

### Methods

#### __init__(self)
Initializes the `HierarchicalVisualizer` object.

**Returns:** None

---

#### create_tree_structure(self, data: Dict[str, Any], **kwargs) -> Any
Generates a tree structure visualization from the provided data.

**Args:**
- `data` (dict): The data representing the hierarchy.
- `**kwargs`: Additional customization options for the tree structure (e.g., style settings).

**Returns:** 
- Visualization object representing the tree structure.

---

#### create_org_chart(self, data: Dict[str, Any], **kwargs) -> Any
Generates an organizational chart from the provided data.

**Args:**
- `data` (dict): The data representing the organizational hierarchy.
- `**kwargs`: Additional customization options for the organizational chart.

**Returns:** 
- Visualization object representing the organizational chart.

---

#### export_to_powerpoint(self, filepath: str, slide_layout: int = 5) -> None
Exports the created visualization into a PowerPoint slide.

**Args:**
- `filepath` (str): The path to save the PowerPoint file.
- `slide_layout` (int): Specify which layout to use for the PowerPoint slide (default is 5).

**Returns:** None

---

#### _generate_tree_visual(self, data: Dict[str, Any], **kwargs) -> Any
Internal method to generate a tree visualization (to be implemented).

**Args:**
- `data` (dict): The data representing the hierarchy.
- `**kwargs`: Additional customization options.

**Returns:** 
- Placeholder for tree visualization object.

---

#### _generate_org_chart_visual(self, data: Dict[str, Any], **kwargs) -> Any
Internal method to generate an org chart visualization (to be implemented).

**Args:**
- `data` (dict): The data representing the organizational hierarchy.
- `**kwargs`: Additional customization options.

**Returns:** 
- Placeholder for org chart visualization object.


# Node Documentation

## Class: Node

### Description
The `Node` class represents a node in a hierarchical structure. It holds information such as a label, styling attributes, and references to parent and child nodes.

### Methods

#### __init__(self, id, label, parent=None, **kwargs)
Initializes the Node object with an identifier, label, and optional parent node and styling attributes.

**Args:**
- `id`: Unique identifier for the node.
- `label` (str): Represents the name or description of the node.
- `parent` (Node, optional): The parent node reference; defaults to None.
- `**kwargs`: Additional style attributes such as color, shape, and line thickness.

**Returns:** None

---

#### set_style(self, color, shape, line_thickness)
Customizes the appearance of the node, including its color, shape, and line thickness.

**Args:**
- `color` (str): Specifies the color of the node.
- `shape` (str): String or enumeration specifying the shape of the node.
- `line_thickness` (int/float): Defines the thickness of the node's border line.

**Returns:** None

---

#### add_child(self, child_node)
Adds a child node to the current node, extending the hierarchy.

**Args:**
- `child_node` (Node): A Node object representing the child node to be added.

**Returns:** None


# PowerPointExporter Documentation

## Class: PowerPointExporter

### Description
The `PowerPointExporter` class is designed to facilitate the creation and manipulation of PowerPoint slides using a specified template. It provides methods to initialize a presentation, add slides with visual data, and save the presentation to a file.

### Methods

#### __init__(self, slide_template: str)
Initializes the PowerPointExporter object with a specified slide template.

**Args:**
- `slide_template` (str): Path to a PowerPoint file to be used as a template.

**Returns:** None

---

#### init_powerpoint(self)
Initializes the PowerPoint presentation object using the provided slide template.

**Returns:** None

---

#### add_slide(self, visual_data: Any, layout: int = 5)
Adds a new slide to the PowerPoint presentation with specified layout, incorporating the provided visualization data.

**Args:**
- `visual_data` (Any): Data object or structure containing the visual components to be added to the slide.
- `layout` (int): Integer specifying the layout of the slide (default is 5).

**Returns:** None

**Raises:**
- `ValueError`: If the presentation has not been initialized before calling this method.

---

#### save_presentation(self, filepath: str)
Saves the PowerPoint presentation to the specified file path.

**Args:**
- `filepath` (str): The path to save the PowerPoint file.

**Returns:** None

**Raises:**
- `ValueError`: If the presentation has not been initialized before calling this method.


# Function Documentation

## Function: load_data_from_csv

### Description
Loads hierarchical data from a CSV file. Parses and converts it into a suitable data structure for visualizations.

### Args
- `file_path` (str): The path to the CSV file containing hierarchical data.

### Returns
- `data` (list or dict): A data structure representing the hierarchy parsed from the CSV file.

### Raises
- `FileNotFoundError`: If the specified file path does not exist.
- `ValueError`: If the CSV content cannot be parsed into the expected hierarchical format.
- `csv.Error`: For issues specifically related to CSV formatting.

---

## Function: convert_to_hierarchy

### Description
Converts a flat list of dictionaries into a nested hierarchical structure.

### Args
- `data` (list of dict): The parsed CSV data.

### Returns
- `dict`: A nested dictionary representing the hierarchy.


# Function Documentation

## Function: load_data_from_json

### Description
Loads hierarchical data from a JSON file. Parses and converts it into a suitable data structure for visualizations.

### Args
- `file_path` (str): The path to the JSON file containing hierarchical data.

### Returns
- `data` (dict): A nested dictionary representing the hierarchy parsed from the JSON file.

### Raises
- `FileNotFoundError`: If the specified file path does not exist.
- `json.JSONDecodeError`: If the content is not valid JSON.
- `ValueError`: If the JSON content cannot be converted into the expected hierarchical format.


# Function Documentation

## Function: load_data_from_xml

### Description
Loads hierarchical data from an XML file. Parses and converts it into a suitable data structure for visualizations.

### Args
- `file_path` (str): The path to the XML file containing hierarchical data.

### Returns
- `data` (dict): A nested dictionary representing the hierarchy parsed from the XML file.

### Raises
- `FileNotFoundError`: If the specified file path does not exist.
- `xml.etree.ElementTree.ParseError`: If the content is not valid XML.
- `ValueError`: If the XML content cannot be converted into the expected hierarchical format.


# Function Documentation

## Function: fetch_data_from_api

### Description
Fetches hierarchical data from a specified API endpoint. Retrieves data in JSON format and converts it into a suitable data structure for visualizations.

### Args
- `api_endpoint` (str): The URL of the API endpoint from which to fetch hierarchical data.

### Returns
- `data` (dict): A nested dictionary representing the hierarchy parsed from the JSON response.

### Raises
- `requests.exceptions.RequestException`: For network issues (e.g., connection error, timeout).
- `json.JSONDecodeError`: If the API response is not valid JSON.
- `ValueError`: If the JSON content cannot be converted into the expected hierarchical format.
- `HTTPError`: For unsuccessful HTTP responses (status codes indicating an error).


# Function Documentation

## Function: validate_hierarchy_data

### Description
Validates the hierarchical data structure to ensure it meets predefined criteria for integrity and structure.

### Args
- `data` (dict): The hierarchical data structure to be validated.

### Returns
- `bool`: True if the data is valid, False otherwise.

### Inner Functions
#### is_valid_node(node: Dict[str, Any]) -> bool
- Validates a single node within the hierarchy.
- Checks for required keys ('attributes', 'children') and ensures 'children' is a list.
- Recursively validates all child nodes.

#### validate_no_cycles(node: Dict[str, Any], seen: set) -> bool
- Checks for circular references in the hierarchy.
- Uses depth-first traversal to ensure no node is revisited.

### Raises
- None: This function does not throw exceptions but instead returns False for invalid input structures.


# Function Documentation

## Function: format_visual_elements

### Description
Formats the visual elements of a hierarchical visualization based on a provided style configuration. Applies styling attributes such as colors, shapes, font sizes, and other visual properties to the nodes and connections in the visualization.

### Args
- `style_config` (dict): A dictionary containing style attributes and their respective values. Expected keys include:
  - `node_color`: Color of the nodes in the visualization (string).
  - `node_shape`: Shape of the nodes (string).
  - `line_thickness`: Thickness of the lines connecting nodes (int or float).
  - `font_size`: Size of the font used in the visualization (int).
  - `font_color`: Color of the font used in the visualization (string).

### Returns
- `dict`: A dictionary containing the formatted visual elements with their styles applied.

### Raises
- `KeyError`: If any expected style attribute is missing in the `style_config`.
- `TypeError`: If the provided values for style attributes are of incorrect types.
