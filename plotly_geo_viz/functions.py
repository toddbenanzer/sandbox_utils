lotly.graph_objects as go

def customize_highlighted_regions(fig, trace_name, color, opacity):
    """
    Customize the color and opacity of the highlighted regions in a Plotly figure.

    Args:
        fig (plotly.graph_objects.Figure): The Plotly figure object.
        trace_name (str): The name of the trace to be customized.
        color (str): The desired color for the highlighted regions.
        opacity (float): The desired opacity value for the highlighted regions.

    Returns:
        plotly.graph_objects.Figure: The updated Plotly figure object.
    """
    # Find the index of the trace with the given name
    trace_index = None
    for i, trace in enumerate(fig.data):
        if trace.name == trace_name:
            trace_index = i
            break

    # If no trace with the given name is found, raise an error
    if trace_index is None:
        raise ValueError(f"No trace found with name '{trace_name}'")

    # Update the color and opacity of the specified trace
    fig.update_traces(
        patch={
            'marker': {'color': [color]},
            'marker.opacity': [opacity]
        },
        selector={'name': trace_name}
    )

    return fi