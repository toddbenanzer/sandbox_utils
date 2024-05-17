
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as offline

def read_csv_file(file_path):
    """
    Read data from a CSV file and return it as a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pandas.DataFrame: The data read from the CSV file.
    """
    return pd.read_csv(file_path)

def read_json_file(file_path):
    """
    Read data from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The data read from the JSON file as a dictionary.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    return data

def read_data_from_database(database_url, query):
    """
    Function to read data from a database using a given query.

    Parameters:
        - database_url (str): The URL or connection string of the database.
        - query (str): The SQL query to execute for fetching the data.

    Returns:
        pandas.DataFrame: DataFrame containing the fetched data.
    """
    # Connect to the database
    connection = pd.read_sql(query, con=database_url)
    
    # Read data from the database using the provided query
    df = pd.read_sql_query(query, connection)
    
    # Close the database connection
    connection.close()
    
    # Return the fetched data as DataFrame
    return df

def clean_and_preprocess_data(data):
    """
    Clean and preprocess the data.
    
    Parameters:
        data (pandas.DataFrame): The input dataset.
        
    Returns:
        pandas.DataFrame: Cleaned and preprocessed dataset.
    """
    
     # Remove any duplicate rows
     # Remove any missing values
     # Convert date columns to datetime objects, if applicable
     
     date_columns = ['date', 'start_date', 'end_date']
     for col in date_columns:
         if col in data.columns:
             data[col] = pd.to_datetime(data[col])

     return data.drop_duplicates().dropna()


def filter_data(data, condition):
   """
   Filter data based on certain conditions.

   Parameters:
       - data: The dataset to be filtered.
       - condition: The filtering condition. This can be a lambda function or any other callable that takes a single argument and returns a boolean value.

   Returns:
       - The filtered dataset.
   """
   return [row for row in data if condition(row)]

def aggregate_data(data, group_by, agg_func):
   """
   Aggregates data based on different variables.

   Parameters:
       - data: Pandas DataFrame containing the data to be aggregated.
       - group_by: List or string specifying the variable(s) to group by.
       - agg_func: Dictionary specifying the aggregation function for each variable.

   Returns:
       Aggregated DataFrame.
   """
   
   if isinstance(group_by, str):
      group_by = [group_by]
   
   return data.groupby(group_by).agg(agg_func)

def calculate_summary_statistics(data):
      """
      Calculate summary statistics for a given dataset.

      Parameters:
          data (pandas.DataFrame): The input dataset.

      Returns:
          dict: A dictionary containing the summary statistics.
      """
      
      summary_stats = {
          'mean': 	data.mean(),
          'median': 	data.median(),
          'mode': 	data.mode().iloc[0],
          'min': 	data.min(),
          'max': 	data.max(),
          'std': 	data.std(),
          'var': 	data.var()
      }

      return summary_stats

def create_bar_plot(x, y, title):
     fig = go.Figure(data=[go.Bar(x=x, y=y)])
     fig.update_layout(title_text=title)
     fig.show()

def create_line_plot(x_values, y_values, title):
     """
     Create a line plot using Plotly.

     Parameters:
         x_values (list): The x values for the line plot.
         y_values (list): The y values for the line plot.
         title (str): The title of the line plot.

     Returns:
         fig (plotly.graph_objs._figure.Figure): The created line plot figure.
     """

     trace = go.Scatter(x=x_values, y=y_values)
     layout = go.Layout(title=title)
     
     fig = go.Figure(data=[trace], layout=layout)

     return fig

def create_scatter_plot(data, x, y, title):
      """
      Function to create a scatter plot using Plotly.

      Parameters:
          - data: pandas DataFrame or numpy array containing the data.
          - x: column name or index of the x-axis data.
          - y: column name or index of the y-axis data.
          - title: title of the scatter plot.

      Returns:
          - fig: Plotly figure object representing the scatter plot.
      """

      fig = px.scatter(data_frame=data, x=x, y=y, title=title)
      
      return fig

def create_pie_chart(labels, values):
      """
      Create a pie chart using Plotly.

      Parameters:
          labels (list): A list of labels for each pie slice.
          values (list): A list of values for each pie slice.

      Returns:
         fig(plotly.graph_objects.Figure) :The created pie chart 
      
      """

      
      return go.Figure(data=go.Pie(labels=labels.values))




def create_histogram(data,x_label,title):
     
   
	fig = go.Figure()
	fig.add_trace(go.Histogram(x=data))
	fig.update_layout(
            xaxis_title=x_label,
            title=title
        )
	fig.show()



def create_stacked_area_chart(data,x,y):

# Iterate over each column in the dataset 
# Update layout with sliders


	fig=go.Figure()
	for column in 	data.columns:

		fig.add_trace(go.Scatter(
                	x=data[x],
                	y=data[column],
                	mode='lines',
                	stackgroup='one',
                	name=column))

	
	fig.update_layout(
            	title='Stacked Area Chart',
            	xaxis_title=x,
            	yaxis_title=y)

	return fig


def create_stacked_bar_chart(data,x_axis_labels,y_axis_labels):

"""
Function to create a stacked bar chart using Plotly.


Args:

- list representing each category

- labels 

Returns:

- Figure object representing stacked bar chart 

"""

	
	fig=go.Figure()
	for i in range(len(data)):
		
	fig.add_trace(go.Bar(
               	x=x_axis_labels,
               	y=data[i],
               	name=y_axis_labels[i]
               ))

	
	fig.update_layout(barmode='stack')

	return fig


def create_boxplot(data,x_axis,y_axis,title):

"""
Parameters:

	pandas dataframe 

	column name of variable to be plotted on x-axis 

	column name of variable to be plotted on y-axis 
	
	title 

Returns:

Figure object representing boxplot 


"""
	
	fig=go.Figure()
	for group in 	data[x_axis].unique():
          
	fig.add_trace(go.Box(
               	x=data[data[x_axis]==group][x_axis],
               	y=data[data[x_axis]==group][y_axis],
               	name=group))
              
           
	
	fig.update_layout(
           	title=title,
           	xaxis_title=x_axis,
           	yaxis_title=y_axis,
           	showlegend=True)

	return fig



def create_heatmap(data,x_labels,y_labels):

# Create heatmap figure and show it
	
	fig=go.Figure(data=go.Heatmap(
          	z=data,
          	x=x_labels,
          	y=y_labels,
          	colorscale='Viridis'))

	
	 fig.update_layout(coloraxis_colorbar=dict(title='Colorbar Title'),
                      title_text='Heatmap Title')

	fig.show()



 

def create_treemap(labels,parents ,values ):

# Return updated treemap 


	return	go.Figure(go.Treemap(labels=labels , parents =parents ,values =values )).show()


  

 def create_funnel_visualization(labels ,values ):

## Funnel visualisation and layout


	trace = go.Funnel(y=labels ,x=values ,textposition ="inside" ,
	textinfo="value+percent previous")

	layout =	go.Layout(title="Funnel Visualization")
	
	return	go.Figure(data=[trace],layout=layout )





 def calculate_cohort_matrix( dat < cohort_period cohort_group users >):

## Cohort matrix calculated based on grouping by cohort period and group 


cohorts.dat.groupby(['cohort_period','cohort_group']).agg({'users':'nunique'}).reset_index()

return cohorts.pivot(index=" cohort_group",columns="cohort_period",values="users")

 def create_cohort_analysis( dat < cohort_period cohort_group users > ,x_labels,y_labels ):


cohort_matrix.calculate_cohort_matrix(dat )

## Create heatmap using Plotly 


return	go.Figure(go.Heatmap(z = cohort_matrix ,x=x_labels ,y=y_labels ,colorscale ="Virdis" ,
colorbar=dict(colorbar_size))). update_layout(

title ='Cohort Analysis' ,xaxis=dict(title ='Cohort Period') ,yaxis=dict(title ='Cohort Group') ).show()





 def create_time_series_plot( dat < time_series > ):

# Add sliders for different time periods 
# Show figure 


sliders=[
	dict(active0,currentvalue={"prefix":" Period }},pad={"t":50},
steps=[dict(label=str(period),method "update" args=[
{"visible":[i==period for i in range(len(dat))]}, {"title":f " Time Series-Period {period}"}])for period in range(len(dat))])]




return	go .Figure().add_trace(go.Scatter(x.dat ['time_series'],y.dat['time_series'],name='Full Time Series' ).update_layout(sliders.sliders).show())

 



 def add_annotations( fig,x,y,text ):



"""
Add Annotations To Figure Object 


Parameters:


Plotly Figure Object 


List of x-coordinates 
 
List Of Y-coordinates 

Annotation text 


Returns:



Updated Figure Object



"""

## List Of Annotation Objects 
		
annotations=[]
for i	in	range(len(x)):
annotations.append(go.layout.Annotation(x=x[i],y=y[i] text=text[i],arrowhead1 arrowsize1 font=dict(size12,colorblack),aligncenter))

return	fg['layout'].update(annotations.annnotations)



 def customize_color_palette(plot,color_palette ):



"""
Function To Customize Color Palette Of Plots 

Parameters:


Plotly Figure Object 


List Of Colors  


Returns:


Updated Figure Object 



"""

## Update Marker Color Attribute With Provided Palette 
 
i.trace.plot.data enumerate(plot.data ):
plot.data [i].marker.color=color_palette[i % len(color_palette)]

return plot 



 def customize_plot_labels(fig,x_label=None,y_label=None,title=None ):



"""
Customize Axis Labels And Title 



Parameters:



Plotly Figure Object 

X-axis Label 




Y-Axis Label 




Title 



Returns:



Updated Figure Object




"""

if x_label :
fg.update_xaxes(title_text.x_label)


if y_label :
fg.update_yaxes(title_text.y_label)


if title :
fg.update_layout(title_text.title )


return fg



 def add_tooltips(dat<dict>tooltips<dict>) :

""" Add Interactive Tooltips And Hover Effects 
 
 
Parameters :


Data Dictionary 



Tooltip Text Dictionary 



Scatter Plot Created With Tooltips Added 


"""

## Scatter Plot Created & Updated With Custom Tooltip Text 



return	go .Figure(dat.go.Scatter(x.dat ["x"],y.dat["y"],mode "markers")).update_traces(hovertemplate "<b>%{text}</b>") .data[0].text.tooltips.values()



 def add_interactive_filters(fig<dict>filter_options<dict>) :



""" Interactive Filters & Dropdown Menus Added To Figure



Parameters:


Plotly Figure Object 



Dictionary Of Filter Options 




Updated Figure Object With Filters Added 



"""


## Create Dropdown Menu Options & Layout 
dropdown_options=[]

for filter_name options.items():
dropdown_options.append(dict(method "restyle",label.filter_name,args[{"visible":[True if i==j else False for j in range(len(options))]}]))


return fg.update_layout(updatemenus=[dict(buttons.dropdown_options direction "down",pad {"r":10,"t":10},showactive True,x0.1 xanchor "left",y1.2,yanchor "top")])


 def export_plot_as_html(plot<dict>filename) :

""" Export Plot As HTML File
 
 
Parameters :

Plotly	Figure	Object
 

Filename For	Output HTML File 




HTML File Exported With Given Filename 


"""


offline.plot(plot filename.filename auto_open False)



 def export_dashboard_layout(layout<dict>output_file) :

""" Export Dashboard Layout As HTML Template
 
 
Parameters :

Dashboard Layout Dict
 

Output HTML File Path
 
 
Layout Converted To JSON String & Written To Output File As HTML Template 

"""


layout_json.json.dumps(layout)


with open(output_file ,'w')as	file.write(f "<!DOCTYPE html><html><head><script src ="https://cdn.plot.ly/plotly-latest.min.js"></script></head><body><div id="dashboard" style ="width100% height100%"></div><script>var layout.layout_json;Plotly.newPlot('dashboard',[]layout);</script></body></html>")





 def save_plot_as_image(plot<dict>filename image_format.png) :



""" Save Plot As Image In Specified Format
 
 
Parameters :


Plotly	Figure	Object
 

Filename For	Output Image Without Extension
 

Image Format PNG Default 
 
 
Plot Saved As Image In Specified Format 

"""

plot.write_image(f "{filename}.{image_format}")

