
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot
import plotly.io as pio
from geopy import distance

def plot_scatter_map(data, lat_column, lon_column, marker_size=8, marker_color='blue'):
    """
    Function to plot a scatter plot on a map using Plotly.

    Parameters:
        - data: pandas DataFrame - The input data containing latitude and longitude columns.
        - lat_column: str - The name of the column containing latitude values.
        - lon_column: str - The name of the column containing longitude values.
        - marker_size: int (optional) - The size of the markers on the map. Default is 8.
        - marker_color: str (optional) - The color of the markers on the map. Default is 'blue'.

    Returns:
        - fig: plotly.graph_objects.Figure - The resulting scatter plot on a map.
    """
    fig = go.Figure(data=go.Scattermapbox(
        lat=data[lat_column],
        lon=data[lon_column],
        mode='markers',
        marker=dict(
            size=marker_size,
            color=marker_color,
            opacity=0.7
        )
    ))

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=3,
        mapbox_center={"lat": data[lat_column].mean(), "lon": data[lon_column].mean()},
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )

    return fig

def plot_line_on_map(latitude, longitude):
    scatter_trace = go.Scattergeo(
        lat=latitude,
        lon=longitude,
        mode='lines',
        line=dict(color='red', width=2),
        marker=dict(size=4, color='red')
    )

    map_layout = go.Layout(
        geo=dict(
            resolution=110,
            showland=True,
            showlakes=True,
            landcolor='rgb(204, 204, 204)',
            countrycolor='rgb(204, 204, 204)',
            lakecolor='rgb(255, 255, 255)',
            projection_type="equirectangular",
            coastlinewidth=2,
            lataxis=dict(range=[min(latitude), max(latitude)], showgrid=True, dtick=10),
            lonaxis=dict(range=[min(longitude), max(longitude)], showgrid=True, dtick=20),
        )
    )

    fig = go.Figure(data=scatter_trace, layout=map_layout)
    fig.show()

def plot_bar_chart_on_map(data, latitudes, longitudes):
    fig = go.Figure(data=go.Scattergeo(
        lat=latitudes,
        lon=longitudes,
        mode='markers',
        marker=dict(
            size=data,
            color=data,
            colorscale='Viridis',
            reversescale=True,
            colorbar=dict(title='Value'),
        )
    ))

    fig.update_layout(
        title_text='Bar Chart on Map',
        showlegend=False,
        geo=dict(
            resolution=50,
            showland=True,
            showlakes=True,
            landcolor='rgb(204, 204, 204)',
            countrycolor='rgb(204, 204, 204)',
            lakecolor='rgb(255, 255, 255)',
            projection_type="equirectangular",
            coastlinewidth=2,
            lataxis=dict(range=[-90, 90], showgrid=True, dtick=10),
            lonaxis=dict(range=[-180, 180], showgrid=True, dtick=20)
        )
    )

    fig.show()

def plot_choropleth(data, locations, values, title):
    """
    Function to plot a choropleth map using Plotly.

    Parameters:
      - data (pandas.DataFrame): The data containing the values for each location.
      - locations (str): The column name in the data that contains the locations (e.g. country names).
      - values (str): The column name in the data that contains the values to be visualized.
      - title (str): The title of the choropleth map.
    """
    
    fig = px.choropleth(data_frame=data,
                        locations=locations,
                        locationmode='country names',
                        color=values,
                        hover_name=locations,
                        title=title)
    
    fig.show()

def add_markers_on_map(map_fig, latitudes, longitudes):
    
  
def add_markers_on_map(map_fig: Go.Figure(), latitudes: list(), longitutes:list() ,maker_text:list())
"""
Function to add markers to a map

Parameters:
map_fig(Go.Figure()):the input figure object representing a figure object
latitude(list()):A list of latitude values for markers
longitude(list()):A list of longitude value for markers 
maker_text(list()):A list of text labels for markers

Returns:
Go.Figure():The updated figure object with added makers 
"""
#Create a scatter trace for markers 
marker_trace = Go.scattermapbox(latitude = latitude,longitute = longitude,modes ='markers',marker={'size':10,'color':'red'},text = maker_text)

#Add trace to figure object 
map_fig.add_trace(marker_trace)

#Return updated figure object 
return map_fig

def customize_color_scale(data_frame , geojson ,feature_property,color_scale)
"""
Customize color scale for choropeth map 

Parameters:
data_frame(pandas.DataFrame) :The input dataframe containing geographical regions and their corresponding values 
geojson(dict):The GeoJSON file representing geographical regions on the choropeth maps
feature_property(str):Name of property in GeoJSON file that matches with dataframe 
color_scale(list):A list of RGB colors 

Returns:
Go.Figure():The updated figure object with customized color scale 
"""
fig = px.choropeth(data_frame=data_frame ,geojson = geojson ,locations = feature_property,color_continous_scale=color_scale)

return fig

def customize_marker_size(data_frame,pandas.DataFrame(),latitude(str()),longitude(str()),maker_size(str())):
"""
Function to customize maker's size on a map 

Parameters:
data(pandas.DataFrame()):The input Dataframe containing geographical regions and their corresponding values 
latitude(str()):Name of latitude column in dataframe 
longitude(str()):Name of longitude column in dataframe 
maker_size(str()):Name of column containing maker size 

Returns:
Go.Figure():Updated figure object   
"""
fig = Go.Figure()
fig.add_trace(Go.scattermapbox(latitude=dataframe[latitude] ,longitude=dataframe[longitude],mode ='makers',marker={'size':dataframe[maker_size]},text=dataframe['text']))

fig.update_layout(mapbox_style="open-street-map",mapbox_zoom =3,mapbox_center={'latitude':38.9072,'longitude':-77.0369})

return fig 

def customize_marker_opacity(fig:Go.Fig(),opacity:int()):
"""
Customize opacity level for makers 

Parameters;
fig(GO.fig()):figure object representing scatter plots on geographical maps 
opacity(int);Opacity level (between o and1)

Returns;
GO.fig();Updated figure with customized opacity level for makers 
"""

for trace in fig.data:
if isinstance(trace.Go.scattermapbox) and hasattr(trace,'marker'):
trace.marker.opacity = opacity


return fig 


def add_tooltip_maker(map_fig:GO.fig(),maker_label:str())
""""
Add tooltips to makers 

Parameters;
map_fig(GO.fig()):figure object representing scatter plots on geographical maps .
maker_label(str());Dictionary mapping maker coordinates with tooltip labels .

returns;
GO.fig();updated figure with tooltip added to makers .
"""

for i,label in enumerate(map_figure.data):
if isinstance(maker.Go.scattermapbox) and 'maker' in maker :
latitude = maker.latitude
longitude = maker.longitude
if (latitude ,longitude )in maker_label :
tooltip_label = maker_label[(latitude ,longititude)]
map_figure.data[i].update(text=mker_label)


return fg 



def highlight_regions(map_data:list(),highlighted_regions:list())
""""
Highlight specific regions on maps .

Parameters;
maps_data(list);Data for base maps .
highlighted_regions(list);Data for highlighted regions .

returns;
GO.fig();Plotly's Figure objects with highlighted regions .
"""

fig = GO.fig()
for i,data in enumerate(maps_data):
fig.add_trace(GO.Choroplethmapsbox(geojson=data['GeoJson'],location=['location'],z=['values'],colorscale ='Viridis',zmin=min('value'),zmax=max('value'),marker_opacity=.5,colorbar={'title':'Color bar'}))

for i,data in enumerate(highlighted_regions):
fig.add_trace(GO.scattermaps(mode ='makers',latitude=['lat'],longtude=['long'],maker={'size':'8','color':'red','opacity':'1'},hover_template=['hover_templat']))

fig.update_layout(map_style ='carto-posistron',zoom ='3'center={'latitude':28.0902,'longtude':-195.7129})

return fig 


def create_drill_down(data:dict(),level:list())
""""
Function to create interactive drill downs using Plotly

Parameters;
data(dict);dictionary containing information about each drill down level .keys represent levels while value represent information about each level .
level(list);list representing drill down levels


returns;
GO.fig();Plotly's Figure objects representing interactive drill down visualization .  
"""

figs_go.figs()
for location in data[level[0]]:
fig.add_trace(GO.scattermaps(lontude=['long'],latitude=['lat'],text=['name'],mode ='makers',maker_color ='blue',hoverlabel={'name length':'-1'}))

for i in range(1,len(level)):
parent_level = level[i-1]
child_level = level[i]


drill_down(event.point.indx['0']):
selected_location;point.point_indx['0']
children_locations;data[children_level][selected_location]['childern']

new_traces;[]
for location in children_locations :
new_traces.append(GO.scattermaps(lontude=['long'],latitude=['lat'],text=['name']mode ='makers',maker_color ='blue',hoverlabel{'name length':'-1'}))
add_traces(new_traces)

trace[i-1].on_click(drill_down)


return figs 


def zoom_maps(figs_go.figs(),zoom:int()):
""""
Zoom In/out function .

Parameter;
zoom(int);Zoom factor.Positive value means zoom-in while negative means zoom-out .

returns ;
GO.figs();Updated Figure objects after zooming .
"""
update_geos(projection_scale=int())

return figs 



def pan_maps(coordintes:tuple(),pan_direction:str()):
""""
Panning function.

Parameter;
coordintes(tuple());Current coordinates(latitude,longitute)
pan_direction(str());Panning direction(up,left,right and down)

returns ;
tuple();Coordinates after panning(latitude,longitute)
"""

pan_amount;.1 #assuming .1 degree panning

if pan_direction =='up':
cordinates+=pan_amount

elif pan_direction =='down'
cordinates-=pan_amount


elif pan_direction=='right'
cordinates+=pan_amount


elif pan_direction=='left'
cordinates-=pan_amount


return coordinates 



def display_multiple_layers(layers:list()):
""""
Function to display multiple layers/maps .

parameter ;
layers(list());List dictionaries containing information about each layer .Each dictionary will have longtude,lattiude and name keys.

returns ;
GO.figs();Plotly's Figure objects displaying multiple layers/maps.  
"""

fg_go.fgs()
for i,data enumerate(layers):
fg.add_trace(GO.scattermaps(lat=data['lat'],lon=data['lon']mode ='makers',maker={'size':'10','color':'blue'},name=data['name']))
fg.update_layout(hovermode_closet,mapbbox_style-'open-street-map',center={'lat':layer['lat'][0],'lon'=layers['lon'][0]},zoom-'10')

return fg 




def create_heat_map(df:pandas.DataFrame(),latitude:str(),longtude:str()):
""""
Heat Map function.

parameter;
df(pandas.DataFrame());Input DataFrame consisting heat maps information .
latitude(str());Name latitude column .
longtude(str());Name longtude column .

returns ;
GO.fgs():Figure objects containing heat maps  
"""


fg_go.fgs(go.densitymaps(lat=df[latitude],lon=df[longtude],z=df,radius-'10'))

fg.update_layout(fg,mapbbox_style-'stamen-terrain',center_lon=sum(df[long])/len(df),center_lat=sum(df)/len(df),zoom-'10')

return fg



def create_contour_maps(df:pandas.DataFrame(),location,pandas.Series,z_value,pandas.Series,mapbbox_token:str())
""""
Contour Maps function.

parameter ;
df(pandas.DataFrame());Input DataFrame consisting contours information .
location(pandas.Series());Series representing various locations over maps .
z_value(pandas.Series());Series representing numerical/z_value over contours maps.
mapbbox_token(str());MapsToken


returns ;
go.fgs().Figure objects containing contours over maps   
"""


fg_go.fgs()

for i,data enumerate(location,z_value):
trace_go.choroplethmaps(locations=(df[i]),z_value(i),colorscale-'viridis'reversescale=True,zmin=min(z_value),zmax=max(z_value),contours=dict(start-min(z_value),end-max(z_value)),size-'1'),marker_opacity-.5,colorbar_title-(text-'Contour levels'))
add_traces(trace)


fg.update_layout(mapbbox_style-'carto-posistron',accesstoken-str(mapbbox_token),zoom-'2.5',center-lat-"38.72490",center-lon-"95.61446")

return fg 




def create_bubble_plot(df:pandas.Dataframe(),lat_col:str(),lon_col:str(),size_col:str())
""""
Functions creating bubble plots using Plotly .

parameter ;
df(pandas.DataFrame());Input DataFrame consisting bubble plots information .
lat_col(str())'column name storing latitude information within Dataframe.
lon_col(str());column storing longtude within Dataframe.
size_col(str())*column storing bubbles sizes within Dataframe.


returns ;
go.figs().Figure objects displaying bubble plots over geographical maps   
"""

fgs-go.fgs(go.scattergeo(lat-df(lat_col),lon-df(lon_col),mode-makers,maker-size=df(size_col)))

fgs.update_layout(go_scope-world)

return fgs()



def create_3d_geospatial_visualization(df.pandas.data.frame(lat_col.str().lon_col.str().values.str()))
""""
Functions creating three dimensional visualisation using Plotly .

parameter ;
df(pandas.Data.Frame())*Input Data.Frame consisting three dimensional plots information .
lat_col.str())*Name columns storing latitude within Data.Frame.
lon_col.str())*Name columns storing longtudes within Data.Frame.values.str())*Name columns storing z_values .

returns ;
go.figs().three dimensional visualisation using Plotly.  
"""


fgs-go.fgs(go.scatter3d(x-df_lon.col,y-df_lat.col,z-df-value.col.mode-makers.maker-size-min12-color-df.value.col.colorscale-viridis-opacity-.8))

fgs.update_layout(scene.dict(x_axis-title_lon.col,y.axis-title_lat.col,z.axis.title-value.col))

fgs.show()




