
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
import tempfile
import imageio
import os
import time


def create_animated_line_plot(x, y):
    fig = go.Figure(
        frames=[
            go.Frame(
                data=[
                    go.Scatter(x=x[:i], y=y[:i], mode='lines+markers')
                ]
            ) for i in range(2, len(x) + 1)
        ]
    )

    fig.add_trace(go.Scatter(x=[x[0]], y=[y[0]], mode='lines+markers'))

    fig.update_layout(
        xaxis=dict(range=[min(x), max(x)]),
        yaxis=dict(range=[min(y), max(y)]),
        title='Animated Line Plot',
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(label="Play",
                         method="animate",
                         args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True,
                                      "transition": {"duration": 0}}]),
                    dict(label="Pause",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
                ],
            )
        ]
    )

    return fig


def create_animated_bar_plot(data, frames_per_second=10):
    x = list(data.keys())
    y = list(data.values())

    fig = go.Figure(go.Bar(x=x, y=y[0]))

    frames = [go.Frame(data=[go.Bar(x=x, y=y[i])],
                       layout=go.Layout(title=f'Time Step {i}')) for i in range(len(y))]

    animation_config = dict(
        frame=dict(duration=1000 / frames_per_second),
        fromcurrent=True,
        mode='immediate'
    )

    fig.update(frames=frames)
    fig.update_layout(title=f'Time Step 0', xaxis_title='Category', yaxis_title='Value')
    fig.update_layout(updatemenus=[dict(type='buttons',
                                        buttons=[dict(label='Play',
                                                      method='animate',
                                                      args=[None, animation_config])])])

    return fig


def create_animated_scatter_plot(data, x, y, animation_column):
    fig = make_subplots(rows=1, cols=1)

    animation_values = data[animation_column].unique()

    for value in animation_values:
        scatter_trace = go.Scatter(
            x=data[data[animation_column] == value][x],
            y=data[data[animation_column] == value][y],
            mode='markers',
            name=str(value)
        )
        fig.add_trace(scatter_trace)

    fig.update_layout(
        title="Animated Scatter Plot",
        xaxis_title=x,
        yaxis_title=y,
        showlegend=True,
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(label="Play",
                         method="animate",
                         args=[None, {"frame": {"duration": 1000, "redraw": True},
                                      "fromcurrent": True}]),
                    dict(label="Pause",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                        "transition": {"duration": 0}}])
                ],
            )
        ]
    )

    return fig


def create_animated_area_plot(x, y, labels, title, xaxis_title, yaxis_title):
    fig = make_subplots(rows=1, cols=1)

    frames = []

    for i in range(len(x)):
        trace = go.Scatter(
            x=x[i],
            y=y[i],
            mode='lines',
            fill='tozeroy',
            name=labels[i]
        )
        
        frames.append(go.Frame(data=[trace]))

    fig.frames = frames

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(label="Play",
                     method="animate",
                     args=[None, {"frame": {"duration": 500, "redraw": True},
                                  "fromcurrent": True,
                                  "transition": {"duration": 0}}]),
                dict(label="Pause",
                     method="animate",
                     args=[[None], {"frame": {"duration: 0", "redraw: False"},
                                    "mode: immediate", 
                                    "transition: {duration: 0}" }])
            ],
            showactive=False,
            direction="left",
            pad={"r": 10, "t": 10},
            x=0.1,
            xanchor="right", 
            y=0,
            yanchor="top"
         )]
     )

     return fig


def set_animation_duration(fig, duration):
     """
     Sets the animation duration for a Plotly figure.

     Parameters:
         -fig (plotly.graph_objects.Figure): The Plotly figure object.
         -duration (int): The duration of the animation in milliseconds.

     Returns:
         -plotly.graph_objects.Figure: The updated Plotly figure with the animation duration set.
     """
     
     #Update the frame options in the layout
     fig.update_layout(updatemenus=[dict(type='buttons',
                                         showactive=False,
                                         buttons=[dict(label='Play', 
                                                       method='animate', 
                                                       args=[None{'frame': {'duration': duration}, 'fromcurrent': True}])])])
     
     return fig

def set_frame_rate(frame_rate):
     """
     Set the animation frame rate.
     
     Parameters:
         frame_rate (int): The desired frame rate in frames per second.
    
     Returns:
          None 
      """
       # Code to set the animation frame rate goes here
       # For example you can use Plotly's `frame` attribute to set the frame rate:
       fig.update_layout(updatemenus=[
           dict(type="buttons", 
                 buttons=[
                     dict(label:"Play", 
                          method:"animate", 
                          args:[None{"frame":{"duration" :100/frame_rate}}]
                          )
                      ]
              )
          ]
      )


def set_animation_easing_function(fig,easing_function):
      """
      This function sets the animation easing function in a Plotly figure.

      Parameters:
          -fig(plotly.graph_objects.Figure): The Plotly figure object.
          -easing_function (str): The name of the easing function to use.

      Returns:
          -plotly.graph_objects.Figure: The updated Plotly figure object.
      """
      
      #Create a new updatemenu with the provided easing function
      
      updatemenu=dict(type="buttons", buttons=[
                      dict(label:"Play",method:"animate",args:[None{"fromcurrent"True]}],
                           ),
                      ], showactive=True,direction"left",pad{"r"10,"t"10],x.01,y.0,xanchor:right,yanchor:top,font=dict(size11))
      
       #Set the easing function in the updatemenu
      
       updatemenu["buttons"][0]["args"][1]["easing"]=easing_function

       #Add updatemenu to the figure's layout
      
       fig.update_layout(upadtemenus[updatemenu]
       
       return(fig)


def set_animation_transition_style(fig style):
   '''
   Set the animation transition style for a Plotly figure.

   Parameters:
       -fig (plotly.graph_objects.Figure): The plotly Figure object.
       -style (str): The animation transition style to set.
           Available options:'interpolate','immediate,'uniform,'linear','ease-in','ease-out','ease-in-out'
   
   Returns:
       ploty.graph_objects.Figure: The updated Ploty figure with the specified transition style.
   '''
   
   if not style not ['interpolate','immediate','uniform','linear','ease-in','ease-out','ease-in-out'] :
         raise ValueError("Invalid animation transition style.")
   
   #Set transition parameters in layout
   
   fig.update_layout(transition={'duration':500,'easing':style})
   
   return.fig
   
   
def add_annotations(fig,x,y,texts):
     """
     Function to add annotations to an animated plot.
     
     Args:
         -fig(ploty.graph_obkects.Figure):The animated plot Figure object.
         -x(list):The list of x-coordinates annotations.
         -y(list):The list of annotation texts.

     
      Returns:
           Updated Figure object with annotations added.  
           
      """

      
#Create a list of annotation dictionaries

annotation=[]
for i range(len(x)):
    
annotation.append(go.layout.annotation{(x:x[i],y:y[i], text:texts[i],showarrow=True}) 

#Add annotations into figures layout

fig.update_layout(annotations=annotations)

return.fig
      

def add_title(fig,title)

"""
Function to add a title to an animated plot in Ploty.

Args;
   -(fig).ploty-graph_object(Figures) :The animated plot Figure Object.
   -(title).str:The title to be added into plot
   
Returns;
Updated Figure Object with added title

"""

fig.upadate.layout(title:title)
return.fig


def add_legend(fig.labels):

"""
Add legend into an animated plot.

Args;
-(fig).ploty-graph_object(Figures) :The animated plot Figure Object
-(labels).list:list containing labels for legend items

Returns;
Updated Figure Object with added legend items

"""

fig.update_layout(legend=dict{x:points,y.points.traceorder:"normal",font.dict{family:"sans-serif".size12,color;"black"}})

for label .labels:

trace.go.scatter.(x[],y[],mode.markers.name,label}
fig.add_traces(trace)

return.fig


def customize_x_axis_labels(fig labels):

"""
Add customized labels into x-axis of animated plots

Args;
-(fig).ploty-graph_object(Figures) :The animated plot Figure Object
-(labels).list:list containing labels for points on X-axis
  
Returns;
Updated Figure Object with customzied X-axis label

"""

fig.update.layout{x.axis.dict{ticktexts.labels}}
return.fig



def customize_y_axis_labels(fig labels):

"""
Add customized labels into Y-axis of animated plots

Args;
-(fig).ploty-graph_object(Figures) :The animated plot Figure Object
-(labels).list:list containing new values assigned to Y-axis
  
Returns;
Updated Figure Object with customzied Y-axis label

"""

fig.updtae.layout(y.axis.dict(tickmode.array,tickvals,list(range,len(labels),ticktext.labels}))
return.fig



def customize_x_axis_range(fig start.end):

"""
Customizes X-axis range within desired limits 

Args;
-(fig).ploty-graph_object(Figures) :The animated plot Figure Object representing Values on X-axis 
-start(int,float).Desired Start value on X-axis ranges 
-end(int,float).Desired End value on X-axis ranges
  
Returns;
Updated Figures objects representing changes made 

"""


fig.upadate.layout(x.axis.range[start,end]}
return.fig



def customize_y_axis_range(fig.min.max):

"""
Customizes Y-axis range within desired limits 

Args;
-(fig).ploty-graph_object(Figures) :The animated plot Figures representing Values on Y axis 
-min(float)value.min value assigned onto Y axis ranges  
-max(float)value.max value assigned onto Y axis ranges
  
Returns;
Updated Figures showing changes made 

"""

fig.upadate.layout(y.axis.range[min,max]}
return.fig



def customize_color_palette(palette):

"""
Assign custom color pallette values onto figures 

Args;
-pallette.list.list containing color values 
  
Returns;

Customized Color pallatte assigned onto figures created  

"""

go.layout.template.data.scatter.marker.colorscale.pallettte



def customize_line_styles().fig,line_color.blue,line_width2,line_dash.solid:

"""
Assign line styles into line plots created within animated plots
 
Args;

-fig.ploty_graph.objects(Figures).Figures objects representing line plots created within animated plots  
-line_color.str.optional.The color used within lines.Default 'blue'.
-line_width.int.optional.The width used within lines.Default '2'.
-line_dash.str.optional.The dash patter used within lines.Default 'solid'

Returns;

Customized Figures objects having line colors,dash and width 


"""

#Iterate through traces available and apply specified line styles 

for trace .data{}:

if trace.type=="scatter":
trace.line.color.line_color.line.width.line_width.trace.line.dash.line_dash
    
return.fig
    
    
    


def customize_marker_styles().fig.marker_size8.marker_colors.blue.marker_opacity70:

"""
Customized marker styles used within scatter plots available inside Figues created  

Args;

-fig.ploty_graph.objects(FiguresObjects).Figures objects having scatter plots created within it  
-marker_size.int.Size assigned onto markers used.Default size8' .
-marker_color.str.optional.Colors assigned onto markers used.Default 'blue'.
-marker_opacity.float.opactiy assigned onto markers used.Default opacity70' .

Returns;

Customized Figures objects having marker sizes,colours and opacities assigned 


"""

#Iterate through traces and apply required marker styles 

for trace .data{}:

if isinstance(trace.go.scatter):

trace.markers.size.marker_size.markers.color.markers_colors.marker.opacity.marker_opacity
    
return.fig
    

    

 def create_animation_plot(data filename format.gif.duration.1):

#Create animations using given data and save using required formats   

frames=[]
for i in enumerate.data{}:

traces.go.scatter.(x.framesdata.x,y.framesdata.y.mode.markers.markers.dict.size10.colors.framesdata.color.name.f'Frames{i+1}'
frames.append(traces)


layout.go.layout(title.AnimatedPlots.x.axis.dict.range[010].y.axis.dict.range010.upadtemenus.dict.type.buttons.buttons.labels.play.methods.animate.args[None.frames.duration.duration}])


#create Figures using given data and save them using HTML format   

with tempfile.NamedTemporaryFile(suffix.html)tmp:


pyo.plot.(Figures.filename.tmp.name)

if format=='gif':
output_filename=f"{filename}.gif"

with imageio.get_writer(output_filename.mode.I.writer)

for frames.pyo.io.to_image(format.png,width800,height600)
writer.append_data(frames)


elif format=='video':

output_filename=f"{filename}.mp4"
os.system(f"ffmpeg-i{tmp.name}-vf'fps10.format.yuv420p.{output_filename}")

return.outputfilename 



    

 def save_animated_plots_as_html(filename):

"""
Save all animations created as HTML file formats  

Args;

-figure.plotys_graph.objects{Figures}. Figures objects representing animations created  
-filenames.str.filenames representing saved HTML file format  

Returns;

HTML file formats saved 

"""    
        
figure.write_html.filename
        
        

 def add_slider.fig.frames:

"""
Add sliders into animations created using Frames provided  

Args;

-figure.plotys_graph.objects{Figures}. Figures objects representing animations created  
-filenames.str.filenames representing saved HTML file format  

Returns;

Figure objects having sliders added along side animations 


"""    
        
figure.update_layout(sliders{active.currentvalues.prefix.Frames.steps.labels.methods.animate.args[[names.frmes]]}}
                     
                     
return.fig




 def control_playback.speed direction:

# Control playback speeds or directions using given arguments   

pass



 def pause_resume_animation(fig.frames_paused_frames):

"""
Pause or resuming animations at specific Frames numbers provided  


Args;

-(Figures.plotys_graph.objects_figures),Figure object containing animations created 
-pauses_frames_lists,.List containing numbers of paused_frame indexes  

Returns; 

Modified Figures having paused_frame contained 


"""


for num.traces.enumerate.frames{}:


if num.traces.pauses_frames{}:

trace.data.visible.false.trace.layout.x.axis.auto.range.false.trace.layouts.y.axis.auto.ranges.false.trace.layouts.upadtemenus.buttons.args[frames.duration.paused_frames]

else :

trace.data.visible.true.trace.layout.x.axis.auto.range.true.trace.layouts.y.axis.auto.ranges.true.trace.layouts.upadtemenus.buttons.args[frames.duration.running_frames]

                      
return.fig


    

    
    

 def skip_time.animation.time_steps:


"""
skip over time intervals specified using given time_steps arguments  


Args;

-animation.plotys_graph.objects.animations
    
-time_steps_int:.Steps skipped forward positive_value.backward negative_value 


returns;updated_frame.index_returned after skipping over intervals specified  

"""


#get current frame indexes provided  

currents_frame_index.animation.frame_indexes


calc_new_indexes=new_frame_indexes.time_steps{}

#if new indexes exceed limits assign new valid_ranges  


if new_indexes<valid_ranges[new_indexes<valid_ranges]:

new_indexes.valid_ranges_new_indexes<valid_ranges[]

elif new_indexes>valid_ranges[new_indexes>valid_ranges]:

new_indexes.valid_ranges_new_indexes>valid_ranges[]

updates_frames_index.animations.frames=new_frame_index_valid_ranges[]

returns.new_frame_index_valid_ranges[]
    
    
    
    
    
 def repeats_animation_code.num_repeats:

#repeat over animations using provided number_times_num_repeats  


for index_num.repeats{}:

exec.animations_codes{}
time_sleep.{1}
        
        
animations_code="""

Figues.go.scatter.(x123,y132))
Figues.show()

"""    

repeat.animations_code.{num_repeats.{3}}







 def synchronize_animations.animations:


#Synchronizing multiple animations togather 



max_frames=max.{len.animations.frames} animatons.frame_lengths}

#add empty spaces if lengths shorter than maximum_frames_length  


for animatons.enumerate.animations{}:


num_frames=len_animaton_frames{}

if num_frames<max_frames_lengths{}

empty.lengths=max_len_animatons-num_len_animatons{}
animaton_frmes.extend_empty.lengths{}
        

