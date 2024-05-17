
import re
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Theme and Style Settings
def set_theme_style():
    sns.set_theme(style="whitegrid", font="Georgia")
    sns.set_context("paper", font_scale=1.4)
    plt.rcParams.update({'font.size': 10})
    plt.rcParams['legend.fontsize'] = 10

def set_font_face_georgia():
    plt.rcParams['font.family'] = 'Georgia'

def set_title_font_size(size=14):
    plt.rcParams['axes.titlesize'] = size

def set_label_font_size(size=10):
    plt.rcParams['axes.labelsize'] = size

def set_legend_position_bottom(plot):
    plot.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3)

def set_reference_line_color(color='lightgray'):
    ax = plt.gca()
    ax.axhline(y=0, color=color, linestyle='--')
    ax.axvline(x=0, color=color, linestyle='--')

def disable_grid_lines():
    sns.set(style="ticks", rc={"axes.grid": False})

# Chart Creation Functions
def create_stacked_bar_chart(data, colors):
    sns.set(style="whitegrid")
    sns.set_palette(colors)
    
    # Create the stacked bar chart using seaborn
    ax = sns.barplot(data=data)

    # Move the legend to the bottom
    set_legend_position_bottom(ax)

    # Add light gray reference lines for clarity
    set_reference_line_color()

    # Remove unnecessary grid lines around bounding box
    disable_grid_lines()

    # Display the chart
    plt.show()

def create_custom_line_chart(data, x, y, colors):
    sns.set(style="ticks", font='Georgia')
    
    ax = sns.lineplot(x=x, y=y, data=data, palette=colors)
    
    # Set the font sizes for title and labels
    plt.title('Custom Line Chart', fontsize=14)
    
    # Show the legend at the bottom
    set_legend_position_bottom(ax)
    
    # Show the plot
    plt.show()

# Helper Functions for Validation and Calculation
def validate_color_list(color_list):
    validated_colors = []
    color_regex = r'^#(?:[0-9a-fA-F]{3}){1,2}$'
    
    for color in color_list:
        if re.match(color_regex, color):
            validated_colors.append(color)
    
    return validated_colors

def calculate_num_bars(data):
    return len(data[0])

def calculate_number_of_lines(data):
     return len(set(data))

# Label and Legend Customization Functions
def create_chart_title(title):
     plt.title(title, fontsize=14)

def create_x_axis_labels(data, tick_labels):
     formatted_labels = [label.upper() for label in tick_labels]
     return formatted_labels

def create_y_axis_labels(labels):
     plt.yticks(fontsize=10)
     plt.xticks(rotation=90)
     plt.ylabel('Y-Axis', fontsize=10)
     plt.show()

def create_legend(labels):
     plt.legend(labels, loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=len(labels))

# Tick Label Formatting Functions
def format_xaxis_tick_labels(ax):
     ax.tick_params(axis='x', labelsize=10)
     ax.xaxis.label.set_size(10)

def format_y_axis_tick_labels(ax):
     ax.tick_params(axis='y', labelsize=10)
     y_ticks = ax.get_yticks()
     ax.set_yticklabels([f'{tick:,.0f}' for tick in y_ticks])
     
# Reference Line Functions     
def add_reference_lines(chart, reference_positions):
     for position in reference_positions:
         chart.axvline(x=position, color='lightgray', linestyle='--')

def add_reference_lines_y(y_values, color='lightgray', linestyle='dashed', linewidth=1):
     ax = plt.gca()
     for value in y_values:
         ax.axhline(value, color=color, linestyle=linestyle, linewidth=linewidth)

# Dataset Scaling and Series Detection Functions     
def calculate_max_value(dataset):
     return max(dataset)

def calculate_min_value(data):
      return min(data)

 def calculate_range(dataset):
      return max(dataset) - min(dataset)

 def scale_dataset(dataset): 
      max_value = max(dataset) 
      min_value = min(dataset) 
      scaled_dataset = [(value - min_value) / (max_value - min_value) for value in dataset] 
      return scaled_dataset

 def has_multiple_series(data): 
      num_unique_values = len(set(data.index.unique()).union(set(data.columns.unique())))
      return num_unique_values > 1 

 def has_multiple_lines(dataset,x_axis_column): 
       unique_values = dataset[x_axis_column].nunique() 
       if unique_values > 1: 
           return True 
       else: 
           return False 

# Additional Customization Functions     
  def add_data_labels(ax): 
        for container in ax.containers: 
            for rect in container.patches: 
                height = rect.get_height() 
                ax.text(rect.get_x() + rect.get_width() / 2,height,f'{height:.0f}',ha='center',va='bottom',fontsize=10) 

 def customize_bar_width(ax,width): 
        for container in ax.containers: 
            for patch in container.patches: 
                current_width = patch.get_width() 
                diff=current_width-width 

                patch.set_width(width) 

                patch.set_x(patch.get_x()+diff*0.5) 

  def customize_line_width(line_width):  
         ax=plt.gca()  
         lines=ax.lines  
         for line in lines:  
              line.set_linewidth(line_width) 

  def customize_marker_size(line_chart ,marker_size):  
         for line in line_chart.lines:  
              line.set_markersize(marker_size) 

  def customize_marker_shape(marker_shape):  
         if marker_shape=='circle':  
              sns.set(style='ticks' ,rc={'lines.markersize':6 ,'scatter.marker':'o'})  
         elif marker_shape=='square':  
              sns.set(style='ticks' ,rc={'lines.markersize':6 ,'scatter.marker':'s'})  
         elif marker_shape=='diamond':  
              sns.set(style='ticks' ,rc={'lines.markersize':6 ,'scatter.marker':'d'})  
         elif marker_shape=='triangle_up':  
              sns.set(style='ticks' ,rc={'lines.markersize':6 ,'scatter.marker':'^'})  
         elif marker_shape=='triangle_down':  
              sns.set(style='ticks' ,rc={'lines.markersize':6 ,'scatter.marker':'v'}) 

   def customize_line_style(x,y ,linestyle='-' ,linewidth=1.5):  
          sns.set(style='white' ,font='Georgia' ,rc={'axes.titlesize':14 ,'axes.labelsize':10})  
          ax=sns.lineplot(x=x,y=y ,linestyle=linestyle ,linewidth=linewidth)   
          set_legend_position_bottom(ax )  

   def customize_legend_title_font_size(size):   
          plt.rcParams['legend.title_fontsize']=size   

   def set_legend_label_font_size(font_size):   
          sns.set_context(rc={"legend.fontsize":font_size})   

   def customize_chart_background_color(color):   
          sns.set(rc={"figure.facecolor":color})   

   def set_transparency(transparency ):   
          sns .set_context(rc={"patch.linewidth":0,"lines.linewidth":0 ,"patch.facecolor":(1 ,1 ,1 transparency )})

   # Example usage of some functions to create a custom plot with specified settings.
if __name__ == "__main__":
   data=pd.DataFrame({'x':[0.5 .1.5 .2.5],'y':[4 .5 .6],'hue':['A','B','C']})
   
   colors=['#FF0000','#00FF00','#0000FF']
   
   validate_color_list(colors )
   
   create_stacked_bar_chart (data[['x ','y']],colors )
