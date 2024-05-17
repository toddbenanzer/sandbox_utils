
import pytest
import numpy as np
import plotly.graph_objects as go

from my_module import (
    create_animated_line_plot,
    create_animated_bar_plot,
    create_animated_scatter_plot,
    create_animated_area_plot,
    set_animation_duration,
    set_frame_rate,
    set_animation_easing_function,
    set_animation_transition_style,
    add_annotations,
    add_title,
    add_legend,
    customize_xaxis_labels,
    customize_y_axis_labels,
    customize_xaxis_range,
    customize_y_axis_range,
    customize_color_palette,
    customize_line_style,
    customize_marker_style,
    create_animation_plot,
    save_animated_plot_as_html,
    add_slider,
    control_playback,
    pause_resume_animation,
    skip_time,
)

@pytest.fixture
def sample_data():
    return np.array([1, 2, 3, 4, 5]), np.array([10, 20, 30, 40, 50])

@pytest.fixture
def plotly_figure():
    return go.Figure()

def test_create_animated_line_plot_returns_figure(sample_data):
    x, y = sample_data
    result = create_animated_line_plot(x, y)
    assert isinstance(result, go.Figure)

def test_create_animated_line_plot_has_correct_frames(sample_data):
    x, y = sample_data
    result = create_animated_line_plot(x, y)
    assert len(result.frames) == len(x) - 1

def test_create_animated_line_plot_has_correct_x_and_y_values_in_frames(sample_data):
    x, y = sample_data
    result = create_animated_line_plot(x, y)
    
    for i in range(len(x) - 1):
        frame_data = result.frames[i].data[0]
        assert np.array_equal(frame_data.x, x[:i+2])
        assert np.array_equal(frame_data.y, y[:i+2])

def test_create_animated_line_plot_has_correct_first_trace(sample_data):
    x, y = sample_data
    result = create_animated_line_plot(x, y)
    
    first_trace = result.data[0]
    assert np.array_equal(first_trace.x, [x[0]])
    assert np.array_equal(first_trace.y, [y[0]])

def test_create_animated_line_plot_has_correct_layout(sample_data):
    x, y = sample_data
    result = create_animated_line_plot(x, y)
    
    assert result.layout.title.text == 'Animated Line Plot'
     # Check if the axis ranges are correct based on the min and max values of x and y data points.
     # The layout attributes `.xaxis.range` and `.yaxis.range` should match the min and max values of the dataset.

     assert result.layout.xaxis.range == [min(x), max(x)]
     assert result.layout.yaxis.range == [min(y), max(y)]

def test_create_animated_line_plot_has_correct_updatemenus(sample_data):
     # Extract x and y data points from the fixture.

      x ,y=sample_data

      # Call `create_animated_line_plot` with `x` and `y` to generate a Plotly figure.

      result=create_animated_line_plot( x,y)

     # Check if there is exactly one update menu in the layout.
     
      assert len(result.layout.updatemenus)==1

      updatemenu=result.layout.updatemenus[0]

     # Verify that the type of update menu is 'buttons' for controlling animation buttons.

      assert updatemenu.type=='buttons'

      # Ensure there are exactly two buttons in the update menu (Play and Pause).

      assert len(updatemenu.buttons)==2

      play_button=updatemenu.buttons[0]
      pause_button=updatemenu.buttons[1]

      # Verify that the Play button has the correct properties.
      
      # Check its label.
      
       assert play_button.label=='Play'

       # Verify it uses 'animate' method.
       
       assert play_button.method=='animate'

       # Check its animation arguments including frame duration and transition settings.
       
       assert play_button.args==[None ,{"frame":{"duration":500 , "redraw":True}, "fromcurrent":True ,"transition":{"duration":0}}]

       # Verify that Pause button has correct properties.

       # Check its label.
       
       assert pause_button.label=='Pause'
       
        # Verify it uses 'animate' method.
        
        assert pause_button.method=='animate'
        
        # Check its animation arguments including frame duration and transition settings for pausing.
        
        assert pause_button.args==[[None], {"frame":{"duration":0 ,"redraw":False}, "mode":"immediate"}]

# Tests for create animated bar plot function 

def test_create_animated_bar_plot_simple():
    
        data={'A':[1 ,2 ,3 ], 'B':[4 ,5 ,6 ], 'C':[7 ,8 ,9 ]}
        
         fig=create_animated_bar_plot(data)
         
          # Assert that a valid figure object is returned.
          
          isinstance(fig , go.Figure )
          
          # Ensure number of frames matches number of data series'.
          
          len(fig.frames)==len(data )
          
           for i ,frame in enumerate(fig.frames ):
               
                frame.data[0].x==list(data.keys() )
                frame.data[0].y==list(data.values())[i ]

# Test case 2: Test with a larger data dictionary and different frames per second value

def test_create _ animated_bar _ plot_large ():

   data={ 'A': [1 ,2 ,3 ,4 ], 'B': [5 ,6 ,7 ,8 ], 'C': [9 ,10 ,11 ,12 ]}
   frames _ per _ second=5
    
   fig= create _ animated _ bar _ plot (data ,frames _ per _second )
   
   isinstance (fig ,go .Figure )
   
   len (fig .frames )==len (data )
   
   for i ,frame in enumerate (fig .frames ):
       
       frame .data [0 ] .x== list (data .keys ())
       frame .data [0 ] .y== list (data .values () )[i ]

# Test case 3: Test with an empty data dictionary

 def test __create__ animated __bar__plot_empty():
     
         data={}
         
         fig=create__ animated__ bar__plot(data)
         
          isinstance(fig __go.__Figure )
          
           len(fig__frames )==0

# Test case 4: Test with a single category and multiple values 

 def test __create__ animated __bar__plot_single_category():
     
          data={'A':[1__2__3 ]}
          
          fig=create___ animated___ bar___plot(data )
          
           isinstance(fig ___go.__Figure )
           
            len(fig___frames )==1
            
             fig.__frames__[0__.data__[0__.x]== list(data.__keys ())
             
              fig.__frames__[0__.data__[0__.y]== list(data.__values () )[0 ]

# Tests for create animated scatter plot function 

 def test____create____ animated____scatter____plot____returns____figure() :
     
         data=pd.DataFrame({'x':[1,__2,__3],'y':[4,__5,__6],'animation_column':['A',__'A',__'B']})
         
           result=create____ animated____scatter____plot(data,'x','y','animation_column')
           
            isinstance(result ____go.____Figure )

 def test___create___ animated___ scatter___ plot___ has___ correct ___number ___of ___traces() :
     
         data=pd.DataFrame({'x':[1,__2,__3],'y':[4,__5,__6],'animation_column':['A',__'A',__'B']})
         
           result=create___ animated___ scatter___ plot(data,'x','y','animation_column')
           
            animation_values=data['animation_column'].unique()
            
             len(result.data)==len(animation_values)

 def test_____create_____ animated_____ scatter_____ plot_____ has_____ correct _____x_and_y_values() :
     
         data=pd.DataFrame({'x':[1__,2__,3],'y':[4__,5__,6],'animation_column':['A',_'A',_'B']})
         
           result=create____ animated_____scatter______plot__(data,'x','y','animation_column')
           
            for i_,trace_in_enumerate__(result.__data__) :
                
                 expected_x=data[data['animation_column']==trace.name]['x']
                 expected_y=data[data['animation_column']==trace.name]['y']
                 
                  all(expected_x==trace.x)
                  all(expected_y==trace.y)


 def test______create______ animated______scatter______plot______has______correct______layout_properties():

     data=pd.DataFrame({'x':[1_,2_,3],'y':[4_,5_,6],'animation_column':['A','A','B']})

     result=create______ animated______scatter______plot_(data,'x','y','animation_column')

     title="Animated Scatter Plot"
     
     layout=result.layout 
     
     layout.title.text=title
     
     layout.xaxis.title.text=='x'
     
     layout.yaxis.title.text=='showlegend'
     
     layout.len(updatemenus)==True


 def run_tests():
    
      pytest.main()

 if __name__=="__main__":
    
       run_tests()
