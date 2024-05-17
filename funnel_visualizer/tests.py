
import pytest
from marketing_funnel import MarketingFunnel
from funnel import Funnel

@pytest.fixture
def marketing_funnel():
    stages = ['Stage 1', 'Stage 2', 'Stage 3']
    return MarketingFunnel(stages)

@pytest.fixture
def funnel():
    return Funnel()

def test_update_drop_off(marketing_funnel):
    marketing_funnel.update_drop_off(0)
    assert marketing_funnel.drop_offs[0] == 1

def test_update_conversion(marketing_funnel):
    marketing_funnel.update_conversion(1)
    assert marketing_funnel.conversions[1] == 1

def test_set_metric(marketing_funnel):
    marketing_funnel.set_metric('metric_name', 'value')
    assert marketing_funnel.metrics['metric_name'] == 'value'

def test_add_stage(funnel):
    # Add a stage to the funnel
    funnel.add_stage("Stage 1")
    assert funnel.stages == ["Stage 1"]

    # Add another stage to the funnel
    funnel.add_stage("Stage 2")
    assert funnel.stages == ["Stage 1", "Stage 2"]

@pytest.fixture
def mock_funnel():
    return {
        'prospects': 100,
        'leads': 50,
        'opportunities': 20,
        'customers': 10
    }

def test_remove_stage_exists(mock_funnel):
    stage = 'leads'
    expected_funnel = {
        'prospects': 100,
        'opportunities': 20,
        'customers': 10
    }
    assert remove_stage(mock_funnel, stage) == expected_funnel

def test_remove_stage_not_exists(mock_funnel):
    stage = 'invalid_stage'
    assert remove_stage(mock_funnel, stage) == mock_funnel

def test_remove_stage_empty_funnel():
    empty_funnel = {}
    stage = 'leads'
    assert remove_stage(empty_funnel, stage) == {}

def test_remove_stage_empty_string(mock_funnel):
    stage = ''
    assert remove_stage(mock_funnel, stage) == mock_funnel

def test_remove_stage_none(mock_funnel):
    stage = None
    assert remove_stage(mock_funnel, stage) == mock_funnel

def test_remove_stage_multiple_stages(mock_funnel):
    mock_funnel['leads'] = 30
    expected_result = {
        'prospects': 100,
        'opportunities': 20,
        'customers': 10,
        'leads': 30
    }
    
    assert remove_stage(mock_funnel, 'leads') == expected_result

from your_module import get_funnel_stages

@pytest.fixture
def expected_stages():
    return [
        "Awareness",
        "Interest",
        "Consideration",
        "Conversion",
        "Retention",
        "Advocacy"
    ]

def test_get_funnel_stages(expected_stages):
    stages = get_funnel_stages()
    assert stages == expected_stages

from your_module import set_conversion_rate

funnel_conversion_rates = {}

def test_set_conversion_rate():
    
     # Test case: Setting conversion rate for a new stage and existing stages
    
     stages_and_rates = [
         ("stage1", 0.2),
         ("stage2", 0.5),
         ("stage3", 0.8)
     ]
     
     for stage, rate in stages_and_rates:
         set_conversion_rate(stage, rate)
         assert funnel_conversion_rates[stage] == rate
    
def test_set_conversion_rate_invalid_input():
    
     # Test case: Setting conversion rate with invalid inputs
    
     invalid_inputs = [
         ("", 0.5),   # Invalid stage name: empty string
         ("stage4", -0.1)   # Invalid conversion rate: negative value
     ]
     
     for stage, rate in invalid_inputs:
         with pytest.raises((KeyError, ValueError)):
             set_conversion_rate(stage, rate)

from your_module import calculate_conversion_rate

def test_calculate_conversion_rate():
    
     # Test cases for calculate_conversion_rate
    
     conversion_cases = [
         ("stage1", (0,0), (True), (ZeroDivisionError)),
         ("stage2", (100,0), (False), (None)),
         ("stage3", (50,25), (False), (50.0)),
         ("stage4", (1000000,500000), (False), (50.0)),
         ("stage5", (-100,-10), (True), (ZeroDivisionError)),
     ]
     
     for name,(users,c_users),(is_err,is_err_type) in conversion_cases:
         
         if is_err:
             with pytest.raises(is_err_type):
                 calculate_conversion_rate(name, users,c_users)
                 
         else:
             result=calculate_conversion_rate(name,users,c_users)
             exptected_result=c_users/users*100 if users!=0 else result==expected_result<|vq_10530|>             assert result==expected_result


@pytest.fixture
def funnel_with_initial_stages():
    
     # Returns an instance of Funnel class with pre-defined values
    
     initial_data={
            {'stage1',10},
            {'stage2',20}
          }
          
      return Funnel(initial_data)

@pytest.mark.parametrize("f_values",[({'initial'},{'updated'})])
      
 def test_set_users(f_values,funnels_with_initial_stages):

 @pytest.mark.parametrize(
 
 "func_input,expected_output"
 [
   ('initial',{'initial'}==set_users({'initial'})),

   ('updated',{'updated'}==set_users({'updated'}))
]
 )
 
 
 def tests(func_input,funnels_with_initial_stgaes):

       func_input=f_values['initial']
       
       exptected_output=f_values['initital']
       
       set_users(funnels_with_initial-stages,fucn_input)
       
       current_values=funnels_with_initial-stgaes.get_current()
       
       asser current_values==exptected_output
   
   
 
 def tests(func_input,funnels_with_intial_stges)
 
 
      func_input=f_valies['updated']
       
       exptected_output=f_values['initital']
       
       set_users(funnels_with_initial-stages,fucn_input)
       
       current_values=funnels_with_initial-stgaes.get_current()
       
       asser current_values==exptected_output


# Run tests via pytest framework using `pytest` command within terminal/IDE.
pytest.main()
