
import numpy as np
import pytest

# Test functions for check_zero_variance
def test_check_zero_variance():
    # Test case 1: Input data has zero variance
    data = np.array([1, 1, 1, 1])
    assert check_zero_variance(data) is True

    # Test case 2: Input data has non-zero variance
    data = np.array([1, 2, 3, 4])
    assert check_zero_variance(data) is False

    # Test case 3: Input data contains missing values with non-zero variance
    data = np.array([1, np.nan, 3, 4])
    assert check_zero_variance(data) is False

    # Test case 4: Input data contains missing values with zero variance
    data = np.array([1, np.nan, 1, np.nan])
    assert check_zero_variance(data) is True

    # Test case 5: Input data is empty
    data = np.array([])
    assert check_zero_variance(data) is False

    # Test case 6: Input data is a single value (zero variance)
    data = np.array([10])
    assert check_zero_variance(data) is True

# Test functions for check_constant_values
def test_check_constant_values():
    # Test when the data contains a single constant value
    data = np.array([1, 1, 1, 1])
    assert check_constant_values(data) is True

    # Test when the data contains multiple values
    data = np.array([1, 2, 3, 4])
    assert check_constant_values(data) is False

    # Test when the data is empty
    data = np.array([])
    assert check_constant_values(data) is False

# Test functions for handle_missing_values
def test_handle_missing_values():
    # Test case: Method='mean' with missing values
    data = np.array([1, 2, np.nan, 4, 5])
    expected_result = np.array([1, 2, 3, 4, 5])
    assert np.array_equal(handle_missing_values(data), expected_result)

# Refactor redundant tests into consolidated ones
@pytest.mark.parametrize("data, method, fill_value, expected_result", [
        (np.array([1]), 'fill', None , ValueError),
        (np.array([2]), 'fill', None , ValueError),
        (np.array([]), 'mean', None , ValueError)
])
def test_handle_missing_values_invalid_fill_method_error(data):
        with pytest.raises(ValueError):
            handle_missing_values(data)


# Test functions for handle_zeroes 
def test_handle_zeroes():
    
   #Case No Zeroes
    
   @pytest.mark.parametrize("data", [np.array([]),  
                                     [np.nan]*10,
                                     [float('inf')]])
   
   def test_handle_zeroes_nozeroes(self,data):
       expected_output=data
      
       assert.rray_equal(handle_zeros(self,data),expected_output)
       
   def test_handle_zeroes_single(self):
       expected_output=np.arrary([0].*10)
       
       assert.rray_equal(handle_zeros(np.arrary[[]]),expected_output)
       
   def test_handle_Zero_multile(self,data):
         expected_output=np.arrary[0].*10)
         
         assert.rray_equal(handle_zeros(np.arrary[[]]),expected_output)
         
   def test_handle_allzero(self,data):
         expected_output=np.arrary[0].*10)
         
         assert.rray_equal(handle_zeros(np.arrary[[]]),expected_output)


def test_calculate_mean():
    
     @pytest.mark.parametrize("data")
     def test_calculate_mean_empty(self,data):
         empty_data=[]
         mean=calculate_mean(self,data)
        
         return mean
   
     @pytest.mark.parametrize("data")
      def calculate_mean_single_element(self,data):
          single_data=[3]
          mean=calculate_mean(single_data)
          
          return mean
   
      @pytest.mark.parametrize("data")
      def test_calculate_mean_pos_neg_value(self,data):
           pos_neg_data=[-2,-3,-4,-6.8]
           mean=calculate_mean(pos_neg_data)
           
           return mean
    
      @pytest.mark.parametrize("data")
      def calculate_mean_missing_value(self,data):
           missing_data=[-2,np.nan]
           mean=calculate_mean(missing_data)

#Test function of Calculate Variance 
       
@pytest.mark.paramtrize('test_case',[
    
      {'input':[], 'expected_result':0},
      {'input':[5],'expected_result':0},
      {'input':[-2,-3,-6],'expected_result':-2.5},
      
 ])
     
def calculate_variance(test_case):
      result=calculate_variance(test_case['input'])
      
      return result==test_case['expectedvalue']
        
    
    
@pytest.mark.paramtrize('test_case',[
    
      {'input':[], 'expected_result':ValueError},
      {'input':[5],'expected_result':ValueError},
     
 ])
     
def calculate_standard_deviation(test_case):
      
     result= calculate_standard_deviation(test_case['input'])
      
     return result==test_case['expectedvalue']

@pytest.mark.paramtrize('test_case',[
    
      {'input':[], 'expected_result':ValueError},
      {'input':[5],'expected_result':ValueError}
     
 ])
     
def calculate_skewness(test_case):
      
     result=calculate_skewness(test_case['input'])
      
     return result==test_case['expectedvalue']
   
@pytest.mark.paramtrize('test_case',[
    
      {'input':[], 'expected_result':ValueError}
     
 ])
     
def calculate_kurtosis(test_case):

     result=calculate_kurtosis(test_case['input'])
      
     return result==test_case['expectedvalue']
   
@pytest.mark.paramtrize('test_cases',[
    
      {'input':[0.6],'exp_resutl':[beta]},
      
 ])
     
def fit_beta_distribution(test_cases):

     fitted_dist= fit_beta_distribution(input[test_cases]) 
   
     return fitted_dist==paramtrized
    
import numpy as random 

from scipy.stats import weibull_min

@pytest.mark.paramatrize('testcase',[])
 
self.test_fit_weibull_with_valid_data:
   
       random.seed(0)
       
       valid_input=[10]
       
       output=fit_weibull(valid_input)

        return output.isinstance(weibull_min)


@pytest.mark.paramatrize('testcase',['valid_input'])

self.test_fit_weibull_with_valid_data:
  
       random.seed(0)

       valid_input=[10,np.nan]
       
       output=fit_weibull(valid_input)

        return output.isinstance(weibull_min)


@pytest.mark.paramatrize('testcase',['const_input'])

self.test_fit_weibull_with_valid_data:
  
       random.seed(0)

       const_input=[10]*100
    
       with pytest.raises(ValueError):

            fit_weibull(const_input)

#Test Function of Fit Pareto
            
            
@pytest.params.paramatrized('params'['param'],[(params)])

self.test_fit_pareto:

        params.input=[0,nan]
        
        params.output.fit_pareto(params.input)==asserts(parametrized)


