from configure_integration import configure_integration
from data_handler import DataHandler
from results_interpreter import ResultsInterpreter
from setup_logging import setup_logging
from statistical_tests import StatisticalTests
from visualizer import Visualizer
import logging
import numpy as np
import pandas as pd


# Initialize the class
stats_test = StatisticalTests()

# Example 1: Performing an independent t-test
group_a = np.array([5.5, 7.8, 8.3, 5.6, 6.2])
group_b = np.array([8.4, 7.4, 6.9, 7.1, 8.0])
t_test_result = stats_test.perform_t_test(group_a, group_b, paired=False)
print("T-Test Result:", t_test_result)

# Example 2: Performing a paired t-test
pre_launch = np.array([115, 120, 135, 150, 160])
post_launch = np.array([120, 125, 145, 155, 165])
paired_t_test_result = stats_test.perform_t_test(pre_launch, post_launch, paired=True)
print("Paired T-Test Result:", paired_t_test_result)

# Example 3: Performing ANOVA
group1 = np.array([1.2, 1.3, 1.4, 1.5])
group2 = np.array([1.1, 1.4, 1.5, 1.7])
group3 = np.array([1.3, 1.5, 1.6, 1.8])
anova_result = stats_test.perform_anova(group1, group2, group3)
print("ANOVA Result:", anova_result)

# Example 4: Performing a chi-square test
observed_counts = np.array([10, 20, 30])
expected_counts = np.array([15, 15, 30])
chi_square_result = stats_test.perform_chi_square(observed_counts, expected_counts)
print("Chi-Square Result:", chi_square_result)

# Example 5: Performing a regression analysis
data = pd.DataFrame({
    'Sales': [10, 20, 30, 40, 50],
    'Marketing_Spend': [100, 150, 230, 300, 350],
    'Price_Discount': [5, 7, 8, 10, 12]
})
regression_result = stats_test.perform_regression_analysis(
    data, 
    independent_vars=['Marketing_Spend', 'Price_Discount'], 
    dependent_var='Sales'
)
print("Regression Analysis Result:", regression_result)



# Initialize the DataHandler class
handler = DataHandler()

# Example 1: Load data from a CSV file
csv_file_path = 'path/to/your/data.csv'
data_frame = handler.load_data(csv_file_path)
print("Loaded Data:")
print(data_frame.head())

# Example 2: Cleanse data by handling missing values and duplicates
raw_data = pd.DataFrame({
    'A': [1, None, 2, 2, 4],
    'B': [5, 6, None, 8, 5],
    'C': ['cat', 'dog', 'cat', 'dog', 'dog']
})
cleansed_data = handler.cleanse_data(raw_data)
print("Cleansed Data:")
print(cleansed_data)

# Example 3: Preprocess data for further analysis
preprocessed_data = handler.preprocess_data(cleansed_data)
print("Preprocessed Data:")
print(preprocessed_data.head())



# Initialize the ResultsInterpreter
interpreter = ResultsInterpreter()

# Example 1: Interpret t-test results
t_test_result = {
    't_stat': 2.1,
    'p_value': 0.04,
    'degrees_of_freedom': 20
}
interpretation = interpreter.interpret_t_test(t_test_result)
print("T-test Interpretation:", interpretation)

# Example 2: Interpret ANOVA results
anova_result = {
    'f_stat': 4.5,
    'p_value': 0.03,
    'df_between': 3,
    'df_within': 36
}
anova_interpretation = interpreter.interpret_anova(anova_result)
print("ANOVA Interpretation:", anova_interpretation)

# Example 3: Calculate Cohen's d as effect size for a t-test
effect_size = interpreter.calculate_effect_size(t_test_result)
print("Effect Size (Cohen's d):", effect_size)

# Example 4: Compute confidence interval for a t-test result
confidence_interval = interpreter.compute_confidence_interval(t_test_result, confidence_level=0.95)
print("95% Confidence Interval:", confidence_interval)



# Initialize the Visualizer
visualizer = Visualizer()

# Example 1: Plot the distribution of a dataset
data = np.random.normal(loc=0, scale=1, size=1000)
visualizer.plot_distribution(data, test_type="Normal Distribution")

# Example 2: Create a summary plot of statistical results
results = {
    'categories': ['A', 'B', 'C'],
    'means': [1.5, 2.0, 1.8],
    'conf_int_lower': [0.1, 0.2, 0.15],
    'conf_int_upper': [0.15, 0.25, 0.2]
}
visualizer.create_result_summary_plot(results)

# Example 3: Generate a correlation matrix from a DataFrame
data_frame = pd.DataFrame({
    'var1': np.random.rand(100),
    'var2': np.random.rand(100),
    'var3': np.random.rand(100)
})
visualizer.generate_correlation_matrix(data_frame)



# Example 1: Set logging to INFO level
setup_logging("INFO")
logging.info("This is an info message.")
logging.debug("This debug message will not appear because the level is set to INFO.")

# Example 2: Set logging to DEBUG level
setup_logging("DEBUG")
logging.debug("This debug message will now appear because the level is set to DEBUG.")

# Example 3: Set logging to ERROR level
setup_logging("ERROR")
logging.error("This is an error message.")
logging.info("This info message will not appear because the level is set to ERROR.")



# Initialize logging
logging.basicConfig(level=logging.INFO)

# Example 1: Configure integration for common libraries
configure_integration(['os', 'sys'])

# Example 2: Attempt to configure integration for a nonexistent library
configure_integration(['nonexistent_lib'])

# Example 3: Configure integration and custom configuration for a specific library
configure_integration(['example_lib'])
