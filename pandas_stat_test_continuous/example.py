from data_analyzer import DataAnalyzer  # Adjust the import according to your project structure
from statistical_tests import StatisticalTests  # Adjust the import according to your project structure
import numpy as np
import pandas as pd


# Sample DataFrame
df = pd.DataFrame({
    'A': [10, 20, np.nan, 40, 50],
    'B': [15, np.inf, 35, 40, 45],
    'C': [5, 5, 5, 5, 5],
    'D': [np.nan, np.nan, np.nan, np.nan, np.nan],
    'E': [1, 2, 3, 4, 5]
})

# Initialize DataAnalyzer with specific columns
analyzer = DataAnalyzer(df, ['A', 'B', 'C', 'D', 'E'])

# Calculate descriptive statistics
stats = analyzer.calculate_descriptive_stats()
print("Descriptive Statistics:")
print(stats)

# Handle missing data by filling with mean
analyzer.detect_and_handle_missing_data(method='fill_mean')
print("\nDataFrame after handling missing data (filled with mean):")
print(analyzer.dataframe)

# Handle infinite data by replacing with NaN
analyzer.detect_and_handle_infinite_data(method='replace_nan')
print("\nDataFrame after handling infinite data (replaced with NaN):")
print(analyzer.dataframe)

# Exclude null and trivial columns
excluded_columns = analyzer.exclude_null_and_trivial_columns()
print("\nExcluded columns (null or trivial):", excluded_columns)
print("Remaining columns for analysis:", analyzer.columns)



# Sample DataFrame
df = pd.DataFrame({
    'Sample1': [10, 12, 9, 11, 10],
    'Sample2': [8, 9, 7, 8, 10],
    'Group': ['A', 'A', 'B', 'B', 'B'],
    'Category1': ['X', 'Y', 'X', 'Y', 'X'],
    'Category2': ['M', 'M', 'N', 'N', 'N']
})

# Initialize StatisticalTests with specific columns
tests = StatisticalTests(df, ['Sample1', 'Sample2'])

# Perform a one-sample t-test
one_sample_results = tests.perform_t_tests(test_type='one-sample', popmean=10)
print("One-sample t-test results:")
print(one_sample_results)

# Perform an independent t-test
independent_t_test_results = tests.perform_t_tests(test_type='independent', group_column='Group')
print("\nIndependent t-test results:")
print(independent_t_test_results)

# Perform a paired t-test
paired_t_test_results = tests.perform_t_tests(test_type='paired')
print("\nPaired t-test results:")
print(paired_t_test_results)

# Perform ANOVA
anova_results = tests.perform_anova(group_column='Group')
print("\nANOVA test results:")
print(anova_results)

# Perform Chi-Squared test
chi_squared_results = tests.perform_chi_squared_test(columns=['Category1', 'Category2'])
print("\nChi-squared test results:")
print(chi_squared_results)


# Example 1: Generating a descriptive statistics report
data_example_1 = {
    'Descriptive Statistics': {
        'Mean': 10.123456,
        'Median': 10.0,
        'Standard Deviation': 2.3456
    }
}
report_1 = generate_summary_report(data_example_1)
print(report_1)

# Example 2: Generating a report for t-test results
data_example_2 = {
    'T-Test Results': {
        't-statistic': 1.962,
        'p-value': 0.049
    }
}
report_2 = generate_summary_report(data_example_2)
print("\n", report_2)

# Example 3: Generating a report for ANOVA test outcomes
data_example_3 = {
    'ANOVA Test': {
        'F-statistic': 4.678,
        'p-value': 0.012
    }
}
report_3 = generate_summary_report(data_example_3)
print("\n", report_3)

# Example 4: Generating a report with mixed data
data_example_4 = {
    'Mixed Data': {
        'Count': 100,
        'Mean': 7.89,
        'Std Dev': 1.234
    },
    'Additional Notes': "This dataset includes outliers."
}
report_4 = generate_summary_report(data_example_4)
print("\n", report_4)

# Example 5: Generating a report with no data
data_example_5 = {}
report_5 = generate_summary_report(data_example_5)
print("\n", report_5)


# Example 1: Visualizing a bar plot
data_for_bar = {
    'Apples': 50,
    'Bananas': 35,
    'Cherries': 75
}
visualize_statistics(data_for_bar, plot_type='bar', title='Fruit Count', xlabel='Fruit', ylabel='Count')

# Example 2: Visualizing a scatter plot
data_for_scatter = {
    'Height': [150, 160, 170, 180, 190],
    'Weight': [50, 60, 70, 80, 90]
}
visualize_statistics(data_for_scatter, plot_type='scatter', title='Height vs. Weight', xlabel='Height (cm)', ylabel='Weight (kg)')

# Example 3: Visualizing a histogram
data_for_histogram = {
    'Days spent on project': [1, 2, 3, 2, 3, 1, 4, 5, 2, 3, 2]
}
visualize_statistics(data_for_histogram, plot_type='histogram', title='Project Duration', xlabel='Days', ylabel='Frequency')

# Example 4: Visualizing multiple histograms
data_multiple_histograms = {
    'Dataset 1': [1, 2, 2, 3, 3, 4],
    'Dataset 2': [2, 3, 3, 3, 4, 5]
}
visualize_statistics(data_multiple_histograms, plot_type='histogram', title='Comparison of Two Datasets', xlabel='Value', ylabel='Frequency')

# Example 5: Bar plot with different figure size
another_bar_data = {
    '2020 Sales': 120,
    '2021 Sales': 135,
    '2022 Sales': 150
}
visualize_statistics(another_bar_data, plot_type='bar', title='Annual Sales', xlabel='Year', ylabel='Sales in 1000$', figsize=(8, 4))


# Example 1: Installing dependencies from a valid requirements file
try:
    install_dependencies('requirements.txt')
except Exception as e:
    print(f"An error occurred: {e}")

# Example 2: Handling a missing requirements file
try:
    install_dependencies('missing_requirements.txt')
except FileNotFoundError as e:
    print(f"File not found error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

# Example 3: Catching installation errors
try:
    # Assuming 'faulty_requirements.txt' is a file with incorrect package names
    install_dependencies('faulty_requirements.txt')
except FileNotFoundError as e:
    print(f"File not found error: {e}")
except Exception as e:
    print(f"An error occurred during installation: {e}")
