from data_standardizer_module import DataStandardizer  # Assuming the class is saved in a module named data_standardizer_module
from validation_module import integrate_with_pandas, ValidationRuleManager
import numpy as np
import pandas as pd


# Sample DataFrame
data = {
    'A': [1, 2, np.nan, 4, 100, 200],
    'B': [5, 6, 1, 8, 2, np.nan],
    'C': ['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04', 'Invalid Date', '2021-01-06']
}
df = pd.DataFrame(data)

# Initialize DataValidator
validator = DataValidator(df)

# Check for missing values
missing_values = validator.check_missing_values()
print("Missing Values:\n", missing_values)

# Detect outliers using z-score
outliers_z_score = validator.detect_outliers(method="z-score", threshold=2)
print("Outliers using z-score:\n", outliers_z_score)

# Detect outliers using IQR
outliers_iqr = validator.detect_outliers(method="IQR")
print("Outliers using IQR:\n", outliers_iqr)

# Find inconsistencies using rules
# Example: Date in column 'C' must be a valid date format
consistency_rules = {'C': "C.apply(lambda x: pd.to_datetime(x, errors='coerce').notna())"}
inconsistencies = validator.find_inconsistencies(consistency_rules)
print("Inconsistencies:\n", inconsistencies)



# Sample DataFrame with missing values and potential outliers
data = {
    'A': [1, 2, np.nan, 4, 100],
    'B': [5, 6, np.nan, 8, 2],
    'C': ['x', 'y', 'y', 'x', 'y']
}
df = pd.DataFrame(data)

# Initialize the DataCorrector
corrector = DataCorrector(df)

# Example 1: Impute missing values with the mean of each column
imputed_df_mean = corrector.impute_missing_values(method="mean")
print("Imputed DataFrame with mean:\n", imputed_df_mean)

# Example 2: Impute missing values with the median of each column
imputed_df_median = corrector.impute_missing_values(method="median")
print("Imputed DataFrame with median:\n", imputed_df_median)

# Example 3: Impute missing values with the mode of each column
imputed_df_mode = corrector.impute_missing_values(method="mode")
print("Imputed DataFrame with mode:\n", imputed_df_mode)

# Example 4: Handle outliers by removing them
df_no_outliers = corrector.handle_outliers(method="remove")
print("DataFrame with outliers removed:\n", df_no_outliers)

# Example 5: Handle outliers by clipping
df_clipped = corrector.handle_outliers(method="clip")
print("DataFrame with outliers clipped:\n", df_clipped)



# Sample DataFrame
data = {
    'Date': ['2021-01-01', '02/14/2021', 'March 3, 2021', 'Invalid Date'],
    'Price': ['$100.00', '$250', '300', '£400.50'],  # Includes an invalid currency format
    'Discount': ['5%', '10%', '15%', '20%']
}
df = pd.DataFrame(data)

# Initialize the DataStandardizer
standardizer = DataStandardizer(df)

# Example 1: Standardize date format
df_dates_standardized = standardizer.standardize_column_format("Date", "date")
print("Standardized Dates:\n", df_dates_standardized['Date'])

# Example 2: Standardize currency format
df_price_standardized = standardizer.standardize_column_format("Price", "currency")
print("Standardized Prices:\n", df_price_standardized['Price'])

# Example 3: Standardize percentage format
df_discount_standardized = standardizer.standardize_column_format("Discount", "percentage")
print("Standardized Discounts:\n", df_discount_standardized['Discount'])


# Example 1: Define and retrieve a validation rule
rule_manager = ValidationRuleManager()
rule_manager.define_rule("positive_values", "x > 0")
print("Defined Rules:", rule_manager.rules)

# Example 2: Overwrite an existing rule
rule_manager.define_rule("positive_values", "x >= 0")
print("Updated Rule:", rule_manager.rules["positive_values"])

# Example 3: Save rules to a file
rule_manager.define_rule("non_empty", "len(x) > 0")
rules_file_path = "rules.json"
rule_manager.save_rules(rules_file_path)

# Example 4: Load rules from a file
new_manager = ValidationRuleManager()
new_manager.load_rules(rules_file_path)
print("Loaded Rules:", new_manager.rules)


# Use Case 1: Create a ValidationReport and generate a text summary
validation_data = [
    {"column": "age", "error": "Out of range"},
    {"column": "salary", "error": "Negative value"}
]

report = ValidationReport(validation_data)
text_summary = report.generate_summary(output_format="text")
print("Text Summary:")
print(text_summary)

# Use Case 2: Generate an HTML summary
html_summary = report.generate_summary(output_format="html")
print("\nHTML Summary:")
print(html_summary)

# Use Case 3: Export the validation results to a CSV file
csv_file_path = "validation_report.csv"
report.export_report(csv_file_path, format="csv")
print(f"\nValidation results exported to {csv_file_path}")

# Use Case 4: Export the validation results to a JSON file
json_file_path = "validation_report.json"
report.export_report(json_file_path, format="json")
print(f"\nValidation results exported to {json_file_path}")

# Use Case 5: Export the validation results to an Excel file
xlsx_file_path = "validation_report.xlsx"
report.export_report(xlsx_file_path, format="xlsx")
print(f"\nValidation results exported to {xlsx_file_path}")



# Integrate custom methods with Pandas
integrate_with_pandas()

# Create a sample DataFrame
data = {
    'A': [1, 2, None, 4, 100],
    'B': [5, 6, 1, 8, 2],
    'C': ['string1', 'string2', 'string2', 'string1', 'string2']
}
df = pd.DataFrame(data)

# Example 1: Validate the DataFrame
validation_results = df.validate()
print("Validation Results:", validation_results)

# Example 2: Correct the DataFrame
df_corrected = df.correct()
print("\nCorrected DataFrame:\n", df_corrected)

# Example 3: Standardize a column in the DataFrame
df_standardized = df.standardize('A', 'currency')
print("\nStandardized DataFrame:\n", df_standardized)

# Example 4: Apply validation rules using a ValidationRuleManager
rule_manager = ValidationRuleManager()
rule_manager.define_rule('positive_B', 'B > 0')
df_with_rules = df.apply_rules(rule_manager)
print("\nDataFrame with Applied Rules:\n", df_with_rules)

# Example 5: Generate a validation report
report_summary = df.generate_report(format="text")
print("\nValidation Report Summary:\n", report_summary)

# Example 6: Export validation results to a CSV file
df.export_report("validation_report.csv", format="csv")
print("\nValidation result exported to 'validation_report.csv'")



# Example 1: Basic validation and correction without any rules
data = {
    'A': [1, None, 3, 4, None],
    'B': [1, 200, 3, None, 5]
}
df = pd.DataFrame(data)

corrected_df, report = validate_and_correct(df, {})
print("Corrected DataFrame:\n", corrected_df)
print("Validation and Correction Report:\n", report)

# Example 2: Validation and correction with specific rules
data = {
    'X': [10, 20, None, 40, 50],
    'Y': [None, 1, 2, None, 5]
}
df = pd.DataFrame(data)
rules = {'X': "X > 0", 'Y': "Y <= 5"}

corrected_df, report = validate_and_correct(df, rules)
print("\nCorrected DataFrame with Rules:\n", corrected_df)
print("Validation and Correction Report with Rules:\n", report)

# Example 3: Handling a DataFrame with missing values and outliers
data = {
    'Temperature': [23, 29, None, 100, 35, 24],
    'Pressure': [1012, 1015, 986, 1040, None, 1018]
}
df = pd.DataFrame(data)

corrected_df, report = validate_and_correct(df, {'Temperature': "Temperature < 100", 'Pressure': "Pressure < 1050"})
print("\nCorrected DataFrame with Temperature and Pressure:\n", corrected_df)
print("Validation and Correction Report for Temperature and Pressure:\n", report)
