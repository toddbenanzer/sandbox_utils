from boolean_stats import BooleanDescriptiveStats
import pandas as pd


# Example 1: Basic Usage
df = pd.DataFrame({'flag': [True, False, True, False, True, None]})
stats = BooleanDescriptiveStats(df, 'flag')
print("Mean of 'flag':", stats.calculate_mean())
print("Count of True values in 'flag':", stats.calculate_true_count())
print("Count of False values in 'flag':", stats.calculate_false_count())

# Example 2: Handling Missing Values
print("Missing prevalence in 'flag':", stats.calculate_missing_prevalence())
print("Empty prevalence in 'flag':", stats.calculate_empty_prevalence())

# Example 3: Most Common Value
most_common = stats.find_most_common_value()
if most_common is not None:
    print("Most common value in 'flag':", most_common)
else:
    print("No mode found in 'flag'.")

# Example 4: Handling Non-Boolean DataFrame
df_invalid = pd.DataFrame({'flag': [1, 0, 1, 0, 1]})
try:
    BooleanDescriptiveStats(df_invalid, 'flag')
except ValueError as e:
    print(e)


# Example 1: Display statistics in dictionary format
stats = {
    'mean': 0.6,
    'true_count': 15,
    'false_count': 10,
    'mode': True,
    'missing_prevalence': 0.05,
    'empty_prevalence': 0.05
}
display_statistics(stats, 'dictionary')

# Example 2: Display statistics in JSON format
display_statistics(stats, 'json')

# Example 3: Display statistics in table format (requires 'tabulate' package)
try:
    display_statistics(stats, 'table')
except ImportError:
    print("Tabulate library is not installed. Please install it to use the 'table' format.")

# Example 4: Attempt to use an unsupported format
try:
    display_statistics(stats, 'xml')
except ValueError as e:
    print("Caught error:", e)

# Example 5: Use with missing keys in stats
incomplete_stats = {
    'mean': 0.6,
    'true_count': 15
}
try:
    display_statistics(incomplete_stats, 'dictionary')
except ValueError as e:
    print("Caught error:", e)
