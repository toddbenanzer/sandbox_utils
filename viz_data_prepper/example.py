

# Example usage with a list
data_list = [10, 20, 30, 40, 50]
normalizer_list = DataNormalizer(data_list)

# Min-Max normalization on list
normalized_min_max_list = normalizer_list.normalize_data('min-max')
print("Min-Max Normalized List:", normalized_min_max_list)

# Z-Score normalization on list
normalized_z_score_list = normalizer_list.normalize_data('z-score')
print("Z-Score Normalized List:", normalized_z_score_list)


# Example usage with a pandas DataFrame
data_frame = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50]
})
normalizer_df = DataNormalizer(data_frame)

# Min-Max normalization on DataFrame
normalized_min_max_df = normalizer_df.normalize_data('min-max')
print("\nMin-Max Normalized DataFrame:\n", normalized_min_max_df)

# Z-Score normalization on DataFrame
normalized_z_score_df = normalizer_df.normalize_data('z-score')
print("\nZ-Score Normalized DataFrame:\n", normalized_z_score_df)


# Example usage with a numpy ndarray
data_ndarray = np.array([[5, 10], [15, 20], [25, 30]])
normalizer_ndarray = DataNormalizer(data_ndarray)

# Min-Max normalization on ndarray
normalized_min_max_ndarray = normalizer_ndarray.normalize_data('min-max')
print("\nMin-Max Normalized ndarray:\n", normalized_min_max_ndarray)

# Z-Score normalization on ndarray
normalized_z_score_ndarray = normalizer_ndarray.normalize_data('z-score')
print("\nZ-Score Normalized ndarray:\n", normalized_z_score_ndarray)


# Example usage with a list of numbers
data_list = [2.5, 5.1, 7.5, 10.0, 15.0]
binner_list = DataBinner(data_list)

# Binning the list into 3 equal-width bins
binned_list_3_bins = binner_list.bin_data(3)
print("Binned list into 3 equal-width bins:", binned_list_3_bins)

# Binning the list using specific bin edges
binned_list_edges = binner_list.bin_data([0, 5, 10, 20])
print("Binned list using specific edges:", binned_list_edges)


# Example usage with a pandas DataFrame
data_df = pd.DataFrame({
    'A': [1, 4, 7, 10],
    'B': [12, 18, 25, 30]
})
binner_df = DataBinner(data_df)

# Binning the DataFrame into 2 bins
binned_df_2_bins = binner_df.bin_data(2)
print("\nBinned DataFrame into 2 bins:\n", binned_df_2_bins)

# Binning the DataFrame using specific bin edges
binned_df_edges = binner_df.bin_data([0, 15, 30])
print("\nBinned DataFrame using specific edges:\n", binned_df_edges)


# Example usage with a numpy ndarray
data_ndarray = np.array([[2, 8], [4, 10], [6, 12], [8, 14]])
binner_ndarray = DataBinner(data_ndarray)

# Binning the ndarray into 3 bins
binned_ndarray_3_bins = binner_ndarray.bin_data(3)
print("\nBinned ndarray into 3 bins:\n", binned_ndarray_3_bins)

# Binning the ndarray using specific bin edges
binned_ndarray_edges = binner_ndarray.bin_data([0, 5, 10, 15])
print("\nBinned ndarray using specific edges:\n", binned_ndarray_edges)


# Example usage with a simple DataFrame
data = pd.DataFrame({
    'Category': ['A', 'A', 'B', 'B'],
    'Value': [10, 20, 30, 40]
})
aggregator = DataAggregator(data)

# Aggregating by 'Category' and summing the 'Value'
result_sum = aggregator.aggregate_data('Category', 'sum')
print("Sum aggregation by Category:\n", result_sum)

# Example with multiple aggregation functions
result_multiple = aggregator.aggregate_data('Category', {'Value': ['sum', 'mean']})
print("\nMultiple aggregations by Category:\n", result_multiple)

# Example aggregation by multiple columns
data_multi = pd.DataFrame({
    'Category': ['A', 'A', 'B', 'B'],
    'Type': ['X', 'Y', 'X', 'Y'],
    'Value': [10, 20, 30, 40]
})
aggregator_multi = DataAggregator(data_multi)

# Aggregating by 'Category' and 'Type' with sum
result_multi_group = aggregator_multi.aggregate_data(['Category', 'Type'], 'sum')
print("\nSum aggregation by Category and Type:\n", result_multi_group)

# Example with different aggregation functions for different columns
data_diff_agg = pd.DataFrame({
    'Category': ['A', 'A', 'B', 'B'],
    'Value1': [10, 20, 30, 40],
    'Value2': [5, 15, 25, 35]
})
aggregator_diff_agg = DataAggregator(data_diff_agg)

# Aggregating by 'Category' with sum for 'Value1' and mean for 'Value2'
result_diff_agg = aggregator_diff_agg.aggregate_data('Category', {'Value1': 'sum', 'Value2': 'mean'})
print("\nDifferent aggregations on the same group:\n", result_diff_agg)


# Example usage with a pandas DataFrame
data_df = pd.DataFrame({
    'Age': [25, 30, np.nan, 40],
    'Score': [88, np.nan, 95, 78]
})
validator_df = DataValidator(data_df)
report_df = validator_df.validate_data()
print("Validation Report for DataFrame:")
print("Missing Values:", report_df['missing_values'])
print("Data Types:", report_df['data_type_inconsistencies'])

# Example usage with a numpy ndarray
data_ndarray = np.array([5.5, np.nan, 8.5, 9.0])
validator_ndarray = DataValidator(data_ndarray)
report_ndarray = validator_ndarray.validate_data()
print("\nValidation Report for ndarray:")
print("Missing Values:", report_ndarray['missing_values'])
print("Data Type:", report_ndarray['data_type_inconsistencies'])

# Example usage with a list
data_list = [1.0, 2.5, np.nan, 3.6]
validator_list = DataValidator(data_list)
report_list = validator_list.validate_data()
print("\nValidation Report for List:")
print("Missing Values:", report_list['missing_values'])
print("Data Type:", report_list['data_type_inconsistencies'])
