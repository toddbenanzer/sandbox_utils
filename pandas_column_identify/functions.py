andas as pd
import numpy as np

def check_column_string(df, column_name):
    column = df[column_name]
    return column.apply(lambda x: isinstance(x, str)).all()

def check_column_numeric(column):
    return pd.to_numeric(column, errors='coerce').notnull().all()

def check_column_integer(df, column_name):
    column = df[column_name]
    return column.astype(float).apply(lambda x: x.is_integer()).all()

def check_column_float(column):
    return pd.api.types.is_float_dtype(column)

def check_column_boolean(column):
    return column.dtype == bool

def check_column_date(column):
    return pd.api.types.is_datetime64_dtype(column)

def check_column_datetime(column):
    return pd.api.types.is_datetime64_dtype(column)

def handle_missing_data(column):
    column = column.replace(['', 'NA', 'N/A', 'nan', 'NaN'], float('nan'))
    column = column.replace([float('inf'), float('-inf')], float('nan'))
    column = column.dropna()
    
    return column

def handle_infinite_data(column):
    if np.isinf(column).any():
        column.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return column

def check_null_or_empty(column):
    if column.isnull().any() or column.astype(str).str.strip().empty:
        return True
    
    return False

def is_trivial_column(column):
    unique_values = column.unique()
    
    if len(unique_values) == 1:
        return True
    
    if set(unique_values) == {0, 1}:
        return True
    
    return False

def determine_most_likely_data_type(df, column_name):
    column = df[column_name]
    
    if check_null_or_empty(column) or is_trivial_column(column):
        return "Null or Trivial"
    
    if check_column_boolean(column):
        return "Boolean"
    
    if pd.api.types.is_categorical_dtype(column):
        return "Categorical"
    
    if pd.api.types.is_datetime64_any_dtype(column):
        return "Datetime"
    
    if pd.api.types.is_datetime64_dtype(column):
        return "Date"
    
    if pd.api.types.is_float_dtype(column):
        return "Float"
    
    if pd.api.types.is_integer_dtype(column):
        return "Integer"
    
    return "String"

def check_null_column(column):
    if column.isnull().all():
        return True
    
    return False

def is_trivial_column(column):
    unique_values = column.unique()
    
    if len(unique_values) == 1:
        return True
    
    if set(unique_values) == {0, 1}:
        return True
    
    return False

def check_categorical(column):
    return column.dtype.name == 'category'

def is_boolean_column(column):
    return all(value in [True, False, pd.NA] for value in column)

def is_categorical(column):
    return column.dtype.name == 'category'

def handle_missing_data(column, method='impute', value=None):
    if method == 'impute':
        column = column.fillna(value)
    elif method == 'remove':
        column = column.dropna()
    
    return column

def handle_infinite_data(data, remove_values=True):
    if data.isin([np.inf, -np.inf]).any():
        if remove_values:
            data = data.replace([np.inf, -np.inf], np.nan).dropna()
        else:
            data = data.replace([np.inf, -np.inf], np.nan)
            
    return data

def determine_data_type(column):
    if column.isnull().all() or len(column.unique()) == 1:
        return "Null or Trivial"
    
    if set(column.unique()) == {True, False}:
        return "Boolean"
    
    if pd.api.types.is_categorical_dtype(column):
        return "Categorical"
    
    try:
        pd.to_datetime(column, errors='raise')
        return "Datetime"
    except:
        pass
    
    try:
        pd.to_datetime(column, format='%Y-%m-%d', errors='raise')
        return "Date"
    except:
        pass
    
    if np.issubdtype(column.dtype, np.floating):
        return "Float"
    
    if np.issubdtype(column.dtype, np.integer):
        return "Integer"
    
    return "String"

def handle_mixed_data(df, column_name):
    unique_types = df[column_name].apply(lambda x: type(x)).unique()
    
    if len(unique_types) == 1 and np.isnan(unique_types[0]):
        df[column_name] = np.nan
        return df
    
    if len(unique_types) == 2 and np.nan in unique_types:
        df[column_name] = df[column_name].apply(lambda x: np.nan if x is None else x)
        return df
    
    if bool in unique_types:
        df[column_name] = df[column_name].apply(lambda x: str(x) if isinstance(x, bool) else x)
        return df
    
    if pd.api.types.is_categorical_dtype(df[column_name]):
        df[column_name] = df[column_name].astype(str)
        return df
    
    if pd.api.types.is_datetime64_any_dtype(df[column_name]):
        df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
        return df
    
    if np.issubdtype(df[column_name].dtype, np.floating) and np.isinf(df[column_name]).any():
        df[column_name] = df[column_name].replace([np.inf, -np.inf], np.nan)
        return df
    
    df[column_name] = df[column_name].astype(str)    
    return d