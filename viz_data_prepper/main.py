from typing import Union, List
from typing import Union, List, Dict
from typing import Union, List, Dict, Any
import numpy as np
import pandas as pd


class DataNormalizer:
    """
    A class for normalizing datasets.
    """

    def __init__(self, data: Union[List[float], pd.DataFrame, np.ndarray]):
        """
        Initialize the DataNormalizer with the input data.

        Args:
            data (Union[List[float], pd.DataFrame, np.ndarray]): The raw data to be normalized.
        """
        if not isinstance(data, (list, pd.DataFrame, np.ndarray)):
            raise ValueError("Data must be a list, pandas DataFrame, or numpy ndarray.")
        
        self.data = data

    def normalize_data(self, method: str) -> Union[List[float], pd.DataFrame, np.ndarray]:
        """
        Normalize the data using the specified method.

        Args:
            method (str): The normalization method to apply ('min-max' or 'z-score').

        Returns:
            Union[List[float], pd.DataFrame, np.ndarray]: The normalized data.

        Raises:
            ValueError: If an unsupported normalization method is provided.
        """
        if method == 'min-max':
            return self._min_max_normalize()
        elif method == 'z-score':
            return self._z_score_normalize()
        else:
            raise ValueError(f"Unsupported normalization method '{method}'. Choose 'min-max' or 'z-score'.")

    def _min_max_normalize(self) -> Union[List[float], pd.DataFrame, np.ndarray]:
        """
        Apply min-max normalization.

        Returns:
            Union[List[float], pd.DataFrame, np.ndarray]: The min-max normalized data.
        """
        if isinstance(self.data, pd.DataFrame):
            return (self.data - self.data.min()) / (self.data.max() - self.data.min())
        elif isinstance(self.data, np.ndarray):
            return (self.data - self.data.min(axis=0)) / (self.data.max(axis=0) - self.data.min(axis=0))
        elif isinstance(self.data, list):
            data_array = np.array(self.data)
            return ((data_array - data_array.min()) / (data_array.max() - data_array.min())).tolist()

    def _z_score_normalize(self) -> Union[List[float], pd.DataFrame, np.ndarray]:
        """
        Apply z-score normalization.

        Returns:
            Union[List[float], pd.DataFrame, np.ndarray]: The z-score normalized data.
        """
        if isinstance(self.data, pd.DataFrame):
            return (self.data - self.data.mean()) / self.data.std()
        elif isinstance(self.data, np.ndarray):
            return (self.data - self.data.mean(axis=0)) / self.data.std(axis=0)
        elif isinstance(self.data, list):
            data_array = np.array(self.data)
            return ((data_array - data_array.mean()) / data_array.std()).tolist()



class DataBinner:
    """
    A class for binning continuous data into discrete categories.
    """

    def __init__(self, data: Union[List[float], pd.DataFrame, np.ndarray]):
        """
        Initialize the DataBinner with the input data.

        Args:
            data (Union[List[float], pd.DataFrame, np.ndarray]): The raw continuous data to be binned.
        """
        if not isinstance(data, (list, pd.DataFrame, np.ndarray)):
            raise ValueError("Data must be a list, pandas DataFrame, or numpy ndarray.")
        
        self.data = data

    def bin_data(self, bins: Union[int, List[float]]) -> Union[List[int], pd.DataFrame, np.ndarray]:
        """
        Bin the continuous data into discrete categories based on specified bin edges or intervals.

        Args:
            bins (Union[int, List[float]]): The number of bins or the specific bin edges to use.

        Returns:
            Union[List[int], pd.DataFrame, np.ndarray]: The binned data.

        Raises:
            ValueError: If bins is neither an int nor a list of scalars.
        """
        if isinstance(bins, int) and bins <= 0:
            raise ValueError("Number of bins must be a positive integer.")
        if isinstance(bins, list) and not all(isinstance(b, (int, float)) for b in bins):
            raise ValueError("Bin edges must be a list of scalars.")
        
        if isinstance(self.data, pd.DataFrame):
            return self.data.apply(lambda col: pd.cut(col, bins, labels=False, include_lowest=True))
        elif isinstance(self.data, np.ndarray):
            return np.apply_along_axis(lambda col: np.digitize(col, bins=bins, right=False), axis=0, arr=self.data) - 1
        elif isinstance(self.data, list):
            data_array = np.array(self.data)
            binned_array = np.digitize(data_array, bins=bins, right=False)
            return (binned_array - 1).tolist()



class DataAggregator:
    """
    A class for performing aggregation operations on datasets.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the DataAggregator with the input DataFrame.

        Args:
            data (pd.DataFrame): The data to be aggregated.

        Raises:
            ValueError: If the provided data is not a pandas DataFrame.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")
        
        self.data = data

    def aggregate_data(self, group_by: Union[str, List[str]], aggregate_func: Union[Dict[str, str], str]) -> pd.DataFrame:
        """
        Perform aggregation on the data by grouping based on specified columns and applying aggregate functions.

        Args:
            group_by (Union[str, List[str]]): Column name(s) to group the data by.
            aggregate_func (Union[Dict[str, str], str]): The aggregation function(s) to apply. Can be a dictionary mapping column names to functions, or a single function to apply to all columns.

        Returns:
            pd.DataFrame: The aggregated DataFrame.

        Raises:
            ValueError: If the `group_by` columns do not exist in the DataFrame.
            ValueError: If the `aggregate_func` is not compatible with the data columns.
        """
        # Validate group_by columns
        if isinstance(group_by, str):
            group_by = [group_by]
        
        for col in group_by:
            if col not in self.data.columns:
                raise ValueError(f"Group by column '{col}' does not exist in the DataFrame.")
        
        # Perform aggregation
        try:
            aggregated_data = self.data.groupby(group_by).agg(aggregate_func)
        except Exception as e:
            raise ValueError(f"Aggregation function failed with error: {e}")

        return aggregated_data



class DataValidator:
    """
    A class for validating datasets to ensure consistency, check for missing values, and verify data types.
    """

    def __init__(self, data: Union[List[float], pd.DataFrame, np.ndarray]):
        """
        Initialize the DataValidator with the input data.

        Args:
            data (Union[List[float], pd.DataFrame, np.ndarray]): The data to be validated.

        Raises:
            ValueError: If the provided data is not a list, pandas DataFrame, or numpy ndarray.
        """
        if not isinstance(data, (list, pd.DataFrame, np.ndarray)):
            raise ValueError("Data must be a list, pandas DataFrame, or numpy ndarray.")
        
        self.data = data

    def validate_data(self) -> Dict[str, Any]:
        """
        Validate the data for consistency issues, check for missing values, and ensure data types are appropriate.

        Returns:
            Dict[str, Any]: A validation report containing the results such as missing values and data type inconsistencies.
        """
        validation_report = {
            'missing_values': None,
            'data_type_inconsistencies': None
        }
        
        if isinstance(self.data, pd.DataFrame):
            validation_report['missing_values'] = self.data.isnull().sum().to_dict()
            validation_report['data_type_inconsistencies'] = self.data.dtypes.to_dict()
            
        elif isinstance(self.data, np.ndarray):
            validation_report['missing_values'] = np.sum(pd.isnull(self.data))
            validation_report['data_type_inconsistencies'] = self.data.dtype
            
        elif isinstance(self.data, list):
            data_array = np.array(self.data)
            validation_report['missing_values'] = np.sum(pd.isnull(data_array))
            validation_report['data_type_inconsistencies'] = data_array.dtype

        return validation_report
