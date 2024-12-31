from datetime import datetime
from scipy import stats
from types import MethodType
from typing import Any, Dict
from typing import Any, Union
from typing import Dict
from typing import Dict, Tuple, Any
from typing import Dict, Union
from typing import Union, Callable
import json
import numpy as np
import pandas as pd


class DataValidator:
    """
    A class to validate a Pandas DataFrame by checking for missing values, detecting outliers,
    and finding inconsistencies according to specified rules.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the DataValidator object with the provided DataFrame.

        Args:
            dataframe (pd.DataFrame): The dataset to perform validations on.
        """
        self.dataframe = dataframe

    def check_missing_values(self) -> pd.Series:
        """
        Identify and report missing values within the DataFrame.

        Returns:
            pd.Series: A series indicating the count of missing values per column.
        """
        missing_values = self.dataframe.isnull().sum()
        print("Missing values found in the following columns:")
        print(missing_values[missing_values > 0])
        return missing_values

    def detect_outliers(self, method: str = "z-score", threshold: float = 3.0) -> Dict[str, pd.DataFrame]:
        """
        Detect outliers in the numerical columns of the DataFrame based on a specified method and threshold.

        Args:
            method (str): The method to use for detecting outliers ('z-score' or 'IQR').
            threshold (float): The threshold beyond which a data point is considered an outlier.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary with column names as keys and DataFrames of detected outliers as values.
        """
        outliers = {}
        if method == "z-score":
            for column in self.dataframe.select_dtypes(include=[np.number]):
                z_scores = np.abs(stats.zscore(self.dataframe[column].dropna()))
                outliers[column] = self.dataframe[column][z_scores > threshold]
                print(f"Outliers detected in column '{column}':")
                print(outliers[column])

        elif method == "IQR":
            for column in self.dataframe.select_dtypes(include=[np.number]):
                Q1 = self.dataframe[column].quantile(0.25)
                Q3 = self.dataframe[column].quantile(0.75)
                IQR = Q3 - Q1
                filter = (self.dataframe[column] < (Q1 - 1.5 * IQR)) | (self.dataframe[column] > (Q3 + 1.5 * IQR))
                outliers[column] = self.dataframe[column][filter]
                print(f"Outliers detected in column '{column}':")
                print(outliers[column])

        else:
            raise ValueError(f"Unsupported method '{method}'. Choose 'z-score' or 'IQR'.")

        return outliers

    def find_inconsistencies(self, column_rules: Dict[str, str]) -> pd.DataFrame:
        """
        Identify data inconsistencies within the DataFrame based on user-defined rules for specific columns.

        Args:
            column_rules (Dict[str, str]): Dictionary where keys are column names and values are rules/conditions.

        Returns:
            pd.DataFrame: A DataFrame highlighting inconsistent records.
        """
        inconsistencies = pd.DataFrame()

        for column, rule in column_rules.items():
            if column in self.dataframe.columns:
                # Evaluate the rule as a boolean condition on the column
                condition_met = self.dataframe.eval(rule)
                inconsistency = self.dataframe[~condition_met]
                
                if not inconsistency.empty:
                    print(f"Inconsistencies found in column '{column}':")
                    print(inconsistency)

                inconsistencies = pd.concat([inconsistencies, inconsistency])

        return inconsistencies



class DataCorrector:
    """
    A class to correct a Pandas DataFrame by imputing missing values and handling outliers
    according to specified methods.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the DataCorrector object with the provided DataFrame.

        Args:
            dataframe (pd.DataFrame): The dataset to perform corrections on.
        """
        self.dataframe = dataframe

    def impute_missing_values(self, method: Union[str, Callable[[pd.Series], Union[int, float]]] = "mean") -> pd.DataFrame:
        """
        Fill in missing values in the DataFrame using a specified imputation method.

        Args:
            method (str or callable): The method to use for imputing missing values. Options include
                                      'mean', 'median', 'mode', or a user-defined function.

        Returns:
            pd.DataFrame: A DataFrame with missing values imputed.
        """
        if method == "mean":
            return self.dataframe.fillna(self.dataframe.mean())
        elif method == "median":
            return self.dataframe.fillna(self.dataframe.median())
        elif method == "mode":
            return self.dataframe.fillna(self.dataframe.mode().iloc[0])
        elif callable(method):
            return self.dataframe.fillna(method)
        else:
            raise ValueError(f"Unsupported method '{method}'. Choose 'mean', 'median', 'mode', or provide a custom function.")

    def handle_outliers(self, method: str = "remove") -> pd.DataFrame:
        """
        Manage outliers in the DataFrame using a specified method.

        Args:
            method (str): The method to use for handling outliers. Options include 'remove', 'clip', or 'replace'.

        Returns:
            pd.DataFrame: A DataFrame with outliers handled according to the specified method.
        """
        def is_outlier(series: pd.Series, threshold: float = 1.5) -> pd.Series:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (threshold * IQR)
            upper_bound = Q3 + (threshold * IQR)
            return (series < lower_bound) | (series > upper_bound)

        df_corrected = self.dataframe.copy()
        for column in df_corrected.select_dtypes(include=[np.number]):
            if method == "remove":
                df_corrected = df_corrected[~is_outlier(df_corrected[column])]
            elif method == "clip":
                df_corrected[column] = df_corrected[column].clip(lower=df_corrected[column].quantile(0.25), upper=df_corrected[column].quantile(0.75))
            elif callable(method):
                df_corrected[column] = method(df_corrected[column])
            else:
                raise ValueError(f"Unsupported method '{method}'. Choose 'remove', 'clip', or provide a custom function.")
        return df_corrected



class DataStandardizer:
    """
    A class to standardize the format of specified columns in a Pandas DataFrame.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the DataStandardizer object with the provided DataFrame.

        Args:
            dataframe (pd.DataFrame): The dataset to perform standardization on.
        """
        self.dataframe = dataframe
    
    def standardize_column_format(self, column_name: str, format_type: str) -> pd.DataFrame:
        """
        Standardize the format of a specified column in the DataFrame.

        Args:
            column_name (str): The name of the column to standardize.
            format_type (str): The format type to standardize to, options include 'date', 'currency', 'percentage'.

        Returns:
            pd.DataFrame: A DataFrame with the specified column standardized to the desired format.
        """
        if column_name not in self.dataframe.columns:
            raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
        
        df_standardized = self.dataframe.copy()

        if format_type == 'date':
            df_standardized[column_name] = pd.to_datetime(df_standardized[column_name], errors='coerce')
        elif format_type == 'currency':
            df_standardized[column_name] = df_standardized[column_name].replace('[\$,]', '', regex=True).astype(float)
        elif format_type == 'percentage':
            df_standardized[column_name] = df_standardized[column_name].replace('%', '', regex=True).astype(float) / 100
        else:
            raise ValueError(f"Unsupported format type '{format_type}'. Supported formats: 'date', 'currency', 'percentage'.")

        return df_standardized



class ValidationRuleManager:
    """
    A class to manage validation rules for data processing. Allows defining, saving, and loading rules.
    """

    def __init__(self):
        """
        Initialize the ValidationRuleManager object with an empty dictionary to store rules.
        """
        self.rules: Dict[str, str] = {}

    def load_rules(self, rules_file: str) -> None:
        """
        Load validation rules from a specified file into the rule manager.

        Args:
            rules_file (str): Path to the file containing validation rules, expected in JSON format.

        Returns:
            None
        """
        try:
            with open(rules_file, 'r') as file:
                self.rules = json.load(file)
            print("Rules loaded successfully from", rules_file)
        except FileNotFoundError:
            print(f"The file {rules_file} was not found.")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from the file {rules_file}.")

    def save_rules(self, rules_file: str) -> None:
        """
        Save the current set of validation rules to a specified file.

        Args:
            rules_file (str): Path to the file where the validation rules will be saved in JSON format.

        Returns:
            None
        """
        try:
            with open(rules_file, 'w') as file:
                json.dump(self.rules, file, indent=4)
            print("Rules saved successfully to", rules_file)
        except IOError as e:
            print(f"An error occurred while writing to the file {rules_file}: {e}")

    def define_rule(self, rule_name: str, rule_definition: str) -> None:
        """
        Define a new validation rule by specifying its name and definition.

        Args:
            rule_name (str): The name of the rule to be added.
            rule_definition (str): The definition or condition of the rule.

        Returns:
            None
        """
        if rule_name in self.rules:
            print(f"Rule '{rule_name}' already exists. It will be updated.")
        else:
            print(f"Defining new rule '{rule_name}'.")
        
        self.rules[rule_name] = rule_definition
        print(f"Rule '{rule_name}' defined as: {rule_definition}")



class ValidationReport:
    """
    A class to generate summaries and reports based on validation results for datasets.
    """

    def __init__(self, validation_results: Any):
        """
        Initialize the ValidationReport object with the given validation results.

        Args:
            validation_results (Any): The results from a data validation process.
        """
        self.validation_results = validation_results

    def generate_summary(self, output_format: str = "text") -> Union[str, pd.DataFrame]:
        """
        Generate a summary of the validation results in the specified format.

        Args:
            output_format (str): The format of the summary output; options include 'text' or 'html'.

        Returns:
            Union[str, pd.DataFrame]: A summary in the specified format.
        """
        if output_format == "text":
            summary = str(self.validation_results)
        elif output_format == "html":
            summary = pd.DataFrame(self.validation_results).to_html()
        else:
            raise ValueError(f"Unsupported output format '{output_format}'. Supported formats: 'text', 'html'.")
        
        return summary

    def export_report(self, file_path: str, format: str = "csv") -> None:
        """
        Export the validation results to a specified file in the desired format.

        Args:
            file_path (str): Path to the file where the report will be saved.
            format (str): The file format to use for exporting the report; options include 'csv', 'json', or 'xlsx'.

        Returns:
            None
        """
        if format == "csv":
            pd.DataFrame(self.validation_results).to_csv(file_path, index=False)
        elif format == "json":
            pd.DataFrame(self.validation_results).to_json(file_path, orient='records', lines=True)
        elif format == "xlsx":
            pd.DataFrame(self.validation_results).to_excel(file_path, index=False)
        else:
            raise ValueError(f"Unsupported export format '{format}'. Supported formats: 'csv', 'json', 'xlsx'.")



# Assuming the classes from the module have been imported: DataValidator, DataCorrector, DataStandardizer, ValidationRuleManager, ValidationReport

def integrate_with_pandas():
    """
    Integrates custom validation, correction, and standardization functionalities with the Pandas DataFrame.
    """

    def validate(self, column_rules: Dict[str, str] = None) -> Dict[str, Any]:
        validator = DataValidator(self)
        missing_values = validator.check_missing_values()
        outliers = validator.detect_outliers()
        inconsistencies = validator.find_inconsistencies(column_rules) if column_rules else None
        return {
            "missing_values": missing_values,
            "outliers": outliers,
            "inconsistencies": inconsistencies
        }

    def correct(self) -> pd.DataFrame:
        corrector = DataCorrector(self)
        df_corrected = corrector.impute_missing_values()
        df_corrected = corrector.handle_outliers()
        return df_corrected

    def standardize(self, column_name: str, format_type: str) -> pd.DataFrame:
        standardizer = DataStandardizer(self)
        return standardizer.standardize_column_format(column_name, format_type)

    def apply_rules(self, rule_manager: ValidationRuleManager) -> pd.DataFrame:
        for rule_name, rule_definition in rule_manager.rules.items():
            self.eval(rule_definition, inplace=True)
        return self

    def generate_report(self, format: str = "text") -> str:
        report = ValidationReport(self)
        return report.generate_summary(format)

    def export_report(self, file_path: str, format: str = "csv") -> None:
        report = ValidationReport(self)
        report.export_report(file_path, format)

    # Attaching methods to the DataFrame
    pd.DataFrame.validate = MethodType(validate, None, pd.DataFrame)
    pd.DataFrame.correct = MethodType(correct, None, pd.DataFrame)
    pd.DataFrame.standardize = MethodType(standardize, None, pd.DataFrame)
    pd.DataFrame.apply_rules = MethodType(apply_rules, None, pd.DataFrame)
    pd.DataFrame.generate_report = MethodType(generate_report, None, pd.DataFrame)
    pd.DataFrame.export_report = MethodType(export_report, None, pd.DataFrame)



def validate_and_correct(dataframe: pd.DataFrame, rules: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate and correct a dataset based on specified rules.

    Args:
        dataframe (pd.DataFrame): The dataset to be validated and corrected.
        rules (Dict[str, str]): A dictionary of validation rules defining conditions to check against the data.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: A tuple containing the corrected DataFrame and a report of the validation and correction processes.
    """

    # Initialize validator and perform validation
    validator = DataValidator(dataframe)
    missing_values = validator.check_missing_values()
    outliers = validator.detect_outliers()
    inconsistencies = validator.find_inconsistencies(rules)

    validation_report = {
        "missing_values": missing_values,
        "outliers": outliers,
        "inconsistencies": inconsistencies
    }

    # Initialize corrector and perform corrections
    corrector = DataCorrector(dataframe)
    df_corrected = corrector.impute_missing_values()
    df_corrected = corrector.handle_outliers()

    # Compile a comprehensive report
    complete_report = {
        "validation_report": validation_report,
        "corrections": "Missing values imputed and outliers handled."
    }

    return df_corrected, complete_report
