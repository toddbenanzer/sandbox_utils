from data_corrector_module import DataCorrector  # Assuming the class is saved in a module named data_corrector_module
from data_standardizer_module import DataStandardizer  # Assuming the class is saved in a module named data_standardizer_module
from data_validator_module import DataValidator  # Assuming the class is saved in a module named data_validator_module
from unittest.mock import MagicMock
from validation_and_correction_module import validate_and_correct  # Assuming the function is in a module named validation_and_correction_module
from validation_module import integrate_with_pandas, ValidationRuleManager  # Assuming the integrate function and needed classes are in validation_module
from validation_report_module import ValidationReport  # Assuming the class is saved in a module named validation_report_module
from validation_rule_manager import ValidationRuleManager  # Assuming the class is saved in a module named validation_rule_manager
import json
import numpy as np
import os
import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    data = {
        'A': [1, 2, np.nan, 4],
        'B': [1, 1, 1, 100],
        'C': ['a', 'a', 'b', 'c']
    }
    return pd.DataFrame(data)

def test_check_missing_values(sample_data):
    validator = DataValidator(sample_data)
    missing_values = validator.check_missing_values()
    
    assert missing_values['A'] == 1
    assert missing_values['B'] == 0
    assert missing_values['C'] == 0

def test_detect_outliers_z_score(sample_data):
    validator = DataValidator(sample_data)
    outliers = validator.detect_outliers(method="z-score", threshold=1.5)
    
    assert 'B' in outliers
    assert outliers['B'].iloc[0] == 100

def test_detect_outliers_iqr(sample_data):
    validator = DataValidator(sample_data)
    outliers = validator.detect_outliers(method="IQR")
    
    assert 'B' in outliers
    assert outliers['B'].iloc[0] == 100

def test_find_inconsistencies(sample_data):
    validator = DataValidator(sample_data)
    rules = {'C': "C == 'a'"}  # Fake rule to find inconsistencies
    inconsistencies = validator.find_inconsistencies(rules)
    
    assert len(inconsistencies) == 2  # Two records do not meet the condition



@pytest.fixture
def test_data():
    data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, np.nan, 8, 2],
        'C': ['a', 'b', 'b', 'a', 'b']
    }
    return pd.DataFrame(data)

def test_impute_missing_values_mean(test_data):
    corrector = DataCorrector(test_data)
    imputed_df = corrector.impute_missing_values(method="mean")
    assert imputed_df['A'].isnull().sum() == 0
    assert imputed_df['B'].isnull().sum() == 0

def test_impute_missing_values_median(test_data):
    corrector = DataCorrector(test_data)
    imputed_df = corrector.impute_missing_values(method="median")
    assert imputed_df['A'].isnull().sum() == 0
    assert imputed_df['B'].isnull().sum() == 0

def test_impute_missing_values_mode(test_data):
    corrector = DataCorrector(test_data)
    imputed_df = corrector.impute_missing_values(method="mode")
    assert imputed_df['A'].isnull().sum() == 0
    assert imputed_df['B'].isnull().sum() == 0

def test_handle_outliers_remove(test_data):
    corrector = DataCorrector(test_data)
    df_no_outliers = corrector.handle_outliers(method="remove")
    assert not df_no_outliers['A'].eq(100).any()

def test_handle_outliers_clip(test_data):
    corrector = DataCorrector(test_data)
    df_clipped = corrector.handle_outliers(method="clip")
    assert df_clipped['A'].max() <= test_data['A'].quantile(0.75)
    assert df_clipped['A'].min() >= test_data['A'].quantile(0.25)



@pytest.fixture
def sample_data():
    data = {
        'dates': ['2020-01-01', '01/02/2021', 'March 3, 2022', 'Invalid Date'],
        'currency': ['$100.00', '$200.99', '300', '$400.50'],
        'percentages': ['10%', '20%', '30%', '40%']
    }
    return pd.DataFrame(data)

def test_standardize_date_format(sample_data):
    standardizer = DataStandardizer(sample_data)
    df_standardized = standardizer.standardize_column_format("dates", "date")
    assert pd.to_datetime('2020-01-01') == df_standardized['dates'][0]
    assert pd.to_datetime('2021-01-02') == df_standardized['dates'][1]
    assert pd.to_datetime('2022-03-03') == df_standardized['dates'][2]
    assert pd.isna(df_standardized['dates'][3])

def test_standardize_currency_format(sample_data):
    standardizer = DataStandardizer(sample_data)
    df_standardized = standardizer.standardize_column_format("currency", "currency")
    assert df_standardized['currency'].iloc[0] == 100.00
    assert df_standardized['currency'].iloc[1] == 200.99
    assert df_standardized['currency'].iloc[2] == 300.00
    assert df_standardized['currency'].iloc[3] == 400.50

def test_standardize_percentage_format(sample_data):
    standardizer = DataStandardizer(sample_data)
    df_standardized = standardizer.standardize_column_format("percentages", "percentage")
    assert df_standardized['percentages'].iloc[0] == 0.10
    assert df_standardized['percentages'].iloc[1] == 0.20
    assert df_standardized['percentages'].iloc[2] == 0.30
    assert df_standardized['percentages'].iloc[3] == 0.40

def test_invalid_column(sample_data):
    standardizer = DataStandardizer(sample_data)
    with pytest.raises(ValueError, match="Column 'invalid' does not exist in the DataFrame."):
        standardizer.standardize_column_format("invalid", "date")

def test_unsupported_format_type(sample_data):
    standardizer = DataStandardizer(sample_data)
    with pytest.raises(ValueError, match="Unsupported format type 'unsupported'. Supported formats: 'date', 'currency', 'percentage'."):
        standardizer.standardize_column_format("dates", "unsupported")



@pytest.fixture
def rule_manager():
    return ValidationRuleManager()

def test_define_rule(rule_manager):
    rule_manager.define_rule("rule1", "value > 10")
    assert "rule1" in rule_manager.rules
    assert rule_manager.rules["rule1"] == "value > 10"

def test_define_existing_rule_overwrites(rule_manager):
    rule_manager.define_rule("rule1", "value > 10")
    rule_manager.define_rule("rule1", "value > 20")
    assert rule_manager.rules["rule1"] == "value > 20"

def test_save_rules_to_file(rule_manager, tmp_path):
    rule_manager.define_rule("rule1", "value > 10")
    rule_manager.define_rule("rule2", "value < 5")
    file_path = tmp_path / "rules.json"
    rule_manager.save_rules(file_path)

    with open(file_path, 'r') as file:
        saved_rules = json.load(file)

    assert saved_rules == rule_manager.rules

def test_load_rules_from_file(rule_manager, tmp_path):
    file_path = tmp_path / "rules.json"
    rules_data = {"rule1": "value > 10", "rule2": "value < 5"}
    
    with open(file_path, 'w') as file:
        json.dump(rules_data, file)
        
    rule_manager.load_rules(file_path)

    assert rule_manager.rules == rules_data

def test_load_rules_file_not_found(rule_manager):
    with pytest.raises(FileNotFoundError):
        rule_manager.load_rules("non_existent_file.json")

def test_load_rules_invalid_json(rule_manager, tmp_path):
    file_path = tmp_path / "invalid.json"
    with open(file_path, 'w') as file:
        file.write("{ not: a valid json }")
    
    with pytest.raises(json.JSONDecodeError):
        rule_manager.load_rules(file_path)



@pytest.fixture
def sample_data():
    return [
        {'field1': 'value1', 'field2': 'value2'},
        {'field1': 'value3', 'field2': 'value4'}
    ]

def test_generate_summary_text(sample_data):
    report = ValidationReport(sample_data)
    summary = report.generate_summary(output_format="text")
    assert isinstance(summary, str)
    assert "value1" in summary

def test_generate_summary_html(sample_data):
    report = ValidationReport(sample_data)
    summary = report.generate_summary(output_format="html")
    assert isinstance(summary, str)
    assert summary.startswith("<table")
    assert "value1" in summary

def test_generate_summary_invalid_format(sample_data):
    report = ValidationReport(sample_data)
    with pytest.raises(ValueError, match="Unsupported output format 'invalid'. Supported formats: 'text', 'html'."):
        report.generate_summary(output_format="invalid")

def test_export_report_csv(sample_data, tmp_path):
    report = ValidationReport(sample_data)
    file_path = tmp_path / "report.csv"
    report.export_report(file_path, format="csv")
    
    df = pd.read_csv(file_path)
    assert len(df) == len(sample_data)

def test_export_report_json(sample_data, tmp_path):
    report = ValidationReport(sample_data)
    file_path = tmp_path / "report.json"
    report.export_report(file_path, format="json")
    
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    
    assert len(data) == len(sample_data)

def test_export_report_xlsx(sample_data, tmp_path):
    report = ValidationReport(sample_data)
    file_path = tmp_path / "report.xlsx"
    report.export_report(file_path, format="xlsx")
    
    df = pd.read_excel(file_path)
    assert len(df) == len(sample_data)

def test_export_report_invalid_format(sample_data):
    report = ValidationReport(sample_data)
    with pytest.raises(ValueError, match="Unsupported export format 'xml'. Supported formats: 'csv', 'json', 'xlsx'."):
        report.export_report("report.xml", format="xml")



@pytest.fixture(scope='module', autouse=True)
def setup_module():
    integrate_with_pandas()

@pytest.fixture
def sample_dataframe():
    data = {
        'A': [1, 2, None, 4, 100],
        'B': [5, 6, 1, 8, 2],
        'C': ['a', 'b', 'b', 'a', 'b']
    }
    return pd.DataFrame(data)

def test_validate_method(sample_dataframe):
    result = sample_dataframe.validate()
    assert 'missing_values' in result
    assert 'outliers' in result
    assert 'inconsistencies' in result

def test_correct_method(sample_dataframe):
    df_corrected = sample_dataframe.correct()
    assert df_corrected.isnull().sum().sum() == 0

def test_standardize_method(sample_dataframe):
    df_standardized = sample_dataframe.standardize('A', 'currency')
    assert isinstance(df_standardized['A'].iloc[0], float)

def test_apply_rules_method(sample_dataframe):
    rule_manager = ValidationRuleManager()
    rule_manager.define_rule('rule1', 'A > 0')
    df_with_rules = sample_dataframe.apply_rules(rule_manager)
    assert (df_with_rules['A'] > 0).all()

def test_generate_report_method(sample_dataframe):
    summary = sample_dataframe.generate_report(format="text")
    assert isinstance(summary, str)
    assert "A" in summary

def test_export_report_method(sample_dataframe, tmp_path):
    file_path = tmp_path / "report.csv"
    sample_dataframe.export_report(file_path, format="csv")
    assert file_path.exists()
    df_loaded = pd.read_csv(file_path)
    assert not df_loaded.empty



# Sample DataFrame to use in tests
@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'A': [1, 2, None, 4, 100],
        'B': [5, 6, None, 8, 2],
        'C': ['x', 'y', 'z', 'x', 'y']
    })

# Mock DataValidator for testing
class MockDataValidator:
    def __init__(self, dataframe):
        pass

    def check_missing_values(self):
        return {'A': 1, 'B': 1, 'C': 0}

    def detect_outliers(self):
        return {'A': [100], 'B': []}

    def find_inconsistencies(self, rules):
        return {'A': "No inconsistencies"}

# Mock DataCorrector for testing
class MockDataCorrector:
    def __init__(self, dataframe):
        pass

    def impute_missing_values(self):
        return pd.DataFrame({
            'A': [1, 2, 2.75, 4, 100],  # Assuming mean imputation
            'B': [5, 6, 5.25, 8, 2],
            'C': ['x', 'y', 'z', 'x', 'y']
        })

    def handle_outliers(self):
        return pd.DataFrame({
            'A': [1, 2, 2.75, 4, 4],  # Assuming outlier handling
            'B': [5, 6, 5.25, 8, 2],
            'C': ['x', 'y', 'z', 'x', 'y']
        })

# Patching DataValidator and DataCorrector with Mocks
@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    monkeypatch.setattr('validation_and_correction_module.DataValidator', MockDataValidator)
    monkeypatch.setattr('validation_and_correction_module.DataCorrector', MockDataCorrector)

# Unit tests
def test_validate_and_correct(sample_dataframe):
    rules = {'A': "A > 0"}
    corrected_df, report = validate_and_correct(sample_dataframe, rules)
    
    assert isinstance(corrected_df, pd.DataFrame)
    assert corrected_df['A'].iloc[-1] != 100  # Ensure outlier was handled
    
    assert 'validation_report' in report
    assert 'corrections' in report
    assert report['validation_report']['missing_values'] == {'A': 1, 'B': 1, 'C': 0}
    assert report['corrections'] == "Missing values imputed and outliers handled."
