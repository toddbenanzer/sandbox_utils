from io import StringIO
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from unittest.mock import patch
from your_module_path import DatasetManager  # Replace with actual import path
from your_module_path import EngagementAnalyzer  # Replace with your actual module path
from your_module_path import Evaluator, perform_cross_validation, perform_ab_testing  # Replace with your module path
from your_module_path import RecommendationEngine, CollaborativeFiltering, ContentBasedFiltering, HybridApproach
from your_module_path import Visualizer  # Replace with the actual module path
from your_module_path import setup_logging, load_config  # Replace with the actual module path
import json
import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
import pytest


@pytest.fixture
def sample_csv():
    data = """feature1,feature2,feature3
               1,A,100
               2,B,200
               2,B,200
               3,C,300
               ,D,400"""
    return StringIO(data)

@pytest.fixture
def dataset_manager(sample_csv):
    dm = DatasetManager(sample_csv)
    dm.load_data()
    return dm

def test_load_data(dataset_manager):
    assert isinstance(dataset_manager.data, pd.DataFrame)
    assert not dataset_manager.data.empty

def test_clean_data(dataset_manager):
    cleaned_data = dataset_manager.clean_data()
    assert cleaned_data.isnull().sum().sum() == 0
    assert len(cleaned_data) == 4  # After removing duplicates, initial 5 entries

def test_preprocess_data(dataset_manager):
    dataset_manager.clean_data()
    preprocessed_data = dataset_manager.preprocess_data()

    # Check that numeric columns are scaled
    numeric_features = ['feature3']
    for feature in numeric_features:
        assert preprocessed_data[feature].mean() < 1e-6  # Mean should be close to zero after scaling
        assert abs(preprocessed_data[feature].std() - 1) < 1e-6  # Std should be close to 1 after scaling

    # Check that categorical columns are one-hot encoded
    assert 'feature2_B' in preprocessed_data.columns
    assert 'feature2_C' in preprocessed_data.columns
    assert 'feature2_D' in preprocessed_data.columns



@pytest.fixture
def setup_recommendation_engine():
    engine = RecommendationEngine()
    cf_algo = CollaborativeFiltering()
    cbf_algo = ContentBasedFiltering()
    engine.add_algorithm("collaborative", cf_algo)
    engine.add_algorithm("content_based", cbf_algo)
    return engine

def test_add_algorithm(setup_recommendation_engine):
    engine = setup_recommendation_engine
    assert "collaborative" in engine.algorithms
    assert "content_based" in engine.algorithms
    assert isinstance(engine.algorithms["collaborative"], CollaborativeFiltering)
    assert isinstance(engine.algorithms["content_based"], ContentBasedFiltering)

def test_configure_algorithm(setup_recommendation_engine):
    engine = setup_recommendation_engine
    config = {"param1": "value1"}
    engine.configure_algorithm("collaborative", config)
    assert engine.algorithms["collaborative"].param1 == "value1"

def test_generate_recommendations(setup_recommendation_engine):
    engine = setup_recommendation_engine
    recommendations = engine.generate_recommendations(user_id="user1")
    assert set(recommendations) == {"item1", "item2", "item3", "item4", "item5", "item6"}

def test_hybrid_approach():
    hybrid = HybridApproach()
    cf_algo = CollaborativeFiltering()
    cbf_algo = ContentBasedFiltering()
    hybrid.integrate_algorithms([cf_algo, cbf_algo])
    recommendations = hybrid.generate_recommendations(user_id="user1")
    assert set(recommendations) == {"item1", "item2", "item3", "item4", "item5", "item6"}



@pytest.fixture
def sample_evaluator():
    recommendations = ['item1', 'item2', 'item3', 'item4']
    ground_truth = ['item2', 'item3', 'item5']
    return Evaluator(recommendations, ground_truth)

def test_calculate_precision(sample_evaluator):
    evaluator = sample_evaluator
    precision = evaluator.calculate_precision()
    assert precision == 0.5  # 2 relevant recommendations out of 4

def test_calculate_recall(sample_evaluator):
    evaluator = sample_evaluator
    recall = evaluator.calculate_recall()
    assert recall == 2/3  # 2 relevant recommendations out of 3 ground truth

def test_calculate_f1_score(sample_evaluator):
    evaluator = sample_evaluator
    f1_score = evaluator.calculate_f1_score()
    assert f1_score == 2 * (0.5 * 2/3) / (0.5 + 2/3)  # Harmonic mean of precision and recall

def test_calculate_mean_squared_error():
    evaluator = Evaluator([], [])
    predicted_scores = [3.5, 4.0, 2.0, 5.0]
    true_scores = [3.0, 4.5, 2.0, 5.0]
    mse = evaluator.calculate_mean_squared_error(predicted_scores, true_scores)
    assert mse == ((0.5)**2 + (-0.5)**2 + (0.0)**2 + (0.0)**2) / 4

def test_perform_cross_validation():
    # Placeholder test for cross-validation
    example_algorithm = MockAlgorithm()  # You might need to create a mock class for testing
    data = ['dummy_data']
    results = perform_cross_validation(example_algorithm, data, n_splits=5)
    assert 'precision' in results
    assert 'recall' in results
    assert 'f1_score' in results

def test_perform_ab_testing():
    # Placeholder test for A/B testing
    algorithm_a = MockAlgorithm()  # You might need to create a mock class for testing
    algorithm_b = MockAlgorithm()
    data = ['dummy_data']
    results = perform_ab_testing(algorithm_a, algorithm_b, data)
    assert 'algorithm_a' in results
    assert 'algorithm_b' in results
    assert 'precision' in results['algorithm_a']
    assert 'recall' in results['algorithm_b']



@pytest.fixture
def sample_user_data():
    return {
        'user1': {'views': 10, 'clicks': 2, 'time_spent': 30},
        'user2': {'views': 15, 'clicks': 3, 'time_spent': 45},
        'user3': {'views': 5, 'clicks': 1, 'time_spent': 20},
    }

@pytest.fixture
def empty_user_data():
    return {}

def test_track_click_through_rate(sample_user_data):
    analyzer = EngagementAnalyzer(sample_user_data)
    ctr = analyzer.track_click_through_rate()
    assert ctr == pytest.approx(25.0)  # 6 clicks out of 30 views

def test_track_click_through_rate_empty(empty_user_data):
    analyzer = EngagementAnalyzer(empty_user_data)
    ctr = analyzer.track_click_through_rate()
    assert ctr == 0.0

def test_track_time_spent(sample_user_data):
    analyzer = EngagementAnalyzer(sample_user_data)
    average_time = analyzer.track_time_spent()
    assert average_time == pytest.approx(31.6666667)  # (30 + 45 + 20) / 3

def test_track_time_spent_empty(empty_user_data):
    analyzer = EngagementAnalyzer(empty_user_data)
    average_time = analyzer.track_time_spent()
    assert average_time == 0.0

def test_generate_engagement_report(sample_user_data):
    analyzer = EngagementAnalyzer(sample_user_data)
    report = analyzer.generate_engagement_report()
    assert report['click_through_rate'] == pytest.approx(25.0)
    assert report['average_time_spent'] == pytest.approx(31.6666667)
    assert report['total_users'] == 3

def test_generate_engagement_report_empty(empty_user_data):
    analyzer = EngagementAnalyzer(empty_user_data)
    report = analyzer.generate_engagement_report()
    assert report['click_through_rate'] == 0.0
    assert report['average_time_spent'] == 0.0
    assert report['total_users'] == 0



@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'Category': ['A', 'B', 'C', 'D'],
        'Values': [10, 20, 30, 40],
        'Time': ['Q1', 'Q2', 'Q3', 'Q4'],
        'Metrics': [1, 3, 2, 4]
    })

def test_create_bar_chart(sample_data):
    visualizer = Visualizer(sample_data)
    # Instead of displaying, we use a non-interactive backend to test
    plt.switch_backend('Agg')
    visualizer.create_bar_chart(x_column='Category', y_column='Values', title='Test Bar Chart')
    assert plt.gca().get_title() == 'Test Bar Chart'
    assert plt.gca().get_xlabel() == 'Category'
    assert plt.gca().get_ylabel() == 'Values'
    plt.close()

def test_create_line_graph(sample_data):
    visualizer = Visualizer(sample_data)
    # Use a non-interactive backend for testing
    plt.switch_backend('Agg')
    visualizer.create_line_graph(x_column='Time', y_column='Metrics', title='Test Line Graph')
    assert plt.gca().get_title() == 'Test Line Graph'
    assert plt.gca().get_xlabel() == 'Time'
    assert plt.gca().get_ylabel() == 'Metrics'
    plt.close()



def test_setup_logging():
    with patch('logging.basicConfig') as mock_logging:
        setup_logging('DEBUG')
        mock_logging.assert_called_with(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    with pytest.raises(ValueError):
        setup_logging('INVALID')

def test_load_config(tmp_path):
    # Setup a temporary JSON config file
    config_data = {"key1": "value1", "key2": "value2"}
    config_file = tmp_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config_data, f)

    # Test loading config
    loaded_config = load_config(str(config_file))
    assert loaded_config == config_data

    # Test for FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_config.json")

    # Test for JSONDecodeError
    bad_json_file = tmp_path / "bad_config.json"
    with open(bad_json_file, 'w') as f:
        f.write("{bad json}")

    with pytest.raises(json.JSONDecodeError):
        load_config(str(bad_json_file))
