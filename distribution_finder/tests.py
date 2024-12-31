from distribution_fitter import DistributionFitter
from distribution_sampler import DistributionSampler
from main_module import main  # Assuming main function is located in a file named main_module.py
from plot_distribution_fit import plot_distribution_fit
import numpy as np
import pytest


def test_empty_data():
    with pytest.raises(ValueError, match="Input data array is empty."):
        DistributionFitter(np.array([]))

def test_non_numeric_data():
    with pytest.raises(ValueError, match="Input data contains non-numeric values."):
        DistributionFitter(np.array(['a', 'b', 'c']))

def test_zero_variance_data():
    with pytest.raises(ValueError, match="Input data array has zero variance."):
        DistributionFitter(np.array([1, 1, 1]))

def test_missing_values_handling():
    df = DistributionFitter(np.array([1, np.nan, 3, 4]))
    assert not np.isnan(df.data).any(), "Missing values were not handled."

def test_fit_distribution_normal():
    data = np.random.normal(loc=0, scale=1, size=1000)
    df = DistributionFitter(data)
    best_fit = df.fit_distribution()
    assert best_fit == 'norm', f"Expected 'norm', but got '{best_fit}'."

def test_fit_distribution_exponential():
    data = np.random.exponential(scale=1.0, size=1000)
    df = DistributionFitter(data)
    best_fit = df.fit_distribution()
    assert best_fit == 'expon', f"Expected 'expon', but got '{best_fit}'."

def test_fit_distribution_uniform():
    data = np.random.uniform(low=0.0, high=1.0, size=1000)
    df = DistributionFitter(data)
    best_fit = df.fit_distribution()
    assert best_fit == 'uniform', f"Expected 'uniform', but got '{best_fit}'."



def test_invalid_distribution():
    with pytest.raises(ValueError, match="Distribution 'invalid' is not recognized."):
        DistributionSampler("invalid", {})

def test_invalid_sample_size():
    sampler = DistributionSampler("norm", {"loc": 0, "scale": 1})
    with pytest.raises(ValueError, match="Size must be a positive integer."):
        sampler.generate_samples(0)

def test_normal_distribution():
    sampler = DistributionSampler("norm", {"loc": 0, "scale": 1})
    samples = sampler.generate_samples(100)
    assert len(samples) == 100
    assert isinstance(samples, np.ndarray)

def test_exponential_distribution():
    sampler = DistributionSampler("expon", {"scale": 1.0})
    samples = sampler.generate_samples(50)
    assert len(samples) == 50
    assert isinstance(samples, np.ndarray)

def test_runtime_error_incorrect_params():
    sampler = DistributionSampler("norm", {"loc": 0})  # Missing 'scale' parameter
    with pytest.raises(RuntimeError):
        sampler.generate_samples(10)



def test_plot_distribution_fit_invalid_distribution():
    data = np.random.normal(0, 1, 1000)
    with pytest.raises(ValueError, match="Distribution 'invalid' is not supported for plotting."):
        plot_distribution_fit(data, 'invalid', {})

def test_plot_distribution_fit_normal():
    data = np.random.normal(0, 1, 1000)
    try:
        plot_distribution_fit(data, 'norm', {'loc': 0, 'scale': 1})
    except Exception as e:
        pytest.fail(f"plot_distribution_fit raised an exception unexpectedly: {e}")

def test_plot_distribution_fit_exponential():
    data = np.random.exponential(1.0, 1000)
    try:
        plot_distribution_fit(data, 'expon', {'loc': 0, 'scale': 1})
    except Exception as e:
        pytest.fail(f"plot_distribution_fit raised an exception unexpectedly: {e}")

def test_plot_distribution_fit_uniform():
    data = np.random.uniform(0.0, 1.0, 1000)
    try:
        plot_distribution_fit(data, 'uniform', {'loc': 0, 'scale': 1})
    except Exception as e:
        pytest.fail(f"plot_distribution_fit raised an exception unexpectedly: {e}")



def test_main_normal_distribution():
    data = np.random.normal(loc=0, scale=1, size=1000)
    result = main(data)
    assert result['best_distribution'] == 'norm', "Expected 'norm' for normally distributed data."
    assert 'loc' in result['params'] and 'scale' in result['params'], "Params must include 'loc' and 'scale'."
    assert len(result['samples']) == 1000, "Samples size must be 1000."

def test_main_exponential_distribution():
    data = np.random.exponential(scale=1.0, size=1000)
    result = main(data)
    assert result['best_distribution'] == 'expon', "Expected 'expon' for exponentially distributed data."
    assert 'loc' in result['params'] and 'scale' in result['params'], "Params must include 'loc' and 'scale'."
    assert len(result['samples']) == 1000, "Samples size must be 1000."

def test_main_uniform_distribution():
    data = np.random.uniform(low=0.0, high=1.0, size=1000)
    result = main(data)
    assert result['best_distribution'] == 'uniform', "Expected 'uniform' for uniformly distributed data."
    assert 'loc' in result['params'] and 'scale' in result['params'], "Params must include 'loc' and 'scale'."
    assert len(result['samples']) == 1000, "Samples size must be 1000."

def test_main_invalid_data():
    with pytest.raises(ValueError, match="Input data array is empty."):
        main(np.array([]))

def test_main_with_nans():
    data_with_nans = np.array([1, 2, np.nan, 4, 5])
    try:
        result = main(data_with_nans)
        assert isinstance(result, dict), "Result should be a dictionary."
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")
