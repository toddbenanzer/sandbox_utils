umpy as np
import pytest
from scipy.stats import norm, expon, gamma

def check_zero_variance(data):
    return np.var(data) == 0

def check_constant_values(data):
    return np.unique(data).size == 1

def handle_missing_values(data):
    return data

def handle_zeroes(data):
    epsilon = np.finfo(float).eps
    data[data == 0] = epsilon
    return data

def calculate_statistics(data):
    statistics = {
        'mean': np.mean(data),
        'median': np.median(data),
        'mode': stats.mode(data).mode[0],
        'standard_deviation': np.std(data),
        'variance': np.var(data)
    }
    return statistics

def fit_distribution(data):
    if check_constant_values(data):
        return "Constant Distribution"
    if check_zero_variance(data):
        return "Zero Distribution"
    
    result = {}
    
    if len(np.unique(data)) <= 2:
        result["constant"] = {
            "params": (),
            "pdf": None,
            "cdf": None
        }
    
    try:
        params = norm.fit(data)
        result["normal"] = {
            "params": params,
            "pdf": norm.pdf,
            "cdf": norm.cdf
        }
    except:
        pass
    
    try:
        params = gamma.fit(data)
        result["gamma"] = {
            "params": params,
            "pdf": gamma.pdf,
            "cdf": gamma.cdf
        }
    except:
        pass
    
    try:
        params = expon.fit(data)
        result["exponential"] = {
            "params": params,
            "pdf": expon.pdf,
            "cdf": expon.cdf
        }
    except:
        pass
    
    return result

def select_best_distribution(data):
    distributions = fit_distribution(data)
    
    if "constant" in distributions:
        return "Constant Distribution"
    if "normal" in distributions:
        return "Normal Distribution"
    if "gamma" in distributions:
        return "Gamma Distribution"
    
    return "Other Distribution"

def generate_random_samples(data):
    if check_zero_variance(data):
        raise ValueError("Data has zero variance")
    if check_constant_values(data):
        raise ValueError("Data has constant values")
    if np.isnan(data).any():
        raise ValueError("Data contains missing values")
    
    samples = np.random.choice(data, size=len(data))
    return samples

@pytest.fixture
def random_data():
    return np.random.randn(1000)

def plot_histogram(data):
    plt.hist(data, bins='auto')
    plt.show()

def plot_density(data, distribution):
    x = np.linspace(min(data), max(data), 100)
    y = distribution.pdf(x)
    plt.plot(x, y)
    plt.show()

@pytest.mark.parametrize(
    "data",
    [
        np.ones(1000),
        np.zeros(1000),
        np.random.randn(1000),
        np.concatenate((np.random.randn(500), [np.nan], np.zeros(500))),
        np.random.randn(1000),
        np.random.exponential(scale=1, size=1000)
    ]
)
def test_fit_distribution(random_data, data):
    distribution = fit_distribution(data)
    
    if check_constant_values(data):
        assert distribution == "Constant Distribution"
    
    elif check_zero_variance(data):
        assert distribution == "Zero Distribution"
        
    else:
        assert isinstance(distribution, dict)

@pytest.mark.parametrize(
    "data",
    [
        np.array([1, 1, 1, 1]),
        np.array([1, 1, 1]),
        random_data()
    ]
)
def test_select_best_distribution(random_data, data):
    distribution = select_best_distribution(data)
    
    if check_constant_values(data):
        assert distribution == "Constant Distribution"
        
    elif check_zero_variance(data):
        assert distribution == "Zero Distribution"
    
    elif distribution in ["Normal", "Gamma"]:
        assert distribution in ["Normal Distribution", "Gamma Distribution"]
    
    else:
        assert distribution == "Other Distribution"

@pytest.mark.parametrize(
    "data",
    [
        np.ones(1000),
        np.zeros(1000),
        random_data(),
        np.repeat(1, 100),
        np.arange(100),
        np.concatenate((random_data(), [np.nan], random_data()))
    ]
)
def test_generate_random_samples(random_data, data):
    if check_zero_variance(data) or check_constant_values(data) or np.isnan(data).any():
        with pytest.raises(ValueError):
            generate_random_samples(data)
    else:
        samples = generate_random_samples(data)
        assert len(samples) == len(data)

@pytest.mark.parametrize("data", [random_data()])
def test_plot_histogram(random_data, data):
    plot_histogram(data)

@pytest.mark.parametrize("data", [random_data()])
def test_plot_density(random_data, data):
    plot_density(data, norm()