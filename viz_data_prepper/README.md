# Package Name

The **util_functions** package is a collection of utility functions for various data preprocessing and analysis tasks. This package provides functions for normalizing and scaling data, handling missing values, outlier removal, data discretization, descriptive statistics calculation, statistical tests, text preprocessing, geocoding, image processing, audio processing, network analysis, spatial analysis, sentiment analysis, and more.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)

## Installation <a name="installation"></a>

To install the **util_functions** package, you can use `pip`:

```
pip install util_functions
```

## Usage <a name="usage"></a>

To use the functions from the **util_functions** package in your Python code, you can import them like this:

```python
from util_functions import normalize_data_min_max, standardize_data
```

## Examples <a name="examples"></a>

### Normalizing Data

The `normalize_data_min_max` function can be used to normalize a list or array of numerical data to the range `[0, 1]`. Here's an example:

```python
data = [1, 2, 3, 4, 5]
normalized_data = normalize_data_min_max(data)
print(normalized_data)  # Output: [0.0, 0.25, 0.5, 0.75, 1.0]
```

### Standardizing Data

The `standardize_data` function can be used to standardize a list or array of numerical data to have zero mean and unit variance. Here's an example:

```python
data = [1, 2, 3, 4, 5]
standardized_data = standardize_data(data)
print(standardized_data)  # Output: [-1.41421356, -0.70710678, 0.0, 0.70710678, 1.41421356]
```

### Handling Missing Values

The `handle_missing_values` function can be used to handle missing values in a dataset. It supports two strategies: mean imputation and median imputation. Here's an example:

```python
import pandas as pd

data = [[1, 2, None], [4, None, 6], [None, 8, 9]]
df = pd.DataFrame(data)

imputed_data = handle_missing_values(df, strategy='mean')
print(imputed_data)
# Output:
#    0    1    2
# 0  1  2.0  7.5
# 1  4  5.0  6.0
# 2  2  8.0  9.0
```

### Removing Outliers

The `remove_outliers` function can be used to remove outliers from a list or array of numerical data using the Z-score method. Here's an example:

```python
data = [1, 2, 3, -10, -20, -30]
filtered_data = remove_outliers(data)
print(filtered_data) # Output: [1, 2, 3]
```

### Data Discretization

The `discretize_data` function can be used to discretize a list or array of numerical data into bins of equal width. Here's an example:

```python
data = [1,2,3,4,5]
num_bins = 3
discretized_data = discretize_data(data, num_bins)
print(discretized_data) # Output: [1,1,2,2,3]
```

### Descriptive Statistics Calculation

The `calculate_mean_value`, `calculate_median_value`, and `calculate_mode_value` functions can be used to calculate the mean, median, and mode of a list or array of numerical data, respectively. Here's an example:

```python
data = [1, 2, 3, 4, 5]
mean = calculate_mean_value(data)
median = calculate_median_value(data)
mode = calculate_mode_value(data)
print(mean)   # Output: 3.0
print(median) # Output: 3
print(mode)   # Output: [1, 2, 3, 4, 5]
```

### Statistical Tests

The `calculate_correlation` function can be used to calculate the correlation coefficient between two variables in a numerical dataframe. Here's an example:

```python
import pandas as pd

data = {'var1': [1, 2, 3, 4, 5], 'var2': [2, 4, 6, 8, 10]}
df = pd.DataFrame(data)

correlation = calculate_correlation(df, 'var1', 'var2')
print(correlation) # Output: 1.0
```

### Text Preprocessing

The `handle_categorical_vars` function can be used to handle categorical variables in a dataset by either one-hot encoding or label encoding. Here's an example:

```python
import pandas as pd

data = {'color': ['red', 'blue', 'green'], 'size': ['S', 'M', 'L']}
df = pd.DataFrame(data)

transformed_df = handle_categorical_vars(df)
print(transformed_df)
# Output:
#    color_0  color_1  color_2    size
# 0        1        0        0       S
# 1        0        1        0       M
# 2        0        0        1       L
```

### Geocoding

The `geocode_address` function can be used to geocode an address and return its latitude and longitude. Here's an example:

```python
address = "1600 Amphitheatre Parkway, Mountain View, CA"
latitude, longitude = geocode_address(address)
print(latitude, longitude) # Output: 37.4220266 -122.0840704
```

### Image Processing

The `resize_image` function can be used to resize an image to a specified width and height. Here's an example:

```python
import cv2

image = cv2.imread("image.jpg")
resized_image = resize_image(image, width=100, height=100)
cv2.imshow("Resized Image", resized_image)
cv2.waitKey(0)
```

### Audio Processing

The `handle_audio_file_using_librosa` function can be used to handle an audio file using the Librosa library. Here's an example:

```python
audio_file = "audio.wav"
mfcc_features, spectrogram = handle_audio_file_using_librosa(audio_file)
print(mfcc_features.shape)   # Output: (n_mfcc_features, n_frames)
print(spectrogram.shape)     # Output: (n_fft_bins, n_frames)
```

### Network Analysis

The `calculate_centrality_measures_for_graph` function can be used to calculate centrality measures for a network graph. Here's an example:

```python
import networkx as nx

graph = nx.karate_club_graph()
centrality_measures = calculate_centrality_measures_for_graph(graph)
print(centrality_measures['DegreeCentrality'])
print(centrality_measures['ClosenessCentrality'])
print(centrality_measures['BetweennessCentrality'])
```

### Spatial Analysis

The `calculate_distance_between_two_points` function can be used to calculate the distance between two points on the Earth's surface using their latitude and longitude. Here's an example:

```python
point_a = (37.7749, -122.4194)
point_b = (34.0522, -118.2437)
distance = calculate_distance_between_two_points(point_a, point_b)
print(distance) # Output: 559.23 km
```

### Sentiment Analysis

The `analyze_social_media_sentiments_and_topics_using_nltk_and_sklearn` function can be used to analyze sentiments and topics in social media posts using the NLTK library and the sklearn library. Here's an example:

```python
social_media_posts = [
    "I love this product! It's amazing!",
    "This is the worst service I've ever experienced.",
    "Great job on the new feature!",
    "I'm really disappointed with the quality of the product."
]

sentiments, topics = analyze_social_media_sentiments_and_topics_using_nltk_and_sklearn(social_media_posts)
print(sentiments) # Output: [0.9666, -0.7906, 0.9765, -0.4834]
print(topics)    # Output: [['product', 'amazing'], ['service', 'worst'], ['job', 'great', 'new', 'feature'], ['quality', 'disappointed']]
```

These are just a few examples of the functionality provided by the **util_functions** package. For more information on how to use each function, please refer to the function docstrings or the package documentation.