# Marketing Funnel

This package provides functionality for analyzing and visualizing marketing funnels. It includes classes and functions to calculate drop-off rates, conversion rates, and visualize funnel data.

## Overview
A marketing funnel is a series of stages that a user goes through in the customer journey, from initial awareness to conversion and beyond. The MarketingFunnel class represents a marketing funnel and provides methods to update drop-offs, conversions, and set additional metrics. The Funnel class is used to create a new marketing funnel by adding stages.

## Usage
To use this package, follow these steps:

1. Create a new Funnel object.
2. Add stages to the funnel using the `add_stage(stage_name)` method.
3. Create a MarketingFunnel object by passing the stages of the Funnel object as an argument to the constructor.
4. Update drop-offs and conversions using the `update_drop_off(stage_idx, count)` and `update_conversion(stage_idx, count)` methods.
5. Set additional metrics using the `set_metric(metric_name, value)` method.
6. Use various calculation functions provided in the package to analyze the funnel data.
7. Visualize the funnel data using the visualization functions.

## Examples

### Creating a Marketing Funnel

```python
funnel = Funnel()
funnel.add_stage("Awareness")
funnel.add_stage("Interest")
funnel.add_stage("Consideration")
funnel.add_stage("Conversion")
funnel.add_stage("Retention")
funnel.add_stage("Advocacy")

marketing_funnel = MarketingFunnel(funnel.stages)
```

### Updating Drop-offs and Conversions

```python
marketing_funnel.update_drop_off(0)  # Increase drop-offs at stage 0 (Awareness) by 1
marketing_funnel.update_conversion(2)  # Increase conversions at stage 2 (Consideration) by 1
```

### Setting Additional Metrics

```python
marketing_funnel.set_metric("Revenue", 1000)
marketing_funnel.set_metric("ROI", 0.5)
```

### Calculating Conversion Rate

```python
total_users = calculate_total_users(funnel.stages)
converted_users = funnel.stages[3]  # Number of users at stage 3 (Conversion)
conversion_rate = calculate_conversion_rate(3, total_users, converted_users)
print(f"Conversion rate at stage 3: {conversion_rate}%")
```

### Visualizing Funnel Data

#### Drop-offs at Each Stage

```python
visualize_dropoffs(marketing_funnel.drop_offs)
```

![Drop-offs Visualization](dropoffs.png)

#### User Conversions at Each Stage

```python
visualize_user_conversions(funnel.stages, marketing_funnel.conversions)
```

![User Conversions Visualization](user_conversions.png)

#### User Drop-offs and Conversions at Each Stage

```python
visualize_funnel_dropoffs_conversions(funnel.stages, marketing_funnel.drop_offs, marketing_funnel.conversions)
```

![Funnel Drop-offs and Conversions Visualization](funnel_dropoffs_conversions.png)

### Exporting Funnel and Metrics Data

#### Exporting Visualization to Image File

```python
fig = visualize_funnel_dropoffs_conversions(funnel.stages, marketing_funnel.drop_offs, marketing_funnel.conversions)
export_visualization(fig, "funnel_visualization.png")
```

#### Saving Funnel to Pickle File

```python
save_funnel(funnel.stages, "funnel.pickle")
```

#### Loading Funnel from Pickle File

```python
loaded_funnel = load_funnel_from_pickle("funnel.pickle")
print(loaded_funnel)
```

#### Exporting Funnel to JSON String

```python
json_data = export_funnel_to_json(funnel.stages)
print(json_data)
```

#### Importing Funnel from JSON String

```python
imported_funnel = import_funnel_from_json(json_data)
print(imported_funnel)
```

#### Exporting Metrics to CSV File

```python
metrics = {
    "Awareness": {"drop-offs": 10, "conversions": 5},
    "Interest": {"drop-offs": 8, "conversions": 3},
    "Consideration": {"drop-offs": 12, "conversions": 6},
}
export_metrics_to_csv(metrics, "funnel_metrics.csv")
```