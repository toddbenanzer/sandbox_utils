# Python Date Utility Package

This package provides a set of utility functions for working with dates in Python. It includes functionality for converting date formats, adding and subtracting days, weeks, months, and years from a date, checking if a year is a leap year, getting the day of the week for a given date, finding the week ending on a Friday for a given date, calculating the difference between two dates in days or weeks, finding the next or previous occurrence of a specific day of the week, comparing if two dates fall on the same day of the week, checking if a meeting end time falls within a specified range, validating if a datetime string is in a valid format, getting the current date and time, rounding valid dates to the nearest end point (e.g. nearest month end), checking if one value is greater than another, sorting values in ascending or descending order, and more.

## Usage

To use this package, you can import it in your Python script as follows:

```python
import date_utils
```

You can then call any of the available functions using the `date_utils` prefix followed by the function name. For example:

```python
date = "2022-01-01"
formatted_date = date_utils.convert_to_yyyymm(date)
print(formatted_date)  # Output: 202201
```

## Examples

### Converting Date Formats

```python
from datetime import datetime
import date_utils

date = "2022-01-01"
formatted_date = date_utils.convert_to_yyyymm(date)
print(formatted_date)  # Output: 202201

date = "2022/01/01"
formatted_date = date_utils.convert_to_yyyy_mm_dd(date)
print(formatted_date)  # Output: 2022-01-01
```

### Adding and Subtracting Days

```python
from datetime import datetime
import date_utils

date = "2022-01-01"
new_date = date_utils.add_days_to_date(date, 5)
print(new_date)  # Output: 2022-01-06

date = "2022-01-01"
new_date = date_utils.subtract_days(date, 5)
print(new_date)  # Output: 2021-12-27
```

### Adding and Subtracting Weeks

```python
from datetime import datetime
import date_utils

date = "2022-01-01"
new_date = date_utils.add_weeks(date, 2)
print(new_date)  # Output: 2022-01-15

date = "2022-01-15"
new_date = date_utils.subtract_weeks(date, 2)
print(new_date)  # Output: 2022-01-01
```

### Adding and Subtracting Months

```python
from datetime import datetime
import date_utils

date = datetime(2022, 1, 1)
new_date = date_utils.add_months(date, 3)
print(new_date)  # Output: 2022-04-01

date = datetime(2022, 4, 1)
new_date = date_utils.subtract_months(date, 3)
print(new_date)  # Output: 2022-01-01
```

### Adding and Subtracting Years

```python
from datetime import datetime
import date_utils

date = "2022-01-01"
new_date = date_utils.add_years(date, 1)
print(new_date)  # Output: 2023-01-01

date = "2023-01-01"
new_date = date_utils.subtract_years(date, 1)
print(new_date)  # Output: 2022-01-01
```

### Checking if a Year is a Leap Year

```python
import date_utils

year = 2024
is_leap_year = date_utils.is_leap_year(year)
print(is_leap_year)  # Output: True

year = 2022
is_leap_year = date_utils.is_leap_year(year)
print(is_leap_year)  # Output: False
```

### Getting the Day of the Week

```python
from datetime import datetime
import date_utils

date = "2022-01-01"
day_of_week = date_utils.get_day_of_week(date)
print(day_of_week)  # Output: Saturday
```

### Finding the Week Ending on a Friday

```python
from datetime import datetime
import date_utils

date = "2022-01-01"
week_ending = date_utils.get_week_ending_on_friday(date)
print(week_ending)  # Output: 2022-01-07
```

### Calculating the Difference Between Two Dates

```python
from datetime import datetime
import date_utils

start_date = "2022-01-01"
end_date = "2022-02-01"
difference_in_days = date_utils.calculate_difference_between_dates(start_date, end_date)
print(difference_in_days)  # Output: 31

start_date = "2022-01-01"
end_date = "2022-02-15"
difference_in_weeks = date_utils.calculate_difference_in_week_between_dates(start_date, end_date)
print(difference_in_weeks)  # Output: 6

start_date = "2022-01-01"
end_date = "2023-01-01"
difference_in_months = date_utils.calculate_difference_in_month(start_date, end_date)
print(difference_in_months)  # Output: 12

start_date = "2022-01-01"
end_date = "2023-01-01"
difference_in_years = date_utils.calculate_difference_in_year(start_date, end_date)
print(difference_in_years)  # Output: 1
```

These are just a few examples of how to use the functions provided by the `date_utils` package. For more information on each function and its parameters, please refer to the function docstrings.