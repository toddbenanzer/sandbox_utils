# Date Manipulation Package

## Overview
The Date Manipulation package provides functions for manipulating dates in various ways. It includes functions for adding or subtracting days, weeks, months, and years from a given date. It also includes functions for converting dates between different formats and performing common date calculations.

The package is implemented in Python and requires the `datetime` module.

## Usage
To use the Date Manipulation package, you can import it into your Python script or interactive session using the following code:

```python
from date_manipulation import *
```

### Functions

#### `add_days(date_str: str, num_days: int) -> str`

This function adds the specified number of days to a given date.

Arguments:
- `date_str` (str): The date in 'yyyy-mm-dd' format.
- `num_days` (int): The number of days to add.

Returns:
- `str`: The new date after adding the specified number of days in 'yyyy-mm-dd' format.

Example:

```python
new_date = add_days('2022-01-01', 7)
print(new_date)  # Output: '2022-01-08'
```

#### `subtract_days(date_str: str, num_days: int) -> str`

This function subtracts the specified number of days from a given date.

Arguments:
- `date_str` (str): The date in 'yyyy-mm-dd' format.
- `num_days` (int): The number of days to subtract.

Returns:
- `str`: The new date after subtracting the specified number of days in 'yyyy-mm-dd' format.

Example:

```python
new_date = subtract_days('2022-01-08', 7)
print(new_date)  # Output: '2022-01-01'
```

#### `add_weeks(date: datetime, num_weeks: int) -> datetime`

This function adds the specified number of weeks to a given date.

Arguments:
- `date` (datetime): The initial date.
- `num_weeks` (int): The number of weeks to add.

Returns:
- `datetime`: The new date after adding the specified number of weeks.

Example:

```python
from datetime import datetime

date = datetime(2022, 1, 1)
new_date = add_weeks(date, 1)
print(new_date)  # Output: datetime.datetime(2022, 1, 8, 0, 0)
```

#### `subtract_weeks(date: datetime, num_weeks: int) -> datetime`

This function subtracts the specified number of weeks from a given date.

Arguments:
- `date` (datetime): The initial date.
- `num_weeks` (int): The number of weeks to subtract.

Returns:
- `datetime`: The new date after subtracting the specified number of weeks.

Example:

```python
from datetime import datetime

date = datetime(2022, 1, 8)
new_date = subtract_weeks(date, 1)
print(new_date)  # Output: datetime.datetime(2022, 1, 1, 0, 0)
```

#### `add_months(date_str: str, num_months: int) -> str`

This function adds the specified number of months to a given date, considering month-end dates.

Arguments:
- `date_str` (str): The date in 'yyyy-mm-dd' format.
- `num_months` (int): The number of months to add.

Returns:
- `str`: The new date after adding the specified number of months in 'yyyy-mm-dd' format.

Example:

```python
new_date = add_months('2022-01-31', 1)
print(new_date)  # Output: '2022-02-28'
```

#### `subtract_months(date_str: str, num_months: int) -> str`

This function subtracts the specified number of months from a given date, considering month-end dates.

Arguments:
- `date_str` (str): The date in 'yyyy-mm-dd' format.
- `num_months` (int): The number of months to subtract.

Returns:
- `str`: The new date after subtracting the specified number of months in 'yyyy-mm-dd' format.

Example:

```python
new_date = subtract_months('2022-02-28', 1)
print(new_date)  # Output: '2022-01-31'
```

#### `add_years(date_str: str, years: int) -> str`

This function adds the specified number of years to a given date.

Arguments:
- `date_str` (str): The date in 'yyyy-mm-dd' format.
- `years` (int): The number of years to add.

Returns:
- `str`: The new date after adding the specified number of years in 'yyyy-mm-dd' format.

Example:

```python
new_date = add_years('2022-01-01', 1)
print(new_date)  # Output: '2023-01-01'
```

#### `subtract_years(date_str: str, years: int) -> str`

This function subtracts the specified number of years from a given date.

Arguments:
- `date_str` (str): The date in 'yyyy-mm-dd' format.
- `years` (int): The number of years to subtract.

Returns:
- `str`: The new date after subtracting the specified number of years in 'yyyy-mm-dd' format.

Example:

```python
new_date = subtract_years('2023-01-01', 1)
print(new_date)  # Output: '2022-01-01'
```

#### `convert_date_to_yyyymm(date: str) -> str`

This function converts a date from 'yyyy-mm-dd' format to 'yyyymm' format.

Arguments:
- `date` (str): The date in 'yyyy-mm-dd' format.

Returns:
- `str`: The converted date in 'yyyymm' format.

Example:

```python
new_date = convert_date_to_yyyymm('2022-01-01')
print(new_date)  # Output: '202201'
```

#### `convert_date_to_yyyymmdd(yyyymm_date: str) -> str`

This function converts a date from 'yyyymm' format to 'yyyy-mm-dd' format.

Arguments:
- `yyyymm_date` (str): The date in 'yyyymm' format.

Returns:
- `str`: The converted date in 'yyyy-mm-dd' format.

Example:

```python
new_date = convert_date_to_yyyymmdd('202201')
print(new_date)  # Output: '2022-01-01'
```

#### `is_leap_year(year: int) -> bool`

This function checks if a given year is a leap year or not.

Arguments:
- `year` (int): The year to check.

Returns:
- `bool`: True if the given year is a leap year, False otherwise.

Example:

```python
is_leap = is_leap_year(2024)
print(is_leap)  # Output: True
```

#### `is_end_of_month(date: str) -> bool`

This function checks if a given date is the end of the month.

Arguments:
- `date` (str): The date in 'yyyy-mm-dd' format.

Returns:
- `bool`: True if the given date is the end of the month, False otherwise.

Example:

```python
is_end = is_end_of_month('2022-01-31')
print(is_end)  # Output: True
```

#### `is_friday(date: str) -> bool`

This function checks if a given date falls on a Friday.

Arguments:
- `date` (str): The date in either 'yyyy-mm-dd' or 'yyyymmdd' format.

Returns:
- `bool`: True if the given date falls on a Friday, False otherwise.

Raises:
- `ValueError`: If the date is not provided in the correct format.

Example:

```python
is_fri = is_friday('2022-01-07')
print(is_fri)  # Output: True
```

#### `find_month_end(date: str) -> str`

This function finds the month end of a given date.

Arguments:
- `date` (str): The date in 'yyyy-mm-dd' format.

Returns:
- `str`: The month end of the given date in 'yyyy-mm-dd' format.

Example:

```python
month_end = find_month_end('2022-01-15')
print(month_end)  # Output: '2022-01-31'
```

#### `find_week_ending_on_friday(date: datetime) -> datetime`

This function finds the week ending on Friday for a given date.

Arguments:
- `date` (datetime): The date.

Returns:
- `datetime`: The date of the week ending on Friday.

Example:

```python
from datetime import datetime

date = datetime(2022, 1, 1)
week_ending = find_week_ending_on_friday(date)
print(week_ending)  # Output: datetime.datetime(2022, 1, 7, 0, 0)
```

#### `add_subtract_days(date_str: str, days: int) -> str`

This function adds or subtracts the specified number of days from a given date, taking into account week ending on Friday.

Arguments:
- `date_str` (str): The date in 'yyyy-mm-dd' format.
- `days` (int): The number of days to add or subtract.

Returns:
- `str`: The new date after adding or subtracting the specified number of days in 'yyyy-mm-dd' format.

Example:

```python
new_date = add_subtract_days('2022-01-07', 3)
print(new_date)  # Output: '2022-01-10'
```

#### `add_subtract_weeks(date_str: str, num_weeks: int) -> str`

This function adds or subtracts the specified number of weeks from a given date, taking into account week ending on Friday.

Arguments:
- `date_str` (str): The date in 'yyyy-mm-dd' format.
- `num_weeks` (int): The number of weeks to add or subtract.

Returns:
- `str`: The new date after adding or subtracting the specified number of weeks in 'yyyy-mm-dd' format.

Example:

```python
new_date = add_subtract_weeks('2022-01-07', 2)
print(new_date)  # Output: '2022-01-21'
```

#### `get_next_month_end_date(date: datetime) -> datetime`

This function gets the next month end date after a given date.

Arguments:
- `date` (datetime): The initial date.

Returns:
- `datetime`: The next month end date after the given date.

Example:

```python
from datetime import datetime

date = datetime(2022, 1, 15)
next_month_end = get_next_month_end_date(date)
print(next_month_end)  # Output: datetime.datetime(2022, 2, 28, 0, 0)
```

#### `get_previous_month_end_date(date: str) -> str`

This function gets the previous month end date before a given date.

Arguments:
- `date` (str): The date in 'yyyy-mm-dd' format.

Returns:
- `str`: The previous month end date before the given date in 'yyyy-mm-dd' format.

Example:

```python
previous_month_end = get_previous_month_end_date('2022-02-15')
print(previous_month_end)  # Output: '2022-01-31'
```

#### `get_next_week_ending_on_friday(date: datetime) -> datetime`

This function gets the next week ending on Friday after a given date.

Arguments:
- `date` (datetime): The initial date.

Returns:
- `datetime`: The next week ending on Friday after the given date.

Example:

```python
from datetime import datetime

date = datetime(2022, 1, 1)
next_friday = get_next_week_ending_on_friday(date)
print(next_friday)  # Output: datetime.datetime(2022, 1, 7, 0, 0)
```

#### `get_previous_week_ending_on_friday(date: datetime) -> datetime`

This function gets the previous week ending on Friday before a given date.

Arguments:
- `date` (datetime): The initial date.

Returns:
- `datetime`: The previous week ending on Friday before the given date.

Example:

```python
from datetime import datetime

date = datetime(2022, 1, 15)
previous_friday = get_previous_week_ending_on_friday(date)
print(previous_friday)  # Output: datetime.datetime(2022, 1, 7, 0, 0)
```

#### `calculate_days_between_dates(start_date: str, end_date: str) -> int`

This function calculates the number of days between two given dates.

Arguments:
- `start_date` (str): The start date in 'yyyy-mm-dd' format.
- `end_date` (str): The end date in 'yyyy-mm-dd' format.

Returns:
- `int`: The number of days between the two dates.

Raises:
- `ValueError`: If the start_date or end_date is not provided in the correct format.

Example:

```python
days = calculate_days_between_dates('2022-01-01', '2022-01-31')
print(days)  # Output: 30
```

#### `calculate_weeks_between_dates(start_date: str, end_date: str) -> int`

This function calculates the number of weeks between two given dates, considering week ending on Friday.

Arguments:
- `start_date` (str): The start date in 'yyyy-mm-dd' format.
- `end_date` (str): The end date in 'yyyy-mm-dd' format.

Returns:
- `int`: The number of weeks between the two dates, considering week ending on Friday.

Raises:
- `ValueError`: If the start_date or end_date is not provided in the correct format.

Example:

```python
weeks = calculate_weeks_between_dates('2022-01-01', '2022-01-31')
print(weeks)  # Output: 5
```

## Examples

### Example 1: Adding Days

```python
new_date = add_days('2022-01-01', 7)
print(new_date)  # Output: '2022-01-08'
```

### Example 2: Subtracting Days

```python
new_date = subtract_days('2022-01-08', 7)
print(new_date)  # Output: '2022-01-01'
```

### Example 3: Adding Weeks

```python
from datetime import datetime

date = datetime(2022, 1, 1)
new_date = add_weeks(date, 1)
print(new_date)  # Output: datetime.datetime(2022, 1, 8, 0, 0)
```

### Example 4: Subtracting Weeks

```python
from datetime import datetime

date = datetime(2022, 1, 8)
new_date = subtract_weeks(date, 1)
print(new_date)  # Output: datetime.datetime(2022, 1, 1, 0, 0)
```

### Example 5: Adding Months

```python
new_date = add_months('2022-01-31', 1)
print(new_date)  # Output: '2022-02-28'
```

### Example 6: Subtracting Months

```python
new_date = subtract_months('2022-02-28', 1)
print(new_date)  # Output: '2022-01-31'
```

### Example 7: Adding Years

```python
new_date = add_years('2022-01-01', 1)
print(new_date)  # Output: '2023-01-01'
```

### Example 8: Subtracting Years

```python
new_date = subtract_years('2023-01-01', 1)
print(new_date)  # Output: '2022-01-01'
```

### Example 9: Converting Date to YYYMM Format

```python
new_date = convert_date_to_yyyymm('2022-01-01')
print(new_date)  # Output: '202201'
```

### Example 10: Converting YYYMM Date to YYYY-MM-DD Format

```python
new_date = convert_date_to_yyyymmdd('202201')
print(new_date)  # Output: '2022-01-01'
```

### Example 11: Checking Leap Year

```python
is_leap = is_leap_year(2024)
print(is_leap)  # Output: True
```

### Example 12: Checking End of Month

```python
is_end = is_end_of_month('2022-01-31')
print(is_end)  # Output: True
```

### Example 13: Checking if Date Falls on Friday

```python
is_fri = is_friday('2022-01-07')
print(is_fri)  # Output: True
```

### Example 14: Finding Month End

```python
month_end = find_month_end('2022-01-15')
print(month_end)  # Output: '2022-01-31'
```

### Example 15: Finding Week Ending on Friday

```python
from datetime import datetime

date = datetime(2022, 1, 1)
week_ending = find_week_ending_on_friday(date)
print(week_ending)  # Output: datetime.datetime(2022, 1, 7, 0, 0)
```

### Example 16: Adding or Subtracting Days with Week Ending on Friday

```python
new_date = add_subtract_days('2022-01-07', 3)
print(new_date)  # Output: '2022-01-10'
```

### Example 17: Adding or Subtracting Weeks with Week Ending on Friday

```python
new_date = add_subtract_weeks('2022-01-07', 2)
print(new_date)  # Output: '2022-01-21'
```

### Example 18: Getting Next Month End Date

```python
from datetime import datetime

date = datetime(2022, 1, 15)
next_month_end = get_next_month_end_date(date)
print(next_month_end)  # Output: datetime.datetime(2022, 2, 28, 0, 0)
```

### Example 19: Getting Previous Month End Date

```python
previous_month_end = get_previous_month_end_date('2022-02-15')
print(previous_month_end)  # Output: '2022-01-31'
```

### Example 20: Getting Next Week Ending on Friday

```python
from datetime import datetime

date = datetime(2022, 1, 1)
next_friday = get_next_week_ending_on_friday(date)
print(next_friday)  # Output: datetime.datetime(2022, 1, 7, 0, 0)
```

### Example 21: Getting Previous Week Ending on Friday

```python
from datetime import datetime

date = datetime(2022, 1, 15)
previous_friday = get_previous_week_ending_on_friday(date)
print(previous_friday)  # Output: datetime.datetime(2022, 1, 7, 0, 0)
```

### Example 22: Calculating Days Between Dates

```python
days = calculate_days_between_dates('2022-01-01', '2022-01-31')
print(days)  # Output: 30
```

### Example 23: Calculating Weeks Between Dates

```python
weeks = calculate_weeks_between_dates('2022-01-01', '2022-01-31')
print(weeks)  # Output: 5
```

## Conclusion
The Date Manipulation package provides a set of functions for manipulating dates in various ways. It allows you to easily perform common date calculations and conversions.