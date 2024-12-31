# DateManipulator Class Documentation

## Overview
The `DateManipulator` class provides methods to manipulate dates by adding or subtracting days, weeks, months, and years. It also includes functionality to handle month-end scenarios and to find the last occurrence of a Friday relative to a given date.

## Methods

### __init__(self)
Initializes a new instance of the `DateManipulator` class.

### add_days(date: datetime, days: int) -> datetime
Adds a specified number of days to the given date.
- **Parameters:**
  - `date`: A `datetime` object representing the initial date.
  - `days`: An integer representing the number of days to add.
- **Returns:** A `datetime` object representing the new date.

### subtract_days(date: datetime, days: int) -> datetime
Subtracts a specified number of days from the given date.
- **Parameters:**
  - `date`: A `datetime` object representing the initial date.
  - `days`: An integer representing the number of days to subtract.
- **Returns:** A `datetime` object representing the new date.

### add_weeks(date: datetime, weeks: int) -> datetime
Adds a specified number of weeks to the given date.
- **Parameters:**
  - `date`: A `datetime` object representing the initial date.
  - `weeks`: An integer representing the number of weeks to add.
- **Returns:** A `datetime` object representing the new date.

### subtract_weeks(date: datetime, weeks: int) -> datetime
Subtracts a specified number of weeks from the given date.
- **Parameters:**
  - `date`: A `datetime` object representing the initial date.
  - `weeks`: An integer representing the number of weeks to subtract.
- **Returns:** A `datetime` object representing the new date.

### add_months(date: datetime, months: int) -> datetime
Adds a specified number of months to the given date.
- **Parameters:**
  - `date`: A `datetime` object representing the initial date.
  - `months`: An integer representing the number of months to add.
- **Returns:** A `datetime` object representing the new date after adding months, correctly handling overflow into the next month.

### subtract_months(date: datetime, months: int) -> datetime
Subtracts a specified number of months from the given date.
- **Parameters:**
  - `date`: A `datetime` object representing the initial date.
  - `months`: An integer representing the number of months to subtract.
- **Returns:** A `datetime` object representing the new date after subtracting months, correctly handling underflow into the previous month.

### add_years(date: datetime, years: int) -> datetime
Adds a specified number of years to the given date.
- **Parameters:**
  - `date`: A `datetime` object representing the initial date.
  - `years`: An integer representing the number of years to add.
- **Returns:** A `datetime` object representing the new date.

### subtract_years(date: datetime, years: int) -> datetime
Subtracts a specified number of years from the given date.
- **Parameters:**
  - `date`: A `datetime` object representing the initial date.
  - `years`: An integer representing the number of years to subtract.
- **Returns:** A `datetime` object representing the new date.

### handle_month_end(date: datetime, months: int) -> datetime
Adjusts and adds a specified number of months to a date while properly handling month-end constraints.
- **Parameters:**
  - `date`: A `datetime` object representing the initial date.
  - `months`: An integer representing the number of months to add.
- **Returns:** A `datetime` object representing the final date after handling month-end constraints.

### find_last_friday(date: datetime) -> datetime
Finds the last occurrence of a Friday relative to a given date.
- **Parameters:**
  - `date`: A `datetime` object representing the initial date.
- **Returns:** A `datetime` object representing the last Friday prior to or on the given date.


# convert_to_yyyymm Function Documentation

## Overview
The `convert_to_yyyymm` function converts a given date, either as a `datetime` object or a string in the format `yyyy-mm-dd`, into a string representing the date in the `yyyymm` format.

## Parameters

### date
- **Type:** `datetime` or `str`
- **Description:** The date to be converted. It can be provided either as a `datetime` object or as a string in the format `yyyy-mm-dd`.

## Returns
- **Type:** `str`
- **Description:** A string representing the date in `yyyymm` format (e.g., for March 2023, the output will be '202303').

## Raises
- **ValueError:** This exception is raised if the input date string is not in the valid `yyyy-mm-dd` format.

## Example Usage


# convert_to_yyyy_mm_dd Function Documentation

## Overview
The `convert_to_yyyy_mm_dd` function converts a given date, either as a `datetime` object or a string in the format `yyyymm`, into a string representing the date in the `yyyy-mm-dd` format.

## Parameters

### date
- **Type:** `datetime` or `str`
- **Description:** The date to be converted. It can be provided either as a `datetime` object or as a string in the format `yyyymm`.

## Returns
- **Type:** `str`
- **Description:** A string representing the date in `yyyy-mm-dd` format (e.g., '2023-03-01' for March 1, 2023).

## Raises
- **ValueError:** This exception is raised if the input date string is not in the valid `yyyymm` format or if it represents an invalid month.

## Example Usage


# validate_date_format Function Documentation

## Overview
The `validate_date_format` function checks whether a given date string matches a specified format.

## Parameters

### date
- **Type:** `str`
- **Description:** A string representing the date to be validated.

### format
- **Type:** `str`
- **Description:** A string that defines the expected date format (e.g., '%Y-%m-%d').

## Returns
- **Type:** `bool`
- **Description:** Returns `True` if the date matches the specified format; otherwise, returns `False`.

## Example Usage


# get_current_date Function Documentation

## Overview
The `get_current_date` function retrieves the current date formatted as a string in the 'yyyy-mm-dd' format.

## Returns
- **Type:** `str`
- **Description:** A string representing the current date in the format 'yyyy-mm-dd'.

## Example Usage


# parse_date_string Function Documentation

## Overview
The `parse_date_string` function converts a date string into a `datetime` object based on the specified format.

## Parameters

### date_string
- **Type:** `str`
- **Description:** A string representing the date to be parsed.

### format
- **Type:** `str`
- **Description:** A string that specifies the expected format of the date (e.g., '%Y-%m-%d').

## Returns
- **Type:** `datetime`
- **Description:** A `datetime` object representing the parsed date string.

## Raises
- **ValueError:** This exception is raised if the provided date string does not match the specified format.

## Example Usage


# format_date Function Documentation

## Overview
The `format_date` function converts a `datetime` object into a string representation based on the specified format.

## Parameters

### date
- **Type:** `datetime`
- **Description:** A `datetime` object representing the date to be formatted.

### format
- **Type:** `str`
- **Description:** A string defining the desired output format of the date (e.g., '%Y-%m-%d').

## Returns
- **Type:** `str`
- **Description:** A string representing the date formatted according to the specified format.

## Raises
- **TypeError:** This exception is raised if the input date is not a `datetime` object.

## Example Usage
