from datetime import datetime
from your_module import DateManipulator  # Replace 'your_module' with the actual module name
from your_module import convert_to_yyyy_mm_dd  # Replace 'your_module' with the actual module name
from your_module import convert_to_yyyymm  # Replace 'your_module' with the actual module name
from your_module import format_date  # Replace 'your_module' with the actual module name
from your_module import get_current_date  # Replace 'your_module' with the actual module name
from your_module import parse_date_string  # Replace 'your_module' with the actual module name
from your_module import validate_date_format  # Replace 'your_module' with the actual module name


# Initialize the DateManipulator
date_manipulator = DateManipulator()

# Example 1: Add 10 days to a given date
date1 = datetime(2023, 3, 1)
new_date1 = date_manipulator.add_days(date1, 10)
print(new_date1)  # Output: 2023-03-11

# Example 2: Subtract 3 weeks from a given date
date2 = datetime(2023, 3, 15)
new_date2 = date_manipulator.subtract_weeks(date2, 3)
print(new_date2)  # Output: 2023-02-22

# Example 3: Add 1 month to a month-end date
date3 = datetime(2023, 1, 31)
new_date3 = date_manipulator.add_months(date3, 1)
print(new_date3)  # Output: 2023-02-28

# Example 4: Subtract 2 years from a leap year date
date4 = datetime(2020, 2, 29)
new_date4 = date_manipulator.subtract_years(date4, 2)
print(new_date4)  # Output: 2018-02-28

# Example 5: Find the last Friday before a specific date
date5 = datetime(2023, 11, 2)  # Assuming this is a Thursday
last_friday = date_manipulator.find_last_friday(date5)
print(last_friday)  # Output: 2023-10-27



# Example 1: Using a datetime object
date1 = datetime(2023, 3, 15)
print(convert_to_yyyymm(date1))  # Output: '202303'

# Example 2: Using a valid date string
date_str2 = '2023-08-22'
print(convert_to_yyyymm(date_str2))  # Output: '202308'

# Example 3: Trying to use an invalid date string
try:
    date_str3 = '2023/08/22'
    print(convert_to_yyyymm(date_str3))
except ValueError as e:
    print(e)  # Output: "Input date is not in 'yyyy-mm-dd' format."

# Example 4: Handling an end of year date
date_str4 = '2023-12-31'
print(convert_to_yyyymm(date_str4))  # Output: '202312'



# Example 1: Using a datetime object
date1 = datetime(2023, 3, 1)
print(convert_to_yyyy_mm_dd(date1))  # Output: '2023-03-01'

# Example 2: Using a valid yyyymm string
date_str2 = '202303'
print(convert_to_yyyy_mm_dd(date_str2))  # Output: '2023-03-01'

# Example 3: Handling an invalid yyyymm string
try:
    date_str3 = '202313'
    print(convert_to_yyyy_mm_dd(date_str3))
except ValueError as e:
    print(e)  # Output: "Invalid year or month in 'yyyymm' format."



# Example 1: Validating a correct date and format
print(validate_date_format("2023-03-15", "%Y-%m-%d"))  # Output: True

# Example 2: Invalid format for the given date
print(validate_date_format("2023/03/15", "%Y-%m-%d"))  # Output: False

# Example 3: Invalid date in a valid format (non-existent day)
print(validate_date_format("2023-02-30", "%Y-%m-%d"))  # Output: False

# Example 4: Valid leap year date
print(validate_date_format("2020-02-29", "%Y-%m-%d"))  # Output: True

# Example 5: Validating an empty date string
print(validate_date_format("", "%Y-%m-%d"))  # Output: False

# Example 6: Validating with different date format
print(validate_date_format("15/03/2023", "%d/%m/%Y"))  # Output: True

# Example 7: Non-date string input
print(validate_date_format("not-a-date", "%Y-%m-%d"))  # Output: False



# Example 1: Get the current date
print(get_current_date())  # Output: Current date in 'yyyy-mm-dd' format, e.g., '2023-11-10'

# Example 2: Store the current date in a variable
today_date = get_current_date()
print(f"Today's date is: {today_date}")  # Output: Today's date is: 2023-11-10

# Example 3: Use current date in a conditional statement
if get_current_date() == '2023-11-10':  # Replace with the current date for testing
    print("Today is the target date!")
else:
    print("Today is not the target date.")



# Example 1: Parsing a date string with the correct format
date_str1 = "2023-03-15"
format1 = "%Y-%m-%d"
parsed_date1 = parse_date_string(date_str1, format1)
print(parsed_date1)  # Output: 2023-03-15 00:00:00

# Example 2: Parsing a date string with a different format
date_str2 = "15/03/2023"
format2 = "%d/%m/%Y"
parsed_date2 = parse_date_string(date_str2, format2)
print(parsed_date2)  # Output: 2023-03-15 00:00:00

# Example 3: Handling a date string with an incorrect format
try:
    date_str3 = "2023/03/15"
    format3 = "%Y-%m-%d"
    parse_date_string(date_str3, format3)
except ValueError as e:
    print(e)  # Output: Date string '2023/03/15' does not match format '%Y-%m-%d': ...

# Example 4: Handling an invalid date
try:
    date_str4 = "2023-02-30"
    format4 = "%Y-%m-%d"
    parse_date_string(date_str4, format4)
except ValueError as e:
    print(e)  # Output: Date string '2023-02-30' does not match format '%Y-%m-%d': ...



# Example 1: Formatting a date to 'yyyy-mm-dd' format
date1 = datetime(2023, 3, 15)
formatted_date1 = format_date(date1, "%Y-%m-%d")
print(formatted_date1)  # Output: '2023-03-15'

# Example 2: Formatting a date to 'dd/mm/yyyy' format
date2 = datetime(2023, 3, 15)
formatted_date2 = format_date(date2, "%d/%m/%Y")
print(formatted_date2)  # Output: '15/03/2023'

# Example 3: Formatting a datetime with time
date3 = datetime(2023, 3, 15, 14, 30)
formatted_date3 = format_date(date3, "%Y-%m-%d %H:%M")
print(formatted_date3)  # Output: '2023-03-15 14:30'

# Example 4: Handling invalid input (non-datetime object)
try:
    invalid_date = "2023-03-15"
    format_date(invalid_date, "%Y-%m-%d")
except TypeError as e:
    print(e)  # Output: "The 'date' parameter must be a datetime object."
