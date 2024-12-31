from datetime import datetime
from datetime import datetime, timedelta
import calendar


class DateManipulator:
    """Class for manipulating dates by adding or subtracting days, weeks, months, and years."""

    def __init__(self):
        """Initialize the DateManipulator object."""
        pass

    def add_days(self, date: datetime, days: int) -> datetime:
        """Add a specified number of days to the given date."""
        return date + timedelta(days=days)

    def subtract_days(self, date: datetime, days: int) -> datetime:
        """Subtract a specified number of days from the given date."""
        return date - timedelta(days=days)

    def add_weeks(self, date: datetime, weeks: int) -> datetime:
        """Add a specified number of weeks to the given date."""
        return date + timedelta(weeks=weeks)

    def subtract_weeks(self, date: datetime, weeks: int) -> datetime:
        """Subtract a specified number of weeks from the given date."""
        return date - timedelta(weeks=weeks)

    def add_months(self, date: datetime, months: int) -> datetime:
        """Add a specified number of months to the given date."""
        new_month = (date.month + months - 1) % 12 + 1
        new_year = date.year + (date.month + months - 1) // 12
        last_day_of_month = calendar.monthrange(new_year, new_month)[1]
        new_day = min(date.day, last_day_of_month)
        return date.replace(year=new_year, month=new_month, day=new_day)

    def subtract_months(self, date: datetime, months: int) -> datetime:
        """Subtract a specified number of months from the given date."""
        new_month = (date.month - months - 1) % 12 + 1
        new_year = date.year + (date.month - months - 1) // 12
        last_day_of_month = calendar.monthrange(new_year, new_month)[1]
        new_day = min(date.day, last_day_of_month)
        return date.replace(year=new_year, month=new_month, day=new_day)

    def add_years(self, date: datetime, years: int) -> datetime:
        """Add a specified number of years to the given date."""
        try:
            return date.replace(year=date.year + years)
        except ValueError:  # Handle leap year case
            return date.replace(month=2, day=28, year=date.year + years)

    def subtract_years(self, date: datetime, years: int) -> datetime:
        """Subtract a specified number of years from the given date."""
        try:
            return date.replace(year=date.year - years)
        except ValueError:  # Handle leap year case
            return date.replace(month=2, day=28, year=date.year - years)

    def handle_month_end(self, date: datetime, months: int) -> datetime:
        """Adjust and add months to handle month-end constraints."""
        result_date = self.add_months(date, months)
        if date.day != result_date.day:
            # This implies that the original date was a month-end date like 31st
            result_date = result_date.replace(day=calendar.monthrange(result_date.year, result_date.month)[1])
        return result_date

    def find_last_friday(self, date: datetime) -> datetime:
        """Find the last occurrence of a Friday relative to the given date."""
        days_since_friday = (date.weekday() - calendar.FRIDAY) % 7
        return date - timedelta(days=days_since_friday)



def convert_to_yyyymm(date):
    """
    Convert a given date to the yyyymm string format.

    Args:
        date (datetime or str): Date in datetime format or as a string 'yyyy-mm-dd'.

    Returns:
        str: A string representing the date in 'yyyymm' format.

    Raises:
        ValueError: If the input date string is not in a valid 'yyyy-mm-dd' format.
    """
    if isinstance(date, str):
        try:
            date = datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Input date is not in 'yyyy-mm-dd' format.")
    
    return date.strftime('%Y%m')



def convert_to_yyyy_mm_dd(date):
    """
    Convert a given date to the yyyy-mm-dd string format.

    Args:
        date (datetime or str): Date in datetime format or as a string 'yyyymm'.

    Returns:
        str: A string representing the date in 'yyyy-mm-dd' format.

    Raises:
        ValueError: If the input date string is not in a valid 'yyyymm' format.
    """
    if isinstance(date, str):
        if len(date) != 6 or not date.isdigit():
            raise ValueError("Input date is not in 'yyyymm' format.")
        
        try:
            date = datetime.strptime(date, '%Y%m')
        except ValueError:
            raise ValueError("Invalid year or month in 'yyyymm' format.")

    return date.strftime('%Y-%m-%d')



def validate_date_format(date, format):
    """
    Validate if a given date matches the specified format.

    Args:
        date (str): A string representing a date.
        format (str): A string defining the expected format of the date, e.g., '%Y-%m-%d'.

    Returns:
        bool: True if the date matches the specified format, False otherwise.
    """
    try:
        parsed_date = datetime.strptime(date, format)
        return parsed_date.strftime(format) == date
    except ValueError:
        return False



def get_current_date():
    """
    Retrieve the current date in 'yyyy-mm-dd' format.

    Returns:
        str: A string representing the current date in 'yyyy-mm-dd' format.
    """
    return datetime.now().strftime('%Y-%m-%d')



def parse_date_string(date_string, format):
    """
    Convert a date string into a datetime object based on the specified format.

    Args:
        date_string (str): A string representing a date to be parsed.
        format (str): A string that specifies the expected format of the date, e.g., '%Y-%m-%d'.

    Returns:
        datetime: A datetime object representing the parsed date string.

    Raises:
        ValueError: If the date string does not match the specified format.
    """
    try:
        return datetime.strptime(date_string, format)
    except ValueError as e:
        raise ValueError(f"Date string '{date_string}' does not match format '{format}': {e}")



def format_date(date, format):
    """
    Convert a datetime object into a string representation based on the specified format.

    Args:
        date (datetime): A datetime object representing the date to be formatted.
        format (str): A string defining the desired output format of the date, e.g., '%Y-%m-%d'.

    Returns:
        str: A string representing the date formatted according to the specified format.

    Raises:
        TypeError: If the input date is not a datetime object.
    """
    if not isinstance(date, datetime):
        raise TypeError("The 'date' parameter must be a datetime object.")
    
    return date.strftime(format)
