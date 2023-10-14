etime import datetime, timedelta


def add_days(date_str: str, num_days: int) -> str:
    """
    Add the specified number of days to the given date.
    
    Args:
        date_str (str): The date in 'yyyy-mm-dd' format.
        num_days (int): The number of days to add.
        
    Returns:
        str: The new date after adding the specified number of days in 'yyyy-mm-dd' format.
    """
    # Convert the date string to a datetime object
    date = datetime.strptime(date_str, '%Y-%m-%d')

    # Add the specified number of days to the date
    new_date = date + timedelta(days=num_days)

    # Return the new date as a string in yyyy-mm-dd format
    return new_date.strftime('%Y-%m-%d')


def subtract_days(date_str: str, num_days: int) -> str:
    """
    Subtract the specified number of days from the given date.
    
    Args:
        date_str (str): The date in 'yyyy-mm-dd' format.
        num_days (int): The number of days to subtract.
        
    Returns:
        str: The new date after subtracting the specified number of days in 'yyyy-mm-dd' format.
    """
    # Convert the date string to a datetime object
    date = datetime.strptime(date_str, '%Y-%m-%d')

    # Subtract the specified number of days from the date
    new_date = date - timedelta(days=num_days)

    # Return the new date as a string in yyyy-mm-dd format
    return new_date.strftime('%Y-%m-%d')


def add_weeks(date: datetime, num_weeks: int) -> datetime:
    """
    Add the specified number of weeks to the given date.
    
    Args:
        date (datetime): The initial date.
        num_weeks (int): The number of weeks to add.
        
    Returns:
        datetime: The new date after adding the specified number of weeks.
    """
    return date + timedelta(weeks=num_weeks)


def subtract_weeks(date: datetime, num_weeks: int) -> datetime:
    """
    Subtract the specified number of weeks from the given date.
    
    Args:
        date (datetime): The initial date.
        num_weeks (int): The number of weeks to subtract.
        
    Returns:
        datetime: The new date after subtracting the specified number of weeks.
    """
    return date - timedelta(weeks=num_weeks)


def add_months(date_str: str, num_months: int) -> str:
    """
    Add the specified number of months to the given date, considering month-end dates.
    
    Args:
        date_str (str): The date in 'yyyy-mm-dd' format.
        num_months (int): The number of months to add.
        
    Returns:
        str: The new date after adding the specified number of months in 'yyyy-mm-dd' format.
    """
    # Convert the date string to a datetime object
    date = datetime.strptime(date_str, '%Y-%m-%d')

    # Calculate the new year and month values
    new_year = date.year + (date.month + num_months - 1) // 12
    new_month = (date.month + num_months) % 12

    # Handle the case when the new month is December
    if new_month == 0:
        new_month = 12
        new_year -= 1

    # Calculate the last day of the new month
    last_day_of_new_month = (datetime(new_year, new_month + 1, 1) - timedelta(days=1)).day

    # Handle the case when the original day is greater than the last day of the new month
    if date.day > last_day_of_new_month:
        new_day = last_day_of_new_month
    else:
        new_day = date.day

    # Create a new datetime object with the updated year, month, and day values
    new_date = datetime(new_year, new_month, new_day)

    # Convert the datetime object back to a string in yyyy-mm-dd format
    new_date_str = new_date.strftime('%Y-%m-%d')

    return new_date_str


def subtract_months(date_str: str, num_months: int) -> str:
    """
    Subtract the specified number of months from the given date, considering month-end dates.
    
    Args:
        date_str (str): The date in 'yyyy-mm-dd' format.
        num_months (int): The number of months to subtract.
        
    Returns:
        str: The new date after subtracting the specified number of months in 'yyyy-mm-dd' format.
    """
    # Convert the input date string to a datetime object
    date = datetime.strptime(date_str, '%Y-%m-%d')

    # Subtract the specified number of months
    new_date = date - timedelta(days=date.day)
    
    for _ in range(num_months):
        new_date = new_date.replace(day=1) - timedelta(days=1)

    # Format the new date as yyyy-mm-dd and return it as a string
    return new_date.strftime('%Y-%m-%d')


def add_years(date_str: str, years: int) -> str:
    """
    Add the specified number of years to the given date.
    
    Args:
        date_str (str): The date in 'yyyy-mm-dd' format.
        years (int): The number of years to add.
        
    Returns:
        str: The new date after adding the specified number of years in 'yyyy-mm-dd' format.
    """
    # Convert the date string to a datetime object
    date = datetime.strptime(date_str, '%Y-%m-%d')

    # Add the specified number of years to the date
    new_date = date + timedelta(days=365 * years)

    # Format the new date as a string in yyyy-mm-dd format
    new_date_str = new_date.strftime('%Y-%m-%d')

    return new_date_str


def subtract_years(date_str: str, years: int) -> str:
    """
    Subtract the specified number of years from the given date.
    
    Args:
        date_str (str): The date in 'yyyy-mm-dd' format.
        years (int): The number of years to subtract.
        
    Returns:
        str: The new date after subtracting the specified number of years in 'yyyy-mm-dd' format.
    """
    # Convert string date to datetime object
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')

    # Subtract years from the date using timedelta
    new_date = date_obj - timedelta(days=years * 365)

    # Convert back to string format
    new_date_str = new_date.strftime('%Y-%m-%d')

    return new_date_str


def convert_date_to_yyyymm(date: str) -> str:
    """
    Convert a date from 'yyyy-mm-dd' format to 'yyyymm' format.
    
    Args:
        date (str): The date in 'yyyy-mm-dd' format.
        
    Returns:
        str: The converted date in 'yyyymm' format.
    """
    return date.replace('-', '')


def convert_date_to_yyyymmdd(yyyymm_date: str) -> str:
    """
    Convert a date from 'yyyymm' format to 'yyyy-mm-dd' format.
    
    Args:
        yyyymm_date (str): The date in 'yyyymm' format.
        
    Returns:
        str: The converted date in 'yyyy-mm-dd' format.
    """
    # Convert the yyyymm_date to a string
    yyyymm_date_str = str(yyyymm_date)

    # Extract the year and month from the string
    year = yyyymm_date_str[:4]
    month = yyyymm_date_str[4:]

    # Create a new date string in yyyy-mm-dd format
    yyyy_mm_dd_date = f"{year}-{month}-01"

    return yyyy_mm_dd_date


def is_leap_year(year: int) -> bool:
    """
    Check if a given year is a leap year or not.
    
    Args:
        year (int): The year to check.
        
    Returns:
        bool: True if the given year is a leap year, False otherwise.
    """
    if year % 4 == 0:
        if year % 100 == 0:
            if year % 400 == 0:
                return True
            else:
                return False
        else:
            return True
    else:
        return False


def is_end_of_month(date: str) -> bool:
    """
    Check if a given date is the end of the month.
    
    Args:
        date (str): The date in 'yyyy-mm-dd' format.
        
    Returns:
        bool: True if the given date is the end of the month, False otherwise.
    """
    # Convert the input to a datetime object
    date_obj = datetime.strptime(date, '%Y-%m-%d')

    # Get the day of the month
    day = date_obj.day

    # Get the total number of days in the month
    total_days = (date_obj.replace(day=28) + timedelta(days=4)).day

    # Check if the day is equal to the total number of days in the month
    if day == total_days:
        return True
    else:
        return False


def is_friday(date: str) -> bool:
    """
    Check if a given date falls on a Friday.
    
    Args:
        date (str): The date in either 'yyyy-mm-dd' or 'yyyymmdd' format.
        
    Returns:
        bool: True if the given date falls on a Friday, False otherwise.
        
    Raises:
        ValueError: If the date is not provided in the correct format.
    """
    # Convert the date to datetime.date object if it is in string format
    if isinstance(date, str):
        try:
            date = datetime.strptime(date, '%Y-%m-%d').date()
        except ValueError:
            try:
                date = datetime.strptime(date, '%Y%m%d').date()
            except ValueError:
                raise ValueError("Invalid date format")

    return date.weekday() == 4


def find_month_end(date: str) -> str:
    """
    Find the month end of a given date.
    
    Args:
        date (str): The date in 'yyyy-mm-dd' format.
        
    Returns:
        str: The month end of the given date in 'yyyy-mm-dd' format.
    """
    # Convert the input to a datetime object
    date_obj = datetime.strptime(date, '%Y-%m-%d')

    year = date_obj.year
    month = date_obj.month

    # Find the last day of the month
    if month == 12:
        next_month = 1
        next_year = year + 1
    else:
        next_month = month + 1
        next_year = year

    end_of_month = datetime(next_year, next_month, 1) - timedelta(days=1)

    return end_of_month.strftime('%Y-%m-%d')


def find_week_ending_on_friday(date: datetime) -> datetime:
    """
    Find the week ending on Friday for a given date.
    
    Args:
        date (datetime): The date.
        
    Returns:
        datetime: The date of the week ending on Friday.
    """
    # Find the day of the week for the given date
    day_of_week = date.weekday()

    # Calculate the number of days to add to reach Friday (assuming Friday is the end of the week)
    days_to_add = (4 - day_of_week) % 7

    # Add the calculated number of days to the given date
    week_ending_on_friday = date + timedelta(days=days_to_add)

    return week_ending_on_friday


def add_subtract_days(date_str: str, days: int) -> str:
    """
    Add or subtract the specified number of days from the given date, taking into account week ending on Friday.
    
    Args:
        date_str (str): The date in 'yyyy-mm-dd' format.
        days (int): The number of days to add or subtract.
        
    Returns:
        str: The new date after adding or subtracting the specified number of days in 'yyyy-mm-dd' format.
    """
    # Convert the date string to a datetime object
    date = datetime.strptime(date_str, '%Y-%m-%d')

    # Subtract one day if the current day is Friday and we're adding days
    if date.weekday() == 4 and days > 0:
        date -= timedelta(days=1)

    # Add or subtract the specified number of days
    date += timedelta(days=days)

    # Add one day if the new date is Saturday and we're subtracting days
    if date.weekday() == 5 and days < 0:
        date += timedelta(days=1)

    # Return the updated date as a string in "yyyy-mm-dd" format
    return date.strftime('%Y-%m-%d')


def add_subtract_weeks(date_str: str, num_weeks: int) -> str:
    """
    Add or subtract the specified number of weeks from the given date, taking into account week ending on Friday.
    
    Args:
        date_str (str): The date in 'yyyy-mm-dd' format.
        num_weeks (int): The number of weeks to add or subtract.
        
    Returns:
        str: The new date after adding or subtracting the specified number of weeks in 'yyyy-mm-dd' format.
    """
    # Convert the input date string to a datetime object
    date = datetime.strptime(date_str, '%Y-%m-%d')

    # Calculate the number of days to be added/subtracted based on the number of weeks
    num_days = num_weeks * 7

    # If the current week ends on Friday, adjust the number of days accordingly
    if date.weekday() == 4:
        num_days -= 2

    # Add/subtract the calculated number of days to the date
    new_date = date + timedelta(days=num_days)

    # Convert the new date back to a string in "YYYY-MM-DD" format
    new_date_str = new_date.strftime('%Y-%m-%d')

    return new_date_str


def get_next_month_end_date(date: datetime) -> datetime:
    """
    Get the next month end date after a given date.
    
    Args:
        date (datetime): The initial date.
        
    Returns:
        datetime: The next month end date after the given date.
    """
    # Get the start of the next month
    next_month_start = datetime(date.year, date.month, 1) + timedelta(days=32)

    # Subtract one day to get the last day of the current month
    current_month_end = next_month_start - timedelta(days=1)

    # Get the start of the next month
    next_month_start = datetime(current_month_end.year, current_month_end.month, 1) + timedelta(days=32)

    # Subtract one day to get the last day of the next month
    next_month_end = next_month_start - timedelta(days=1)

    return next_month_end


def get_previous_month_end_date(date: str) -> str:
    """
    Get the previous month end date before a given date.
    
    Args:
        date (str): The date in 'yyyy-mm-dd' format.
        
    Returns:
        str: The previous month end date before the given date in 'yyyy-mm-dd' format.
    """
    # Convert the input string to a datetime object
    date_obj = datetime.strptime(date, '%Y-%m-%d')

    # Get the first day of the current month
    first_day_of_month = date_obj.replace(day=1)

    # Subtract one day to get the last day of the previous month
    last_day_of_previous_month = first_day_of_month - timedelta(days=1)

    return last_day_of_previous_month.strftime('%Y-%m-%d')


def get_next_week_ending_on_friday(date: datetime) -> datetime:
    """
    Get the next week ending on Friday after a given date.
    
    Args:
        date (datetime): The initial date.
        
    Returns:
        datetime: The next week ending on Friday after the given date.
    """
    # Calculate the weekday of the given date
    weekday = date.weekday()

    # If the given date is already a Friday, add 7 days to get the next week's Friday
    if weekday == 4:
        next_friday = date + timedelta(days=7)

    # If the given date is on or before Thursday, add the number of days to reach the next Friday
    elif weekday <= 3:
        days_to_next_friday = (4 - weekday)
        next_friday = date + timedelta(days=days_to_next_friday)

    # If the given date is after Friday, add the number of days to reach the next Friday in the next week
    else:
        days_to_next_friday = (11 - weekday) % 7
        next_friday = date + timedelta(days=days_to_next_friday)

    return next_friday


def get_previous_week_ending_on_friday(date: datetime) -> datetime:
    """
    Get the previous week ending on Friday before a given date.
    
    Args:
        date (datetime): The initial date.
        
    Returns:
        datetime: The previous week ending on Friday before the given date.
    """
    # Calculate the weekday of the given date
    weekday = date.weekday()

    # If the given date is a Friday, subtract 7 days to get the previous week's Friday
    if weekday == 4:
        previous_friday = date - timedelta(days=7)

    # If the given date is on or after Saturday, subtract the number of days to reach the previous Friday
    elif weekday >= 5:
        days_to_previous_friday = (weekday + 2) % 7
        previous_friday = date - timedelta(days=days_to_previous_friday)

    # If the given date is before Friday, subtract the number of days to reach the previous Friday in the previous week
    else:
        days_to_previous_friday = (weekday + 9) % 7
        previous_friday = date - timedelta(days=days_to_previous_friday)

    return previous_friday


def calculate_days_between_dates(start_date: str, end_date: str) -> int:
    """
    Calculate the number of days between two given dates.
    
    Args:
        start_date (str): The start date in 'yyyy-mm-dd' format.
        end_date (str): The end date in 'yyyy-mm-dd' format.
        
    Returns:
        int: The number of days between the two dates.
        
    Raises:
        ValueError: If the start_date or end_date is not provided in the correct format.
    """
    try:
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        return (end - start).days
    except ValueError:
        raise ValueError("Invalid date format. Expected format is 'yyyy-mm-dd'.")


def calculate_weeks_between_dates(start_date: str, end_date: str) -> int:
    """
    Calculate the number of weeks between two given dates, considering week ending on Friday.
    
    Args:
        start_date (str): The start date in 'yyyy-mm-dd' format.
        end_date (str): The end date in 'yyyy-mm-dd' format.
        
    Returns:
        int: The number of weeks between the two dates, considering week ending on Friday.
        
    Raises:
        ValueError: If the start_date or end_date is not provided in the correct format.
    """
    # Convert the input dates to datetime objects
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    # Add one day to the end date so that week ends on Friday
    end += timedelta(days=1)

    # Calculate the number of days between the two dates
    days_between = (end - start).days

    # Calculate the number of weeks
    num_weeks = days_between // 7

    return num_week