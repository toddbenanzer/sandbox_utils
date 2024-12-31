from datetime import date
from datetime import date, timedelta
import calendar


class DateMath:
    """
    A class for performing date calculations such as adding months
    and generating specific dates like Fridays or the last day of the month.
    """

    def __init__(self, date: date):
        """
        Initializes the DateMath object with a starting date.

        Args:
            date (datetime.date): The initial date for calculations.
        """
        self.date = date

    def add_months(self, months: int) -> date:
        """
        Adds the specified number of months to the current date.

        Args:
            months (int): The number of months to add.

        Returns:
            datetime.date: The new date after adding the given months.
        """
        new_month = (self.date.month + months) % 12
        year_delta = (self.date.month + months - 1) // 12
        new_year = self.date.year + year_delta
        
        if new_month == 0:
            new_month = 12
            new_year -= 1
        
        # Get last day of new month to avoid invalid dates
        last_day = calendar.monthrange(new_year, new_month)[1]
        return date(new_year, new_month, min(self.date.day, last_day))

    def next_friday(self) -> date:
        """
        Finds the next Friday from the current date.

        Returns:
            datetime.date: The date of the next Friday.
        """
        days_ahead = 4 - self.date.weekday()  # 4 corresponds to Friday
        if days_ahead <= 0:
            days_ahead += 7
        return self.date + timedelta(days=days_ahead)

    def last_day_of_month(self) -> date:
        """
        Finds the last day of the month for the current date.

        Returns:
            datetime.date: The date of the last day of the month.
        """
        last_day = calendar.monthrange(self.date.year, self.date.month)[1]
        return date(self.date.year, self.date.month, last_day)

    def generate_next_valid_date(self) -> date:
        """
        Generates the next valid date, either a Friday or the last day of the month.

        Returns:
            datetime.date: The next valid date.
        """
        next_friday_date = self.next_friday()
        last_day = self.last_day_of_month()
        
        if next_friday_date <= last_day:
            return next_friday_date
        else:
            return last_day



def format_date(date_obj: date) -> str:
    """
    Formats a given date into a string in the YYYYMMDD format.

    Args:
        date_obj (datetime.date): The date object to format.

    Returns:
        str: The formatted date string.
    """
    return date_obj.strftime('%Y%m%d')
