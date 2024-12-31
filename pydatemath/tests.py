from datetime import date
from yourmodule.date_math import DateMath  # Adjust the import as necessary
from yourmodule.date_utils import format_date  # Adjust the import as necessary
import pytest


def test_add_months():
    dm = DateMath(date(2023, 1, 15))
    assert dm.add_months(1) == date(2023, 2, 15)
    assert dm.add_months(12) == date(2024, 1, 15)
    assert dm.add_months(-2) == date(2022, 11, 15)
    assert dm.add_months(13) == date(2024, 2, 15)
    assert dm.add_months(1) == date(2023, 2, 15)  # handling for last day of Feb

def test_next_friday():
    dm = DateMath(date(2023, 1, 10))  # Wednesday
    assert dm.next_friday() == date(2023, 1, 13)
    dm = DateMath(date(2023, 1, 13))  # Friday
    assert dm.next_friday() == date(2023, 1, 20)

def test_last_day_of_month():
    dm = DateMath(date(2023, 2, 15))
    assert dm.last_day_of_month() == date(2023, 2, 28)
    dm = DateMath(date(2024, 2, 15))  # Leap year
    assert dm.last_day_of_month() == date(2024, 2, 29)
    dm = DateMath(date(2023, 1, 1))
    assert dm.last_day_of_month() == date(2023, 1, 31)

def test_generate_next_valid_date():
    dm = DateMath(date(2023, 1, 30))  # Monday
    assert dm.generate_next_valid_date() == date(2023, 2, 3)  # Next Friday
    dm = DateMath(date(2023, 1, 31))  # Last day of month
    assert dm.generate_next_valid_date() == date(2023, 1, 31)  # Is last day
    dm = DateMath(date(2023, 2, 27))  # Monday before last day
    assert dm.generate_next_valid_date() == date(2023, 2, 28)  # Last day of month



def test_format_date():
    assert format_date(date(2023, 1, 5)) == '20230105'
    assert format_date(date(2023, 12, 25)) == '20231225'
    assert format_date(date(1999, 9, 9)) == '19990909'
    assert format_date(date(2020, 2, 29)) == '20200229'  # Leap year check
