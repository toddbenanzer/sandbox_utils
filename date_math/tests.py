the refactored code using pytest and PEP-8 standards:

import pytest
from datetime import datetime, timedelta

def add_days(date_str, num_days):
    date = datetime.strptime(date_str, "%Y-%m-%d")
    new_date = date + timedelta(days=num_days)
    return new_date.strftime("%Y-%m-%d")

def test_add_days():
    assert add_days("2020-01-01", 1) == "2020-01-02"
    assert add_days("2020-01-01", 7) == "2020-01-08"
    assert add_days("2020-01-01", 30) == "2020-01-31"
    assert add_days("2020-01-01", 365) == "2021-01-01"

def add_weeks(date, num_weeks):
    new_date = date + timedelta(weeks=num_weeks)
    return new_date

def test_add_weeks():
    date = datetime.date(2022, 1, 1)
    expected_result = datetime.date(2022, 1, 8)
    assert add_weeks(date, 1) == expected_result

    date = datetime.date(2022, 1, 1)
    assert add_weeks(date, 0) == date

    date = datetime.date(2022, 1, 1)
    expected_result = datetime.date(2021, 12, 25)
    assert add_weeks(date, -1) == expected_result

    date = datetime.date(2022, 1, 1)
    expected_result = datetime.date(2022, 1, 29)
    assert add_weeks(date, 4) == expected_result

import pytest
from your_module import subtract_weeks

@pytest.mark.parametrize("date, weeks, expected_output", [
    ("2022-01-01", 2, "2021-12-18"),
    ("2022-01-01", 0, "2022-01-01"),
    ("2022-01-01", -1, "2022-01-08"),
    ("2022-01-01", 52, "2021-01-02"),
    ("2020-02-29", 2, "2020-02-15")
])
def test_subtract_weeks(date, weeks, expected_output):
    assert subtract_weeks(date, weeks) == expected_output

from datetime import datetime
from your_module import add_months

def test_add_months():
    date = "2022-02-28"
    num_months = 3
    expected_result = "2022-05-31"
    
    assert add_months(date, num_months) == expected_result
    
    date = "2020-12-31"
    num_months = 6
    expected_result = "2021-06-30"
    
    assert add_months(date, num_months) == expected_result
    
    date = "2019-07-15"
    num_months = 12
    expected_result = "2020-07-15"
    
    assert add_months(date, num_months) == expected_result

import pytest
from datetime import datetime

def test_subtract_months():
    assert subtract_months('2022-05-20', 1) == '2022-04-20'
    
    assert subtract_months('2022-07-01', 3) == '2022-04-01'
    
    assert subtract_months('2022-01-15', 6) == '2021-07-15'
    
    assert subtract_months('2021-10-31', 12) == '2020-10-31'
    
    assert subtract_months('2022-09-18', 0) == '2022-09-18'
    
    with pytest.raises(ValueError):
        subtract_months('2023-03-10', -2)
        
    with pytest.raises(ValueError):
        subtract_months('2022-02-28', 3)

def test_add_years():
    assert add_years('2020-01-01', 1) == '2021-01-01'

    assert add_years('2020-02-29', 5) == '2025-02-28'

    assert add_years('2022-12-31', 0) == '2022-12-31'

    assert add_years('1999-03-20', -10) == '1989-03-20'


import pytest
from datetime import datetime

from your_module import subtract_years


def test_subtract_years():
    assert subtract_years('2022-01-01', 1) == '2021-01-01'

    assert subtract_years('2024-02-29', 5) == '2019-02-28'

    current_date = datetime.now().strftime('%Y-%m-%d')
    expected_date = (datetime.now() - timedelta(days=3650)).strftime('%Y-%m-%d')
    assert subtract_years(current_date, 10) == expected_date

    assert subtract_years('2022-01-01', 0) == '2022-01-01'

    assert subtract_years('2022-01-01', -3) == '2025-01-

def convert_date(date):
    return date.replace("-", "")


def test_convert_date():
    assert convert_date('2021-
    
    assert convert_date('1999-
    
    assert convert_date('2000-
    
    assert convert_date('2022-
    
    assert convert_date('1987-


import pytest

from my_module import convert_date

def test_convert_date():
    assert convert_date(202201) == "2022-01-01"
    
    assert convert_date(202212) == "2022-12-01"
    
    with pytest.raises(TypeError):
        convert_date("invalid")
    
    with pytest.raises(ValueError):
        convert_date(-202201)

    assert convert_date(000001) == "0-01-01"

from pytest import mark

@mark.parametrize("year, expected_result", [
    (2000, True),
    (2020, True),
    (1900, False),
    (2021, False),
])
def test_is_leap_year(year, expected_result):
    assert is_leap_year(year) == expected_result

def is_end_of_month(date):
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    day = date_obj.day
    total_days = (date_obj.replace(day=28) + timedelta(days=4)).day
    if day == total_days:
        return True
    else:
        return False

def test_is_end_of_month():
    assert is_end_of_month("2021-12-31") == True
    
    assert is_end_of_month("2022-01-15") == False
    
    assert is_end_of_month("2022-02-28") == True
    
    assert is_end_of_month("2022-02-15") == False
    
    assert is_end_of_month("2024-02-29") == True
    
    assert is_end_of_month("2024-02-15") == False

def find_month_end(date):
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    year = date_obj.year
    month = date_obj.month
    if month == 12:
        year += 1
        month = 1
    else:
        month += 1
    next_month_start = datetime(year, month, 1)
    month_end = next_month_start - timedelta(days=1)
    return month_end.strftime("%Y-%m-%d")

def test_find_month_end_valid_string_date():
    assert find_month_end("2021-01-15") == "2021-01-31"

def test_find_month_end_valid_datetime_date():
    date = datetime(2021, 2, 14).date()
    assert find_month_end(date) == "2021-02-28"

def test_find_month_end_last_day_of_december():
    assert find_month_end("2020-12-31") == "2020-12-31"

def test_find_month_end_last_day_of_leap_year_february():
    assert find_month_end("2020-02-29") == "2020-02-29"

def test_find_month_end_last_day_of_non_leap_year_february():
    assert find_month_end("2021-02-28") == "2021--28"

def is_friday(date):
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    return date_obj.weekday() == 4

def test_is_friday():
    assert is_friday('2021-07-02') == True
    
    assert is_friday('20210702') == True
    
    assert is_friday('2021-07-01') == False
    
    assert is_friday('20210701') == False
    
    assert is_friday('2021-07-03') == False
    
    assert is_friday('20210703') == False

def get_previous_month_end_date(date):
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    month_start = date_obj.replace(day=1)
    month_end = month_start - timedelta(days=1)
    return month_end.strftime("%Y-%m-%d")

def test_get_previous_month_end_date():
    assert get_previous_month_end_date('2022-05-15') == '2022-04-30'
    
    assert get_previous_month_end_date('2022-06-01') == '2022-05-31'
    
    assert get_previous_month_end_date('2022-03-31') == '2022-02-28'
    
    assert get_previous_month_end_date('2020-02-15') == '2020-01-31'

def get_next_friday(date):
    days_ahead = 4 - date.weekday()
    if days_ahead <= 0: 
        days_ahead += 7
    next_friday = date + timedelta(days=days_ahead)
    return next_friday.strftime("%Y-%m-%d")

def test_get_next_friday():
    date = datetime(2021, 11, 19)
    assert get_next_friday(date) == "2021-11-19"
    
    date = datetime(2021, 11, 16)
    assert get_next_friday(date) == "2021--19"
    
    date = datetime(2021, 11, 20)
    assert get_next_friday(date) == "2021--26"

    date = datetime(2021, 11, 21)
    assert get_next_friday(date) == "2021--26"

    date = datetime(2021, 11, 22)
    assert get_next_friday(date) == "2021--26"

def get_previous_week_friday(given_date):
    days_before = given_date.weekday() + 1
    if days_before > 5:
        days_before = 5
    previous_week_friday = given_date - timedelta(days=days_before)
    return previous_week_friday

def test_get_previous_week_friday_given_date_is_friday():
    given_date = datetime.date(2022, 1, 7)
    previous_week_friday = get_previous_week_friday(given_date)
    assert previous_week_friday == datetime.date(2021, 12, 31)

def test_get_previous_week_friday_given_date_is_saturday_or_after():
    given_date = datetime.date(2022, 1, 8)
    previous_week_friday = get_previous_week_friday(given_date)
    assert previous_week_friday == datetime.date(2022, 1, 7)

def test_get_previous_week_friday_given_date_is_before_friday():
    given_date = datetime.date(2022, 1, 6)
    previous_week_friday = get_previous_week_friday(given_date)
    assert previous_week_friday == datetime.date(2021, 12, 31)

def test_get_previous_week_friday_given_date_is_monday():
    given_date = datetime.date(2022, 1, 10)
    previous_week_friday = get_previous_week_friday(given_date)
    assert previous_week_friday == datetime.date(2022, 1, 7)

def test_get_previous_week_friday_given_date_is_first_day_of_year():
    given_date = datetime.date(2022, 1, 1)
    previous_week_friday = get_previous_week_friday(given_date)
    assert previous_week_friday == datetime.date(2021, 12, 

from your_module import calculate_days_between_dates

def test_calculate_days_between_dates():
    start_date = '2020-01-01'
    end_date = '2020-01-10'
    expected_result = 9
    assert calculate_days_between_dates(start_date, end_date) == expected_result

    start_date = '2020-01-10'
    end_date = '2020-01-01'
    with pytest.raises(ValueError):
        calculate_days_between_dates(start_date, end_date)

    start_date = '2020/01/01'
    end_date = '2020-01-10'
    with pytest.raises(ValueError):
        calculate_days_between_dates(start_date, end_date)

    start_date = '2020-02-28'
    end_date = '2020-03-01'
    expected_result = 2
    assert calculate_days_between_dates(start_date, end_date) == expected_result

    start_date = '2020-01-01'
    end_date = '2020-01-01'
    expected_result = 0
    assert calculate_days_between_dates(start_date, end_da

def weeks_between_dates(start, end):
    weeks_count = (end - start).days // 7
        if weeks_count < 0:
            weeks_count -= 1
        return weeks_count

def test_weeks_between_dates():
    start_date = datetime.date(2021, 1, 1)
    end_date = datetime.date(2021, 1, 3)
    assert weeks_between_dates(start_, end_) == 0

    start_date = datetime.date(2021, 1, 1)
    end__date_ = datetime.date(2021, 1, 8)
    
  
import pytest
import datetime


def test_get_previous_month_end:
    
  
 if __name__ == '__main__':
     pytest.main(