from datetime import datetime
from your_module import DateManipulator  # Replace 'your_module' with the actual module name
from your_module import convert_to_yyyy_mm_dd  # Replace 'your_module' with the actual module name
from your_module import convert_to_yyyymm  # Replace 'your_module' with the actual module name
from your_module import format_date  # Replace 'your_module' with the actual module name
from your_module import get_current_date  # Replace 'your_module' with the actual module name
from your_module import parse_date_string  # Replace 'your_module' with the actual module name
from your_module import validate_date_format  # Replace 'your_module' with the actual module name
import pytest


@pytest.fixture
def date_manipulator():
    return DateManipulator()

def test_add_days(date_manipulator):
    initial_date = datetime(2023, 3, 1)
    result = date_manipulator.add_days(initial_date, 10)
    expected = datetime(2023, 3, 11)
    assert result == expected

def test_subtract_days(date_manipulator):
    initial_date = datetime(2023, 3, 11)
    result = date_manipulator.subtract_days(initial_date, 10)
    expected = datetime(2023, 3, 1)
    assert result == expected

def test_add_weeks(date_manipulator):
    initial_date = datetime(2023, 3, 1)
    result = date_manipulator.add_weeks(initial_date, 2)
    expected = datetime(2023, 3, 15)
    assert result == expected

def test_subtract_weeks(date_manipulator):
    initial_date = datetime(2023, 3, 15)
    result = date_manipulator.subtract_weeks(initial_date, 2)
    expected = datetime(2023, 3, 1)
    assert result == expected

def test_add_months(date_manipulator):
    initial_date = datetime(2023, 1, 31)
    result = date_manipulator.add_months(initial_date, 1)
    expected = datetime(2023, 2, 28)
    assert result == expected

def test_subtract_months(date_manipulator):
    initial_date = datetime(2023, 3, 31)
    result = date_manipulator.subtract_months(initial_date, 1)
    expected = datetime(2023, 2, 28)
    assert result == expected

def test_add_years(date_manipulator):
    initial_date = datetime(2020, 2, 29)
    result = date_manipulator.add_years(initial_date, 1)
    expected = datetime(2021, 2, 28)
    assert result == expected

def test_subtract_years(date_manipulator):
    initial_date = datetime(2021, 2, 28)
    result = date_manipulator.subtract_years(initial_date, 1)
    expected = datetime(2020, 2, 28)
    assert result == expected

def test_handle_month_end(date_manipulator):
    initial_date = datetime(2021, 1, 31)
    result = date_manipulator.handle_month_end(initial_date, 1)
    expected = datetime(2021, 2, 28)
    assert result == expected

def test_find_last_friday(date_manipulator):
    initial_date = datetime(2023, 11, 1)  # Assuming this is a Tuesday
    result = date_manipulator.find_last_friday(initial_date)
    expected = datetime(2023, 10, 27)  # Last Friday before the given date
    assert result == expected



def test_convert_to_yyyymm_with_datetime():
    date = datetime(2023, 3, 15)
    result = convert_to_yyyymm(date)
    assert result == '202303'

def test_convert_to_yyyymm_with_valid_string():
    date = '2023-03-15'
    result = convert_to_yyyymm(date)
    assert result == '202303'

def test_convert_to_yyyymm_invalid_string():
    date = '2023/03/15'
    with pytest.raises(ValueError, match="Input date is not in 'yyyy-mm-dd' format."):
        convert_to_yyyymm(date)

def test_convert_to_yyyymm_with_edge_dates():
    # Leap year date
    date = '2020-02-29'
    result = convert_to_yyyymm(date)
    assert result == '202002'

    # End of year date
    date = '2023-12-31'
    result = convert_to_yyyymm(date)
    assert result == '202312'



def test_convert_to_yyyy_mm_dd_with_datetime():
    date = datetime(2023, 3, 1)
    result = convert_to_yyyy_mm_dd(date)
    assert result == '2023-03-01'

def test_convert_to_yyyy_mm_dd_with_valid_string():
    date_str = '202303'
    result = convert_to_yyyy_mm_dd(date_str)
    assert result == '2023-03-01'

def test_convert_to_yyyy_mm_dd_invalid_string_length():
    date_str = '2023-03'
    with pytest.raises(ValueError, match="Input date is not in 'yyyymm' format."):
        convert_to_yyyy_mm_dd(date_str)

def test_convert_to_yyyy_mm_dd_invalid_characters():
    date_str = '20A303'
    with pytest.raises(ValueError, match="Input date is not in 'yyyymm' format."):
        convert_to_yyyy_mm_dd(date_str)

def test_convert_to_yyyy_mm_dd_invalid_month():
    date_str = '202313'
    with pytest.raises(ValueError, match="Invalid year or month in 'yyyymm' format."):
        convert_to_yyyy_mm_dd(date_str)



def test_validate_date_format_correct():
    date_str = "2023-03-15"
    format = "%Y-%m-%d"
    assert validate_date_format(date_str, format) == True

def test_validate_date_format_incorrect_format():
    date_str = "2023/03/15"
    format = "%Y-%m-%d"
    assert validate_date_format(date_str, format) == False

def test_validate_date_format_incorrect_date():
    date_str = "2023-02-30"
    format = "%Y-%m-%d"
    assert validate_date_format(date_str, format) == False

def test_validate_date_format_edge_case_leap_year():
    date_str = "2020-02-29"
    format = "%Y-%m-%d"
    assert validate_date_format(date_str, format) == True

def test_validate_date_format_empty_string():
    date_str = ""
    format = "%Y-%m-%d"
    assert validate_date_format(date_str, format) == False

def test_validate_date_format_different_format():
    date_str = "15/03/2023"
    format = "%d/%m/%Y"
    assert validate_date_format(date_str, format) == True

def test_validate_date_format_non_date_string():
    date_str = "not-a-date"
    format = "%Y-%m-%d"
    assert validate_date_format(date_str, format) == False



def test_get_current_date_format():
    current_date = get_current_date()
    try:
        datetime.strptime(current_date, '%Y-%m-%d')
        assert True  # If parsing succeeds, the format is as expected
    except ValueError:
        assert False  # If parsing fails, the format is not correct

def test_get_current_date_value():
    current_date = get_current_date()
    expected_date = datetime.now().strftime('%Y-%m-%d')
    assert current_date == expected_date



def test_parse_date_string_correct_format():
    date_str = "2023-03-15"
    format = "%Y-%m-%d"
    result = parse_date_string(date_str, format)
    expected = datetime(2023, 3, 15)
    assert result == expected

def test_parse_date_string_incorrect_format():
    date_str = "2023/03/15"
    format = "%Y-%m-%d"
    with pytest.raises(ValueError, match=r"Date string '2023/03/15' does not match format '%Y-%m-%d'"):
        parse_date_string(date_str, format)

def test_parse_date_string_invalid_date():
    date_str = "2023-02-30"
    format = "%Y-%m-%d"
    with pytest.raises(ValueError, match=r"does not match format '%Y-%m-%d'"):
        parse_date_string(date_str, format)

def test_parse_date_string_edge_case_leap_year():
    date_str = "2020-02-29"
    format = "%Y-%m-%d"
    result = parse_date_string(date_str, format)
    expected = datetime(2020, 2, 29)
    assert result == expected

def test_parse_date_string_different_format():
    date_str = "15/03/2023"
    format = "%d/%m/%Y"
    result = parse_date_string(date_str, format)
    expected = datetime(2023, 3, 15)
    assert result == expected



def test_format_date_correct_format():
    date = datetime(2023, 3, 15)
    format_str = "%Y-%m-%d"
    result = format_date(date, format_str)
    expected = "2023-03-15"
    assert result == expected

def test_format_date_with_different_format():
    date = datetime(2023, 3, 15)
    format_str = "%d/%m/%Y"
    result = format_date(date, format_str)
    expected = "15/03/2023"
    assert result == expected

def test_format_date_invalid_type():
    date = "2023-03-15"  # Not a datetime object
    format_str = "%Y-%m-%d"
    with pytest.raises(TypeError, match=r"The 'date' parameter must be a datetime object."):
        format_date(date, format_str)

def test_format_date_with_time():
    date = datetime(2023, 3, 15, 14, 30)
    format_str = "%Y-%m-%d %H:%M"
    result = format_date(date, format_str)
    expected = "2023-03-15 14:30"
    assert result == expected
