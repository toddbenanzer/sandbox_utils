
import pytest
from datetime import datetime, timedelta, date


def convert_to_yyyy_mm_dd(input_date):
    # Stub function for demonstration; replace with actual implementation
    if isinstance(input_date, datetime):
        return input_date.strftime('%Y-%m-%d')
    elif isinstance(input_date, str):
        return input_date  # Assuming the input string is already in 'yyyy-mm-dd' format
    else:
        raise ValueError("Invalid date format")


def add_days_to_date(date_obj, num_days):
    # Stub function for demonstration; replace with actual implementation
    return date_obj + timedelta(days=num_days)


def subtract_days(date_str, num_days):
    # Stub function for demonstration; replace with actual implementation
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    return (date_obj - timedelta(days=num_days)).strftime("%Y-%m-%d")


def add_weeks(date_str, num_weeks):
    # Stub function for demonstration; replace with actual implementation
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    new_date = date_obj + timedelta(weeks=num_weeks)
    return new_date.strftime("%Y-%m-%d")


def subtract_weeks(date_str, num_weeks):
    # Stub function for demonstration; replace with actual implementation
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    new_date = date_obj - timedelta(weeks=num_weeks)
    return new_date.strftime("%Y-%m-%d")


def add_months(date_obj, num_months):
    # Stub function for demonstration; replace with actual implementation
    return date_obj + relativedelta(months=num_months)


def subtract_months(date_str, num_months):
    # Stub function for demonstration; replace with actual implementation
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    new_date = date_obj - relativedelta(months=num_months)
    return new_date.strftime("%Y-%m-%d")


def add_years(date_str, num_years):
    # Stub function for demonstration; replace with actual implementation
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    try:
        new_date = date_obj.replace(year=date_obj.year + int(num_years))
        if int(num_years) != num_years:
            fraction_year_days = (datetime(date_obj.year + 1, 1, 1) - datetime(date_obj.year, 1, 1)).days * (num_years % 1)
            new_date += timedelta(days=fraction_year_days)
        return new_date.strftime("%Y-%m-%d")
    except ValueError:
        raise ValueError("Invalid Date")


def subtract_years(date_str, num_years):
    # Stub function for demonstration; replace with actual implementation
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    new_date = date_obj - relativedelta(years=num_years)
    return new_date.strftime("%Y-%m-%d")


def is_leap_year(year):
    # Stub function for demonstration; replace with actual implementation
    if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
        return True
    else:
        return False


def get_day_of_week(date_str):
    # Stub function for demonstration; replace with actual implementation
    try:
        if '-' in date_str:
            date_format = "%Y-%m-%d"
        elif len(date_str) == 8:
            date_format = "%Y%m%d"
        else:
            raise ValueError("Invalid Date Format")

        date_obj = datetime.strptime(date_str, date_format)
        return date_obj.strftime("%A")
    
    except ValueError as e:
        raise ValueError(f"Invalid Date: {e}")


def get_week_ending_on_friday(given_date):
   given_day_of_week=given _date.weekday()
   days_to_friday=4-given_day_of_week if given_day_of_week <=4 else (6-given_day_of_week)+4+1
   next_friday=given _date+timedelta(days=days_to_friday)
   return next_friday

@pytest.mark.parametrize("date_input ,expected",[
         (datetime(2022 ,1 ,10),datetime(2022 ,1 ,14)),
         (datetime(2022 ,1 ,12),datetime(2022 ,1 ,14)),
         (datetime(2022 ,1 ,15),datetime(2022 ,1 ,21)),
         (datetime(2022 ,1 ,20),datetime(2022 ,1 ,21))])

@pytest.mark.parametrize("input_dates",[
      ('2022-01-01','2022-01-10',9),
      ('2022-01-01','2022-01-01',0),
      ('2022-01-10','2022-01-01',-9)])
 
@pytest.mark.parametrize("date_input",[
   ('2023-01-02',True),
   ('2019-12-31',True),
   ('0000-00-00',False),
   ('abcd-ab-cb',False)])

@pytest.mark.parametrize("test_input",[
     ("2019/12/31","Invalid Date Format"),
     ("13/12/22","Invalid Date Format")])

@pytest.mark.parametrize("test_data",[
{"start":"2019/12/31","end":"2019/11/30","result":False},
{"start":"13/12/22","end":"13/12/23" ,"result":False}])

@pytest.mark.parametrize("test_dates",[
{
"input":["2019/12/31",'13/12/22'],
"result":[]
}
{
"input":["abcd-ab-cb","13/22"],
"result":[]}
])
     
 @pytest.mark.parametrize("input_dates",[
 {
 "input":[('2019-11-30')]
 "result":[]
 }
 ])
 
 @pytest.mark.parametrize('date_strings', [
 {"start":"2019/11/30",
 "end":"2019 /10 /20",
 "expected_result":[]}])
 
 @pytest.mark.parametrize('week_dates',
 [
 {"dates":['2019 /11 /30'],"result":[]},
 {"week_dates":[''],"result":[]}
 ])
 
 
 def test_implementation_round_to_month_end():
  
      assert round_to_month_end(datetime(2018 ,6 ,3))==datetime(2018 ,6 ,30)
      assert round_to_month_end(datetime(2000.3.15))==datetime (2000.3.30)
 
 
 def test_round_to_weekending_friday():
 
       assert round_to_week_ending_friday(datetime (2000.3.15))==datetime(2000.3.17)
       assert round_to_week_ending_friday(datetime (1998.5.11))==datetime (1998.5.15)

  
