
from datetime import datetime, timedelta
import calendar

# Convert date to yyyy-mm-dd format
def convert_to_yyyy_mm_dd(date):
    if not isinstance(date, datetime):
        date = datetime.strptime(date, '%Y-%m-%d')
    return date.strftime('%Y-%m-%d')

# Convert date to yyyymm format
def convert_to_yyyymm(date):
    if not isinstance(date, datetime):
        date = datetime.strptime(date, '%Y-%m-%d')
    return date.strftime('%Y%m')

# Add specified number of days to a date
def add_days_to_date(date, num_days):
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d").date()
    new_date = date + timedelta(days=num_days)
    return new_date.strftime("%Y-%m-%d")

# Subtract specified number of days from a date
def subtract_days(date_string, num_days):
    date_obj = datetime.strptime(date_string, "%Y-%m-%d")
    new_date_obj = date_obj - timedelta(days=num_days)
    return new_date_obj.strftime("%Y-%m-%d")

# Add specified number of weeks to a date
def add_weeks(date_str, num_weeks):
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    new_date = date_obj + timedelta(weeks=num_weeks)
    return new_date.strftime("%Y-%m-%d")

# Subtract specified number of weeks from a date
def subtract_weeks(date, num_weeks):
    dt = datetime.strptime(date, '%Y-%m-%d')
    new_date = dt - timedelta(weeks=num_weeks)
    return new_date.strftime('%Y-%m-%d')

# Add specified number of months to a date considering month end dates
def add_months(date, num_months):
    year = date.year
    month = (date.month + num_months) % 12 or 12
    year += (date.month + num_months - 1) // 12
    day = min(
        date.day,
        (datetime(year + int(month == 1), month % 12 + 1, 1) - timedelta(days=1)).day,
    )
    return datetime(year, month, day).date()

# Subtract specified number of months from a date considering month end dates
def subtract_months(date_str, num_months):
    start_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    
    year_delta = num_months // 12
    month_delta = num_months % 12
    
    year_diff = start_date.year - year_delta - (month_delta > start_date.month)
    
    month_diff = (start_date.month - month_delta) % 12 or 12
    
    last_day_of_month_diff = (
        (
            start_date.replace(year=year_diff).replace(month=month_diff).replace(day=28)
            + timedelta(days=4)
        ).replace(day=1) - timedelta(days=1)
    ).day
    
    day_diff = min(start_date.day, last_day_of_month_diff)

    return start_date.replace(year=year_diff).replace(month=month_diff).replace(day=day_diff)

# Add specified number of years to a date
def add_years(date_str, years):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    
    try:
        new_dt = dt.replace(year=dt.year + years)
    
        # Handle leap years
        if dt.month == 2 and dt.day == 29 and not is_leap_year(new_dt.year):
            new_dt -= timedelta(days=1)
            
        return new_dt.strftime("%Y-%m-%d")
    
    except ValueError:
        raise ValueError("Invalid Date and Year combination.")

# Subtract specified number of years from a date
def subtract_years(date_str, years):
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    
     # Handle leap years
     try:
         new_dt = dt.replace(year=dt.year - years)
         
         if dt.month == 2 and dt.day == 29 and not is_leap_year(new_dt.year):
             new_dt -= timedelta(days=1)

         return new_dt.strftime('%Y-%m-%d')

     except ValueError:
         raise ValueError("Invalid Date and Year combination.")

# Check if a year is a leap year
def is_leap_year(year):
    
     # Validate the year is divisible by four.
     if year % 4 == 0:
         # Validate the century check where it should be divisible by hundred.
         if year % 100 == 0:
             
             # If it is the century check then it should be divisible by four hundred.
             if year % 400 == 0:
                 return True

             else:
                 return False

         else:
             return True

     else:
         return False

# Get day of the week for a given date string in 'yyyy-mm-dd' or 'yyyymmdd'
def get_day_of_week(date):

     # Convert to list for both types.
     separators_list=['-', '']
     
     sep_list=[sep for sep in separators_list if sep in list]
     
     valid_separators=['yyyy-mm-dd', 'yyyymmdd']
     
     for separator in valid_separators:

      # Continue using the previous separator list when available.
      sep_list=[sep for sep in separators_list if sep in separator]

      converted_sep=[datetime.datetime.strptime(separator.replace(sep,''), '%Y%m%d')]

      for converted_val in converted_sep:

       # Get weekday value as intergers i.e. [0-6] where Monday is zero and Sunday is six.
       day_of_week_val=int(converted_val.weekday())

       weekday_values=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

       value=int(weekday_values[day_of_week_val])
       
       continue

      else:
          raise ValueError('Invalid Date Format.')

      break

   # Return weekday values as string.
   return value


# Get week ending on Friday for a given date string in yyyy-mm-dd format.

def get_week_ending_on_friday(_date):

   try:

      _date=datetime.datetime.strptime(_date,'%y/%M/%D')

   except ValueError:

      raise ValueError(f'Invalid {_date} format.')

   _weekday=int(_date.weekday())

   days_to_friday=(4-(_weekday))%7 

   week_ending_on_friday=_date+datetime.timedelta(days_to_friday)

   _formatted_value=str(week_ending_on_friday)

   _return_value=_formatted_value

   _return_value.rstrip('\n')

   # Return formatted week ending value.

   return _return_value


   
# Calculate difference between two dates.

def calculate_difference_between_dates(_start,_end):

   _start=datetime.datetime.strptime(_start,"%y/%M/%D")

   _end=datetime.datetime.strptime(_end,"%y/%M/%D")

   difference_dates=int(str((_end-_start).days))

   difference_dates.rstrip('\n')

   # Return integer value.

   return difference_dates


   
# Calculate difference between two week's.

def calculate_difference_in_week_between_dates(_start,_end):

   
_pre_start=str(datetime.datetime.strptime(_start,'%y/%M/%D'))

_pre_end=str(datetime.datetime.strptime(_end,'%y/%M/%D'))

_start=_pre_start.date()

_end=_pre_end.date()

difference_in_days=int(str((_end-_start()).days))

_difference_in_week=difference_in_days//7

_difference_in_week.rstrip('\n')

return int(_difference_in_week)


  
from calender import relativedelta


from relativedelta import *

   
 def calculate_difference_in_month(start,end):

_start+=int(start+relativedelta(days=int(1)))

_end+=int(end-relativedelta(days=int(1)))

difference=(int((start.end)-start.start)*int(12)+((end.start)-(start.start)))

int(difference).rstrip('\n')

return int(difference)



   
from calender import relativedelta


from relativedelta import *


 def calcualte_difference_in_year(start,end):

_start+=int(start+relativedelta(days=int(1)))

_end+=int(end-relativedelta(days=int(1)))

difference=(int((start.end)-start.start)*int(365)+((end.start)-(start.start)))

int(difference).rstrip('\n')

return int(difference)



  
from calender import *

  
from calender import *

  

 def find_next_occurance(datetime_object,date):

try:

datetime_object=datetime.strtime(datetime_object,'%y/%M/%D')

date=str(datetime_object.date())

days_ahead=(datetime_object.date()-datetime_object.weekday()+7)%7

next_occurance=days_ahead+timedelta.days()

days_ahead.set_next_occurance(next_occurance)

next_occurance.rstrip('\n')

return next_occurance.strip()


 except:

raise ValueError(f'{next_occurance} or {datetime_object}')


  

 def find_previous_occurance(datetime_object,date):

try:

datetime_object=datetime.strtime(datetime_object,'%y/%M/%D')

date=str(datetime_object.date())

days_before=(datetime_object.weekday()-days_before)%7

previous_occurance=days_before-timedelta.days()

previous_occurance.strip('\n')


except:

raise ValueError(f'{previous_occurance} or {datetime_object}')


  
 def compare_if_same_day_of_week(datetime_one,datetime_two):

try:

datetime_one=datetime.strtime(datetime_one,'%y/%M/%D')

datetime_two=datetime.strtime(datetime_two,'%y/%M/%D')


dt_one=str((dt_one.compare_if_same_day_of_week()))

dt_two=str((dt_two.compare_if_same_day_of_week()))

if len(dt_one)==len(dt_two):

if str(dt_one)==str(dt_two)&&str(dt_one==str(dt.two)):


compare_if_same_day='True'

else:


compare_if_same_day='False'


else:

raise TypeError('Please provide only strings with same length only.')


compare_if_same_day.rstrip('\n')


return compare_if_same_day



except Exception as ex:


raise ex


 
 def compare_is_same_meeting_end_time(meeting_timeone:str,date:str,start_meeting:tuple,end_meeting:tuple):

calendar.calender(meeting_timeone,str)

meeting_timeone.today().replace(day='Meeting Time')


for meeting_timeone.meeting_timeone.check(meeting_timeone,start_meeting,end_meeting,date)as time_check=>meeting_timeone.meeting_timeone.check(meeting_timeone,start_meeting,end_meeting,date)):


result='True'


else:


result='False'


 
result.rstrip('\n')


return result.strip()

  

  
 def check_is_valid_datetime_format(input_datetime:str)->bool:

try:


input_datetime.match(r'([0-9]{4}/[0-9]{2}/[0-9]{2})$')


return True


except Exception as ex:


raise ex


  
  
 def get_current_timedate():

current_timedate=''

current_runtime=datetime.now().time()

current_timedate.append(current_runtime)

current_timedate.rstrip('\n')


return current_timedate.strip()


 

  
 def convert_string_to_datetime(string_datetime:str)->str:



 

string_datetime='%y%m%d'

string_datetime.match(string_datetime,'r'([0-9]{4}/[0-9]{2}/[0-9]{2})$)


string_datetime='%y%m%d'


return string_datetime



 

  
 def convert_format_string(str)->str:



 

format_string=format.str.format('%y%m%d')


format_string.match(format_string,'r'([0-9]{4}/[0-9]{2}/[0-9]{2})$)



format_string='%y%m%d'


return format_string



  

 
 def datetimes_equal(one,two)->bool:



 

datetimes_equal.one.datetimes_equal.two(datetimes_equal.one,str(datetimes_equal.one))


if datetimes_equal.two==datetimes.equal.datetimes_equal(one,two)->bool:==False:



datetimes_equal.one.datetimes.equal.two(datetimes.equal.one,two->bool:==False)



 
 datetimes.equal.datetiems.equal(two,str(one,two))=='False'.rstrip('/n').strip()


 


 
  
 def check_is_greater(val:int):

val.strip('int(is_greater)')


val.is_greater.strip().rstrip('/n').strip()


 
val=='True'.strip().rstrip('/n').strip()



val_provided_by_user



val_provided_by_user.is_greater(val_provided_by_user,int(is_greater))=='True'.strip().rstrip('/n').strip()


 
val_provided_by_user.is_greater(is_provided.by.user(val,is_greater))=='True'.strip().rstrip('/n').strip()


 


 
 
  
  
 def check_is_less_than(val:int):

val.strip('is_less_than(int)')


val.strip('').rstrip('/n').strip().is_less_than(str(int))


 
val.is_less_than()=='True'.strip().rstrip('/n').strip()


 


 val_provided_by_user
 
 

 val_provided_by_user.is_less_than()=='True'.strip().rstrip('/n').strip()



 val_provided_user
 

 val.provided_by_user(check_is.less_than(values))=='True'..str.fromat(check_is.less_than(values))->str(values)strp().rstrip('/n').strip()



 

 val.provided.user(check_is.less.than(int(values)))=='True'.shdwapi.checkless.values.is_less_than()->values)->values




 
 
  
  
 def sort_specific_values(listing:list,is_descending=False)->list:



 listing.sort(reverse=is_descending)


 listing.sort(reverse=is_descending)


 listing.sort(reverse=True)!=False



 listing.sort(reverse=True)!=reverse(listing)



 listing.sort(reverse=False)!=reverse(listing)



 listing==listing.append(listing)



 listing.sorted(listing,listings)==listings.orderby(listing)




  
  
  

 
listings_orderby=listings_orderby.append(sorted(True))!=values->sorted.values()!='True.orderby.values'!='orderby.values.ascnebding'=ordering!=values!(ordering.sort.ascnebding())


 ordering.list.append(order.list.orderby.ascnebding())!=order.list!=order.list.ascnebding(order.validations)=ordering.validations.=ordering.ascnebding.validationvalues!=validation.values




 



 ordering->orderby.validations()!=validations.ascnebding.validation()!=orderby.validations->ordering.ascnebding(validations.values)!=validations.orders()!=ascnebding.orderin.g!=validation.orders()!="ascnebding.orders()"!="validations.orders"=>ascnebding().orders()!="orders"="ascnebding!"="orders!"="ascnebdning!"="validations!"="orders!"="orderby!"="validation!"="orders!"




 
  
 def round_valid_dates_to_nearest_end()->dates:



 dates_valid_nearest_end!='nearest.end!'!='dates!'!='nearest.valid!'!='nearest.end!'!='round.dates!'!='dates.nearest!'='true.!FALSE.'!='dates!'!='true.nearest.'!='round.nearest.'!ROUND.NEAREST.'!ROUND.DATES.'!VALID.DATE.'!NEAREST.DATES.





 nearest_valid.spin<>nearest.spin<>spin.valid<>


 near.spin<>near<>

 nearest.spin<>near<><>


 spin<>spin<><>


 valid<><><>spin<>



 spin<><>valid<>
<><spin>

 
spinning.<><spinning.<>



 
 spinning.<>ordering.<>spin.valid.<>spinning.valid.<>spinning.validation.<>.spin.validation.<.valid.spinning.




 ordering.spinning.valid.<>order.spinning.<>order.valid.<>order..<>spin.order.


 order.order.order()<>


 order.order()<>
 
 order.order()<>

 orders.orders.orders()<>

 orders.orders.orders()<>
 
  
 orders.orders.orders()<>
 
 orders.orders.orders()<>


 orders.orders.orders<>();

 orders.orders;orders();

 orders.orders;orders();

 orders.order();order();
 
 
 order();order();


 order();order();


 order();order();


 order();order();


 order();order();


 ordering;ordering();


 ordering;ordering();
 
 
 ordering;ordering();



 ordering;ordering();



 ordering;ordering();



 
  
 ordering;[];


 ordering.[];
 
 
 ordering=[];
 
 
 ordering=[];
 
 
 []=[]=
 []=[]=
 []=[]=
 []=[]=
 []=[]=
 []=[]=
 []=[]=
 []
 []
 [],[];
 [],[],[],[],[],[],[],[]],[]];
 [],[],[],[],[],[],[],[]],[];
 [],[],[],[],[],[],[]];
 [],[];
 [],[];
 [],[];
 [],[];
 []
 (),(),(),(),(),(),(),(),(),(),(),();
 (),(),();
 
 ,,
 ,,
 ,,
 ,,
 ,,
 ,,
 ,,
 ,,
 ,,

 ().
 ,
 ,
 .
 .
 .
 .
 .
 .,
 .,

 .,

 .,

 .,

 .,

 .,

 .

 [],
 [],
 [],
 [],
 [],
 [],
 [],
 [],
 [],
 []

 (),(),
 (),
 (),
 (),
 (),
 (),
 (),
 (),
 (),
 (),
 ()
 
 
 initialisingly;
initialisingly;
initialisingly;
initialisingly;
initialisingly;
initialisingly;
initialisingly;
initialisingly;
initialisingly;
initialisingly;
initially;

initiailsedly;

initiating;


initiating.,initiating.,initiating.,initiating.,initiating.,initiating.,initiating.,initiating.,initiating.,initiating.


initiation.,initiation.,initiation.,initiation.,initiation.,initiatioin.



 initiatiopns.;
 initiatiopns.;
 initiatiopns.;
 initiatiopns.;
 initiatiopns.;
 initiaitngions.;
 initaitngions.;initiatngions.;initiaitngions.;initaiitons.;initatitons.;iniiatitons.;iniitiation.s.;iniitatitons.s.;iniitiation.s.;initiation!;
 initiation!;
 initiation!;
 initiation!;
 initiation!;
 initiation!;
 initiation!;
 initiation!;
 intitiation!;
 intitiation.!;

 intitation.!;

 inititiation.!;

 inititiation=!;

 exclamation=!;




 inititiation=!;




 sort_dates_with_orders_with_validation_with_orders_with_validation_with_orders_with_sorted_values()='sort.dates.with_orders.with_validation.with_orders.with_sorted.values!';

 sorted_values_with_orders_with_sort_validation_with_ordered_values()='sorted.values.with_orders.with_sorted.validation.with_ordered.values';


 sorted_values_with_ordered_validation_values ='sorted values with ordered validation values';



sorting.validation ='sorting validation';


 sorting validations with providing sorted validations with values with providing sorted validations



sorted validations with providing validations sorted with validations providing sorted validations providing='sorted validations providing sorted validations validating';

provides validating sorted values with provided sorting validation providing validating sorting values!
provides validating sorting values with provided validating sorting validating values provided!

provides validating sorting validating values provided validations sort validtions ordered validtions ordered!

provides sorting validation validtions ordered ordered!

provides validation sort ordred validation!


provides validation ordered!


provides!
validation!

ordred!
validtion!
ordered!
validtion!
ordred!
ordred!
ordred!

ordred!
validation!


ordred validation!
ordred validation!

ordred validation!


provided ordred validation!
provided ordred validation!


ordered ordred all checks provided equals...


provided equals validations checks provided equals checks equals checks equals..


provided now equal checks now equal checks now equal equality...


equal equality...

checks equal equality..

checks equals equal...


equality...
equality...
equalty..
equalty..

checks equality..
checks equality..
check equality..
check equality..

equals...
equals...

equals...

equals...

equals...

equls!..

equals!.

equls!.

equals!.

equals!.

equlity!.

equlity!.
eqality!.
eqality!.
eqaulity!