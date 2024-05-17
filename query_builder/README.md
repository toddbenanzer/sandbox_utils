# Python SQL Utilities

This Python package provides a set of utility functions for working with SQL queries and databases. It includes functions for generating SQL queries, manipulating data in a database, and retrieving metadata information about database tables.

## Installation

To use this package, you will need to have Python installed. You can install the package using pip:

```bash
pip install python-sql-utils
```

## Usage

To use the functions provided by this package, you will need to import them into your Python script:

```python
from sql_utils import generate_sql_query, build_cte, join_tables, filter_data, aggregate_data, order_result_set, limit_rows, calculate_row_count, calculate_sum, calculate_average, calculate_min_value, calculate_max_value, get_distinct_values, calculate_percentile, calculate_rank, calculate_row_number, calculate_dense_rank, calculate_lag, calculate_lead_value, running_total, pivot_data
```

## Functions

### `generate_sql_query(template_string: str, data: dict) -> str`

This function takes a template string and a dictionary of data and returns a rendered SQL query. The template string can contain placeholders for the data values using double curly braces (e.g. `{{ value }}`). The function uses the Jinja2 templating engine to render the query.

### `build_cte(cte_name: str, query: str) -> str`

This function takes a common table expression (CTE) name and a query and returns a CTE string. The CTE string is in the format `{cte_name} AS ({query})`.

### `join_tables(tables: List[str], join_conditions: List[str]) -> str`

This function takes a list of table names and a list of join conditions and returns a SQL query that joins the tables based on the specified conditions. The join conditions should be in the format `table1.column = table2.column`. The function uses the Jinja2 templating engine to render the query.

### `filter_data(table_name: str, conditions: str) -> str`

This function takes a table name and a set of conditions and returns a SQL query that filters the data based on the specified conditions. The conditions should be in valid SQL syntax. The function uses the Jinja2 templating engine to render the query.

### `aggregate_data(group_by: str, select_columns: str) -> str`

This function takes a grouping column and a set of select columns and returns a SQL query that aggregates the data based on the specified columns. The select columns should be in valid SQL syntax. The function uses the Jinja2 templating engine to render the query.

### `order_result_set(template_str: str, columns: List[str]) -> Any`

This function takes a template string and a list of columns and returns an ordered result set based on the specified columns. The template string should contain placeholders for the column names using double curly braces (e.g. `{{ column }}`). The function uses the Jinja2 templating engine to render the query.

### `limit_rows(query_template: str, limit: int) -> str`

This function takes a query template file path and a limit value and returns a SQL query that limits the number of rows returned by the query. The query template should contain placeholders for the limit value using double curly braces (e.g. `{{ limit }}`). The function uses the Jinja2 templating engine to render the query.

### `calculate_row_count(table_name: str) -> str`

This function takes a table name and returns a SQL query that calculates the number of rows in the table. The function uses the Jinja2 templating engine to render the query.

### `calculate_sum(table_name: str, column_name: str) -> str`

This function takes a table name and a column name and returns a SQL query that calculates the sum of the values in the specified column. The function uses the Jinja2 templating engine to render the query.

### `calculate_average(column: str, table: str) -> str`

This function takes a column name and a table name and returns a SQL query that calculates the average of the values in the specified column. The function uses the Jinja2 templating engine to render the query.

### `calculate_min_value(table_name: str, column_name: str) -> Any`

This function takes a table name and a column name and returns the minimum value in the specified column. The function uses the Jinja2 templating engine to render the query.

### `calculate_max_value(table: str, column: str) -> Any`

This function takes a table name and a column name and returns the maximum value in the specified column. The function uses the Jinja2 templating engine to render the query.

### `get_distinct_values(column: str, table: str) -> List[Any]`

This function takes a column name and a table name and returns a list of distinct values in the specified column.

### `calculate_percentile(table_name: str, column_name: str, percentile: float) -> str`

This function takes a table name, a column name, and a percentile value and returns a SQL query that calculates the value at the specified percentile in the specified column. The function uses the Jinja2 templating engine to render the query.

### `calculate_rank(table_name: str, criteria: str) -> Any`

This function takes a table name and a criteria for ranking and returns a SQL query that calculates the rank based on the specified criteria. The function uses the Jinja2 templating engine to render the query.

### `calculate_row_number(criteria: str) -> str`

This function takes a criteria for row numbering and returns a SQL query that calculates the row number based on the specified criteria. The function uses the Jinja2 templating engine to render the query.

### `calculate_dense_rank(criteria: str) -> str`

This function takes a criteria for dense ranking and returns a SQL query that calculates the dense rank based on the specified criteria. The function uses the Jinja2 templating engine to render the query.

### `calculate_lag(column_name: str, order_by_column: str) -> Any`

This function takes a column name and an order by column and returns a SQL query that calculates the lag value of a column in a table. The function uses the Jinja2 templating engine to render the query.

### `calculate_lead_value(column_name: str, table_name: str) -> str`

This function takes a column name and a table name and returns a SQL query that calculates the lead value of a column in a table. The function uses the Jinja2 templating engine to render the query.

### `running_total(table_name: str, column: str) -> Any`

This function takes a table name and a column name and returns a SQL query that calculates the running total of values in the specified column. The function uses the Jinja2 templating engine to render the query.

### `pivot_data(data: List[dict], pivot_column: str, value_column: str) -> pd.DataFrame`

This function takes a list of dictionaries representing data, a pivot column, and a value column, and returns a pivoted dataframe based on the specified criteria.

## Examples

Here are some examples of how to use these functions:

```python
# Generate an SQL query
template_string = "SELECT * FROM {{ table }} WHERE id = {{ id }}"
data = {'table': 'customers', 'id': 10}
sql_query = generate_sql_query(template_string, data)

# Build a CTE
cte_name = 'cte1'
query = 'SELECT * FROM customers WHERE age > 18'
cte = build_cte(cte_name, query)

# Join tables
tables = ['customers', 'orders']
join_conditions = ['customers.id = orders.customer_id']
sql_query = join_tables(tables, join_conditions)

# Filter data
table_name = 'customers'
conditions = 'age > 18 AND gender = "Male"'
sql_query = filter_data(table_name, conditions)

# Aggregate data
group_by = 'country'
select_columns = 'city, COUNT(*) AS count'
sql_query = aggregate_data(group_by, select_columns)

# Order result set
template_str = 'SELECT * FROM table ORDER BY {{ column }} DESC'
columns = ['name']
result_set = order_result_set(template_str, columns)

# Limit rows
query_template = 'SELECT * FROM table LIMIT {{ limit }}'
limit = 10
sql_query = limit_rows(query_template, limit)

# Calculate row count
table_name = 'customers'
sql_query = calculate_row_count(table_name)

# Calculate sum
table_name = 'transactions'
column_name = 'amount'
sql_query = calculate_sum(table_name, column_name)

# Calculate average
column = 'amount'
table = 'transactions'
sql_query = calculate_average(column, table)

# Calculate min value
table_name = 'products'
column_name= 'price'
min_value= calculate_min_value(table_name,column_name)

# Calculate max value
table= 'products '
column= price 
max_value= calculate_max_value (table,column )


# Get distinct values
column ='category '
table ='products '
distinct_values= get_distinct_values(column , table)


#Calculate percentile 
table name='sales '
column name ='amount '
percentile=0.90

sql qyery=calculate_percentile (table name ,column name, percentile)


# Calculate rank
table_name = 'employees'
criteria = 'salary DESC'
result = calculate_rank(table_name, criteria)

# Calculate row number
criteria = 'age ASC'
sql_query = calculate_row_number(criteria)

# Calculate dense rank
criteria = 'salary DESC'
sql_query = calculate_dense_rank(criteria)

# Calculate lag
column_name = 'sales'
order_by_column = 'date'
result = calculate_lag(column_name, order_by_column)

# Calculate lead value
column_name = 'sales'
table_name = 'transactions'
query = calculate_lead_value(column_name, table_name)

# Running total
table_name= 'sales '
column='amount '

result= running_total(table name,column)



# Pivot data
data = [{'date': '2022-01-01', 'category': 'A', 'value': 10},
        {'date': '2022-01-01', 'category': 'B', 'value': 20},
        {'date': '2022-01-02', 'category': 'A', 'value': 30},
        {'date': '2022-01-02', 'category': 'B', 'value': 40}]
pivot_column = 'date'
value_column = 'value'
df_pivoted = pivot_data(data, pivot_column, value_column)
```

For more detailed usage examples, please refer to the example scripts provided in the `examples` directory.

## Contributing

Contributions to this package are welcome. Please submit a pull request or create an issue if you have any suggestions or improvements.