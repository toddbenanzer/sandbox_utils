import pandas as pd
import sqlite3

# Example 1: Basic Template Rendering
template_dir = '/path/to/templates'
builder = SQLTemplateBuilder(template_dir)

# Load and render a simple template with context
try:
    rendered_query = builder.render_template('customer_query.sql', {'customer_id': 123})
    print(rendered_query)
except Exception as e:
    print(e)

# Example 2: Handling Template Not Found
try:
    rendered_query = builder.render_template('non_existent_template.sql', {'param': 'value'})
    print(rendered_query)
except TemplateNotFound as tnfe:
    print(tnfe)

# Example 3: Using Template with Multiple Variables
try:
    context = {'customer_id': 456, 'account_status': 'active'}
    rendered_query = builder.render_template('account_status_query.sql', context)
    print(rendered_query)
except Exception as e:
    print(e)

# Example 4: Providing Insufficient Context Variables
try:
    rendered_query = builder.render_template('transactions_query.sql', {})
    print(rendered_query)
except Exception as e:
    print(e)


# Example 1: Adding a Single CTE
cte_manager = CTEManager()
cte_manager.add_cte('customers', 'SELECT id, name FROM customer_table')
print(cte_manager.get_cte_query())
# Output: WITH customers AS (SELECT id, name FROM customer_table)

# Example 2: Adding Multiple CTEs
cte_manager.add_cte('accounts', 'SELECT id, customer_id FROM account_table')
print(cte_manager.get_cte_query())
# Output: WITH customers AS (SELECT id, name FROM customer_table), accounts AS (SELECT id, customer_id FROM account_table)

# Example 3: Retrieve Query with No CTEs Added
empty_cte_manager = CTEManager()
print(empty_cte_manager.get_cte_query())
# Output: (an empty string)



# Example 1: Aggregating transaction data to account level
transactions_data = pd.DataFrame({
    'account_id': [101, 101, 102, 103, 103, 103],
    'transaction_amount': [150, 200, 350, 100, 200, 100]
})

query_aggregator = QueryAggregator()
account_level_data = query_aggregator.aggregate_transactions_to_account(transactions_data)
print(account_level_data)
# Output:
#    account_id  total_transactions  avg_transaction_amount
# 0         101                350                    175.0
# 1         102                350                    350.0
# 2         103                400                    133.3...

# Example 2: Aggregating account data to customer level
accounts_data = pd.DataFrame({
    'customer_id': [1, 1, 2, 3, 3],
    'account_balance': [1000, 1500, 2000, 2500, 3500]
})

customer_level_data = query_aggregator.aggregate_accounts_to_customer(accounts_data)
print(customer_level_data)
# Output:
#    customer_id  total_balance
# 0            1           2500
# 1            2           2000
# 2            3           6000

# Example 3: Handling missing columns in data
try:
    incomplete_data = pd.DataFrame({'account_id': [101, 102], 'missing_amount': [10, 20]})
    query_aggregator.aggregate_transactions_to_account(incomplete_data)
except ValueError as e:
    print(e)
# Output: Input data must contain columns: {'account_id', 'transaction_amount'}


# Example 1: Using Table Names Only
query1 = build_join_query('customer', 'account', 'transaction', '2023-10-01')
print(query1)

# Example 2: Using Detailed Dictionary Inputs
customer = {'table': 'customer', 'fields': ['id', 'name']}
account = {'table': 'account', 'fields': ['id', 'balance']}
transaction = {'table': 'transaction', 'fields': ['id', 'amount', 'date']}
query2 = build_join_query(customer, account, transaction, '2023-12-31')
print(query2)

# Example 3: Using None for Date Snapshot to Omit Date Filtering
query3 = build_join_query(customer, account, transaction, None)
print(query3)

# Example 4: Handling an Error Due to Missing Table Information
try:
    faulty_query = build_join_query({'table': 'customer'}, '', 'transaction', '2023-11-01')
except ValueError as e:
    print(e)


# Example 1: Applying a date filter to a query without a WHERE clause
query1 = "SELECT * FROM orders"
filtered_query1 = filter_by_date(query1, '2023-01-01', '2023-12-31')
print(filtered_query1)
# Output: "SELECT * FROM orders WHERE date_column BETWEEN '2023-01-01' AND '2023-12-31'"

# Example 2: Applying a date filter to a query with an existing WHERE clause
query2 = "SELECT * FROM orders WHERE order_status = 'shipped'"
filtered_query2 = filter_by_date(query2, '2023-01-01', '2023-12-31')
print(filtered_query2)
# Output: "SELECT * FROM orders WHERE order_status = 'shipped' AND date_column BETWEEN '2023-01-01' AND '2023-12-31'"

# Example 3: Catching an error with an improperly formatted start date
try:
    faulty_query = "SELECT * FROM transactions"
    filter_by_date(faulty_query, '01-01-2023', '2023-12-31')
except ValueError as e:
    print(e)
# Output: "start_date and end_date must be in 'YYYY-MM-DD' format."

# Example 4: Catching an error with an improperly formatted end date
try:
    faulty_query = "SELECT * FROM transactions"
    filter_by_date(faulty_query, '2023-01-01', '12-31-2023')
except ValueError as e:
    print(e)
# Output: "start_date and end_date must be in 'YYYY-MM-DD' format."


# Example 1: Applying a date snapshot to a query without a WHERE clause
query1 = "SELECT * FROM customers"
snapshot_query1 = specify_date_snapshot(query1, '2023-10-01')
print(snapshot_query1)
# Output: "SELECT * FROM customers WHERE date_column = '2023-10-01'"

# Example 2: Applying a date snapshot to a query with an existing WHERE clause
query2 = "SELECT * FROM orders WHERE order_status = 'completed'"
snapshot_query2 = specify_date_snapshot(query2, '2023-10-01')
print(snapshot_query2)
# Output: "SELECT * FROM orders WHERE order_status = 'completed' AND date_column = '2023-10-01'"

# Example 3: Handling an invalid date format
try:
    faulty_query = "SELECT * FROM transactions"
    specify_date_snapshot(faulty_query, '10/01/2023')
except ValueError as e:
    print(e)
# Output: "date_snapshot must be in 'YYYY-MM-DD' format."

# Example 4: Applying a date snapshot to an empty query
query4 = ""
snapshot_query4 = specify_date_snapshot(query4, '2023-10-01')
print(snapshot_query4)
# Output: "WHERE date_column = '2023-10-01'"



# Example 1: Executing a SELECT query
def example_select_query():
    connection = sqlite3.connect(':memory:')
    cursor = connection.cursor()
    cursor.execute("CREATE TABLE users (id INTEGER, name TEXT)")
    cursor.execute("INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob')")
    connection.commit()

    query = "SELECT * FROM users"
    result = execute_query(query, connection)
    print(result)
    # Output: [(1, 'Alice'), (2, 'Bob')]

    connection.close()

# Example 2: Executing an INSERT query
def example_insert_query():
    connection = sqlite3.connect(':memory:')
    cursor = connection.cursor()
    cursor.execute("CREATE TABLE products (id INTEGER, name TEXT, price REAL)")
    connection.commit()

    query = "INSERT INTO products (id, name, price) VALUES (1, 'Laptop', 999.99)"
    execute_query(query, connection)

    # Verify insertion by selecting
    verify_query = "SELECT * FROM products"
    result = execute_query(verify_query, connection)
    print(result)
    # Output: [(1, 'Laptop', 999.99)]

    connection.close()

# Example 3: Handling query execution error
def example_query_with_error():
    connection = sqlite3.connect(':memory:')
    try:
        query = "SELECT * FROM non_existent_table"
        execute_query(query, connection)
    except Exception as e:
        print(e)
    # Output: An error occurred during query execution: ...

    connection.close()

# Run the examples
example_select_query()
example_insert_query()
example_query_with_error()
