from jinja2 import Environment, FileSystemLoader, TemplateNotFound
import os
import pandas as pd


class SQLTemplateBuilder:
    """
    A class responsible for loading and rendering Jinja templates for SQL queries.
    """

    def __init__(self, template_dir: str):
        """
        Initializes the SQLTemplateBuilder with a directory path for templates.

        Args:
            template_dir (str): The directory path where SQL template files are located.
        """
        self.template_dir = template_dir
        self.env = Environment(loader=FileSystemLoader(self.template_dir))

    def load_template(self, template_name: str):
        """
        Loads a specific SQL template from the template directory.

        Args:
            template_name (str): The name of the template file to be loaded.

        Returns:
            Template: The Jinja template object.

        Raises:
            TemplateNotFound: If the template file does not exist.
        """
        try:
            return self.env.get_template(template_name)
        except TemplateNotFound:
            raise TemplateNotFound(f"Template '{template_name}' not found in directory '{self.template_dir}'.")

    def render_template(self, template_name: str, context: dict) -> str:
        """
        Renders the loaded template using the provided context.

        Args:
            template_name (str): The name of the template file to be loaded and rendered.
            context (dict): A dictionary containing variables and their values for the template rendering.

        Returns:
            str: The rendered SQL query as a string.

        Raises:
            Exception: For errors related to template rendering.
        """
        try:
            template = self.load_template(template_name)
            return template.render(context)
        except Exception as e:
            raise Exception(f"An error occurred during template rendering: {e}")


class CTEManager:
    """
    A class responsible for managing Common Table Expressions (CTEs) for SQL queries.
    """

    def __init__(self):
        """
        Initializes the CTEManager with an empty dictionary to store CTEs.

        """
        self.ctes = {}

    def add_cte(self, cte_name: str, query: str):
        """
        Adds a new CTE to the collection.

        Args:
            cte_name (str): The name assigned to the CTE, serving as a reference key.
            query (str): The SQL query string defining the CTE.
        """
        self.ctes[cte_name] = query

    def get_cte_query(self) -> str:
        """
        Generates the complete SQL string to incorporate all currently stored CTEs into a query.

        Returns:
            str: Concatenated SQL string of all CTE definitions. Returns an empty string if no CTEs exist.
        """
        if not self.ctes:
            return ""
        
        cte_strings = [f"{name} AS ({query})" for name, query in self.ctes.items()]
        return "WITH " + ", ".join(cte_strings)



class QueryAggregator:
    """
    A class responsible for aggregating data from transactions to accounts and from accounts to customers.
    """

    def __init__(self):
        """
        Initializes the QueryAggregator.
        """
        pass

    def aggregate_transactions_to_account(self, transactions_data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates transaction-level data to the account level.

        Args:
            transactions_data (pd.DataFrame): A DataFrame containing transaction details with fields 'account_id' and 'transaction_amount'.

        Returns:
            pd.DataFrame: Data aggregated at the account level, including total and average transaction amounts per account.

        Raises:
            ValueError: If the required columns are not present in the transactions_data DataFrame.
        """
        required_columns = {'account_id', 'transaction_amount'}
        if not required_columns.issubset(transactions_data.columns):
            raise ValueError(f"Input data must contain columns: {required_columns}")

        # Aggregate data
        aggregated_data = transactions_data.groupby('account_id').agg(
            total_transactions=('transaction_amount', 'sum'),
            avg_transaction_amount=('transaction_amount', 'mean')
        ).reset_index()

        return aggregated_data

    def aggregate_accounts_to_customer(self, accounts_data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates account-level data to the customer level.

        Args:
            accounts_data (pd.DataFrame): A DataFrame containing account details with fields 'customer_id' and 'account_balance'.

        Returns:
            pd.DataFrame: Data aggregated at the customer level, including the total balance per customer.

        Raises:
            ValueError: If the required columns are not present in the accounts_data DataFrame.
        """
        required_columns = {'customer_id', 'account_balance'}
        if not required_columns.issubset(accounts_data.columns):
            raise ValueError(f"Input data must contain columns: {required_columns}")

        # Aggregate data
        aggregated_data = accounts_data.groupby('customer_id').agg(
            total_balance=('account_balance', 'sum')
        ).reset_index()

        return aggregated_data


def build_join_query(customer, account, transaction, date_snapshot):
    """
    Constructs a SQL query that joins customer, account, and transaction tables with date snapshot filtering.

    Args:
        customer: dict | str - A dictionary with customer fields/conditions or a string for the customer table name.
        account: dict | str - A dictionary with account fields/conditions or a string for the account table name.
        transaction: dict | str - A dictionary with transaction fields/conditions or a string for the transaction table name.
        date_snapshot: str - A date string for filtering transactions based on a specific snapshot date.

    Returns:
        str: The SQL join query as a string.

    Raises:
        ValueError: If required information for building the query is missing or malformed.
    """

    # Extract table names and fields
    customer_table = customer if isinstance(customer, str) else customer.get('table', '')
    account_table = account if isinstance(account, str) else account.get('table', '')
    transaction_table = transaction if isinstance(transaction, str) else transaction.get('table', '')

    customer_fields = ', '.join(customer.get('fields', '*')) if isinstance(customer, dict) else '*'
    account_fields = ', '.join(account.get('fields', '*')) if isinstance(account, dict) else '*'
    transaction_fields = ', '.join(transaction.get('fields', '*')) if isinstance(transaction, dict) else '*'

    # Validate inputs
    if not all([customer_table, account_table, transaction_table]):
        raise ValueError("Table names for customer, account, and transaction must be provided.")

    # Additional join conditions and where clause
    join_condition = "ON account.customer_id = customer.id AND transaction.account_id = account.id"
    date_condition = f"AND transaction.date <= '{date_snapshot}'" if date_snapshot else ""

    # Build the SQL query
    query = f"""
    SELECT {customer_fields}, {account_fields}, {transaction_fields}
    FROM {customer_table} AS customer
    JOIN {account_table} AS account {join_condition}
    JOIN {transaction_table} AS transaction {join_condition}
    WHERE 1=1 {date_condition}
    """.strip()

    return query


def filter_by_date(query: str, start_date: str, end_date: str) -> str:
    """
    Enhances the given SQL query with date filtering conditions based on the specified start and end dates.

    Args:
        query (str): The SQL query string to be modified.
        start_date (str): The start date of the range in 'YYYY-MM-DD' format.
        end_date (str): The end date of the range in 'YYYY-MM-DD' format.

    Returns:
        str: The modified SQL query string with date filtering conditions.

    Raises:
        ValueError: If the start_date or end_date is not in the correct format.
    """
    # Validate date format
    from datetime import datetime

    try:
        datetime.strptime(start_date, '%Y-%m-%d')
        datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError as ve:
        raise ValueError("start_date and end_date must be in 'YYYY-MM-DD' format.") from ve

    # Determine if there's an existing WHERE clause
    if 'WHERE' in query.upper():
        modified_query = f"{query.strip()} AND date_column BETWEEN '{start_date}' AND '{end_date}'"
    else:
        modified_query = f"{query.strip()} WHERE date_column BETWEEN '{start_date}' AND '{end_date}'"

    return modified_query


def specify_date_snapshot(query: str, date_snapshot: str) -> str:
    """
    Updates the given SQL query by adding a condition to filter results based on a specific date snapshot.

    Args:
        query (str): The SQL query string to be modified.
        date_snapshot (str): The snapshot date in 'YYYY-MM-DD' format.

    Returns:
        str: The modified SQL query string with the date snapshot condition included.

    Raises:
        ValueError: If the date_snapshot is not in the correct format.
    """
    # Validate date format
    from datetime import datetime

    try:
        datetime.strptime(date_snapshot, '%Y-%m-%d')
    except ValueError as ve:
        raise ValueError("date_snapshot must be in 'YYYY-MM-DD' format.") from ve

    # Determine if there's an existing WHERE clause
    if 'WHERE' in query.upper():
        modified_query = f"{query.strip()} AND date_column = '{date_snapshot}'"
    else:
        modified_query = f"{query.strip()} WHERE date_column = '{date_snapshot}'"

    return modified_query


def execute_query(query: str, connection) -> list:
    """
    Executes a SQL query on the provided database connection.

    Args:
        query (str): The SQL query string to be executed.
        connection: A database connection object with an execute-capable cursor.

    Returns:
        list: A list of tuples containing the result set for SELECT queries, or an empty list for non-SELECT queries.

    Raises:
        Exception: If an error occurs during query execution.
    """
    try:
        # Create a cursor from the connection
        cursor = connection.cursor()
        
        # Execute the query
        cursor.execute(query)
        
        # If the query is a SELECT, fetch and return the results
        if query.strip().lower().startswith('select'):
            result = cursor.fetchall()
        else:
            # Commit the transaction for non-SELECT queries
            connection.commit()
            result = []

        # Close the cursor
        cursor.close()
        
        return result

    except Exception as e:
        # Log the error or handle it as required
        print(f"An error occurred during query execution: {e}")
        # Rollback the transaction in case of an error
        connection.rollback()
        raise
