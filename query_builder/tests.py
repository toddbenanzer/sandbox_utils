from jinja2 import TemplateNotFound
from unittest.mock import MagicMock, create_autospec
from your_module import CTEManager  # Replace 'your_module' with the actual module name
from your_module import QueryAggregator  # Replace 'your_module' with the actual module name
from your_module import SQLTemplateBuilder  # Replace 'your_module' with the actual module name
from your_module import build_join_query  # Replace 'your_module' with the actual module name
from your_module import execute_query  # Replace 'your_module' with the actual module name
from your_module import filter_by_date  # Replace 'your_module' with the actual module name
from your_module import specify_date_snapshot  # Replace 'your_module' with the actual module name
import pandas as pd
import pytest


@pytest.fixture
def setup_sql_template_builder(tmp_path):
    # Create a temporary directory to hold templates
    template_dir = tmp_path / "templates"
    template_dir.mkdir()

    # Create a sample template file
    sample_template = template_dir / "sample.sql"
    sample_template.write_text("SELECT * FROM users WHERE id = {{ user_id }};")
    
    # Create the SQLTemplateBuilder instance with the temp directory
    return SQLTemplateBuilder(template_dir=str(template_dir))

def test_load_template_existing(setup_sql_template_builder):
    builder = setup_sql_template_builder
    assert builder.load_template("sample.sql") is not None

def test_load_template_non_existing(setup_sql_template_builder):
    builder = setup_sql_template_builder
    with pytest.raises(TemplateNotFound):
        builder.load_template("non_existing.sql")

def test_render_template_success(setup_sql_template_builder):
    builder = setup_sql_template_builder
    rendered_query = builder.render_template("sample.sql", {"user_id": 1})
    assert rendered_query == "SELECT * FROM users WHERE id = 1;"

def test_render_template_missing_variable(setup_sql_template_builder):
    builder = setup_sql_template_builder
    with pytest.raises(Exception):
        builder.render_template("sample.sql", {})



def test_init_empty_ctes():
    cte_manager = CTEManager()
    assert cte_manager.ctes == {}

def test_add_cte():
    cte_manager = CTEManager()
    cte_manager.add_cte('cte1', 'SELECT * FROM table1')
    assert 'cte1' in cte_manager.ctes
    assert cte_manager.ctes['cte1'] == 'SELECT * FROM table1'

def test_get_cte_query_with_no_ctes():
    cte_manager = CTEManager()
    assert cte_manager.get_cte_query() == ""

def test_get_cte_query_with_single_cte():
    cte_manager = CTEManager()
    cte_manager.add_cte('cte1', 'SELECT * FROM table1')
    expected_query = "WITH cte1 AS (SELECT * FROM table1)"
    assert cte_manager.get_cte_query() == expected_query

def test_get_cte_query_with_multiple_ctes():
    cte_manager = CTEManager()
    cte_manager.add_cte('cte1', 'SELECT * FROM table1')
    cte_manager.add_cte('cte2', 'SELECT * FROM table2')
    expected_query = "WITH cte1 AS (SELECT * FROM table1), cte2 AS (SELECT * FROM table2)"
    assert cte_manager.get_cte_query() == expected_query



@pytest.fixture
def aggregator():
    return QueryAggregator()

def test_aggregate_transactions_to_account_valid_data(aggregator):
    transactions_data = pd.DataFrame({
        'account_id': [1, 1, 2, 2, 3],
        'transaction_amount': [100, 150, 200, 250, 300]
    })
    result = aggregator.aggregate_transactions_to_account(transactions_data)
    expected = pd.DataFrame({
        'account_id': [1, 2, 3],
        'total_transactions': [250, 450, 300],
        'avg_transaction_amount': [125.0, 225.0, 300.0]
    })
    pd.testing.assert_frame_equal(result, expected)

def test_aggregate_transactions_to_account_missing_columns(aggregator):
    transactions_data = pd.DataFrame({
        'account_id': [1, 1, 2, 2, 3]
        # Missing 'transaction_amount' column
    })
    with pytest.raises(ValueError, match="Input data must contain columns"):
        aggregator.aggregate_transactions_to_account(transactions_data)

def test_aggregate_accounts_to_customer_valid_data(aggregator):
    accounts_data = pd.DataFrame({
        'customer_id': [1, 1, 2, 3, 3],
        'account_balance': [1000, 1500, 2000, 2500, 3500]
    })
    result = aggregator.aggregate_accounts_to_customer(accounts_data)
    expected = pd.DataFrame({
        'customer_id': [1, 2, 3],
        'total_balance': [2500, 2000, 6000]
    })
    pd.testing.assert_frame_equal(result, expected)

def test_aggregate_accounts_to_customer_missing_columns(aggregator):
    accounts_data = pd.DataFrame({
        'customer_id': [1, 1, 2, 3, 3]
        # Missing 'account_balance' column
    })
    with pytest.raises(ValueError, match="Input data must contain columns"):
        aggregator.aggregate_accounts_to_customer(accounts_data)



def test_build_join_query_with_full_dictionaries():
    customer = {'table': 'customer', 'fields': ['id', 'name']}
    account = {'table': 'account', 'fields': ['id', 'balance']}
    transaction = {'table': 'transaction', 'fields': ['id', 'amount', 'date']}
    date_snapshot = '2022-01-01'
    
    query = build_join_query(customer, account, transaction, date_snapshot)

    expected_substring = "SELECT id, name, id, balance, id, amount, date FROM customer AS customer " \
                         "JOIN account AS account ON account.customer_id = customer.id AND transaction.account_id = account.id " \
                         "JOIN transaction AS transaction ON account.customer_id = customer.id AND transaction.account_id = account.id " \
                         "WHERE 1=1 AND transaction.date <= '2022-01-01'"
    assert expected_substring in query

def test_build_join_query_with_table_names():
    query = build_join_query('customer', 'account', 'transaction', '2022-01-01')
    expected_substring = "SELECT *, *, * FROM customer AS customer " \
                         "JOIN account AS account ON account.customer_id = customer.id AND transaction.account_id = account.id " \
                         "JOIN transaction AS transaction ON account.customer_id = customer.id AND transaction.account_id = account.id " \
                         "WHERE 1=1 AND transaction.date <= '2022-01-01'"
    assert expected_substring in query

def test_build_join_query_missing_table_name():
    customer = {'table': 'customer', 'fields': ['id', 'name']}
    with pytest.raises(ValueError, match="Table names for customer, account, and transaction must be provided."):
        build_join_query(customer, '', 'transaction', '2022-01-01')

def test_build_join_query_without_date_snapshot():
    customer = {'table': 'customer', 'fields': ['id', 'name']}
    account = {'table': 'account', 'fields': ['id', 'balance']}
    transaction = {'table': 'transaction', 'fields': ['id', 'amount', 'date']}
    
    query = build_join_query(customer, account, transaction, None)
    assert "WHERE 1=1 AND transaction.date <= ''" not in query



def test_filter_by_date_with_existing_where_clause():
    query = "SELECT * FROM transactions WHERE status = 'active'"
    modified_query = filter_by_date(query, '2023-01-01', '2023-12-31')
    expected_query = "SELECT * FROM transactions WHERE status = 'active' AND date_column BETWEEN '2023-01-01' AND '2023-12-31'"
    assert modified_query == expected_query

def test_filter_by_date_without_existing_where_clause():
    query = "SELECT * FROM transactions"
    modified_query = filter_by_date(query, '2023-01-01', '2023-12-31')
    expected_query = "SELECT * FROM transactions WHERE date_column BETWEEN '2023-01-01' AND '2023-12-31'"
    assert modified_query == expected_query

def test_filter_by_date_invalid_start_date_format():
    query = "SELECT * FROM transactions"
    with pytest.raises(ValueError, match="start_date and end_date must be in 'YYYY-MM-DD' format."):
        filter_by_date(query, '01-01-2023', '2023-12-31')

def test_filter_by_date_invalid_end_date_format():
    query = "SELECT * FROM transactions"
    with pytest.raises(ValueError, match="start_date and end_date must be in 'YYYY-MM-DD' format."):
        filter_by_date(query, '2023-01-01', '31-12-2023')



def test_specify_date_snapshot_with_existing_where_clause():
    query = "SELECT * FROM orders WHERE status = 'active'"
    modified_query = specify_date_snapshot(query, '2023-10-01')
    expected_query = "SELECT * FROM orders WHERE status = 'active' AND date_column = '2023-10-01'"
    assert modified_query == expected_query

def test_specify_date_snapshot_without_existing_where_clause():
    query = "SELECT * FROM orders"
    modified_query = specify_date_snapshot(query, '2023-10-01')
    expected_query = "SELECT * FROM orders WHERE date_column = '2023-10-01'"
    assert modified_query == expected_query

def test_specify_date_snapshot_invalid_date_format():
    query = "SELECT * FROM orders"
    with pytest.raises(ValueError, match="date_snapshot must be in 'YYYY-MM-DD' format."):
        specify_date_snapshot(query, '10-01-2023')
        
def test_specify_date_snapshot_empty_query():
    query = ""
    modified_query = specify_date_snapshot(query, '2023-10-01')
    expected_query = "WHERE date_column = '2023-10-01'"
    assert modified_query == expected_query



def test_execute_query_select():
    query = "SELECT * FROM my_table"
    mock_connection = create_autospec(['cursor', 'commit', 'rollback'])
    mock_cursor = MagicMock()
    mock_connection.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = [(1, 'Alice'), (2, 'Bob')]

    result = execute_query(query, mock_connection)
    assert result == [(1, 'Alice'), (2, 'Bob')]
    mock_cursor.execute.assert_called_once_with(query)
    mock_cursor.fetchall.assert_called_once()
    mock_cursor.close.assert_called_once()

def test_execute_query_non_select():
    query = "UPDATE my_table SET name = 'Charlie' WHERE id = 1"
    mock_connection = create_autospec(['cursor', 'commit', 'rollback'])
    mock_cursor = MagicMock()
    mock_connection.cursor.return_value = mock_cursor

    result = execute_query(query, mock_connection)
    assert result == []
    mock_cursor.execute.assert_called_once_with(query)
    mock_connection.commit.assert_called_once()
    mock_cursor.close.assert_called_once()

def test_execute_query_exception():
    query = "INVALID SQL"
    mock_connection = create_autospec(['cursor', 'commit', 'rollback'])
    mock_cursor = MagicMock()
    mock_connection.cursor.return_value = mock_cursor
    mock_cursor.execute.side_effect = Exception("Database error")

    with pytest.raises(Exception, match="Database error"):
        execute_query(query, mock_connection)
    mock_cursor.close.assert_called_once()
    mock_connection.rollback.assert_called_once()
