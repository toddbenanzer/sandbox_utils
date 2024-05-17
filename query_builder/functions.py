
from jinja2 import Template, Environment, FileSystemLoader
import pandas as pd
import teradata
import teradatasql

def generate_sql_query(template_string, data):
    template = Template(template_string)
    sql_query = template.render(data=data)
    return sql_query


def build_cte(cte_name, query):
    cte = f"{cte_name} AS ({query})"
    return cte


def join_tables(tables, join_conditions):
    sql_template = """
    WITH joined_data AS (
        SELECT *
        FROM {{ tables[0] }}
        {% for i in range(1, len(tables)) %}
        {% set table_alias = 't' + loop.index|string %}
        INNER JOIN {{ tables[i] }} AS {{ table_alias }}
        ON {{ join_conditions[i-1] }}
        {% endfor %}
    )
    SELECT *
    FROM joined_data;
    """

    template = Template(sql_template)
    sql_query = template.render(tables=tables, join_conditions=join_conditions)
    return sql_query


def filter_data(table_name, conditions):
    sql_template = """
    WITH filtered_data AS (
        SELECT *
        FROM {{ table_name }}
        WHERE {{ conditions }}
    )
    
    SELECT *
    FROM filtered_data;
    """
    
    env = Environment()
    template = env.from_string(sql_template)
    
    sql_query = template.render(table_name=table_name, conditions=conditions)
    
    return sql_query


def aggregate_data(group_by, select_columns):
    query_template = '''
    WITH aggregated_data AS (
        SELECT {{ group_by }}, {{ select_columns }}, SUM(transaction_amount) AS total_amount
        FROM transactions
        GROUP BY {{ group_by }}, {{ select_columns }}
    )
    SELECT * FROM aggregated_data
    '''

    template = Template(query_template)
    rendered_query = template.render(group_by=group_by, select_columns=select_columns)
    
    return rendered_query


def order_result_set(template_str, columns):
    template = Template(template_str)
    
    rendered_query = template.render(columns=columns)
    
    result_set = execute_sql_query(rendered_query)
    
    return result_set


def limit_rows(query_template, limit):
    with open(query_template, 'r') as file:
        template_content = file.read()
    
    template = Template(template_content)
    
    rendered_query = template.render(limit=limit)
    
    return rendered_query


def calculate_row_count(table_name):
    sql_template = '''
        WITH {{ table_name }}_cte AS (
            SELECT COUNT(*) AS row_count
            FROM {{ table_name }}
        )
        SELECT row_count
        FROM {{ table_name }}_cte;
    '''

    template = Template(sql_template)
    
     rendered_query = template.render(table_name=table_name)

     return rendered_query


def calculate_sum(table_name, column_name):
     env = Environment(loader=FileSystemLoader('templates'))
     template = env.get_template('sum_query.sql')
     
     sql_query = template.render(table=table_name, column=column_name)
     
     return sql_query


def calculate_average(column, table):
     template_str = """
     WITH aggregated_data AS (
         SELECT {{ column }}, AVG({{ column }}) OVER () as average
         FROM {{ table }}
     )
     SELECT average
     FROM aggregated_data;
     """
     
     sql_query = Template(template_str).render(column=column, table=table)

     return sql_query


def calculate_min_value(table_name, column_name):
      query_template_str  = """
      WITH cte AS (
          SELECT MIN({{ column_name }}) AS min_value
          FROM {{ table_name }}
      )
      SELECT min_value
      FROM cte;
      """
      
      query  render(Template(query_template_str).render(table_name=table_name ,column)

       result  teradata_connection.execute(query)

       min_value result.fetchall()

       return min_value

       
 def calculate_max_value(table , column ):
       query _template_str """

       WITH( max_value_cte)AS(
           select max value_cte

           from{{table}}

           """

           query_render (Template(query _template_str).render (table))

          max_value execute_sql _query(sql _query).fetchone()[0]
         

          return max_value

          
           def get_distinct_values (column , table) :

udaExec  teradata.UdaExec(appName="MyApp", version="1.0", logConsole=False)

session  udaExec.connect(method="odbc", system="localhost", username="myuser", password="mypassword")

sql  f"SELECT DISTINCT {column} FROM {table}"

with session.cursor() as cursor:
cursor.execute(sql)


results cursor.fetchall()

session.close()


return [row[0] for row in results]


 def calculate_percentile (table name , column name percentile) :
   template  Template("""

   with ranked data as(
   select{{column name}},
   row_number over(order by{{column name}})as row num,
   count(*)over()as total rows 

 from{{table name}})
 
 select{{column name}}
 from ranked data 
 where row num ceil({{percentile}}*total rows)


 """)
 
sql _query render (template .render (table name-column name-percentile))

return sql-query :



 def calculate_rank (table name ,criteria) :

 rank-query=f"""
with ranked data as(
select *,
rank() over(order by {criteria}})as row rank 
from {table name})

select * from ranked data;

"""

query-template  Template(rank query )

sql-query  query-template.render (criteria-table-name )

result execute_sql -query(sql -query)

return result :



 def calculate_row_number(criteria) :

sql-template'''
WITH cte as(
select *,
row number()over(order by{{criteria}})as row num 
from your_table)

select*
from cte

'''

template  Template(sql-template)


sql-query render (template .render(criteria))

return sql-query:


 def calculate_dense_rank(criteria):

sql-query"""
WITH ranked data as(
select *,
Dense_rank()over(order by{criteria})as Dense_rank 
from your_table)

select *from ranked 

}

result execute _teradata _query (sql -query)

return result:


 def calculate_lag(column-name , order by_column):

query-template"""`
select {column-name}lag({column -name})over(order by{order by -column})
from {table-name}

"""

query format(column-name-column-order by_column-table-name)

result execute_teradata (query):

return result:

 def lead value(column-name-table-name):

"""

function to calculate the lead value of a column in a table.

Args:
Column-name(str):The name of the column to calculate the lead value for.
Table-name(str):The name of the table containing the column.

Returns: str: The teradata SQL query to calculate the lead value of the column.


"""

query=f"""
select{column-name},
lead({column-name})over(order by{column-name})as lead_value}
from {table name}

"""


return query:



 def running_total(table name-column ):

template-Template("""
WITH cte as(
select *,
sum({{Column}}pair over(
order by{{Column}}asc

rows between unbounded preceding and current row 
)) As running_total 

from{{table name }}

)

select*from cte

""")

sql-query render(template .render(Column -table))

return execy_sql-Query(sql-query):





 def pivot-data(row pivot Column-value Column):

"""

pivot data from rows into columns based on specified criteria.

parameters:
Data(list):The input data as a list of dictionaries.
Pivot Column(str):The name of the column to pivot on.
Value Column(str):The generated SQL query for pivoting the datqa.

"""

template -Template("""

CTE(pivoted-data As(

Select{% for Column-value in Column values %)

max(case when{pivot-Column}= '{Column-value}'then{{value-column}}end)As"{Column-value}"
{% if not loop.last %},{% endif %}
{% endfor %}

From data 

)


Select*
From pivoted-data:

""")

Column values-set(row[pivot_column] for-row-in-data )


sql-query render(template.Column-values,pivot_column,value_column)




import pandas as pd 


Unpivot-data(data,id-vars value_vars-var-variable,value-variable):

Unpivot -data=pd.melt(data,id-vars-id-vars-value-vars,var-var-variable,var-variable).

return Unpivoted-data:



 def generate_sql-query(entity-date snapshot-aggregation):

sql-template-Template("""
With customer CTE As(

select*

From customers 
where date snapshot='{:date snapshot}'
),
account_cte As(
select* 
From accounts
where date snapshot='{:date snapshot}'
),
transaction CTE As(

Select*
From_transactions_

Where date snapshot='{:date snapshot}'
)


Select{
{% if aggregation=='customer'%}
customer-id,count(Distinct count id),sum(transaction_amount),num_accounts,total_transaction_amount:
{% elif aggregation=='account' %}

count Distinc transaction id,sum transaction_amount,total transaction_amount:

{% endif %}
}


From Transaction CTE
Inner_join account CTE On transaction CTE.account id-account CTE.account id,
Inner join customer CTE On account CTE.customer id-customer CTE.customer id,

Group By{
if aggregation=='customer'%}customer-id {%elif aggregation-'account'}account {%endif%}


""")





SQL-Render(entity-date_snapshot-aggregation):

Return SQL_QUERY:



Execute Query(Query):

try:
Execute SQL Query using Teradata library:

Replace this line with actual code to execute the Query .

result-Teradata.execute(Query).

Return result if needed.

Except DatabaseError As e:
Custom-error message with original exception message.

error_message-f"Error executing SQL query:{e.message}" .

raise Exception(error_message)



Execute Query(Query):

parameters:
Query(str):The SQL Query to Execute.

Returns list:The Result set retrieved from the Query execution.


"""


Conn-Teradatasql.connect('<your_connection details>')As conn:


Cursor conn.cursor():
Cursor Execute:

cursor Execute(Query)



Result set-cursor.fetchall()

Return Result set:

Connect_to_Teradata(host-username-password):

Connects_Teradata_Database using provided credentials .


Args:

Host:str: The Hostname or IP Address of Teradata Database.
Username:str:The username for Authentication .
Password:str:The password for Authentication.


Returns:

Teradatasql .Connection: The Connection object to Teradata Database.


Conn-Teradatasql.connect(host-host-user-host-password-host).

Return Conn:


Get_table_metadata(Database Table ):


UdaExec-Teradata.UdaExec(Application My App Version-logConsole-False).


session-UdaExec.connect(method system hostname username-password).


try:
Execute a SQL Query Retrieve metadata information .

Query-f"show TABLE(Database.Table};"


Session.cursor()As Cursor Cursor.ExecuteQuery :

Rows-cursor.fetchall():
Cursor description:

Column names-[Column[0] For Column In Cursor.Description]


Return Column names Rows :





finally:
Close Database Connection .
session.close()





