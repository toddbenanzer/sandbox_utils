# SQLTemplateBuilder Documentation

## Overview
`SQLTemplateBuilder` is a class designed for loading and rendering Jinja templates for SQL queries. It streamlines the process of creating dynamic SQL by using template files and substituting values through provided contexts.

## Class Definition

### SQLTemplateBuilder

#### Constructor



# CTEManager Documentation

## Overview
`CTEManager` is a class designed to manage Common Table Expressions (CTEs) for SQL queries. This class allows easy addition and retrieval of CTEs, enhancing the modularity and readability of SQL queries.

## Class Definition

### CTEManager

#### Constructor



# QueryAggregator Documentation

## Overview
`QueryAggregator` is a class designed for aggregating data from transactions to accounts and from accounts to customers. This class provides methods to perform aggregation operations on transactional and account data efficiently using the Pandas library.

## Class Definition

### QueryAggregator

#### Constructor



# build_join_query Documentation

## Overview
The `build_join_query` function constructs a SQL query that performs a join operation between customer, account, and transaction tables, with the option to filter the results based on a specified date snapshot.

## Function Signature



# filter_by_date Documentation

## Overview
The `filter_by_date` function enhances a given SQL query by adding date filtering conditions based on specified start and end dates. This allows for more precise data retrieval from the database by limiting results to a specific date range.

## Function Signature



# specify_date_snapshot Documentation

## Overview
The `specify_date_snapshot` function enhances a SQL query by adding a filtering condition that restricts results to a specific date snapshot. This allows users to retrieve data reflecting the state of the database as of that particular date.

## Function Signature



# execute_query Documentation

## Overview
The `execute_query` function is responsible for executing a given SQL query using the provided database connection. It handles both data retrieval (SELECT queries) and data manipulation (INSERT, UPDATE, DELETE queries) while managing errors and transactions effectively.

## Function Signature

