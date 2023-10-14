# Python Tableau TDE Converter

This is a Python script that allows you to convert a pandas DataFrame into a Tableau Data Extract (TDE) file. The TDE file can then be used in Tableau for data visualization and analysis.

## Overview

Tableau is a powerful data visualization tool that allows users to connect to various data sources and create interactive dashboards and reports. However, the native data formats supported by Tableau are limited. This script provides a solution by allowing you to convert your pandas DataFrame, a popular Python library for data manipulation, into a TDE file that can be directly imported into Tableau.

## Usage

To use this script, you need to have the following dependencies installed:

- pandas
- tableausdk

You can install these dependencies using pip:

```
pip install pandas tableausdk
```

Once you have the dependencies installed, you can import the `convert_to_tde` function from the script and use it to convert your DataFrame into a TDE file.

The `convert_to_tde` function takes two arguments: the DataFrame you want to convert and the name of the output TDE file. Here's an example of how to use it:

```python
import pandas as pd
from tableausdk import *
from tableausdk.Extract import *

# Create a sample DataFrame
df = pd.DataFrame({'Name': ['John', 'Jane', 'Mike'],
                   'Age': [25, 30, 35],
                   'Height': [170.5, 165.2, 180.3],
                   'IsStudent': [False, True, True]})

# Convert the DataFrame to a TDE file
convert_to_tde(df, 'data.tde')
```

In this example, we create a sample DataFrame with four columns: Name (string), Age (integer), Height (float), and IsStudent (boolean). We then call the `convert_to_tde` function with the DataFrame and specify the name of the output TDE file as "data.tde". The script will convert the DataFrame into a TDE file and save it to the specified location.

## Examples

Here are some examples that demonstrate how to use the script:

### Example 1: Converting a DataFrame with string columns

```python
import pandas as pd
from tableausdk import *
from tableausdk.Extract import *

# Create a sample DataFrame
df = pd.DataFrame({'Name': ['John', 'Jane', 'Mike'],
                   'Email': ['john@example.com', 'jane@example.com', 'mike@example.com']})

# Convert the DataFrame to a TDE file
convert_to_tde(df, 'data.tde')
```

In this example, we have a DataFrame with two columns: Name and Email, both of which are strings. The script will convert this DataFrame into a TDE file.

### Example 2: Converting a DataFrame with numeric columns

```python
import pandas as pd
from tableausdk import *
from tableausdk.Extract import *

# Create a sample DataFrame
df = pd.DataFrame({'Age': [25, 30, 35],
                   'Height': [170.5, 165.2, 180.3]})

# Convert the DataFrame to a TDE file
convert_to_tde(df, 'data.tde')
```

In this example, we have a DataFrame with two columns: Age (integer) and Height (float). The script will convert this DataFrame into a TDE file.

### Example 3: Converting a DataFrame with boolean columns

```python
import pandas as pd
from tableausdk import *
from tableausdk.Extract import *

# Create a sample DataFrame
df = pd.DataFrame({'IsStudent': [False, True, True],
                   'IsEmployed': [True, False, True]})

# Convert the DataFrame to a TDE file
convert_to_tde(df, 'data.tde')
```

In this example, we have a DataFrame with two columns: IsStudent and IsEmployed, both of which are booleans. The script will convert this DataFrame into a TDE file.

## Conclusion

This script provides a convenient way to convert pandas DataFrames into Tableau Data Extract (TDE) files. By using this script, you can easily import your data into Tableau for further analysis and visualization.