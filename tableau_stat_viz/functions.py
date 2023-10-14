andas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def read_csv_file(file_path):
    """
    Function to read in a csv file and return a pandas DataFrame.
    
    Parameters:
        - file_path (str): Path to the csv file.
    
    Returns:
        - df (pd.DataFrame): DataFrame containing the data from the csv file.
    """
    df = pd.read_csv(file_path)
    return df


def clean_and_preprocess_data(data):
    """
    Function to clean and preprocess the data.
    
    Parameters:
        - data (pd.DataFrame): The input data to be cleaned and preprocessed.
        
    Returns:
        - cleaned_data (pd.DataFrame): The cleaned and preprocessed data.
    """
    # Remove missing values
    cleaned_data = data.dropna()

    # Remove duplicate rows
    cleaned_data = cleaned_data.drop_duplicates()

    # Convert categorical variables to numerical
    categorical_columns = cleaned_data.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        cleaned_data[column] = pd.factorize(cleaned_data[column])[0]

    # Normalize numerical variables
    numeric_columns = cleaned_data.select_dtypes(include=['float', 'int']).columns
    for column in numeric_columns:
        mean = cleaned_data[column].mean()
        std = cleaned_data[column].std()
        cleaned_data[column] = (cleaned_data[column] - mean) / std

    return cleaned_data


def visualize_distribution(data, distribution_type):
    """
    Function to calculate and visualize different types of distributions.
    
    Parameters:
        - data: A list or numpy array containing the data for visualization.
        - distribution_type: The type of distribution to be visualized ('histogram', 'boxplot', 'density').
        
    Returns:
        None (displays the plot)
    """
    if distribution_type == 'histogram':
        plt.hist(data)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram')
        plt.show()
    elif distribution_type == 'boxplot':
        sns.boxplot(data)
        plt.xlabel('Distribution')
        plt.ylabel('Value')
        plt.title('Box Plot')
        plt.show()
    elif distribution_type == 'density':
        sns.kdeplot(data)
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title('Density Plot')
        plt.show()
    else:
        print("Invalid distribution type. Please choose from 'histogram', 'boxplot', or 'density'.")


def visualize_correlations(data):
    """
    Function to calculate and visualize correlations between variables.
    
    Parameters:
        - data (pd.DataFrame): The input data for calculating correlations.
        
    Returns:
        None (displays the correlation matrix and scatter plots)
    """
    # Calculate the correlation matrix
    corr_matrix = data.corr()
    
    # Plot the correlation matrix as a heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    # Plot scatter plots for each pair of variables
    sns.pairplot(data)
    plt.title('Scatter Plots')
    plt.show()


def create_categorical_visualization(data, variable):
    """
    Function to create statistical summaries and visualizations for categorical variables.
    
    Parameters:
        - data (pandas.DataFrame): The dataset containing the categorical variable.
        - variable (str): The name of the categorical variable in the dataset.
    
    Returns:
        None
    """
    
    # Count the frequency of each category
    category_counts = data[variable].value_counts()
    
    # Create a bar chart
    plt.bar(category_counts.index, category_counts.values)
    plt.xlabel(variable)
    plt.ylabel('Frequency')
    plt.title('Bar Chart of {}'.format(variable))
    plt.show()
    
    # Create a pie chart
    plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
    plt.title('Pie Chart of {}'.format(variable))
    plt.show()


def create_heatmap(data, x, y, values):
    """
    Create a heatmap for exploring complex relationships in the data.
    
    Parameters:
        - data: The input dataset.
        - x: The variable to be displayed on the x-axis.
        - y: The variable to be displayed on the y-axis.
        - values: The variable to represent the values in the heatmap.
        
    Returns:
        None (displays the heatmap)
    """
    pivot_table = data.pivot_table(values=values, index=y, columns=x)
    
    sns.heatmap(pivot_table, cmap='coolwarm')
    plt.show()


def create_interactive_visualization(data, filters=None, tooltips=None, highlight=None):
    """
    Creates interactive visualizations for Tableau.
    
    Parameters:
        - data (pandas DataFrame): The dataset to be visualized.
        - filters (list): List of column names to include as filters in Tableau.
        - tooltips (list): List of column names to include as tooltips in Tableau.
        - highlight (str): Name of the column to highlight in Tableau.
        
    Returns:
        tableau_code (str): The generated Tableau code for the visualization.
    """
    
    # Generate the base Tableau code
    tableau_code = "<tableau_code>"
    
    # Add filters to the Tableau code
    if filters:
        filter_code = ""
        for filter_col in filters:
            filter_code += f"<filter_code>{filter_col}</filter_code>"
        tableau_code = tableau_code.replace("<filter_placeholder>", filter_code)
    
    # Add tooltips to the Tableau code
    if tooltips:
        tooltip_code = ""
        for tooltip_col in tooltips:
            tooltip_code += f"<tooltip_code>{tooltip_col}</tooltip_code>"
        tableau_code = tableau_code.replace("<tooltip_placeholder>", tooltip_code)
    
    # Add highlighting to the Tableau code
    if highlight:
        highlight_code = f"<highlight_code>{highlight}</highlight_code>"
        tableau_code = tableau_code.replace("<highlight_placeholder>", highlight_code)
    
    return tableau_code


def tableau_export(visualizations, output_path):
    """
    Export visualizations as Tableau workbooks or images.
    
    Parameters:
        - visualizations (list): List of visualization file paths.
        - output_path (str): Path to the output directory.
        
    Returns:
        None
    """
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    for visualization in visualizations:
        # Get the file name and extension
        file_name = os.path.basename(visualization)
        file_extension = os.path.splitext(file_name)[1]
        
        # Set the new file path in the output directory
        new_file_path = os.path.join(output_path, file_name)
        
        # Copy or move the visualization to the output directory
        if file_extension == '.twbx':
            shutil.copy(visualization, new_file_path)
        else:
            shutil.move(visualization, new_file_path)


def customize_visualization(visualization, color_scheme=None, labels=None, title=None, annotations=None):
    """
    Customize the appearance of a Tableau visualization.

    Args:
        - visualization (str): The name or ID of the visualization.
        - color_scheme (str): The desired color scheme for the visualization.
        - labels (dict): A dictionary of labels to customize on the visualization.
        - title (str): The desired title for the visualization.
        - annotations (list): A list of annotations to add to the visualization.

    Returns:
        None
    """

    # Code to customize the appearance of the visualization in Tableau
    # ...


def handle_missing_values(data, method='exclude'):
    """
    Handle missing values in the data and provide options for imputation or exclusion from visualizations.
    
    Parameters:
        - data: DataFrame, the input data containing missing values
        - method: str, optional (default='exclude'), specify the method for handling missing values. 
                  Valid options are 'exclude' to exclude rows with missing values,
                  'mean' to fill missing values with mean of respective column,
                  'median' to fill missing values with median of respective column,
                  'mode' to fill missing values with mode of respective column.
    
    Returns:
        - DataFrame, the data with missing values handled according to the specified method.
    """
    
    if method == 'exclude':
        # Exclude rows with any missing values
        return data.dropna()
    
    elif method == 'mean':
        # Fill missing values with mean of respective columns
        return data.fillna(data.mean())
    
    elif method == 'median':
        # Fill missing values with median of respective columns
        return data.fillna(data.median())
    
    elif method == 'mode':
        # Fill missing values with mode of respective columns
        return data.fillna(data.mode().iloc[0])
    
    else:
        raise ValueError("Invalid method. Valid options are: 'exclude', 'mean', 'median', 'mode'.")


def handle_large_dataset(file_path):
    """
    Handle large datasets efficiently by optimizing memory usage and processing speed.
    
    Parameters:
        - file_path: The path to the large dataset file.
        
    Returns:
        - final_result: The processed dataset.
    """
    
    # Read the dataset in chunks to minimize memory usage
    chunk_size = 10000
    chunks = pd.read_csv(file_path, chunksize=chunk_size)
    
    # Process each chunk separately and concatenate the results
    result = []
    
    for chunk in chunks:
        # Perform any necessary operations on the chunk
        processed_chunk = chunk.apply(lambda x: x**2)
        
        # Append the processed chunk to the result list
        result.append(processed_chunk)
    
    # Concatenate all the processed chunks into a single dataframe
    final_result = pd.concat(result)
    
    return final_result


def get_tableau_projects(server_url, username, password):
    """
    Get a list of Tableau projects using Tableau's APIs for seamless interaction with Tableau Desktop or Tableau Server.
    
    Parameters:
        - server_url (str): The URL of the Tableau server.
        - username (str): The username for authentication.
        - password (str): The password for authentication.
        
    Returns:
        - projects (list): A list of Tableau projects.
    """
    
    server = tableau.Server(server_url)
    server.auth.sign_in(username, password)
    
    projects = server.projects.get()
    
    server.auth.sign_out()
    
    return projects


def create_histogram(data, bin_size, output_file):
    """
    Create a histogram visualization in Tableau.
    
    Parameters:
        - data: A pandas Series or DataFrame containing the data.
        - bin_size: The size of each bin in the histogram.
        - output_file: The filename for the Tableau workbook (.twb) file.

    Returns:
        None
    """
    
    # Check if input data is a Series or DataFrame
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)

    # Create a new DataFrame with two columns: 'Value' and 'Bin'
    hist_data = pd.DataFrame({'Value': data.values.flatten()})
    hist_data['Bin'] = pd.cut(hist_data['Value'], bins=bin_size)

    # Generate the count of values in each bin
    hist_counts = hist_data.groupby('Bin').size().reset_index(name='Count')

    # Create a Tableau workbook (.twb) file
    with open(output_file, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<workbook>\n')
        f.write('\t<datasources>\n')
        f.write('\t\t<datasource connection="" name="Histogram Data">\n')
        f.write('\t\t\t<column caption="Bin" datatype="string" name="bin" role="dimension"/>\n')
        f.write('\t\t\t<column caption="Count" datatype="integer" name="count" role="measure"/>\n')

        # Write the histogram counts to the Tableau workbook
        for i, row in hist_counts.iterrows():
            f.write(f'\t\t\t<row bin="{row["Bin"]}" count="{row["Count"]}"/>\n')

        f.write('\t\t</datasource>\n')
        f.write('\t</datasources>\n')
        f.write('\t<sheets>\n')
        f.write('\t\t<sheet name="Histogram" type="worksheet">\n')
        f.write('\t\t\t<views>\n')
        f.write('\t\t\t\t<view alias="Histogram Data" name="Histogram Data">\n')
        f.write('\t\t\t\t\t<pills>\n')
        f.write(
            '\t\t\t\t\t\t<pill aggregation="None" attr="bin" datatype="string" role="dimension"/>\n')
        f.write(
            '\t\t\t\t\t\t<pill aggregation="Sum" attr="count" datatype="integer" role="measure"/>\n')
        f.write('\t\t\t\t\t</pills>\n')
        f.write('\t\t\t\t</view>\n')
        f.write('\t\t\t</views>\n')
        f.write('\t\t</sheet>\n')
        f.write('\t</sheets>\n')
        f.write('</workbook>')

    print(f"Histogram visualization created in Tableau. Please open the '{output_file}' file.")


def create_box_plot(data_frame, x_column, y_column):
    """
    Create a box plot visualization in Tableau.
    
    Parameters:
        - data_frame: Pandas DataFrame containing the data.
        - x_column: Name of the column to be used as the x-axis in the box plot.
        - y_column: Name of the column to be used as the y-axis in the box plot.
    
    Returns:
        None
    """
    
    # Convert the Pandas DataFrame to a Tableau Data Extract
    data_extract = tabpy.TableauDataExtract(data_frame)
    
    # Create a new Worksheet in Tableau
    worksheet = tabpy.Worksheet(data_extract)
    
    # Create a Box Plot on the worksheet
    box_plot = worksheet.create_box_plot(x_column, y_column)
    
    # Save the worksheet as a Tableau workbook
    workbook = tabpy.Workbook([worksheet])
    
    # Export the workbook as a .twbx file
    workbook.export("box_plot.twbx")


def create_density_plot(data, x, y):
    """
    Create a density plot visualization in Tableau.

    Parameters:
        - data: pandas DataFrame
            The input data for the density plot.
        - x: str
            The name of the column to be plotted on the x-axis.
        - y: str
            The name of the column to be plotted on the y-axis.
    """
    
    sns.kdeplot(data=data, x=x, y=y, fill=True)
    plt.show()


def create_violin_plot(data, x, y):
    """
    Create a violin plot visualization in Tableau.
    
    Parameters:
        - data: The input dataset.
        - x: The variable to be displayed on the x-axis.
        - y: The variable to be displayed on the y-axis.
        
    Returns:
        None
    """
    
    # Create a Tableau workbook
    workbook = tableau.Workbook()

    # Create a Tableau worksheet
    worksheet = workbook.add_worksheet("Violin Plot")

    # Add the data to the worksheet
    worksheet.add_data(data)

    # Create a mark type for the violin plot
    mark = tableau.Marks.VIOLIN_PLOT

    # Create a mark specification for the violin plot
    mark_spec = tableau.MarkSpecification(x=x, y=y, mark_type=mark)

    # Add the mark specification to the worksheet
    worksheet.add_mark(mark_spec)

    # Save the workbook as a Tableau file
    workbook.save("violin_plot.twb"