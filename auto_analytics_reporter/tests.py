ytest
import pandas as pd
from my_module import read_data_file

# Test case for reading a CSV file
def test_read_csv_file(tmp_path):
    # Create a temporary CSV file with sample data
    file_path = tmp_path / "data.csv"
    data = "1,2,3\n4,5,6\n7,8,9"
    file_path.write_text(data)

    # Read the CSV file using the function
    df = read_data_file(file_path)

    # Check if the dataframe is not empty
    assert not df.empty

# Test case for reading an Excel file
def test_read_excel_file(tmp_path):
    # Create a temporary Excel file with sample data
    file_path = tmp_path / "data.xlsx"
    data = {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}
    df = pd.DataFrame(data)
    df.to_excel(file_path, index=False)

    # Read the Excel file using the function
    df = read_data_file(file_path)

    # Check if the dataframe is not empty
    assert not df.empty

# Test case for reading a JSON file
def test_read_json_file(tmp_path):
    # Create a temporary JSON file with sample data
    file_path = tmp_path / "data.json"
    data = [{"A": 1, "B": 2}, {"A": 3, "B": 4}]
    file_path.write_text(pd.Series(data).to_json())

    # Read the JSON file using the function
    df = read_data_file(file_path)

    # Check if the dataframe is not empty
    assert not df.empty

# Test case for unsupported file type
def test_read_unsupported_file_type(tmp_path):
    # Create a temporary text file
    file_path = tmp_path / "data.txt"
    data = "Sample text"
    file_path.write_text(data)

    # Check if the function raises a ValueError
    with pytest.raises(ValueError):
        read_data_file(file_path