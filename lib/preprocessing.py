import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# Path to the dataset
dataset_path = 'data/abalone.data'
# pd.set_option('display.max_rows', None)  # Show all rows
# pd.set_option('display.max_columns', None)  # Show all columns

def load_dataset():
    """Load dataset and return headers and data."""
    # Load data as strings
    data = np.genfromtxt(dataset_path, delimiter=',', dtype='str')

    # Determine the data type for each column
    dtypes = [determine_dtype(data[:, i]) for i in range(data.shape[1])]

    # Define data types
    dtype = [(f'col{i}', 'f8' if dt != 'U10' else 'U10') for i, dt in enumerate(dtypes)]

    # Load data with the specified types
    dataset = np.genfromtxt(dataset_path, delimiter=',', dtype=dtype)

    # Convert all numeric values to float
    dataset = convert_numeric_to_float(dataset)

    return dataset

def convert_numeric_to_float(dataset):
    """Convert all numeric values to float."""
    for name in dataset.dtype.names:
        if dataset.dtype[name] == 'f8':
            dataset[name] = dataset[name].astype(float)
    return dataset

def determine_dtype(column):
    """Determine the data type of the column."""
    try:
        _ = column.astype(float)
        return 'f8'  # If true, it's float
    except ValueError:
        return 'U10'  # If there's any error, it's character or string

def set_headers():
    """Set headers based on the dataset file."""
    headers = []
    with open(dataset_path) as file:
        for line in file:
            line = line.strip()  # Remove leading and trailing whitespace
            if line.startswith('#headers'):
                headers_line = next(file).strip("\n")  # Read the next line for actual headers
                headers = headers_line.split(',')
                return headers
    # If headers are still empty, determine the number of columns dynamically
    sample_data = np.genfromtxt(dataset_path, delimiter=',', dtype='str', max_rows=1)
    return [f'col{i}' for i in range(sample_data.shape[0])]  # Create headers based on the number of columns

def create_frame(dataset):
    """Create a DataFrame from the dataset."""
    # Create DataFrame from the dataset
    data_frame = pd.DataFrame({name: dataset[name] for name in dataset.dtype.names})

    # Set headers if provided
    headers = set_headers()
    data_frame.columns = headers

    # Check the available columns
    print("=======================================DataFrame Columns=============================================")
    print(data_frame.columns)
    print("=====================================================================================================")
    return data_frame