import pandas as pd

# Path to the dataset
dataset_path = 'data/abalone.data'

def load_dataset():
    """Load dataset and return a DataFrame."""
    # Load data with pandas
    data_frame = pd.read_csv(dataset_path)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)

    # Display the first 10 rows to check the data
    print("=======================================Loaded Data===================================================")
    print(data_frame.head(10))
    print("=====================================================================================================")

    return data_frame
