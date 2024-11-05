import pandas as pd

# Path to the dataset
dataset_path = 'data/iris.data'

def load_dataset():
    """Load dataset and return a DataFrame."""
    data_frame = pd.read_csv(dataset_path)
    #TODO
    pd.set_option('display.max_columns', None) # If you want to see full columns uncomment this line
    # pd.set_option('display.max_rows', None) # If you want to see full rows uncomment this line

    # Display the first 10 rows to check the data
    print("=======================================Loaded Data===================================================")
    print(data_frame.head(10))
    print("=====================================================================================================")

    return data_frame
