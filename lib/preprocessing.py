import os
import pandas as pd

def load_dataset(path_to_dataset):
    """Load dataset and return a DataFrame."""
    data_frame = pd.read_csv(path_to_dataset)
    # pd.set_option('display.max_columns', None) # If you want to see full columns uncomment this line
    # pd.set_option('display.max_rows', None) # If you want to see full rows uncomment this line

    # Display the first 10 rows to check the data
    print("=======================================Loaded Data===================================================")
    print(data_frame.head(10))
    print("=====================================================================================================")

    return data_frame
def list_datasets(directory="data"):
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        if not files:
            print(f"Directory '{directory}' is empty.")
        return files
    except FileNotFoundError:
        print(f"Directory '{directory}' was not found.")
        return []

def select_dataset(directory="data"):
    files = list_datasets(directory)
    if not files:
        return None

    print("Available datasets:")
    for i, file in enumerate(files):
        print(f"{i + 1}. {file}")

    while True:
        try:
            choice = int(input("Choose a dataset: "))
            if 1 <= choice <= len(files):
                selected_file = os.path.join(directory, files[choice - 1])
                print(f"Chosen file: {selected_file}")
                return selected_file
            else:
                print("Wrong choice. Try again.")
        except ValueError:
            print("Wrong choice. Try again.")
def select_encoding_method():
    methods = ["Label Encoding","Hashing","Word2Vec","GloVe"]

    print("Methods to choose:")
    for i, method in enumerate(methods):
        print(f"{i + 1}. {method}")

    while True:
        try:
            choice = int(input("Choose encoding method: "))
            if 1 <= choice <= len(methods):
                selected_method = methods[choice - 1]
                print(f"Selected method: {selected_method}")
                return selected_method
            else:
                print("Wrong number. Try again.")
        except ValueError:
            print("Incorrect number. Try again.")
def select_correlation_method():
    methods = ["Pearson","Spearman","Kendall"]

    print("Methods to choose:")
    for i, method in enumerate(methods):
        print(f"{i + 1}. {method}")

    while True:
        try:
            choice = int(input("Choose method of correlation computation: "))
            if 1 <= choice <= len(methods):
                selected_method = methods[choice - 1]
                print(f"Selected method: {selected_method}")
                return selected_method.lower()
            else:
                print("Wrong number. Try again.")
        except ValueError:
            print("Incorrect number. Try again.")

