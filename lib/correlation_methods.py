from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd


def calculate_correlation_with_label_encoding(data_frame, method: str):
    """Calculate correlation, including string attributes with Label Encoding."""
    # Label Encoding for string columns
    label_encoders = {}
    for name in data_frame.columns:
        if data_frame[name].dtype == 'object':  # Check if it's a string column
            label_encoder = LabelEncoder()
            data_frame[name] = label_encoder.fit_transform(data_frame[name]) + 1
            label_encoders[name] = label_encoder

    # Calculate correlation matrix
    correlation_matrix = data_frame.corr(method=method)

    # Calculate sigma (standard deviation) for each column
    sigma = correlation_matrix.values.std()

    # Set diagonal elements of the correlation matrix to 0
    np.fill_diagonal(correlation_matrix.values, 0)

    # Replace values outside +/- sigma interval with 0
    filtered_correlation = np.where(
        (correlation_matrix > sigma) | (correlation_matrix < -sigma),
        correlation_matrix,
        0
    )

    # Convert back to DataFrame for better readability
    filtered_correlation_df = pd.DataFrame(filtered_correlation,
                                           index=correlation_matrix.index,
                                           columns=correlation_matrix.columns)

    print("=======================================DataFrame=====================================================")
    print(data_frame.head(10))
    print(f"Sigma (std deviation of correlation values): {sigma}")
    print("=====================================================================================================")

    if label_encoders:
        print("Columns that were label encoded:", list(label_encoders.keys()))
    else:
        print("No columns required label encoding.")

    print("=======================================Correlation matrix============================================")
    print(correlation_matrix)
    print("=====================================================================================================")

    return filtered_correlation_df, label_encoders, sigma, method
