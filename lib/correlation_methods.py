from sklearn.preprocessing import LabelEncoder


def calculate_correlation_with_label_encoding(data_frame, method="pearson"):
    """Calculate correlation, including string attributes with Label Encoding."""
    # Label Encoding for string columns
    label_encoders = {}
    for name in data_frame.columns:
        if data_frame[name].dtype == 'object':  # Check if it's a string column
            label_encoder = LabelEncoder()
            data_frame[name] = label_encoder.fit_transform(data_frame[name]) + 1
            label_encoders[name] = label_encoder

    correlation_matrix = data_frame.corr(method=method)

    #TODO sigma

    print("=======================================DataFrame=====================================================")
    print(data_frame.head(10))
    print("=====================================================================================================")

    if label_encoders:
        print("Columns that were label encoded:", list(label_encoders.keys()))
    else:
        print("No columns required label encoding.")

    print("=======================================Correlation matrix============================================")
    print(correlation_matrix)
    print("=====================================================================================================")

    return correlation_matrix, label_encoders
