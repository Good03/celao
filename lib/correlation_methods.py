from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import FeatureHasher
from gensim.models import Word2Vec
import numpy as np
import pandas as pd


def calculate_correlation_with_label_encoding(data_frame, method: str):
    """Calculate correlation, including string attributes with Label Encoding."""
    print("Started Label Encoding correlation calculation...")
    label_encoders = {}
    encoded_columns = []
    for name in data_frame.columns:
        if data_frame[name].dtype == 'object':
            label_encoder = LabelEncoder()
            data_frame[name] = label_encoder.fit_transform(data_frame[name]) + 1
            label_encoders[name] = label_encoder
            encoded_columns.append(name)

    correlation_matrix = data_frame.corr(method=method)
    print("=======================================Correlation Matrix===================================================")
    print(correlation_matrix)
    print("============================================================================================================")
    print("=======================================Columns that were label encoded===================================================")
    print(encoded_columns)
    print("============================================================================================================")

    sigma = correlation_matrix.values.std()
    np.fill_diagonal(correlation_matrix.values, 0)

    filtered_correlation = np.where(
        (correlation_matrix > sigma) | (correlation_matrix < -sigma),
        correlation_matrix,
        0
    )

    filtered_correlation_df = pd.DataFrame(filtered_correlation,
                                           index=correlation_matrix.index,
                                           columns=correlation_matrix.columns)
    print("Finished Label Encoding correlation calculation.")
    return filtered_correlation_df, label_encoders, sigma


def calculate_correlation_with_hashing(data_frame, method: str, n_features=1):
    """Calculate correlation using Hashing Trick for categorical attributes."""
    print("Started Hashing correlation calculation...")
    hasher = FeatureHasher(n_features=n_features, input_type="string")
    encoded_columns = []
    for name in data_frame.columns:
        if data_frame[name].dtype == 'object':
            hashed_values = hasher.transform(data_frame[name].astype(str).apply(lambda x: [x])).toarray().flatten()
            data_frame[name] = hashed_values
            encoded_columns.append(name)

    correlation_matrix = data_frame.corr(method=method)
    print("=======================================Correlation Matrix===================================================")
    print(correlation_matrix)
    print("============================================================================================================")
    print("=======================================Columns that were hashed===================================================")
    print(encoded_columns)
    print("============================================================================================================")

    correlation_matrix.fillna(0, inplace=True)
    sigma = correlation_matrix.values.std()
    np.fill_diagonal(correlation_matrix.values, 0)

    filtered_correlation = np.where(
        (correlation_matrix > sigma) | (correlation_matrix < -sigma),
        correlation_matrix,
        0
    )

    filtered_correlation_df = pd.DataFrame(filtered_correlation,
                                           index=correlation_matrix.index,
                                           columns=correlation_matrix.columns)
    print("Finished Hashing correlation calculation.")
    return filtered_correlation_df, sigma


def calculate_correlation_with_word2vec(data_frame, method: str):
    """Calculate correlation using Word2Vec encoding for categorical attributes."""
    print("Starting Word2Vec correlation calculation.")
    encoded_columns = []
    for name in data_frame.columns:
        if data_frame[name].dtype == 'object':
            unique_values = data_frame[name]
            sentences = [[str(val)] for val in unique_values]
            model = Word2Vec(sentences, vector_size=1, min_count=1)
            value_to_vector = {val: model.wv[str(val)] for val in unique_values}
            vector_df = data_frame[name].map(value_to_vector).apply(pd.Series)
            vector_df.columns = [f"{name}_vec_{i}" for i in range(1)]
            data_frame = pd.concat([data_frame.drop(columns=[name]), vector_df], axis=1)
            encoded_columns.append(name)

    correlation_matrix = data_frame.corr(method=method)
    print("=======================================Correlation Matrix===================================================")
    print(correlation_matrix)
    print("============================================================================================================")
    print("=======================================Columns that were word2vec encoded===================================================")
    print(encoded_columns)
    print("============================================================================================================")

    sigma = correlation_matrix.values.std()
    np.fill_diagonal(correlation_matrix.values, 0)

    filtered_correlation = np.where(
        (correlation_matrix > sigma) | (correlation_matrix < -sigma),
        correlation_matrix,
        0
    )

    filtered_correlation_df = pd.DataFrame(filtered_correlation,
                                           index=correlation_matrix.index,
                                           columns=correlation_matrix.columns)
    print("Finished Word2Vec correlation calculation.")

    return filtered_correlation_df, sigma


def calculate_correlation_with_pseudo_glove(data_frame, method: str, vector_size=1):
    print("Started GloVe correlation calculation")
    np.random.seed(42)
    encoded_columns = []

    for name in data_frame.columns:
        if data_frame[name].dtype == 'object':
            unique_values = data_frame[name].unique()
            value_to_vector = {val: np.random.rand(vector_size) for val in unique_values}
            vector_df = data_frame[name].map(value_to_vector).apply(pd.Series)
            vector_df.columns = [f"{name}_vec_{i}" for i in range(vector_size)]
            data_frame = pd.concat([data_frame.drop(columns=[name]), vector_df], axis=1)
            encoded_columns.append(name)

    correlation_matrix = data_frame.corr(method=method)
    print("=======================================Correlation Matrix===================================================")
    print(correlation_matrix)
    print("============================================================================================================")
    print("=======================================Columns that were pseudo-glove encoded===================================================")
    print(encoded_columns)
    print("============================================================================================================")

    sigma = correlation_matrix.values.std()
    np.fill_diagonal(correlation_matrix.values, 0)

    filtered_correlation = np.where(
        (correlation_matrix > sigma) | (correlation_matrix < -sigma),
        correlation_matrix,
        0
    )
    filtered_correlation_df = pd.DataFrame(filtered_correlation,
                                           index=correlation_matrix.index,
                                           columns=correlation_matrix.columns)
    print("Finished GloVe correlation calculation")
    return filtered_correlation_df, sigma