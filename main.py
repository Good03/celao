from lib.preprocessing import *
from lib.correlation_methods import *
from lib.draw import draw_graph

# Main part of the program
if __name__ == '__main__':
    selected_dataset = select_dataset()
    if selected_dataset:
        dataset = load_dataset(selected_dataset)
        selected_encoding_method = select_encoding_method()
        if selected_encoding_method:
            selected_correlation_method = select_correlation_method()
            match selected_encoding_method:
                case "Label Encoding":
                    correlation_matrix, _, sigma, = calculate_correlation_with_label_encoding(dataset, method=selected_correlation_method)
                case "Hashing":
                    correlation_matrix, sigma = calculate_correlation_with_hashing(dataset, method=selected_correlation_method)
                case "Word2Vec":
                    correlation_matrix, sigma = calculate_correlation_with_hashing(dataset, method=selected_correlation_method)
                case "GloVe":
                    correlation_matrix, sigma = calculate_correlation_with_hashing(dataset, method=selected_correlation_method)
        draw_graph(correlation_matrix, sigma_value=sigma)

