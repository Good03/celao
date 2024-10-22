import pandas as pd
from lib.correlation_methods import *
from lib.draw import draw_heatmap, draw_plot, draw_graph
from lib.preprocessing import *

# Main part of the program
if __name__ == '__main__':
    # Load the dataset
    dataset = load_dataset()

    # Calculate correlation matrix using label encoding
    correlation_matrix, _ = calculate_correlation_with_label_encoding(dataset, "pearson")

    # Visualize the correlation matrix
    draw_heatmap(correlation_matrix)
    draw_plot(correlation_matrix)
    draw_graph(correlation_matrix, threshold=0.01)
