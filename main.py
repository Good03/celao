import pandas as pd

from lib.correlation_methods import *
from lib.draw import draw_heatmap, draw_plot, draw_graph
from lib.preprocessing import *

# Main part of the program
if __name__ == '__main__':
    dataset = load_dataset()
    correlation_matrix, _, sigma, method = calculate_correlation_with_label_encoding(dataset, "pearson")
    draw_graph(correlation_matrix, sigma, method)

