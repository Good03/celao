import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.preprocessing import LabelEncoder

def draw_heatmap(correlation_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title('Correlation Heatmap')
    plt.show()


def draw_plot(correlation_matrix):
    """Create a line plot from the correlation matrix."""
    # Set the size of the plot
    plt.figure(figsize=(14, 8))

    # Plot each correlation value with a different line
    for i in range(len(correlation_matrix)):
        plt.plot(correlation_matrix.index, correlation_matrix.iloc[:, i], marker='o',
                 label=correlation_matrix.columns[i])

    plt.title('Correlation Plot', fontsize=20)
    plt.xlabel('Attributes', fontsize=14)
    plt.ylabel('Correlation Coefficient', fontsize=14)
    plt.axhline(0, color='grey', linewidth=0.8, linestyle='--')  # Add a horizontal line at y=0 for reference
    plt.xticks(rotation=45, ha='right')  # Rotate x labels for better visibility
    plt.legend(title='Attributes', bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside the plot
    plt.grid()  # Add a grid for better readability
    plt.tight_layout()  # Adjust layout to make room for the legend
    plt.show()


def draw_graph(correlation_matrix, threshold=0.5):
    """Create a graph based on the correlation matrix with a given threshold."""
    # Create a new graph
    G = nx.Graph()

    # Add edges for correlations above the threshold (and below -threshold)
    for i in range(len(correlation_matrix)):
        for j in range(len(correlation_matrix)):
            if i != j:
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > threshold:  # Check for significant correlation
                    G.add_edge(correlation_matrix.index[i], correlation_matrix.columns[j], weight=corr_value)

    # Draw the graph
    plt.figure(figsize=(12, 12))

    # Increase the k parameter to create more space between nodes
    pos = nx.spring_layout(G, seed=42, k=1.5)  # Adjust k value here

    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', edgecolors='black')
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

    # Draw labels for nodes
    nx.draw_networkx_labels(G, pos, font_size=12)

    # Draw edge labels for correlation coefficients
    edge_labels = nx.get_edge_attributes(G, 'weight')
    edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}  # Format for 2 decimal places
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    plt.title('Correlation Network Graph', fontsize=20)
    plt.axis('off')  # Hide axes
    plt.show()