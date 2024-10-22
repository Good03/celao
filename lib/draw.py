import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


def draw_heatmap(correlation_matrix):
    """Draw a heatmap for the correlation matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title('Correlation Heatmap')
    plt.show()


def draw_plot(correlation_matrix):
    """Create a line plot from the correlation matrix."""
    plt.figure(figsize=(14, 8))

    for i in range(len(correlation_matrix)):
        plt.plot(correlation_matrix.index, correlation_matrix.iloc[:, i], marker='o',
                 label=correlation_matrix.columns[i])

    plt.title('Correlation Plot', fontsize=20)
    plt.xlabel('Attributes', fontsize=14)
    plt.ylabel('Correlation Coefficient', fontsize=14)
    plt.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Attributes', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.show()


def draw_graph(correlation_matrix, threshold=0.5):
    """Create a graph based on the correlation matrix with a given threshold."""
    G = nx.Graph()

    for i in range(len(correlation_matrix)):
        for j in range(i + 1, len(correlation_matrix)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > threshold:
                G.add_edge(correlation_matrix.index[i], correlation_matrix.columns[j], weight=corr_value)

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42, k=1.5)
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', edgecolors='black')
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=12)

    edge_labels = nx.get_edge_attributes(G, 'weight')
    edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    plt.title('Correlation Network Graph', fontsize=20)
    plt.axis('off')
    plt.show()
