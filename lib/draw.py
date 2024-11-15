import itertools

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.patches import Rectangle


def draw_heatmap(correlation_matrix, sigma_value=None):
    """Draw a heatmap for the correlation matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='flare', square=True, vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    if sigma_value:
        plt.text(1.1, 1.1, f'Sigma: {sigma_value:.2f}', fontsize=12, ha='right', va='top',
                 transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

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


def find_edge_point(x1, y1, x2, y2, node_width, node_height):
    """Find the intersection point of a line with a rectangle's boundary."""
    dx = x2 - x1
    dy = y2 - y1
    aspect_ratio = node_height / node_width

    # Adjusted to compute intersection with the rectangle's perimeter
    if abs(dy / (dx + 1e-9)) > aspect_ratio:  # Line intersects top/bottom
        if dy > 0:  # Top
            y = y1 + node_height / 2
            x = x1 + dx * (node_height / 2) / dy
        else:  # Bottom
            y = y1 - node_height / 2
            x = x1 - dx * (node_height / 2) / dy
    else:  # Line intersects left/right
        if dx > 0:  # Right
            x = x1 + node_width / 2
            y = y1 + dy * (node_width / 2) / dx
        else:  # Left
            x = x1 - node_width / 2
            y = y1 - dy * (node_width / 2) / dx

    return x, y

def draw_rectangular_nodes(ax, pos, node_width, node_height):
    """Draw rectangular nodes on the provided axis."""
    for node, (x, y) in pos.items():
        rect = Rectangle((x - node_width / 2, y - node_height / 2), node_width, node_height, color='lightpink',
                         ec='white')
        ax.add_patch(rect)
        ax.text(x, y, str(node), fontsize=10, ha='center', va='center')


def draw_edges(ax, pos, G, node_width, node_height):
    """Draw edges for the graph."""
    for edge in G.edges():
        x1, y1 = pos[edge[0]]
        x2, y2 = pos[edge[1]]
        start_x, start_y = find_edge_point(x1, y1, x2, y2, node_width, node_height)
        end_x, end_y = find_edge_point(x2, y2, x1, y1, node_width, node_height)
        weight = G[edge[0]][edge[1]]['weight']  # Get the weight of the edge
        linestyle = 'dashed' if weight == 0 else 'solid'
        ax.plot([start_x, end_x], [start_y, end_y], color='white', alpha=0.7, zorder=1, linestyle=linestyle)


def draw_edge_labels(ax, pos, G):
    """Draw edge labels for the graph."""
    edge_labels = nx.get_edge_attributes(G, 'weight')
    edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
    # Explicitly use the 'pos' for edge label placement
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_color='red', label_pos=0.65, font_size=10,
        bbox=dict(facecolor="grey", ec="grey"), ax=ax
    )


def create_graph(correlation_matrix, sigma_value=None):
    """Create a graph based on correlation matrix, optionally filtering by sigma."""
    G = nx.Graph()
    for i in range(len(correlation_matrix)):
        for j in range(i + 1, len(correlation_matrix)):
            corr_value = correlation_matrix.iloc[i, j]
            if sigma_value is None or abs(corr_value) > sigma_value:
                G.add_edge(correlation_matrix.index[i], correlation_matrix.columns[j], weight=corr_value)
    return G


def find_all_subgraphs(G):
    """Find all possible complete subgraphs (cliques) of size >= 2."""
    nodes = list(G.nodes())
    subgraphs = []

    # Check all subsets of nodes of size 2 or more
    for size in range(3, 5):
        for subset in itertools.combinations(nodes, size):
            subgraph = G.subgraph(subset)
            if nx.complete_graph(subgraph):
                subgraphs.append(subgraph)

    return subgraphs


def draw_graph(correlation_matrix, sigma_value=None, method=None):
    """Main function to draw graphs before and after applying sigma filtering."""
    # Create the graph before filtering
    G = create_graph(correlation_matrix)

    # Create a canvas with 2 subplots (side by side)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), facecolor='grey')

    # Draw the graph before filtering on the first subplot
    pos = nx.circular_layout(G)
    ax = axes[0]
    node_width, node_height = 0.25, 0.1
    draw_rectangular_nodes(ax, pos, node_width, node_height)
    draw_edges(ax, pos, G, node_width, node_height)
    draw_edge_labels(ax, pos, G)
    ax.set_title('Main graph (before filtering)')
    ax.axis('off')
    if sigma_value:
        plt.text(1.1, 1.1, f'Method: {method}', fontsize=12, ha='right', va='top',
                 transform=plt.gca().transAxes, bbox=dict(facecolor='lightgreen', alpha=0.5, edgecolor='black'))
    if sigma_value:
        plt.text(1.1, 1.05, f'Sigma: {sigma_value:.2f}', fontsize=12, ha='right', va='top',
                 transform=plt.gca().transAxes, bbox=dict(facecolor='lightblue', alpha=0.5, edgecolor='white'))
    # Draw the graph after sigma filtering on the second subplot, if sigma is provided
    if sigma_value:
        G_filtered = create_graph(correlation_matrix, sigma_value)
        pos_filtered = nx.circular_layout(G_filtered)
        ax = axes[1]
        draw_rectangular_nodes(ax, pos_filtered, node_width, node_height)
        draw_edges(ax, pos_filtered, G_filtered, node_width, node_height)
        draw_edge_labels(ax, pos_filtered, G_filtered)
        ax.set_title(f'Main graph (after filtering by sigma = {sigma_value:.2f})')
        ax.axis('off')

        # Find all complete subgraphs (cliques) in the filtered graph
        subgraphs = find_all_subgraphs(G_filtered)

        # Create a new figure for subgraphs
        fig_subgraphs, axes_subgraphs = plt.subplots(1, len(subgraphs), figsize=(20, 10), facecolor='grey')
        if len(subgraphs) == 1:
            axes_subgraphs = [axes_subgraphs]  # Make it iterable for a single subgraph

        for i, subgraph in enumerate(subgraphs):
            pos_subgraph = nx.spring_layout(subgraph, seed=42)  # Layout for each subgraph
            ax_subgraph = axes_subgraphs[i]
            draw_rectangular_nodes(ax_subgraph, pos_subgraph, node_width, node_height)
            draw_edges(ax_subgraph, pos_subgraph, subgraph, node_width, node_height)
            draw_edge_labels(ax_subgraph, pos_subgraph, subgraph)
            ax_subgraph.set_title(f'Subgraph {i + 1} with {len(subgraph.nodes)} nodes')
            ax_subgraph.axis('off')

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()








