import itertools

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib.cm as cm
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


def draw_rectangular_nodes(ax, pos):
    """Draw rectangular nodes with custom colors from a colormap."""
    num_nodes = len(pos)
    cmap = cm.get_cmap('Pastel1', num_nodes)  # Получаем цветовую карту Pastel1

    for i, (node, (x, y)) in enumerate(pos.items()):
        color = cmap(i)  # Выбираем цвет для текущей вершины
        ax.text(
            x, y, str(node),
            fontsize=10,
            color='black',
            fontfamily='sans-serif',
            ha='center',
            va='center',
            bbox=dict(facecolor=color, edgecolor='black', boxstyle='round,pad=0.3')
        )


def draw_edges(ax, pos, G):
    """Draw edges for the graph without calculating intersection points."""
    for edge in G.edges():
        x1, y1 = pos[edge[0]]
        x2, y2 = pos[edge[1]]

        weight = G[edge[0]][edge[1]]['weight']
        linestyle = 'dashed' if weight == 0 else 'solid'
        ax.plot([x1, x2], [y1, y2], color='black', alpha=0.7, zorder=1, linestyle=linestyle)


def draw_edge_labels(ax, pos, G):
    """Draw edge labels for the graph."""
    edge_labels = nx.get_edge_attributes(G, 'weight')
    edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
    # Explicitly use the 'pos' for edge label placement
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_color='black', label_pos=0.65, font_size=10,
        bbox=dict(facecolor="white", ec="white"), ax=ax
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
    """Find all possible complete subgraphs (cliques) of size >= 3."""
    nodes = list(G.nodes())
    subgraphs = []

    # Check all subsets of nodes of size 3 or more
    for size in range(3, 6):
        for subset in itertools.combinations(nodes, size):
            subgraph = G.subgraph(subset)
            # Check if the subgraph is a complete graph
            if subgraph.number_of_edges() == size * (size - 1) / 2:
                subgraphs.append(subgraph)

    return subgraphs


def draw_graph(correlation_matrix, sigma_value=None, methodOfCorrelation=None, methodOfEncoding=None):
    G = create_graph(correlation_matrix)

    print("Drawing main graph")
    print(f"Nodes in G: {G.nodes()}")
    print(f"Edges in G: {G.edges()}")

    fig, axes = plt.subplots(1, 2, figsize=(20, 10), facecolor='white')

    pos = nx.circular_layout(G)
    ax = axes[0]
    draw_rectangular_nodes(ax, pos)
    draw_edges(ax, pos, G)
    draw_edge_labels(ax, pos, G)
    ax.set_title('Main graph (before filtering)')
    ax.axis('off')

    if sigma_value:
        G_filtered = create_graph(correlation_matrix, sigma_value)
        print("Drawing filtered graph")
        print(f"Nodes in G_filtered: {G_filtered.nodes()}")
        print(f"Edges in G_filtered: {G_filtered.edges()}")

        pos_filtered = nx.circular_layout(G_filtered)
        ax = axes[1]
        draw_rectangular_nodes(ax, pos_filtered)
        draw_edges(ax, pos_filtered, G_filtered)
        draw_edge_labels(ax, pos_filtered, G_filtered)
        ax.set_title(f'Main graph (after filtering by sigma = {sigma_value:.2f})')
        ax.axis('off')

        subgraphs = find_all_subgraphs(G_filtered)
        print(f"Found {len(subgraphs)} complete subgraphs")

        plt.show()

        # Create a new figure for subgraphs
        subgraph_groups = {}
        for subgraph in subgraphs:
            size = len(subgraph.nodes())
            if size not in subgraph_groups:
                subgraph_groups[size] = []
            subgraph_groups[size].append(subgraph)

        for size, group in subgraph_groups.items():
            if len(group) < 3:
                fig_group, axes_group = plt.subplots(1, len(group), figsize=(10, 5), facecolor='white')
            else:
                fig_group, axes_group = plt.subplots(1, len(group), figsize=(15, 7), facecolor='white')
            if len(group) == 1:
                axes_group = [axes_group]  # Чтобы обеспечить итерабельность
            for i, subgraph in enumerate(group):
                pos_subgraph = nx.spring_layout(subgraph, seed=42, k=0.1, scale=0.1)
                ax_subgraph = axes_group[i]
                draw_rectangular_nodes(ax_subgraph, pos_subgraph)
                draw_edges(ax_subgraph, pos_subgraph, subgraph)
                draw_edge_labels(ax_subgraph, pos_subgraph, subgraph)
                ax_subgraph.set_title(f'Subgraph {i + 1} ({size} nodes)')
                ax_subgraph.axis('off')

            plt.suptitle(f'Subgraphs with {size} nodes', fontsize=14)
            plt.tight_layout()
            plt.show()








