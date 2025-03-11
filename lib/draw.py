import itertools
import math

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib.cm as cm


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
        color = cmap(i)
        node_text = str(node).replace(" ", "\n")# Выбираем цвет для текущей вершины
        ax.text(
            x, y, str(node_text),
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
        G, pos, edge_labels=edge_labels, font_color='black', label_pos=0.65, font_size=7,
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
    for size in range(3, 7):
        for subset in itertools.combinations(nodes, size):
            subgraph = G.subgraph(subset)
            # Check if the subgraph is a complete graph
            if subgraph.number_of_edges() == size * (size - 1) / 2:
                subgraphs.append(subgraph)

    return subgraphs


def draw_graph(correlation_matrix, sigma_value=None, correlation_method=None, encoding_method=None, dataset_name=None):
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
    ax.set_title('Main graph')
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
        ax.set_title(f'Main graph filtered by sigma value')
        ax.axis('off')

        subgraphs = find_all_subgraphs(G_filtered)
        print(f"Found {len(subgraphs)} complete subgraphs")

        method_text = f"Correlation method: {correlation_method}\nEncoding method: {encoding_method}\nSigma value: {sigma_value:.2f}"
        edges_text = f"Edge styles:\n— Dashed: below sigma threshold\n— Solid: above sigma threshold"

        fig.text(0.01, 0.95, edges_text, fontsize=12, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.5))
        fig.text(0.85, 0.95, method_text, fontsize=12, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.5))
        filename = f"Base_graph_{dataset_name}_{correlation_method}_{encoding_method}.png"
        plt.savefig(filename, dpi=1200)
        print(f"Saved: {filename}")
        plt.show()


        subgraph_groups = {}
        for subgraph in subgraphs:
            size = len(subgraph.nodes())
            if size not in subgraph_groups:
                subgraph_groups[size] = []
            subgraph_groups[size].append(subgraph)

        max_cols = 4
        max_rows = 4
        max_graphs_per_fig = max_cols * max_rows

        for size, group in subgraph_groups.items():
            num_graphs = len(group)
            for batch_idx in range(0, num_graphs, max_graphs_per_fig):
                batch = group[batch_idx:batch_idx + max_graphs_per_fig]
                batch_size = len(batch)

                cols = min(max_cols, batch_size)
                rows = math.ceil(batch_size / cols)

                fig_group, axes_group = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), facecolor='white')

                axes_group = axes_group.flatten() if batch_size > 1 else [axes_group]

                for i, subgraph in enumerate(batch):
                    pos_subgraph = nx.circular_layout(subgraph)
                    ax_subgraph = axes_group[i]
                    draw_rectangular_nodes(ax_subgraph, pos_subgraph)
                    draw_edges(ax_subgraph, pos_subgraph, subgraph)
                    draw_edge_labels(ax_subgraph, pos_subgraph, subgraph)
                    ax_subgraph.axis('off')

                for j in range(batch_size, len(axes_group)):
                    fig_group.delaxes(axes_group[j])

                filename = f"{size}ptychs_{dataset_name}_{correlation_method}_{encoding_method}_part{batch_idx // max_graphs_per_fig + 1}.png"
                plt.savefig(filename, dpi=1200)
                print(f"Saved: {filename}")

                plt.tight_layout()
                plt.show()









