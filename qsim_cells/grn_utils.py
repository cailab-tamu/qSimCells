import pandas as pd
import numpy as np
import scanpy as sc
import networkx as nx
import matplotlib.pyplot as plt

# --- 1. Data Preparation: Ensure 'adata' is loaded and ready ---
def prepare_gene_expression_data(adata):
    """
    Extracts the gene expression matrix from the AnnData object and
    converts it into a pandas DataFrame (genes x cells).
    """
    # Check if adata.X is a sparse matrix, convert to dense if necessary
    if hasattr(adata.X, 'toarray'):
        gene_expression_matrix = adata.X.T.toarray()
    else: # assuming it's already a dense numpy array
        gene_expression_matrix = adata.X.T

    # Create the DataFrame (Genes as index, Cells as columns)
    gene_expression_df = pd.DataFrame(
        gene_expression_matrix,
        index=adata.var_names,
        columns=adata.obs_names
    )
    return gene_expression_df

# --- 2. Correlation and Adjacency Matrix Calculation ---
def calculate_correlation_and_adjacency(gene_expression_df, method, threshold=0.2):
    """
    Calculates the gene-gene correlation matrix and an adjacency matrix
    based on a given absolute correlation threshold.
    """
    # Calculate the correlation matrix (Cells as rows, Genes as columns for correlation)
    # df.T ensures we correlate genes across cells
    corr_matrix = gene_expression_df.T.corr(method=method)

    # Filter for strong correlations to create the adjacency matrix.
    adj_matrix = np.where(np.abs(corr_matrix.values) > threshold, corr_matrix.values, 0)
    # Remove self-loops (correlation of a gene with itself)
    np.fill_diagonal(adj_matrix, 0)
    non_zero_count = np.count_nonzero(adj_matrix)
    print(f"Number of non-zero elements (connections): {non_zero_count}")
    
    return corr_matrix.index, adj_matrix

# --- 3. Network Graph Construction and Plotting Function (UPDATED) ---
def plot_correlation_network(adj_matrix, gene_names, title_suffix, num_nodes=20, figsize=(6, 5), filename=None):
    """
    Constructs a NetworkX graph, extracts a sub-network, plots the result,
    and optionally saves the figure.
    
    Args:
        adj_matrix (np.ndarray): The gene-gene adjacency matrix.
        gene_names (pd.Index): The list of gene names.
        title_suffix (str): The correlation method (e.g., 'Pearson').
        num_nodes (int): The number of top-connected genes to plot.
        figsize (tuple): The size of the output figure (width, height).
        filename (str, optional): Filename to save the plot (e.g., 'pearson_network.png'). 
                                   If None, the figure is only displayed.
    """
    # Create a graph object from the adjacency matrix
    G = nx.from_numpy_array(adj_matrix)

    # Map the integer indices back to gene names
    mapping = {i: gene_names[i] for i in range(len(gene_names))}
    G = nx.relabel_nodes(G, mapping)

    # Remove nodes that have no connections (isolated genes)
    G.remove_nodes_from(list(nx.isolates(G)))

    # Find the top 'num_nodes' most highly connected genes
    degrees = dict(G.degree())
    most_connected_genes = sorted(degrees, key=degrees.get, reverse=True)[:num_nodes]
    subgraph = G.subgraph(most_connected_genes)

    # Set a layout for visualization
    subgraph_pos = nx.fruchterman_reingold_layout(subgraph, seed=42)

    # Plotting: Figure size is now an argument input
    plt.figure(figsize=figsize)
    nx.draw(subgraph,
            pos=subgraph_pos,
            with_labels=True,
            node_size=300,# INCREASE THIS VALUE to make nodes larger
            font_size=10,# INCREASE THIS VALUE to make gene labels larger
            font_color='black',
            node_color='skyblue',
            edge_color='gray',
            width=0.5)
    
    plt.title(f"Gene Correlation Network ({title_suffix}): Top {len(subgraph.nodes)} Genes", fontsize=10)

    # Save the file if a filename is provided (New Feature)
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {filename}")

    plt.show()
