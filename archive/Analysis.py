## Import libraries
import pandas as pd
import networkx as nx
from pyvis.network import Network
from networkx.algorithms import community
import matplotlib.colors as mcolors
import numpy as np


def create_graph(filepath):
    ## Load file and data preprocessing
    df = pd.read_csv(filepath)

    ## Create networkX graph
    graph = nx.Graph()
    for i in range(len(df)):
        graph.add_edge(df['Source'][i], df['Target'][i])
    
    return graph

def get_network(graph, save_visualization = False, path_to_save = "graph.html"):
    
    ## Create graph visualizations in html
    nt = Network(notebook=True, select_menu=True)
    nt.from_nx(graph)
    # Set the layout algorithm to Spectral
    pos = nx.spectral_layout(graph)
    # Set node positions manually in Pyvis
    for node, coords in pos.items():
        for node_dict in nt.nodes:
            if node_dict['id'] == node:
                node_dict['x'] = coords[0]
                node_dict['y'] = coords[1]
                break
    # Save the graph with Spectral-like layout
    if save_visualization: 
        nt.save_graph(path_to_save)
    return nt

def get_network_statistics(graph):

    ## Calculating descriptive statistics
    number_of_nodes = graph.number_of_nodes()
    number_of_edges = graph.number_of_edges()
    average_degree = sum(dict(graph.degree()).values())/graph.number_of_nodes()
    density = nx.density(graph)
    clustering_coefficient = nx.average_clustering(graph)
    average_shortest_path_length = nx.average_shortest_path_length(graph)
    diameter = nx.diameter(graph)
    statistics = {
        'Number of nodes': number_of_nodes,
        'Number of edges': number_of_edges,
        'Average degree': average_degree,
        'Density': density,
        'Clustering coefficient': clustering_coefficient,
        'Average shortest path length': average_shortest_path_length,
        'Diameter': diameter
    }
    return statistics

def get_community_statistics(graph):
    ## Community detection
    
    communities = community.louvain_communities(graph)
    number_of_communities = len(communities)
    community_sizes = [len(community) for community in communities]
    largest_community = max(community_sizes)
    smallest_community = min(community_sizes)
    modularity = community.modularity(graph, communities)

    community_statistics = {
        'Number of communities': number_of_communities,
        'Community with the largest size': largest_community,
        'Community with the smallest size': smallest_community,
        'Modularity': modularity,
        'Communities': communities
    }
    return community_statistics

def visualize_communities(network, community_statistics, path_to_save = "communities.html"):
    ## Visualize the graph with communities
    # Generate a LinearSegmentedColormap with pastel colors
    number_of_communities = community_statistics['Number of communities']
    communities = community_statistics['Communities']
    pastel_cmap = mcolors.LinearSegmentedColormap.from_list(
        'pastel_cmap', 
        np.random.rand(number_of_communities, 4), 
        N=number_of_communities
    )

    # Get the pastel colors
    pastel_colors = pastel_cmap(np.linspace(0, 1, number_of_communities))
    # Set node colors based on communities
    node_comunity_color_map = dict()
    for idx, community in enumerate(communities):
        color = mcolors.to_hex(pastel_colors[idx])
        for node in community:
            node_comunity_color_map[node] = color
    for idx, node_dict in enumerate(network.nodes):
        network.nodes[idx]['color'] = node_comunity_color_map[node_dict['id']]
    # Visualize the network
    network.save_graph(path_to_save)



path_to_file = 'Friendships.csv'
graph = create_graph(path_to_file)
network = get_network(graph, save_visualization=True)
statistics = get_network_statistics(graph)
community_statistics = get_community_statistics(graph)
visualize_communities(network, community_statistics, path_to_save = "communities.html")
print(statistics)
print(community_statistics)
