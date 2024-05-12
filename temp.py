import pandas as pd
import networkx as nx
from networkx.algorithms import community
from openpyxl import load_workbook


def create_graph(network_df):
    graph = nx.Graph()
    for i in range(len(network_df)):
        graph.add_edge(network_df['source'][i], network_df['target'][i])
    
    return graph


def calculate_network_statistics(graph):
    
    connected_flag = nx.is_connected(graph)
    if not connected_flag:
        st.warning("The network is not connected. Some network Statistics can not be claculated.")

    number_of_nodes = graph.number_of_nodes()
    number_of_edges = graph.number_of_edges()
    average_degree = sum(dict(graph.degree()).values()) / graph.number_of_nodes()
    density = nx.density(graph)
    clustering_coefficient = nx.average_clustering(graph)
    average_shortest_path_length = nx.average_shortest_path_length(graph) if connected_flag else "Can't calculate Graph is not connected"
    diameter = nx.diameter(graph) if connected_flag else "Can't calculate Graph is not connected"
    degree_centrality = nx.degree_centrality(graph)
    closeness_centrality = nx.closeness_centrality(graph)
    betweenness_centrality = nx.betweenness_centrality(graph)
    eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=500)
    pagerank = nx.pagerank(graph, max_iter=500)
    hits = nx.hits(graph)

    communities = community.louvain_communities(graph)
    number_of_communities = len(communities)
    community_sizes = [len(community) for community in communities]
    largest_community = max(community_sizes)
    smallest_community = min(community_sizes)
    modularity = community.modularity(graph, communities)

    statstics = {
        "Number of Nodes": number_of_nodes,
        "Number of Edges": number_of_edges,
        "Average Degree": average_degree,
        "Density": density,
        "Clustering Coefficient": clustering_coefficient,
        "Average Shortest Path Length": average_shortest_path_length,
        "Diameter": diameter,
        "Degree Centrality": degree_centrality,
        "Closeness Centrality": closeness_centrality,
        "Betweenness Centrality": betweenness_centrality,
        "Eigenvector Centrality": eigenvector_centrality,
        "PageRank": pagerank,
        "HITS Hub Scores": hits[0],
        "HITS Authority Scores": hits[1],
        'Number of communities': number_of_communities,
        'Community with the largest size': largest_community,
        'Community with the smallest size': smallest_community,
        'Modularity': modularity,
        'Communities': communities
    }
    return statstics

if __name__ == '__main__':
    df = pd.read_csv('Freidnships_Data.csv')
    graph = create_graph(df)
    network_statistics = calculate_network_statistics(graph)
    selected_metrics = ['Number of Nodes', 'Number of Edges', 'Average Degree', 'Density', 'Clustering Coefficient', 'Average Shortest Path Length', 'Diameter', 'Number of communities', 'Community with the largest size', 'Community with the smallest size', 'Modularity']
    metrics_dict = {metric: network_statistics.get(metric) for metric in selected_metrics}
    df = pd.DataFrame.from_dict(metrics_dict, orient='index', columns=['Value'])
    print(df)
    # df.to_csv('network_statistics.csv')    