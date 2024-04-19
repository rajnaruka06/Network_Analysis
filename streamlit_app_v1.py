import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
from networkx.algorithms import community
import matplotlib.colors as mcolors
import numpy as np
import pdfkit
from io import BytesIO

def create_graph(df, source_col_name = "source", target_col_name = "target"):
    graph = nx.Graph()
    for i in range(len(df)):
        graph.add_edge(df[source_col_name][i], df[target_col_name][i])
    
    return graph

def calculate_network_statistics(graph):
    number_of_nodes = graph.number_of_nodes()
    number_of_edges = graph.number_of_edges()
    average_degree = sum(dict(graph.degree()).values()) / graph.number_of_nodes()
    density = nx.density(graph)
    clustering_coefficient = nx.average_clustering(graph)
    average_shortest_path_length = nx.average_shortest_path_length(graph)
    diameter = nx.diameter(graph)
    degree_centrality = nx.degree_centrality(graph)
    closeness_centrality = nx.closeness_centrality(graph)
    betweenness_centrality = nx.betweenness_centrality(graph)
    eigenvector_centrality = nx.eigenvector_centrality(graph)
    pagerank = nx.pagerank(graph)
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

def create_network_visualization(graph, selected_visual_metrics, network_statistics):
    nt = Network()
    nt.from_nx(graph)
    pos = nx.spectral_layout(graph)

    max_metric_value = max(network_statistics[selected_visual_metrics].values())
    min_metric_value = min(network_statistics[selected_visual_metrics].values())

    for node, coords in pos.items():
        for node_dict in nt.nodes:
            if node_dict['id'] == node:
                node_dict['x'] = coords[0]
                node_dict['y'] = coords[1]
                node_label = f"ID: {node}"
                metric_value = network_statistics[selected_visual_metrics][node]
                node_label += f", {selected_visual_metrics}: {metric_value:.2f}"
                node_dict['label'] = node_label
                color_density = (metric_value - min_metric_value) / (max_metric_value - min_metric_value)
                node_dict['color'] = f"rgba(0, 0, 255, {color_density})"
                break
    
    output_dir = "./Layouts/app_graph_layout.html"
    nt.save_graph(output_dir)
    return output_dir

def create_community_visualization(graph, network_statistics):
    nt = Network()
    nt.from_nx(graph)
    pos = nx.spectral_layout(graph)

    ## Community coloring
    number_of_communities = network_statistics['Number of communities']
    communities = network_statistics['Communities']
    pastel_cmap = mcolors.LinearSegmentedColormap.from_list(
        'pastel_cmap', 
        np.random.rand(number_of_communities, 4), 
        N=number_of_communities
    )
    pastel_colors = pastel_cmap(np.linspace(0, 1, number_of_communities))
    node_comunity_color_map = dict()
    for idx, community in enumerate(communities):
        color = mcolors.to_hex(pastel_colors[idx])
        for node in community:
            node_comunity_color_map[node] = color
    
    for node, coords in pos.items():
        for node_dict in nt.nodes:
            if node_dict['id'] == node:
                node_dict['x'] = coords[0]
                node_dict['y'] = coords[1]
                node_dict['color'] = node_comunity_color_map[node]
                break
    
    output_dir = "./Layouts/app_community_layout.html"
    nt.save_graph(output_dir)
    return output_dir

def perform_ergm_analysis(df, source_column, destination_column):
    return "Work in Progress"

def perform_alaam_analysis(df, source_column, destination_column):
    return "Work in Progress"

if __name__ == "__main__":

    st.title("Network Analysis App")

    # File Upload FUnctionality
    st.sidebar.title("Upload CSV File")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        progress_bar = st.sidebar.progress(0)
        with st.spinner("Uploading file..."):
            df = pd.read_csv(uploaded_file)
            for col in df.columns: df[col] = df[col].astype(str)
        progress_bar.progress(100)
        st.success("File Uploaded Successfully!")
    
    source_column = st.sidebar.text_input("Enter Source Column Name", "source")
    destination_column = st.sidebar.text_input("Enter Destination Column Name", "target")

    if uploaded_file is not None:

        graph = create_graph(df, source_column, destination_column)

        # Network Visualization and Metrics
        st.sidebar.title("Select Visual Metrics")
        visual_metrics_list = ("Degree Centrality", "Closeness Centrality", "Betweenness Centrality", "Eigenvector Centrality",
                            "PageRank", "HITS Hub Scores", "HITS Authority Scores")
        selected_visual_metrics = st.sidebar.radio("Select Visual Metrics", visual_metrics_list)

        # Network Statistics
        st.sidebar.title("Select Network Statistics")
        metrics_list = ("Number of Nodes", "Number of Edges", "Average Degree", "Density", "Clustering Coefficient", "Average Shortest Path Length", "Diameter"
                        , 'Number of communities', 'Community with the largest size', 'Community with the smallest size', 'Modularity')
        selected_metrics = st.sidebar.multiselect("Select Metrics", metrics_list)
        network_statistics = calculate_network_statistics(graph)
        if selected_metrics: 
            st.header("Network Analysis Metrics")
        for metric in selected_metrics:
            st.write(f"**{metric}:** {network_statistics.get(metric):.2f}")

        st.header("Network Visualization")
        st.markdown("Zoom in/out, Drag or select to see individual node and its attributes.")
        viz_path = create_network_visualization(graph, selected_visual_metrics, network_statistics)
        st.components.v1.html(open(viz_path, 'r', encoding='utf-8').read(), height=800)

        # Community Visualization
        show_community_visualization = st.checkbox("Show Community Visualization")
        if show_community_visualization:
            st.header("Community Visualization")
            st.markdown("Zoom in/out, Drag or select to see individual node and its attributes.")
            community_viz_path = create_community_visualization(graph, network_statistics)
            st.components.v1.html(open(community_viz_path, 'r', encoding='utf-8').read(), height=800)

        
        # Statistical Modeling
        st.sidebar.title("Select Statistical Model")
        selected_model = st.sidebar.radio("Choose Model", ("ERGM", "ALAAM"))
        if selected_model == "ERGM":
            st.header("ERGM Analysis")
            ergm_results = perform_ergm_analysis(df, source_column, destination_column)
            st.write("ERGM Results:")
            st.write(ergm_results)
        elif selected_model == "ALAAM":
            st.header("ALAAM Analysis")
            alaam_results = perform_alaam_analysis(df, source_column, destination_column)
            st.write("ALAAM Results:")
            st.write(alaam_results)

        # # Download report
        st.sidebar.title("Download Report")
        report_button = st.sidebar.button("Download Report")
        if report_button:
            st.write("Work in Progress...")
    else:
        st.warning("Please Upload a CSV File")