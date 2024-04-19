# Import necessary libraries
import pandas as pd
import networkx as nx
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import Formula



# Function to load and prepare data
def load_and_prepare_data(file_path):
    xls = pd.ExcelFile(file_path)
    participants_df = xls.parse('participants')
    net_0_friends_df = xls.parse('net_0_Friends')

    # Create a directed graph from the network data
    G = nx.from_pandas_edgelist(net_0_friends_df, source='Source', target='Target', create_using=nx.DiGraph())

    # Map participant data to nodes
    attributes = participants_df.set_index('Participant-ID').to_dict('index')
    nx.set_node_attributes(G, attributes)
    return G


# Function to setup R environment and run ERGM
def setup_r_and_run_ergm(G):
    # Activate automatic conversion
    pandas2ri.activate()

    # Import R packages
    network_r = importr('network', on_conflict="warn")
    ergm = importr('ergm', on_conflict="warn")

    # Convert NetworkX graph to a dense adjacency matrix
    adj_matrix = nx.to_numpy_array(G)  # This is the correct method for newer NetworkX versions
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_matrix = ro.r.matrix(adj_matrix, nrow=adj_matrix.shape[0])

    # Create R network object
    r_network = network_r.network(r_matrix, directed=True, loops=True, multiple=True)

    # Assign attributes
    # Ensure vertex IDs in R start from 1 (R uses 1-based indexing)
    for i, node in enumerate(G.nodes(data=True), start=1):
        if 'House' in node[1]:  # node is a tuple (node_id, attributes)
            network_r.set_vertex_attribute(r_network, 'House', node[1]['House'], vertex=i)

    # Define and fit ERGM
    formula = ro.Formula('network ~ edges + mutual + nodematch("House") + gwdegree(0.5)')
    formula.environment['network'] = r_network
    fit_ergm = ergm.ergm(formula, control=ergm.control_ergm(MCMC_burnin=100000))

    return fit_ergm

# Example usage
G = nx.DiGraph()
# Add nodes, edges, and 'House' attributes to G before calling setup_r_and_run_ergm
# Example: G.add_node(1, House='Alpha')
# Call the function
fit_ergm = setup_r_and_run_ergm(G)



# Main function to control the flow of the application
def main():
    file_path = 'Student Survey - Jan.xlsx'
    G = load_and_prepare_data(file_path)
    fit_ergm = setup_r_and_run_ergm(G)

    # Print the summary of the model
    print(ro.r('summary')(fit_ergm))

    # Plot the network (optional, requires R setup for plotting)
    # ro.r('plot')(r_network)


# Run the main function
if __name__ == "__main__":
    main()

