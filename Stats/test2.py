import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr, isinstalled
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

def perform_ergm_analysis(file_path, source_col='Source', target_col='Target', output_summary='ergm_summary.txt'):
    # Check and handle package imports
    network = importr('network', on_conflict='warn') if isinstalled('network') else None
    ergm = importr('ergm', on_conflict='warn') if isinstalled('ergm') else None

    # Load data using pandas
    net_data = pd.read_csv(file_path)
    net_data = net_data[net_data[source_col] != net_data[target_col]].drop_duplicates()

    # Aggregate parallel edges if any (counting the occurrences of each edge)
    net_data['weight'] = 1  # Assign a weight of 1 to each edge initially
    net_data = net_data.groupby([source_col, target_col]).weight.sum().reset_index()

    # Ensure pandas2ri is activated for the session
    pandas2ri.activate()

    # Convert pandas DataFrame to R DataFrame
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_net_data = ro.conversion.py2rpy(net_data)

    # Assign the R DataFrame to a variable in R's global environment
    ro.globalenv['df'] = r_net_data

    # Execute R code for network analysis
    ro.r('''
    df$Source <- as.numeric(df$Source)
    df$Target <- as.numeric(df$Target)
    # Set multiple=TRUE to handle parallel edges if needed
    net <- network::network(df, directed = FALSE, loops = TRUE, multiple = TRUE)
    ergm_model <- ergm::ergm(net ~ edges)
    summary_ergm <- summary(ergm_model)
    ''')

    # Retrieve the summary from R
    summary_ergm = ro.r('summary_ergm')
    
    # Write the summary to a text file
    with open(output_summary, 'w') as f:
        f.write(str(summary_ergm))

    print("ERGM summary has been saved to:", output_summary)

# Example usage
file_path = ''  # Path to your CSV file
perform_ergm_analysis(file_path)
