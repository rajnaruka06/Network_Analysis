import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr, isinstalled
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# Check and handle package imports
network = importr('network', on_conflict='warn') if isinstalled('network') else None
ergm = importr('ergm', on_conflict='warn') if isinstalled('ergm') else None

# Load data using pandas
net_data = pd.read_excel("Student Survey - Jan.xlsx", sheet_name="net_0_Friends")
net_data = net_data[net_data['Source'] != net_data['Target']].drop_duplicates()

# Aggregate parallel edges if any (counting the occurrences of each edge)
net_data['weight'] = 1  # Assign a weight of 1 to each edge initially
net_data = net_data.groupby(['Source', 'Target']).weight.sum().reset_index()

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

# Retrieve and print the summary from R
summary_ergm = ro.r('summary_ergm')
print(summary_ergm)
