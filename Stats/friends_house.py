import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# Import R packages
utils = importr('utils', on_conflict='warn')
network = importr('network', on_conflict='warn')
ergm = importr('ergm', on_conflict='warn')
readxl = importr('readxl')

# Load data using pandas
net_friends = pd.read_excel("Student Survey - Jan.xlsx", sheet_name="net_0_Friends")
participants = pd.read_excel("Student Survey - Jan.xlsx", sheet_name="participants")

# Data cleaning and preparation
participants['Participant-ID'] = participants['Participant-ID'].astype(str)
net_friends['Source'] = net_friends['Source'].astype(str)
net_friends['Target'] = net_friends['Target'].astype(str)

# Merge data
net_data = pd.merge(net_friends, participants[['Participant-ID', 'House']], left_on='Source', right_on='Participant-ID', how='left')
net_data.rename(columns={'House': 'Source_House'}, inplace=True)
net_data = pd.merge(net_data, participants[['Participant-ID', 'House']], left_on='Target', right_on='Participant-ID', how='left')
net_data.rename(columns={'House': 'Target_House'}, inplace=True)

# Filter out self-loops
net_data = net_data[net_data['Source'] != net_data['Target']]

# Activate conversion and load data to R
with localconverter(ro.default_converter + pandas2ri.converter):
    r_net_data = ro.conversion.py2rpy(net_data)

ro.globalenv['df'] = r_net_data

# R code execution to create network and run ERGM
ro.r('''
df$Source <- as.character(df$Source)
df$Target <- as.character(df$Target)
net <- network::network(df, directed = TRUE, loops = FALSE)

# Assign House attributes
net %v% "House" <- df$Source_House  # Assuming the House attribute is correctly mapped

# Fit the ERGM
ergm_model <- ergm::ergm(net ~ edges + nodematch("House", diff = FALSE))
summary_ergm <- summary(ergm_model)
print(summary_ergm)
''')
