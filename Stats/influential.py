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
net_data = pd.read_excel("Student Survey - Jan.xlsx", sheet_name="net_1_Influential")
participants_data = pd.read_excel("Student Survey - Jan.xlsx", sheet_name="participants")

# Data cleaning and preparation
participants_data['Participant-ID'] = participants_data['Participant-ID'].astype(str)
net_data['Source'] = net_data['Source'].astype(str)
net_data['Target'] = net_data['Target'].astype(str)

# Convert and handle non-numeric entries
participants_data['Perc_Academic'] = pd.to_numeric(participants_data['Perc_Academic'], errors='coerce')
participants_data['Perc_Effort'] = pd.to_numeric(participants_data['Perc_Effort'], errors='coerce')
participants_data.fillna(0, inplace=True)  # Replace NAs with 0

# Normalize Perc_Academic and Perc_Effort
participants_data['Perc_Academic_Norm'] = (participants_data['Perc_Academic'] - participants_data['Perc_Academic'].mean()) / participants_data['Perc_Academic'].std()
participants_data['Perc_Effort_Norm'] = (participants_data['Perc_Effort'] - participants_data['Perc_Effort'].mean()) / participants_data['Perc_Effort'].std()

# Merge data
net_data = pd.merge(net_data, participants_data[['Participant-ID', 'Perc_Academic_Norm', 'Perc_Effort_Norm']], left_on='Source', right_on='Participant-ID', how='left')
net_data.rename(columns={'Perc_Academic_Norm': 'Source_Perc_Academic_Norm', 'Perc_Effort_Norm': 'Source_Perc_Effort_Norm'}, inplace=True)
net_data = pd.merge(net_data, participants_data[['Participant-ID', 'Perc_Academic_Norm', 'Perc_Effort_Norm']], left_on='Target', right_on='Participant-ID', how='left')
net_data.rename(columns={'Perc_Academic_Norm': 'Target_Perc_Academic_Norm', 'Perc_Effort_Norm': 'Target_Perc_Effort_Norm'}, inplace=True)

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

# Assign normalized scores as node attributes
net %v% "Source_Perc_Academic_Norm" <- df$Source_Perc_Academic_Norm
net %v% "Source_Perc_Effort_Norm" <- df$Source_Perc_Effort_Norm
net %v% "Target_Perc_Academic_Norm" <- df$Target_Perc_Academic_Norm
net %v% "Target_Perc_Effort_Norm" <- df$Target_Perc_Effort_Norm

# Fit the ERGM
ergm_model <- ergm::ergm(net ~ edges + nodecov("Source_Perc_Academic_Norm") + nodecov("Source_Perc_Effort_Norm") + nodecov("Target_Perc_Academic_Norm") + nodecov("Target_Perc_Effort_Norm"))
summary_ergm <- summary(ergm_model)
print(summary_ergm)
''')
