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
net_data = pd.read_excel("Student Survey - Jan.xlsx", sheet_name="net_3_MoreTime")
responses = pd.read_excel("Student Survey - Jan.xlsx", sheet_name="responses")

# Data cleaning and preparation
responses['Participant-ID'] = responses['Participant-ID'].astype(str)
net_data['Source'] = net_data['Source'].astype(str)
net_data['Target'] = net_data['Target'].astype(str)

# Convert GrowthMindset to numeric and handle NAs
responses['GrowthMindset'] = pd.to_numeric(responses['GrowthMindset'], errors='coerce')
responses['GrowthMindset'].fillna(0, inplace=True)  # Replace NAs with 0

# Merge data
net_data = pd.merge(net_data, responses[['Participant-ID', 'GrowthMindset']], left_on='Source', right_on='Participant-ID', how='left')
net_data.rename(columns={'GrowthMindset': 'Source_GrowthMindset'}, inplace=True)
net_data = pd.merge(net_data, responses[['Participant-ID', 'GrowthMindset']], left_on='Target', right_on='Participant-ID', how='left')
net_data.rename(columns={'GrowthMindset': 'Target_GrowthMindset'}, inplace=True)

# Activate conversion and load data to R
with localconverter(ro.default_converter + pandas2ri.converter):
    r_net_data = ro.conversion.py2rpy(net_data)

ro.globalenv['df'] = r_net_data

# R code execution to create network and run ERGM
ro.r('''
df$Source <- as.character(df$Source)
df$Target <- as.character(df$Target)
net <- network::network(df, directed = TRUE, loops = FALSE)

# Assign GrowthMindset scores as node attributes
net %v% "Source_GrowthMindset" <- df$Source_GrowthMindset
net %v% "Target_GrowthMindset" <- df$Target_GrowthMindset

# Fit the ERGM
ergm_model <- ergm::ergm(net ~ edges + nodecov("Source_GrowthMindset") + nodecov("Target_GrowthMindset"))
summary_ergm <- summary(ergm_model)
print(summary_ergm)
''')
