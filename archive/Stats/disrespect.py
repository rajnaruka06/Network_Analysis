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
net_data = pd.read_excel("Student Survey - Jan.xlsx", sheet_name="net_5_Disrespect")
participants_data = pd.read_excel("Student Survey - Jan.xlsx", sheet_name="participants")

# Convert identifiers to string for accurate ID matching
net_data['Source'] = net_data['Source'].astype(str)
net_data['Target'] = net_data['Target'].astype(str)
participants_data['Participant-ID'] = participants_data['Participant-ID'].astype(str)

# Convert Attendance to numeric and handle NAs
participants_data['Attendance'] = pd.to_numeric(participants_data['Attendance'], errors='coerce')
participants_data['Attendance'].fillna(participants_data['Attendance'].mean(), inplace=True)

# Merge Attendance scores into net_data using pandas
net_data = pd.merge(net_data, participants_data[['Participant-ID', 'Attendance']], left_on='Source', right_on='Participant-ID', how='left')
net_data.rename(columns={'Attendance': 'Source_Attendance'}, inplace=True)
net_data = pd.merge(net_data, participants_data[['Participant-ID', 'Attendance']], left_on='Target', right_on='Participant-ID', how='left')
net_data.rename(columns={'Attendance': 'Target_Attendance'}, inplace=True)

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

# Assign attendance scores as node attributes
net %v% "Source_Attendance" <- df$Source_Attendance
net %v% "Target_Attendance" <- df$Target_Attendance

# Fit the ERGM
ergm_model <- ergm::ergm(net ~ edges + nodecov("Source_Attendance") + nodecov("Target_Attendance"))
summary_ergm <- summary(ergm_model)
print(summary_ergm)
''')