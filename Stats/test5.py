import tkinter as tk
from tkinter import filedialog, simpledialog
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

def perform_ergm_analysis():
    root = tk.Tk()
    root.withdraw()

    net_file_path = filedialog.askopenfilename(title="Select the Network Data CSV file", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    attr_file_path = filedialog.askopenfilename(title="Select the Attribute Data Excel file", filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")])

    if not net_file_path or not attr_file_path:
        print("File selection cancelled, exiting...")
        return

    net_data = pd.read_csv(net_file_path)
    sheet_name = simpledialog.askstring("Input", "Enter the sheet name for participant data:", parent=root)
    attr_data = pd.read_excel(attr_file_path, sheet_name=sheet_name)

    attribute_name = simpledialog.askstring("Input", "Enter the attribute name to join on (e.g., House):", parent=root)
    participant_id_col = simpledialog.askstring("Input", "Enter the participant identifier column name (e.g., Participant-ID):", parent=root)

    attr_data[participant_id_col] = attr_data[participant_id_col].astype(str)
    net_data['Source'] = net_data['Source'].astype(str)
    net_data['Target'] = net_data['Target'].astype(str)

    net_data = net_data[net_data['Source'] != net_data['Target']].drop_duplicates()

    net_data = pd.merge(net_data, attr_data[[participant_id_col, attribute_name]], left_on='Source', right_on=participant_id_col, how='left')
    net_data.rename(columns={attribute_name: 'Source_' + attribute_name}, inplace=True)
    net_data = pd.merge(net_data, attr_data[[participant_id_col, attribute_name]], left_on='Target', right_on=participant_id_col, how='left')
    net_data.rename(columns={attribute_name: 'Target_' + attribute_name}, inplace=True)

    pandas2ri.activate()
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_net_data = ro.conversion.py2rpy(net_data)

    ro.globalenv['df'] = r_net_data
    ro.globalenv['attribute_name'] = attribute_name

    ro.r('''
    library(network)
    library(ergm)
    df$Source <- as.character(df$Source)
    df$Target <- as.character(df$Target)
    net <- network::network(df, directed = TRUE, loops = FALSE)

    # Assign attributes correctly and check them
    attr_name <- paste("Source_", attribute_name, sep="")
    net %v% attr_name <- as.character(df[,attr_name])
    if (!all(attr_name %in% network::list.vertex.attributes(net))) {
        stop("Attribute assignment failed or attribute does not exist in the network object")
    }

    formula <- paste("net ~ edges + nodematch('", attr_name, "', diff = FALSE)", sep="")
    ergm_model <- ergm::ergm(as.formula(formula))
    summary_ergm <- summary(ergm_model)
    print(summary_ergm)
    ''')

    print("ERGM analysis completed. Check R output for results.")

perform_ergm_analysis()
