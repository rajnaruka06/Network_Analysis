import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

def perform_ergm_analysis(network_df, attribute_df, selected_attribute, edges_only=False):
    output_file_path="ergm_analysis_results.txt"
    status = 'Error Occured'
    if edges_only:

        pandas2ri.activate()
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_net_data = ro.conversion.py2rpy(network_df)

        ro.globalenv['df'] = r_net_data

        try:
            ro.r(f'''
            library(network)
            library(ergm)
            df$Source <- as.character(df$source)
            df$Target <- as.character(df$target)
            net <- network::network(df, directed = TRUE, loops = FALSE)

            # ERGM formula for edges only
            formula <- "net ~ edges"
            ergm_model <- ergm::ergm(as.formula(formula))
            summary_ergm <- summary(ergm_model)
            writeLines(capture.output(summary_ergm), "{output_file_path}")
            ''')
            status = f"ERGM analysis completed. Results saved to {output_file_path}"

        except Exception as e:
            status = f"An error occurred: {e}"
    else:
        attribute_df = attribute_df[['NodeID', selected_attribute]]
        attribute_df.dropna(subset=[selected_attribute], inplace=True)

        net_data = pd.merge(network_df, attribute_df, left_on='source', right_on='NodeID', how='left')
        # net_data = pd.merge(net_data, attribute_df, left_on='target', right_on='NodeID', how='left', suffixes=('', '_target'))
        # net_data.dropna(subset=[selected_attribute, selected_attribute + '_target'], inplace=True)
        # net_data.drop(columns=['NodeID', 'NodeID_target'], inplace=True)
        net_data.drop(columns=['NodeID'], inplace=True)
        
        pandas2ri.activate()
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_net_data = ro.conversion.py2rpy(net_data)

        ro.globalenv['df'] = r_net_data
        ro.globalenv['selected_attribute'] = selected_attribute

        try:
            ro.r(f'''
            library(network)
            library(ergm)
                 
            # net <- network::network(df, directed = TRUE, loops = FALSE)
            # formula <- paste("net ~ edges + nodematch('", selected_attribute, "', diff = FALSE)", sep="")
                 
            net <- network::network(df, vertex.attr = list(Attendance = df$Attendance), directed = TRUE, loops = FALSE)

            formula <- paste("net ~ edges + nodematch('", "Attendance", "', diff = FALSE)", sep="")

            
            ergm_model <- ergm::ergm(as.formula(formula))
            summary_ergm <- summary(ergm_model)
                 
            writeLines(capture.output(summary_ergm), "{output_file_path}")
            ''')
            
            status = f"ERGM analysis completed. Results saved to {output_file_path}"

        except Exception as e:
            status = f"An error occurred: {e}"
    
    return status