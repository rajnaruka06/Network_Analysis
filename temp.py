# import rpy2.robjects as ro
# from rpy2.robjects import pandas2ri
# from rpy2.robjects.conversion import localconverter

# import pandas as pd

# def perform_ergm_analysis(network_df, attribute_df, selected_attribute, edges_only=False):
#     output_file_path="ergm_analysis_results.txt"

#     if edges_only:

#         pandas2ri.activate()
#         with localconverter(ro.default_converter + pandas2ri.converter):
#             r_net_data = ro.conversion.py2rpy(network_df)

#         ro.globalenv['df'] = r_net_data

#         try:
#             ro.r(f'''
#             library(network)
#             library(ergm)
#             df$Source <- as.character(df$source)
#             df$Target <- as.character(df$target)
#             net <- network::network(df, directed = TRUE, loops = FALSE)

#             # ERGM formula for edges only
#             formula <- "net ~ edges"
#             ergm_model <- ergm::ergm(as.formula(formula))
#             summary_ergm <- summary(ergm_model)
#             writeLines(capture.output(summary_ergm), "{output_file_path}")
#             ''')
#             print(f"ERGM analysis completed. Results saved to {output_file_path}")

#         except Exception as e:
#             print(f"An error occurred: {e}")
#     else:
#         attribute_df = attribute_df[['NodeID', selected_attribute]]
#         attribute_df.dropna(subset=[selected_attribute], inplace=True)

#         net_data = pd.merge(network_df, attribute_df, left_on='source', right_on='NodeID', how='left')
#         # net_data = pd.merge(net_data, attribute_df, left_on='target', right_on='NodeID', how='left', suffixes=('', '_target'))
#         # net_data.dropna(subset=[selected_attribute, selected_attribute + '_target'], inplace=True)
#         # net_data.drop(columns=['NodeID', 'NodeID_target'], inplace=True)
#         net_data.drop(columns=['NodeID'], inplace=True)
        
#         pandas2ri.activate()
#         with localconverter(ro.default_converter + pandas2ri.converter):
#             r_net_data = ro.conversion.py2rpy(net_data)

#         ro.globalenv['df'] = r_net_data
#         ro.globalenv['selected_attribute'] = selected_attribute

#         try:
#             ro.r(f'''
#             library(network)
#             library(ergm)
                 
#             # net <- network::network(df, directed = TRUE, loops = FALSE)
#             # formula <- paste("net ~ edges + nodematch('", selected_attribute, "', diff = FALSE)", sep="")
                 
#             net <- network::network(df, vertex.attr = list(Attendance = df$Attendance), directed = TRUE, loops = FALSE)

#             formula <- paste("net ~ edges + nodematch('", "Attendance", "', diff = FALSE)", sep="")

            
#             ergm_model <- ergm::ergm(as.formula(formula))
#             summary_ergm <- summary(ergm_model)
                 
#             writeLines(capture.output(summary_ergm), "{output_file_path}")
#             ''')
#             print(f"ERGM analysis completed. Results saved to {output_file_path}")

#         except Exception as e:
#             print(f"An error occurred: {e}")



# # df = pd.read_csv("Freidnships_Data.csv")
# # df.columns = ['source', 'target']
# # df = df.astype(str)
# # for col in df.columns:
# #     df[col] = df[col].str.strip()
# # df.dropna(inplace=True)
# # df.drop_duplicates(inplace=True)
# # df = df[~(df['source'] == df['target'])]
# # df.reset_index(drop=True, inplace=True)

# # perform_ergm_analysis(df, None, None, edges_only=True)


# attributes = pd.read_excel('df_excel.xlsx', sheet_name='participants')
# attributes = attributes.astype(str)
# attributes.columns = ['NodeID'] +  list(attributes.columns[1:])


# edges = pd.read_excel('df_excel.xlsx', sheet_name='net_0_Friends')
# edges = edges.astype(str)
# edges.columns = ['source', 'target']
# for col in edges.columns:
#     edges[col] = edges[col].str.strip()
# edges.dropna(inplace=True)
# edges.drop_duplicates(inplace=True)
# edges  = edges[~(edges['source'] == edges['target'])]
# edges.reset_index(drop=True, inplace=True)

# perform_ergm_analysis(edges, attributes, 'Attendance', edges_only=False)


import pandas as pd

ergm_file_path = 'ergm_analysis_results.txt'
with open(ergm_file_path, 'r') as f:
    summary_text = f.read().strip()

call_section, results_section = summary_text.split("\n\nMaximum Likelihood Results:")
results_lines = results_section.splitlines()
headers = ['Estimate', 'Std. Error', 'MCMC %', 'z value', 'Pr(>|z|)']
data = []
for line in results_lines[3:-7]:
    row = []
    for val in line.split()[1:-1]:
        try:
            row.append(float(val))
        except ValueError:
            row.append(float(val[1:]))
        else:
            continue
    data.append(row)
            
summary_df = pd.DataFrame(data, columns=headers, index=['edges', 'nodematch'])

null_deviance_line = results_lines[-4].split(": ")[1].split()
null_deviance_value, null_deviance_df  = null_deviance_line[0], null_deviance_line[2]

residual_deviance_line = results_lines[-3].split(": ")[1].split()
residual_deviance_value, residual_deviance_df = residual_deviance_line[0], residual_deviance_line[2]

aic_bic_line = results_lines[-1].split(": ")
aic_value = aic_bic_line[1].split()[0]
bic_value = aic_bic_line[2].split()[0]


for idx, line in enumerate(results_lines):
    print('-'*50, str(idx))
    print(line)

print('#'*50)
print(summary_df)
