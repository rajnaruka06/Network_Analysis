import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import tempfile

def perform_alaam_analysis(network_df, attribute_df, selected_attribute, output_file_path="alaam_analysis_results.txt"):
    with tempfile.TemporaryDirectory() as temp_dir:
        r_lib_path = temp_dir

        # Ensure R can handle Pandas DataFrame
        pandas2ri.activate()
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_net_data = ro.conversion.py2rpy(network_df)
            r_attr_data = ro.conversion.py2rpy(attribute_df)

        ro.globalenv['net_df'] = r_net_data
        ro.globalenv['attr_df'] = r_attr_data
        ro.globalenv['selected_attribute'] = selected_attribute

        try:
            ro.r(f'''
            install.packages("xergm", lib="{r_lib_path}")
            library(network)
            library(xergm)

            net <- network::network(net_df, directed = TRUE, loops = FALSE)

            # Ensure that the attribute is correctly formatted and attached to the network
            net %v% "{selected_attribute}" <- attr_df$"{selected_attribute}"

            # Fit the ALAAM using btergm from the xergm package
            # The formula can be adjusted based on specific hypotheses about attribute interactions
            formula <- paste("net ~ edges + nodecov('", "{selected_attribute}", "')", sep="")
            alaam_model <- xergm::btergm(net, formula = as.formula(formula), R = 1000)  # R is the number of bootstraps

            summary_alaam <- summary(alaam_model)
            writeLines(capture.output(summary_alaam), "{output_file_path}")
            ''')

        except Exception as e:
            with open(output_file_path, 'r') as f:
                summary_text = f.read().strip()
            if summary_text is None: 
                print(f"An error occurred: {e}")
                return False
        
    return True


# Load the CSV file
data = pd.read_csv('sample.csv')

# Assuming 'source' and 'target' are columns for network edges and 'NodeID', 'Attendance' for attributes
network_df = data[['source', 'target']]
attribute_df = data[['NodeID', 'Attendance']].drop_duplicates()

# Call the function
result = perform_alaam_analysis(network_df, attribute_df, 'Attendance')
print("Function executed successfully:", result)

def create_excel_report(data, file_path='report.xlsx'):
    # Create a Pandas Excel writer using openpyxl as the engine
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        # Goodness of fit results
        gfit_data = data['goodness_of_fit']
        gfit_data.to_excel(writer, sheet_name='Goodness of Fit')

        # Averages - assume data contains a DataFrame for averages
        averages_data = data['averages']
        averages_data.to_excel(writer, sheet_name='Averages')

        # Top 10s for each attribute
        # Assuming data is a dictionary that contains multiple DataFrames for top 10s, keyed by attribute name
        for attribute, df in data['top_10s'].items():
            # You can use the first 10 rows or apply any method to select top 10
            top_10_df = df.nlargest(10, 'some_column')  # Replace 'some_column' with the column to rank by
            top_10_df.to_excel(writer, sheet_name=f'Top 10 - {attribute}')

# Example data structure expected by the function
data = {
    'goodness_of_fit': pd.DataFrame({
        'Statistic': ['Chi-squared', 'RMSEA'],
        'Value': [0.05, 0.06]
    }),
    'averages': pd.DataFrame({
        'Metric': ['Mean Age', 'Average Score'],
        'Value': [30, 75]
    }),
    'top_10s': {
        'Attribute1': pd.DataFrame({
            'ID': range(1, 21),
            'Score': [x * 5 for x in range(20)]
        }),
        'Attribute2': pd.DataFrame({
            'ID': range(1, 21),
            'Hours': [x * 3 for x in range(20)]
        })
    }
}

# Creating the report
create_excel_report(data)