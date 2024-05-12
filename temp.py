import pandas as pd

with open('ergm_gof_results.txt', 'r') as f:
    lines = f.readlines()
    # for idx, line in enumerate(lines):
    #     print(idx, line)
    
    # Extract the data
    headers = lines[2].split()
    headers = ['degree'] + headers[:4] + headers[5:]
    out_degree_dof_data = lines[3:27]
    in_degree_dof_data = lines[21:51]
    network_data = lines[-2:]
    
    out_degree_dof_data = [line.split() for line in out_degree_dof_data]
    in_degree_dof_data = [line.split() for line in in_degree_dof_data]
    network_data = [line.split() for line in network_data]
    
    out_degree_df = pd.DataFrame(out_degree_dof_data, columns=headers)
    in_degree_df = pd.DataFrame(in_degree_dof_data, columns=headers)
    network_df = pd.DataFrame(network_data, columns=['model statistic'] + headers[1:])
    
    