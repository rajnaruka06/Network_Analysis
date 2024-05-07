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
