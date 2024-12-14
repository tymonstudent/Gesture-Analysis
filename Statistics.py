import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import re

# Function to read ordered results from a structured text file
def read_ordered_results(file_path):
    data_list = []
    occurrence_pattern = re.compile(r"Occurrence (\d+):")
    entry_pattern = re.compile(r"\{'Cluster': '([^']+)', 'Subject': '([^']+)', 'Task-Segment': '([^']+)', 'First Frame Number': '(\d+)', 'Filename': '([^']+)'\}")
    
    with open(file_path, 'r') as file:
        current_occurrence = None
        for line in file:
            line = line.strip()
            # Check if the line indicates an occurrence group
            occurrence_match = occurrence_pattern.match(line)
            if occurrence_match:
                current_occurrence = int(occurrence_match.group(1))
                continue
            # Parse the actual data entries
            entry_match = entry_pattern.match(line)
            if entry_match:
                cluster, subject, task_segment, first_frame_number, filename = entry_match.groups()
                data_list.append({
                    "Cluster": cluster,
                    "Subject": subject,
                    "Task-Segment": task_segment,
                    "First Frame Number": int(first_frame_number),
                    "Filename": filename,
                    "Occurrence": current_occurrence  # Keep track of the occurrence
                })
    return data_list

# Information values of clusters
info_values = {
    'cluster_0': 2.268488835925902,
    'cluster_1': 1.5579954531208868,
    'cluster_2': 5.7279204545632,
    'cluster_3': 1.8210298589546807,
    'cluster_4': 2.920565532505595
}

# File path for the ordered results
file_path = r"C:\Users\kakaz\Documents\Pose estimation\Clustered Results\S1 Clustering 5\ordered_segments_results.txt"

# Read ordered results
data_list = read_ordered_results(file_path)

# Create a DataFrame from the ordered list
df = pd.DataFrame(data_list)

# Map cluster names to information values
df['Info Value'] = df['Cluster'].map(lambda x: info_values.get(x, None))

# Assign an order based on their appearance in the 'Occurrence' field
df['Order'] = df['Occurrence']

# Perform statistical analysis
# Pearson correlation
pearson_corr, pearson_p = stats.pearsonr(df['Order'], df['Info Value'])
# Spearman correlation
spearman_corr, spearman_p = stats.spearmanr(df['Order'], df['Info Value'])

# Linear regression model
slope, intercept, r_value, p_value, std_err = stats.linregress(df['Order'], df['Info Value'])

# Print results
print("Pearson Correlation: ", pearson_corr, "P-value: ", pearson_p)
print("Spearman Correlation: ", spearman_corr, "P-value: ", spearman_p)
print("Linear Regression: Slope: ", slope, "Intercept: ", intercept, "R-squared: ", r_value**2)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(df['Order'], df['Info Value'], color='blue', label='Data points')
plt.plot(df['Order'], intercept + slope * df['Order'], color='red', label='Regression line')
plt.title('Correlation between Information Value and Order')
plt.xlabel('Order of Instances')
plt.ylabel('Information Value')
plt.legend()
plt.grid()
plt.show()
