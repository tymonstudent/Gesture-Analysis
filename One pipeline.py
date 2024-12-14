import os
import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from sklearn.cluster import KMeans
from scipy import stats
import matplotlib.pyplot as plt
import shutil
from collections import defaultdict, Counter
import datetime
import argparse

RELEVANT_LANDMARKS = [
    'NOSE', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP',
    'RIGHT_HIP', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST'
]
def save_text_output(output_text, filename):
    with open(filename, 'w') as f:
        f.write(output_text)

def get_task_order():
    """Returns the expected order of tasks."""
    return [
        "11A", "11B", "11C", "11D", "21D", "21C", "21B", "21A",
        "31", "41", "12A", "12B", "12C", "12D", "22D", "22C", "22B", "22A",
        "32", "42", "13A", "13B", "13C", "13D", "23D", "23C", "23B", "23A",
        "33", "43", "14A", "14B", "14C", "14D"
    ]

# Part 1: Feature Extraction and Clustering
def extract_features_from_csv(file_path):
    print(f'Reading CSV file: {file_path}')
    data = pd.read_csv(file_path)
    print(f'Initial data shape: {data.shape}')
    data = data[data['landmark'].isin(RELEVANT_LANDMARKS)]
    print(f'Data shape after filtering relevant landmarks: {data.shape}')
    data['id'] = data['landmark']
    data['time'] = data['frame_number']
    data = data[['id', 'time', 'x', 'y', 'z']]
    features = extract_features(data, column_id='id', column_sort='time', default_fc_parameters=MinimalFCParameters())
    print(f'Extracted features shape: {features.shape}')
    return features

def find_optimal_clusters(data, max_clusters=10):
    print('Finding optimal number of clusters using the Elbow Method')
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=0).fit(data)
        wcss.append(kmeans.inertia_)
        print(f'Number of clusters: {i}, WCSS: {kmeans.inertia_}')
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Within-cluster sum of squares (WCSS)')
    plt.grid()
    plt.show()
    return wcss

def cluster_data(all_features, num_clusters):
    print(f'Clustering data into {num_clusters} clusters')
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(all_features)
    print(f'Cluster labels assigned: {kmeans.labels_}')
    all_features['cluster'] = kmeans.labels_
    return kmeans.labels_

def save_cluster_results(cluster_labels, file_names, output_folder):
    """Save the clustered file names to text files and copy files to cluster-specific folders."""
    os.makedirs(output_folder, exist_ok=True)
    parser = argparse.ArgumentParser(description="Process a specific folder for the pipeline.")
    parser.add_argument("folder_path", type=str, help="The path of the folder to process.")
    args = parser.parse_args()

    # Use the passed folder_path
    folder_path = args.folder_path
    

    # Initialize a dictionary to count files per cluster in memory
    cluster_groups = defaultdict(list)  # Use defaultdict to store lists of files by cluster

    # Calculate number of rows per file to assign cluster labels accurately
    num_rows_per_file = len(cluster_labels) // len(file_names)

    # Map each file to its majority cluster label
    for i, file_name in enumerate(file_names):
        file_labels = cluster_labels[i * num_rows_per_file: (i + 1) * num_rows_per_file]
        most_common_cluster = Counter(file_labels).most_common(1)[0][0]
        cluster_groups[most_common_cluster].append(file_name)  # Add file to the respective cluster

    # Debug: Print file-to-cluster mapping
    print("\nFile to cluster mapping:")
    for cluster, files in cluster_groups.items():
        for file_name in files:
            print(f"{file_name}: Cluster {cluster}")

    # Save each cluster's file names to a separate text file and create folders
    for cluster, files in cluster_groups.items():
        cluster_folder = os.path.join(output_folder, f'cluster_{cluster}')
        os.makedirs(cluster_folder, exist_ok=True)
        
        # Save cluster file list to a text file
        with open(os.path.join(output_folder, f'cluster_{cluster}.txt'), 'w') as f:
            f.write('\n'.join(files))

        # Copy files into respective cluster folder
        for file_name in files:
            source_file_path = os.path.join(folder_path, file_name)
            destination_file_path = os.path.join(cluster_folder, file_name)
            shutil.copy(source_file_path, destination_file_path)

    # Count the number of files in each cluster and return counts as a list
    cluster_counts = [len(files) for files in cluster_groups.values()]

    # Warning for clusters that have no files assigned
    max_cluster_label = max(cluster_labels)
    for cluster in range(max_cluster_label + 1):
        if cluster not in cluster_groups:
            print(f"Warning: Cluster {cluster} has no files assigned!")

    return cluster_groups, cluster_counts  # Return both cluster groups and counts


def compute_information(total_count, cluster_counts):
    print(f'Computing information values with total count {total_count} and cluster counts {cluster_counts}')
    information_values = {}
    for i, count in enumerate(cluster_counts):
        p_x = count / total_count if total_count > 0 else 0
        I_x = -np.log2(p_x) if p_x > 0 else 0
        information_values[f'Cluster {i}'] = I_x
        print(f'Cluster {i}: Count = {count}, P(X) = {p_x}, I(X) = {I_x}')
    return information_values

    return None  # Return None if task not in the expected order

def analyze_original_order(df, output_folder):
    """
    Analyze how information values behave in the original order of the DataFrame.
    
    Parameters:
    - df: DataFrame containing 'info_value' as a column
    - output_folder: Directory to save the output plot
    
    Returns:
    - Slope, intercept, R-squared, and p-value of the linear regression
    """
    print("Analyzing correlation across all tasks based on original order.")

# Ensure the DataFrame has the required column
    if 'info_value' not in df.columns:
        raise ValueError("The DataFrame must contain an 'info_value' column.")

    # Initialize a list to store the results
    correlation_results = []
    regression_results = []

    # Pearson Correlation
    pearson_corr, pearson_p = stats.pearsonr(df.index, df['info_value'])
    print(f"Pearson Correlation: {pearson_corr:.4f} | P-value: {pearson_p:.4e}")
    correlation_results.append({
        'Correlation Type': 'Pearson',
        'Correlation': pearson_corr,
        'P-value': pearson_p
    })

    # Spearman Correlation
    spearman_corr, spearman_p = stats.spearmanr(df.index, df['info_value'])
    print(f"Spearman Correlation: {spearman_corr:.4f} | P-value: {spearman_p:.4e}")
    correlation_results.append({
        'Correlation Type': 'Spearman',
        'Correlation': spearman_corr,
        'P-value': spearman_p
    })

    # Linear Regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(df.index, df['info_value'])
    print(f"Linear Regression: Slope = {slope:.4f}, Intercept = {intercept:.4f}, R-squared = {r_value**2:.4f}, P-value = {p_value:.4e}")
    regression_results.append({
        'Slope': slope,
        'Intercept': intercept,
        'R-squared': r_value**2,
        'P-value': p_value,
        'Standard Error': std_err
    })

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(df.index, df['info_value'], color='blue', label='Data Points')
    plt.plot(df.index, intercept + slope * df.index, color='red', label='Regression Line')
    plt.title('Information Value vs Original Row Order')
    plt.xlabel('Row Index')
    plt.ylabel('Information Value')
    plt.grid(True)
    plt.legend()

    # Save the plot
    correlation_plot_path = os.path.join(output_folder, "correlation_plot.png")
    plt.savefig(correlation_plot_path)
    plt.close()

    print(f"Plot saved to {correlation_plot_path}")

    # Save the correlation results to a CSV file
    correlation_df = pd.DataFrame(correlation_results)
    correlation_results_path = os.path.join(output_folder, "correlation_results.csv")
    correlation_df.to_csv(correlation_results_path, index=False)
    print(f"Correlation results saved to {correlation_results_path}")

    # Save the regression results to a CSV file
    regression_df = pd.DataFrame(regression_results)
    regression_results_path = os.path.join(output_folder, "regression_results.csv")
    regression_df.to_csv(regression_results_path, index=False)
    print(f"Regression results saved to {regression_results_path}")

    print("Analysis complete. All results have been saved.")

    # Return results for further analysis or logging
    return slope, intercept, r_value**2, p_value

def analyze_grouped_gestures(df, output_folder):
    print("Starting analysis of grouped gestures by trial and gesture position...")

    grouped = df.groupby(['subject', 'task'])
    print(f"Data grouped by 'subject' and 'task'. Found {len(grouped)} groups.")

    gesture_positions = []
    max_gestures = grouped.size().max()
    print(f"Maximum number of gestures in any group: {max_gestures}")

    for _ in range(max_gestures):
        gesture_positions.append([])

    for (subject, task), group in grouped:
        print(f"Processing group: Subject = {subject}, Task = {task}, Group Size = {len(group)}")
        sorted_group = group.sort_values(by='frame_number')
        for i in range(len(sorted_group)):
            gesture_positions[i].append(sorted_group.iloc[i]['info_value'])

    print("Converting gesture positions to DataFrame...")
    gesture_df = pd.DataFrame(gesture_positions).transpose()
    print(f"Initial gesture DataFrame shape: {gesture_df.shape}")

    gesture_df = gesture_df.dropna(axis=1, how='all')
    print(f"After dropping all-NaN columns, shape: {gesture_df.shape}")

    gesture_df = gesture_df.dropna(axis=0, how='all')
    print(f"After dropping all-NaN rows, shape: {gesture_df.shape}")

    # Prepare a list to store the correlation results
    trend_corrs = []
    regression_results = []

    for col in gesture_df.columns:
        print(f"Analyzing Gesture Position {col}...")
        position = [col] * len(gesture_df)
        info_values = gesture_df[col]
        if len(info_values.dropna()) < 2:
            print(f"Skipping Gesture Position {col} due to insufficient data (less than 2 valid points).")
            continue
        if np.all(info_values.dropna() == info_values.dropna().iloc[0]):
            print(f"Skipping Gesture Position {col} due to constant data.")
            continue
        
        # Compute Pearson's correlation
        print(f"Computing Pearson's correlation for Gesture Position {col}...")
        pearson_corr, pearson_p_value = stats.pearsonr(position, info_values)
        print(f"Gesture Position {col} (Pearson): Correlation = {pearson_corr} | P-value = {pearson_p_value}")
        
        # Compute Spearman's correlation
        print(f"Computing Spearman's correlation for Gesture Position {col}...")
        spearman_corr, spearman_p_value = stats.spearmanr(position, info_values)
        print(f"Gesture Position {col} (Spearman): Correlation = {spearman_corr} | P-value = {spearman_p_value}")
        
        trend_corrs.append({
            'Gesture Position': col,
            'Pearson Correlation': pearson_corr,
            'Pearson P-value': pearson_p_value,
            'Spearman Correlation': spearman_corr,
            'Spearman P-value': spearman_p_value
        })

    print("Correlation analysis complete. Results:")
    for result in trend_corrs:
        print(result)

    # Save the correlation results to a CSV file
    correlation_df = pd.DataFrame(trend_corrs)
    correlation_path = os.path.join(output_folder, "gesture_correlation_results.csv")
    correlation_df.to_csv(correlation_path, index=False)
    print(f"Correlation results saved to {correlation_path}")

    print("Starting linear regression analysis...")
    plt.figure(figsize=(10, 6))
    for col in gesture_df.columns:
        print(f"Plotting Gesture Position {col} data...")
        plt.plot([col] * len(gesture_df), gesture_df[col], 'o', label=f'Position {col}')

    positions = np.concatenate([[col] * len(gesture_df[col]) for col in gesture_df.columns])
    info_values = np.concatenate([gesture_df[col].values for col in gesture_df.columns])

    valid_positions = positions[~np.isnan(info_values)]
    valid_info_values = info_values[~np.isnan(info_values)]

    print("Performing linear regression...")
    slope, intercept, r_value, p_value, std_err = stats.linregress(valid_positions, valid_info_values)
    print(f"Linear Regression Results: Slope = {slope}, Intercept = {intercept}, "
        f"R-squared = {r_value**2}, P-value = {p_value}")

    # Save the regression results
    regression_results.append({
        'Slope': slope,
        'Intercept': intercept,
        'R-squared': r_value**2,
        'P-value': p_value,
        'Standard Error': std_err
    })

    # Save the regression results to a CSV file
    regression_df = pd.DataFrame(regression_results)
    regression_path = os.path.join(output_folder, "linear_regression_results.csv")
    regression_df.to_csv(regression_path, index=False)
    print(f"Regression results saved to {regression_path}")

    plt.plot(valid_positions, intercept + slope * valid_positions, 'r-', label='Linear Regression Line')
    plt.title('Information Value vs Gesture Position (with Regression)')
    plt.xlabel('Gesture Position')
    plt.ylabel('Information Value')
    plt.grid(True)
    plt.xticks(range(1, len(gesture_df.columns) + 1))
    plt.legend(title="Gesture Position")

    plot_path = os.path.join(output_folder, "info_value_vs_gesture_position_with_regression.png")
    print(f"Saving plot to {plot_path}...")
    plt.savefig(plot_path)
    plt.close()

    print("Analysis complete. Returning regression results.")
    return slope, intercept, r_value, p_value, std_err
def analyze_correlation(df, info_values, output_folder):
    # Save the information values to a text file
    info_values_text = "\n".join([f"{key}: {value}" for key, value in info_values.items()])
    save_text_output(info_values_text, os.path.join(output_folder, "info_values.txt"))

    # Ensure the DataFrame contains necessary columns
    required_columns = {'frame_number', 'info_value', 'subject', 'task'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"The DataFrame must contain the columns: {required_columns}")

    # Save DataFrame to a CSV file for reference
    df.to_csv(os.path.join(output_folder, "gesture_info.csv"), index=False)

    # Perform analyses
    analyze_original_order(df, output_folder)
    analyze_grouped_gestures(df, output_folder)

def process_file(filename, folder_path):
    """Processes an individual CSV file and returns its data if valid."""
    # Create the full path by joining folder_path with filename
    full_path = os.path.join(folder_path, filename)

    parts = filename.split('.')
    if len(parts) < 3 or 'bag_chest_pose_data_segment_' not in filename:
        print(f"  Skipping file '{filename}' - does not match the expected pattern.")
        return None
    
    # Extract subject and task from the filename
    task_info = parts[0].split('_')
    
    # Check if the filename has the required parts
    if len(task_info) < 2:  # At least S1 and task
        print(f"  Skipping file '{filename}' - invalid naming format.")
        return None
    
    subject = task_info[0]  # Assuming the first part is always 'S1'
    task = task_info[1]  # Get the task number

    # Check if there's a task letter
    if len(task_info) > 2:  # If there's a third part, it's a letter
        task += task_info[2]  # Append the letter (e.g., '11A')
    
    print(f"Processing file: {filename}")
    print(f"  Subject: {subject}, Task: {task}")

    # Read the CSV file using the full path
    try:
        # Load only the frame_number column
        df = pd.read_csv(full_path, usecols=['frame_number'])
        
        # Check for expected columns
        if 'frame_number' not in df.columns:
            print(f"    Error: Could not read '{filename}' due to missing 'frame_number' column.")
            return None

        # Get the first and last frame numbers
        first_frame_number = df['frame_number'].iloc[0]
        last_frame_number = df['frame_number'].iloc[-1]

        # Return the necessary information including the filename
        return (subject, task, first_frame_number, last_frame_number, filename)

    except Exception as e:
        print(f"    Error: Could not read '{filename}' due to {str(e)}.")
        return None

def check_file_exists(file_path):
    """Helper function to check if a file already exists."""
    return os.path.exists(file_path)

def main():
    parser = argparse.ArgumentParser(description="Process a specific folder for the pipeline.")
    parser.add_argument("folder_path", type=str, help="The path of the folder to process.")
    args = parser.parse_args()

    # Use the passed folder_path
    folder_path = args.folder_path
    print(f"Processing folder: {folder_path}")

    feature_list = []
    file_names = []

    # Get task order
    task_order = get_task_order()
    print(f"Task order retrieved: {task_order}")

    # Prepare a variable to store output logs
    output_log = []

    # List all CSV files in the folder
    all_files = [file_name for file_name in os.listdir(folder_path) if file_name.endswith('.csv')]
    print(f"Total CSV files found: {len(all_files)}")
    output_log.append(f"Total CSV files found: {len(all_files)}\n")

    file_data = []

    # Process each file and associate it with task and frame number
    for file_name in all_files:
        df = process_file(file_name, folder_path)
        
        if df is not None:
            file_data.append(df)
            file_names.append(file_name)
        else:
            output_log.append(f"File {file_name} was skipped (invalid format or processing error).\n")

    output_log.append(f"Total files successfully processed: {len(file_data)}\n")
    print(f"Total files successfully processed: {len(file_data)}")

    # Sorting files based on task order and frame numbers
    sorted_data = sorted(file_data, key=lambda x: (
        task_order.index(x[1].upper()) if x[1].upper() in [task.upper() for task in task_order] else len(task_order),
        x[2]  # Sort by frame number
    ))


    output_log.append(f"Sorted data:\n{sorted_data}\n")
    print(sorted_data)

    # Define the output folder for storing results
    
    output_folder = folder_path
    
    # Check if the results already exist (i.e., check for previously saved clustering results)
    if check_file_exists(os.path.join(output_folder, "cluster_results.txt")):
        print("Cluster results already exist, skipping clustering.")
    else:
        # Extract features from sorted files (only if not done already)
        all_features = pd.concat([extract_features_from_csv(os.path.join(folder_path, f[4])) for f in sorted_data], axis=0).fillna(0)
        
        # Finding optimal clusters and clustering the data
        wcss = find_optimal_clusters(all_features)
        num_clusters = int(input("Check the elbow here: ")) # This could be chosen based on the elbow method or predetermined
        cluster_labels = cluster_data(all_features, num_clusters)

        # Save the clustered results
        os.makedirs(output_folder, exist_ok=True)

        cluster_groups, cluster_counts = save_cluster_results(cluster_labels, file_names, output_folder)

        # Save cluster information and results to a text file
        cluster_info_text = f"Cluster Groups: {cluster_groups}\nCluster Counts: {cluster_counts}\n"
        save_text_output(cluster_info_text, os.path.join(output_folder, "cluster_results.txt"))

        # Compute information values for each cluster
        total_files = sum(cluster_counts)
        info_values = compute_information(total_files, cluster_counts)

        # Save the information values to a text file
        info_values_text = "\n".join([f"{key}: {value}" for key, value in info_values.items()])
        save_text_output(info_values_text, os.path.join(output_folder, "info_values.txt"))
    
    # Construct the gesture DataFrame with correct cluster and information values
    file_info_mapping = {}
    for cluster, files in cluster_groups.items():
        info_value = info_values.get(f'Cluster {cluster}', None)  # Retrieve information value for the cluster
        for file_name in files:
            file_info_mapping[file_name] = {"cluster": cluster, "info_value": info_value}

    gesture_info = []
    for subject, task, frame_number, _, filename in sorted_data:
        file_info = file_info_mapping.get(filename, {"cluster": None, "info_value": None})
        gesture_info.append({
            "subject": subject,
            "task": task,
            "frame_number": frame_number,
            "filename": filename,
            "cluster": file_info["cluster"],
            "info_value": file_info["info_value"]
        })
    
    # Convert gesture_info to a DataFrame for further analysis or correlation
    df = pd.DataFrame(gesture_info)
    print(df)
    output_log.append(f"DataFrame for Gesture Info:\n{df.head()}\n")

    # Save DataFrame to a text file for further analysis
    df_to_save = os.path.join(output_folder, "gesture_info.csv")
    if check_file_exists(df_to_save):
        print("Gesture information CSV already exists, skipping saving.")
    else:
        df.to_csv(df_to_save, index=False)

    # Analyze correlation and generate plot only if not already done
    if check_file_exists(os.path.join(output_folder, "correlation_plot.png")):
        print("Correlation analysis already performed, skipping analysis.")
    else:
        analyze_correlation(df, info_values, output_folder)

    # Save the output log to a file
    save_text_output("\n".join(output_log), os.path.join(output_folder, "process_log.txt"))


if __name__ == '__main__':
    main()