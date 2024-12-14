import os
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import Counter
import shutil

# Define the relevant landmarks
RELEVANT_LANDMARKS = [
    'NOSE',
    'LEFT_SHOULDER',
    'RIGHT_SHOULDER',
    'LEFT_HIP',
    'RIGHT_HIP',
    'LEFT_ELBOW',
    'RIGHT_ELBOW',
    'LEFT_WRIST',
    'RIGHT_WRIST'
]

# Function to extract features from a single CSV file
def extract_features_from_csv(file_path):
    # Load the CSV data
    data = pd.read_csv(file_path)

    # Filter data for only relevant landmarks
    data = data[data['landmark'].isin(RELEVANT_LANDMARKS)]

    # Prepare the DataFrame for tsfresh
    data['id'] = data['landmark']  # Using landmark as an identifier
    data['time'] = data['frame_number']  # Using frame_number as time
    data = data[['id', 'time', 'x', 'y', 'z']]  # Keep only relevant columns

    # Extract features using MinimalFCParameters
    extracted_features = extract_features(
        data,
        column_id='id',
        column_sort='time',
        default_fc_parameters=MinimalFCParameters()
    )
    return extracted_features

def find_optimal_clusters(data, max_clusters):
    """Determine the optimal number of clusters using the Elbow Method."""
    wcss = []  # Within-cluster sum of squares
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)  # Sum of squared distances to closest centroid
    return wcss

def plot_elbow_method(wcss):
    """Plot the Elbow Method results."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(wcss) + 1), wcss, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Within-cluster sum of squares (WCSS)')
    plt.grid()
    plt.show()

def save_cluster_results(cluster_labels, file_names, output_folder):
    """Save the clustered file names to text files and copy files to cluster-specific folders."""
    os.makedirs(output_folder, exist_ok=True)  # Create directory if it doesn't exist
    folder_path = r'E:\Experiment videos\Test folder for hte program\S1'  # Define here
    # Map cluster labels to files
    file_to_cluster_map = {}
    num_rows_per_file = len(cluster_labels) // len(file_names)  # Assume equal rows per file
    
    # Assign cluster labels based on majority vote
    for i, file_name in enumerate(file_names):
        file_labels = cluster_labels[i * num_rows_per_file: (i + 1) * num_rows_per_file]
        most_common_cluster = Counter(file_labels).most_common(1)[0][0]  # Majority vote
        file_to_cluster_map[file_name] = most_common_cluster

    # Debugging: Print the mapping of files to clusters
    print("\nFile to cluster mapping:")
    for file_name, cluster in file_to_cluster_map.items():
        print(f"{file_name}: Cluster {cluster}")

    # Group files by their assigned cluster
    cluster_groups = {}
    for file_name, cluster in file_to_cluster_map.items():
        if cluster not in cluster_groups:
            cluster_groups[cluster] = []
        cluster_groups[cluster].append(file_name)

    # Save each cluster's file names to a separate text file
    for cluster, files in cluster_groups.items():
        print(f"Saving files for cluster {cluster} to {output_folder}/cluster_{cluster}.txt")
        with open(os.path.join(output_folder, f'cluster_{cluster}.txt'), 'w') as f:
            f.write('\n'.join(files))

        # Create a folder for each cluster
        cluster_folder = os.path.join(output_folder, f'cluster_{cluster}')
        os.makedirs(cluster_folder, exist_ok=True)

        # Copy files into the respective cluster folder
        for file_name in files:
            source_file_path = os.path.join(folder_path, file_name)  # Original file location
            destination_file_path = os.path.join(cluster_folder, file_name)
            shutil.copy(source_file_path, destination_file_path)  # Copy file to cluster folder

    # Warning for clusters that have no files assigned
    for cluster in range(max(cluster_labels) + 1):
        if cluster not in cluster_groups:
            print(f"Warning: Cluster {cluster} has no files assigned!")
#
def main():
    # Step 1: Process all CSV files in the specified folder
    folder_path = r'E:\Experiment videos\Test folder for hte program\S1'
    feature_list = []
    file_names = []  # To keep track of file names for clustering

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            print(f'Processing file: {file_path}')
            
            extracted_features = extract_features_from_csv(file_path)
            feature_list.append(extracted_features)
            file_names.append(file_name)  # Append the file name

    # Step 2: Concatenate all extracted features into a single DataFrame
    all_features = pd.concat(feature_list, axis=0)

    # Step 3: Handle missing values if any
    all_features.fillna(0, inplace=True)  # Replace NaN with 0

    # Debugging: Print shape of feature matrix
    print(f"All features shape: {all_features.shape}")

    # Step 4: Elbow Method to determine optimal number of clusters
    max_clusters = 10  # Set a maximum number of clusters to check
    wcss = find_optimal_clusters(all_features, max_clusters)
    
    # Plot the elbow method
    plot_elbow_method(wcss)  # Add this line to call the plotting function

    # Step 5: Choose a number of clusters based on the elbow point
    num_clusters = int(input("Enter the number of clusters based on the elbow point: "))
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(all_features)

    # Step 6: Add cluster labels to the DataFrame
    all_features['cluster'] = kmeans.labels_

    # Debugging: Print assigned labels
    print(f"Assigned cluster labels: {kmeans.labels_}")

    # Step 7: Display the clustering results
    print("Clustering results:")
    print(all_features.groupby('cluster').size())

    # Optional: Visualize the clusters (2D projection for demonstration)
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(all_features.drop(columns=['cluster']))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=all_features['cluster'], cmap='viridis')
    plt.title('K-means Clustering Results')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster')
    plt.show()

    # Step 8: Save clustering results
    output_folder = r'C:\Users\kakaz\Documents\Pose estimation\Clustered Results'
    print(f"Total files processed: {len(file_names)}")
    print(f"Total cluster labels assigned: {len(kmeans.labels_)}")

    # Debugging: Ensure all clusters are saved
    print(f"Saving {num_clusters} clusters to folder: {output_folder}")
    
    save_cluster_results(kmeans.labels_, file_names, output_folder)

if __name__ == '__main__':
    main()
