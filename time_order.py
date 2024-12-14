import os
import pandas as pd

def get_task_order():
    """Returns the expected order of tasks."""
    return [
        "11A", "11B", "11C", "11D", "21D", "21C", "21B", "21A",
        "31", "41", "12A", "12B", "12C", "12D", "22D", "22C", "22B", "22A",
        "32", "42", "13A", "13B", "13C", "13D", "23D", "23C", "23B", "23A",
        "33", "43", "14A", "14B", "14C", "14D"
    ]

def process_file(filename, cluster_path):
    """Processes an individual CSV file and returns its data if valid."""
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

    # Read the CSV file
    try:
        df = pd.read_csv(os.path.join(cluster_path, filename), nrows=1)
        
        # Check for expected columns
        if 'frame_number' not in df.columns:
            print(f"    Error: Could not read '{filename}' due to missing 'frame_number' column.")
            return None

        # Get the first frame number
        first_frame_number = df['frame_number'].iloc[0]  # Access the first row's frame number

        # Return the necessary information including the filename
        return (subject, task, first_frame_number, filename)  # Include the filename

    except Exception as e:
        print(f"    Error: Could not read '{filename}' due to {str(e)}.")
        return None

def process_cluster(cluster_path, task_order, cluster_name):
    """Processes all files in a cluster directory."""
    valid_files = []
    
    for filename in os.listdir(cluster_path):
        if filename.endswith('.csv'):
            result = process_file(filename, cluster_path)
            if result:
                valid_files.append((*result, cluster_name))  # Add cluster name to results

    # Sort valid_files based on the defined task order and first frame number
    valid_files.sort(key=lambda x: (
        task_order.index(x[1][:3]),  # Sort by task number (first three characters)
        x[2]  # Use first frame number directly
    ))

    return valid_files

def save_results(valid_files, cluster_name, base_directory):
    """Saves the results to a text file."""
    output_file = os.path.join(base_directory, f"ordered_results_{cluster_name}.txt")
    with open(output_file, 'w') as f:
        for subject, task_segment, first_frame_number, filename, cluster in valid_files:  # Unpack to include filename
            f.write(f"Cluster: {cluster}, Subject: {subject}, Task-Segment: {task_segment}, First Frame Number: {first_frame_number}, Filename: {filename}\n")

    print(f"Results saved to {output_file}")

def process_pose_data(base_directory):
    """Main function to process pose data across all cluster directories."""
    task_order = get_task_order()
    all_valid_files = []  # List to accumulate valid files from all clusters

    # Iterate over each cluster directory
    for cluster in os.listdir(base_directory):
        cluster_path = os.path.join(base_directory, cluster)
        if os.path.isdir(cluster_path):
            print(f"Processing cluster directory: {cluster}")
            valid_files = process_cluster(cluster_path, task_order, cluster)

            if valid_files:
                all_valid_files.extend(valid_files)  # Combine all valid files
            else:
                print("No valid pose data files were processed in this cluster. Check file names and folder structure.")

    # Sort all valid files across clusters after processing all clusters
    if all_valid_files:
        all_valid_files.sort(key=lambda x: (
            task_order.index(x[1][:3]),  # Sort by task number (first three characters)
            x[2]  # Use first frame number directly
        ))
        save_results(all_valid_files, "combined_results", base_directory)  # Save once
    else:
        print("No valid pose data files were processed across all clusters.")

# Replace with the path to your clustered results
base_directory = r'C:\Users\kakaz\Documents\Pose estimation\Clustered Results\S1 Clustering 5'
process_pose_data(base_directory)
