import os
import subprocess

# Define the root directory
root_directory = r'C:\Users\kakaz\Documents\Pose estimation\Segmented Data'

# Path to the main script
main_script_path = r'C:\Users\kakaz\Documents\Pose estimation\One pipeline baby.py'  # Replace with the path to your main script

# Iterate through each subdirectory
for subdir in os.listdir(root_directory):
    subdir_path = os.path.join(root_directory, subdir)
    if os.path.isdir(subdir_path):  # Ensure it's a directory
        print(f"Processing folder: {subdir_path}")

        # Dynamically set the folder_path argument and execute the script
        try:
            subprocess.run(
                ['python', main_script_path, subdir_path],
                check=True  # This ensures any errors during execution will raise an exception
            )
            print(f"Finished processing {subdir_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while processing {subdir_path}: {e}")

print("All folders processed.")
