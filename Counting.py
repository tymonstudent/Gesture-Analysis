import os

def count_files_in_folder(root_folder, output_file='file_counts.txt'):
    total_files = 0
    with open(output_file, 'w') as file:
        for root, dirs, files in os.walk(root_folder):
            folder_name = os.path.basename(root)
            file_count = len(files)
            total_files += file_count
            file.write(f'Folder "{folder_name}" contains {file_count} files.\n')
        
        file.write(f'\nTotal number of files across all folders: {total_files}\n')
    print(f'Results saved to {output_file}')
    return total_files

# Usage example
root_folder = r'C:\Users\kakaz\Documents\Pose estimation\Clustered Results\S1 Clustering 5'  # Replace with the path to your main folder
total_files = count_files_in_folder(root_folder)
print(f'Total number of files: {total_files}')
