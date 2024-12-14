from itertools import groupby
from operator import itemgetter
import os

# File paths
input_file_path = r"C:\Users\kakaz\Documents\Pose estimation\Clustered Results\S1 Clustering 5\ordered_results_combined_results.txt"
output_file_path = r"C:\Users\kakaz\Documents\Pose estimation\Clustered Results\S1 Clustering 5\ordered_segments_results.txt"

print("Starting the program...")

# Load data from the file
data = []
print(f"Reading data from {input_file_path}...")
with open(input_file_path, "r") as file:
    for line in file:
        # Parse each line assuming format "Cluster: ..., Subject: ..., Task-Segment: ..., First Frame Number: ..., Filename: ..."
        parts = [part.strip() for part in line.split(",")]
        entry = {k.strip(): v.strip() for k, v in (item.split(":") for item in parts)}
        data.append(entry)
print(f"Loaded {len(data)} entries from the file.")

# Sort by Task-Segment and First Frame Number
print("Sorting data by Task-Segment and First Frame Number...")
data.sort(key=lambda x: (x["Task-Segment"], int(x["First Frame Number"])))
print("Sorting complete.")

# Group by Task-Segment
print("Grouping data by Task-Segment...")
grouped_data = {k: list(v) for k, v in groupby(data, key=lambda x: x["Task-Segment"])}
print(f"Grouped into {len(grouped_data)} task segments.")

# Filter out Task-Segments with only one item
print("Filtering out Task-Segments with only one item...")
filtered_data = {k: v for k, v in grouped_data.items() if len(v) > 1}
print(f"{len(filtered_data)} task segments retained after filtering.")

# Transpose to get the nth occurrence across Task-Segments
print("Organizing data by occurrence across task segments...")
max_segments = max(len(v) for v in filtered_data.values())
ordered_segments = [[] for _ in range(max_segments)]

for task, segments in filtered_data.items():
    for i, segment in enumerate(segments):
        ordered_segments[i].append(segment)
print("Data organized by occurrence.")

# Save ordered segments to a text file
print(f"Saving ordered segments to {output_file_path}...")
with open(output_file_path, "w") as output_file:
    for idx, occurrence in enumerate(ordered_segments):
        output_file.write(f"Occurrence {idx + 1}:\n")
        for segment in occurrence:
            output_file.write(f"{segment}\n")
        output_file.write("\n")

print(f"Results saved to {output_file_path}")
print("Program completed successfully.")
