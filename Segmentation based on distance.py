import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt 
from scipy.signal import cheby1, filtfilt 
# Low-pass filter configuration
def butter_lowpass_filter(data, cutoff, fs, order=4):
    ripple=0.5
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist
    b, a = cheby1(order, ripple, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)  # Apply forward-backward filter
    return y

# Function to smooth the data
def smooth_data(data, cutoff_frequency, sampling_rate):
    # Apply low-pass filter on each coordinate (x, y, z)
    data.loc[:, 'x'] = butter_lowpass_filter(data['x'], cutoff_frequency, sampling_rate)
    data.loc[:, 'y']  = butter_lowpass_filter(data['y'], cutoff_frequency, sampling_rate)
    data.loc[:, 'z']  = butter_lowpass_filter(data['z'], cutoff_frequency, sampling_rate)
    return data

# List of relevant landmarks for distance calculation
relevant_landmarks = ['LEFT_HIP', 'LEFT_WRIST', 'RIGHT_WRIST']

# Paths
input_folder = r'E:\Experiment videos\Test folder for hte program\Output'
output_folder = "C:/Users/kakaz/Documents/Pose estimation/Segmented Data"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# List of CSV files to process
csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

# Parameters for filtering
cutoff_frequency = 4  # Adjust this based on desired smoothness (in Hz)
sampling_rate = 30.0    # Adjust according to your frame rate (in Hz)

# Function to calculate Euclidean distance from LEFT_HIP to another joint
def calculate_euclidean_distance(df, joint1, joint2):
    # Extract the coordinates for LEFT_HIP and the specified joint
    joint1_coords = df.loc[df['landmark'] == joint1, ['x', 'y', 'z']]
    joint2_coords = df.loc[df['landmark'] == joint2, ['x', 'y', 'z']]
    
    # Calculate Euclidean distance
    distance = np.sqrt((joint1_coords['x'].values - joint2_coords['x'].values) ** 2 + 
                       (joint1_coords['y'].values - joint2_coords['y'].values) ** 2 + 
                       (joint1_coords['z'].values - joint2_coords['z'].values) ** 2)
    
    return distance

for file in csv_files:
    file_path = os.path.join(input_folder, file)
    print(f"Processing file: {file_path}")
    
    # Read the CSV file
    data = pd.read_csv(file_path)

    # Check for missing values
    if data[['x', 'y', 'z']].isnull().any().any():
        print("Missing values found in x, y, or z columns. Filling missing values.")
        data[['x', 'y', 'z']] = data[['x', 'y', 'z']].fillna(method='ffill')

    # Check if the file has more than one frame
    if data['frame_number'].nunique() <= 1:
        print(f"Skipping {file} due to insufficient frames.")
        continue

    # Filter data to include only relevant landmarks for distance calculation
    relevant_data = data[data['landmark'].isin(relevant_landmarks)]

    # Apply the low-pass filter to smooth the data
    #relevant_data = smooth_data(relevant_data, cutoff_frequency, sampling_rate)

    # Group by frame_number and calculate mean positions for the relevant landmarks
    mean_data = relevant_data.groupby('frame_number').agg({
        'landmark': 'first', 
        'x': 'mean', 
        'y': 'mean', 
        'z': 'mean'
    }).reset_index()

    # Calculate distances from LEFT_HIP to LEFT_WRIST and RIGHT_WRIST
    mean_data['distance_left'] = calculate_euclidean_distance(relevant_data, 'LEFT_HIP', 'LEFT_WRIST')
    mean_data['distance_right'] = calculate_euclidean_distance(relevant_data, 'LEFT_HIP', 'RIGHT_WRIST')

    Q3 = np.nanquantile(mean_data['distance_right'], 0.55)
    
    distance_threshold= Q3
    print(f"Calculated threshold for {file}: {distance_threshold}")

    # Count valid distance entries
    valid_distance_count = mean_data[['distance_left', 'distance_right']].notna().sum().sum()
    print(f"Valid distance entries: {valid_distance_count}")
    mean_data['distance_left'] = butter_lowpass_filter(mean_data['distance_left'], cutoff_frequency, sampling_rate)
    mean_data['distance_right'] = butter_lowpass_filter(mean_data['distance_right'], cutoff_frequency, sampling_rate)

    

    # Segment data based on threshold
    segments = []
    current_segment = []

    for index, row in mean_data.iterrows():
        if abs(row['distance_left']) > distance_threshold or abs(row['distance_right']) > distance_threshold:
            current_segment.append(row)
        else:
            if current_segment:
                # Capture all rows corresponding to the segment
                segment_start_frame = current_segment[0]['frame_number']
                segment_end_frame = row['frame_number'] - 1  # Use the last frame before threshold drop
                segment_data = data[(data['frame_number'] >= segment_start_frame) & (data['frame_number'] <= segment_end_frame)]
                segments.append(segment_data)
                current_segment = []
    
    # Add the last segment if it exists
    if current_segment:
        segment_start_frame = current_segment[0]['frame_number']
        segment_data = data[data['frame_number'] >= segment_start_frame]
        segments.append(segment_data)

    # Save segmented data
    # Save segmented data
    for i, segment in enumerate(segments):
        # Check if the segment has at least 15 unique frames
        if segment['frame_number'].nunique() >= 7:
            output_file_path = os.path.join(output_folder, f"{file[:-4]}_segment_{i}.csv")
            segment.to_csv(output_file_path, index=False)
            print(f"Saved segmented data to: {output_file_path}")
        else:
            print(f"Segment {i} skipped due to insufficient frames (< 15).")


print("Processing complete!")
