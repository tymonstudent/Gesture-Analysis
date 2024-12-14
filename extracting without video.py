import os
import cv2
import mediapipe as mp
import csv
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def write_landmarks_to_csv(landmarks, frame_number, csv_data, pose_landmark):
    for idx, landmark in enumerate(landmarks):
        csv_data.append([frame_number, pose_landmark.PoseLandmark(idx).name, landmark.x, landmark.y, landmark.z])

def process_video(video_path, output_dir, pose_landmark, model_complexity=2, enable_segmentation=False, smooth_landmarks=True):
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    csv_data = []

    # Create a Pose object with adjustable parameters
    with pose_landmark.Pose(
        model_complexity=model_complexity,
        enable_segmentation=enable_segmentation,
        smooth_landmarks=smooth_landmarks
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Reached the end of the video.")
                break

            print(f"Processing frame {frame_number}")
            
            # Convert the frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe Pose
            result = pose.process(frame_rgb)

            # Add the landmark coordinates to the list
            if result.pose_landmarks:
                print(f"Landmarks found for frame {frame_number}")
                write_landmarks_to_csv(result.pose_landmarks.landmark, frame_number, csv_data, pose_landmark)
            else:
                print(f"No landmarks found for frame {frame_number}")

            frame_number += 1

        cap.release()

    # Create a reasonably named CSV file
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_csv = os.path.join(output_dir, f"{video_name}_pose_data.csv")

    # Save the CSV data to a file
    print(f"Saving data to CSV: {output_csv}")
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['frame_number', 'landmark', 'x', 'y', 'z'])
        csv_writer.writerows(csv_data)

    print(f"Finished processing video: {video_path}")

def process_videos_in_folder(input_folder, output_folder, model_complexity=2, enable_segmentation=False, smooth_landmarks=True):
    mp_pose = mp.solutions.pose

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    for filename in os.listdir(input_folder):
        if filename.endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(input_folder, filename)
            video_name = os.path.splitext(filename)[0]
            output_csv = os.path.join(output_folder, f"{video_name}_pose_data.csv")
            
            # Check if the output CSV file already exists
            if os.path.exists(output_csv):
                print(f"Skipping {filename} as it has already been processed.")
                continue

            print(f"Found video file: {filename}")
            process_video(video_path, output_folder, mp_pose, model_complexity, enable_segmentation, smooth_landmarks)

    print("Finished processing all videos.")
 
   


# Example usage
input_folder = 'E:/Experiment videos'
output_folder = 'E:/Experiment videos/Test folder for hte program/Output'
process_videos_in_folder(input_folder, output_folder, model_complexity=2, enable_segmentation=False, smooth_landmarks=True)
