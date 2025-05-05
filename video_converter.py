import cv2
import os

def video_to_frames(video_path, output_dir):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    frame_number = 0
    
    while True:
        # Read the next frame from the video
        ret, frame = cap.read()
        
        # If the frame was not retrieved successfully, break the loop
        if not ret:
            break
        
        # Construct the output file path
        frame_filename = os.path.join(output_dir, f"frame_{frame_number:04d}.jpg")
        
        # Save the frame as a .jpg file
        cv2.imwrite(frame_filename, frame)
        
        print(f"Saved {frame_filename}")
        
        # Increment the frame number
        frame_number += 1
    
    # Release the video capture object
    cap.release()
    print("Frame extraction completed.")

# Example usage
video_path = "./video_input/pramlee.mp4"  # Replace with your video file path
output_dir = "./video_output"    # Replace with your desired output directory

video_to_frames(video_path, output_dir)
