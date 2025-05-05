import cv2
import os

def frames_to_video(frame_dir, output_video_path, fps=30):
    """
    Combines all `.jpg` frames in a directory into a `.mp4` video.

    Args:
        frame_dir (str): Path to the directory containing `.jpg` frames.
        output_video_path (str): Path to save the output video.
        fps (int): Frames per second for the output video (default: 30).
    """
    # Get list of `.jpg` files in the directory
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])

    if not frame_files:
        print(f"No `.jpg` frames found in directory: {frame_dir}")
        return

    # Read the first frame to get its dimensions
    first_frame_path = os.path.join(frame_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        print(f"Error: Unable to read frame {first_frame_path}")
        return

    height, width, _ = first_frame.shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for `.mp4` video
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not video_writer.isOpened():
        print(f"Error: Unable to create video writer for {output_video_path}")
        return

    # Iterate over frames and write to video
    print("Combining frames into video...")
    for frame_file in frame_files:
        frame_path = os.path.join(frame_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Error: Unable to read frame {frame_path}")
            continue
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()
    print(f"Video saved to {output_video_path}")


# Example usage
if __name__ == '__main__':
    frame_dir = './video_output/pramlee1_colorized'  # Directory containing `.jpg` frames
    output_video_path = './video_output/output_video.mp4'  # Path to save the output video
    fps = 30  # Frames per second for the output video

    # Combine frames into a video
    frames_to_video(frame_dir, output_video_path, fps)
