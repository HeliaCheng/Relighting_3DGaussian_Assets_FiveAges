import cv2
import os
import glob
import argparse

def create_video_from_frames(frame_folder, output_video_path, frame_rate=20.0):
    # Get paths of all image frames
    frame_paths = sorted(glob.glob(os.path.join(frame_folder, '*.png')))

    # Ensure that there is at least one frame
    if not frame_paths:
        print("No frame images found, please check the path.")
        return

    # Read the first frame to determine the video size
    first_frame = cv2.imread(frame_paths[-1])
    height, width, layers = first_frame.shape
    print(height, width, layers)
    
    # Create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4 format
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    # video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (height, width))

    # Iterate over all frames and write them into the video file
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        
        # Rotate the frame counterclockwise by 90 degrees
        # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)   # !

        video_writer.write(frame)

    # Release resources
    video_writer.release()

    print(f"Video created: {output_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a video from image frames.')
    parser.add_argument('input_imgs_path', type=str, help='Path to the folder containing image frames')
    parser.add_argument('output_video_path', type=str, help='Path to the output video file')
    parser.add_argument('--frame_rate', "-f", type=float, default=20.0, help='Video frame rate (default: 20.0)')
    
    args = parser.parse_args()
    
    create_video_from_frames(args.input_imgs_path, args.output_video_path, args.frame_rate)