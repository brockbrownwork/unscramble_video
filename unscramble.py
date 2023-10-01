import cv2
import numpy as np

def extract_frames(video_path, max_frames=None):
    """
    Extract frames from a video file.

    Parameters:
    - video_path (str): Path to the video file.
    - max_frames (int, optional): Maximum number of frames to extract. 
                                  If None, extracts all frames. Default is None.

    Returns:
    - frames (ndarray): 4D numpy array containing the extracted frames.
                        The dimensions are (num_frames, height, width, channels).
    """
    # Step 1: Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Step 2: Extract frames from the video
    frames = []
    frame_count = 0
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            frames.append(frame)
            frame_count += 1
            # Break the loop if max_frames is specified and the count reaches max_frames
            if max_frames is not None and frame_count >= max_frames:
                break
        else:
            break

    # Step 3: Convert frames to a numpy array
    frames = np.array(frames)

    # Step 4: Release the video file
    video.release()

    return frames

# Usage:
# Specify the path to your video file and the number of frames you wish to extract
video_path = 'cab_ride_trimmed.mkv'
max_frames = 1000  # Set to None to extract all frames
frames = extract_frames(video_path, max_frames)
print("...done!")

def display_frames(frames):
    """
    Display frames using OpenCV.

    Parameters:
    - frames (ndarray): 4D numpy array containing the frames.
                        The dimensions are (num_frames, height, width, channels).
    """
    for i, frame in enumerate(frames):
        cv2.imshow(f'Frame {i}', frame)
        while True:
            key = cv2.waitKey(0)  # Wait indefinitely for a key press
            if key == ord(' '):  # If spacebar is pressed, move to the next frame
                break
            elif key == ord('q'):  # If 'q' is pressed, exit the loop
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()

# Call the function to display the frames
display_frames(frames)

