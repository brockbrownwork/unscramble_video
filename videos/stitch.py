import os
import re
import subprocess
from tqdm import tqdm

# Get the current directory (where the script is located)
video_folder = os.getcwd()
output_file = 'stitched_output.mp4'

# Regex to extract the number from the filenames (assuming the format is consistent)
def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else float('inf')  # Return a large number if no match

# Get the list of all mp4 files that start with 'combined' and sort by the extracted number
video_files = sorted([f for f in os.listdir(video_folder) 
                      if f.startswith('combined') and f.endswith('.mp4')],
                     key=extract_number)


# Create a temporary text file to list the videos for ffmpeg
with open('videos_to_concatenate.txt', 'w') as f:
    for video in video_files:
        f.write(f"file '{os.path.join(video_folder, video)}'\n")

# Function to execute ffmpeg command
def concatenate_videos_ffmpeg(input_list, output):
    command = [
        'ffmpeg',
        '-f', 'concat',      # Telling ffmpeg we are concatenating
        '-safe', '0',        # Allows filenames with special characters
        '-i', input_list,    # Input list file
        '-c', 'copy',        # Copy codec without re-encoding (faster)
        output               # Output file
    ]
    subprocess.run(command)

# Display progress with tqdm based on the number of video files
for _ in tqdm(range(len(video_files)), desc="Processing videos"):
    pass

# Concatenate videos using ffmpeg
concatenate_videos_ffmpeg('videos_to_concatenate.txt', output_file)

# Clean up the temporary text file
os.remove('videos_to_concatenate.txt')

print("Video stitching complete!")
