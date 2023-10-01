from moviepy.editor import *

# Load the video file
video = VideoFileClip("cab_ride.mkv")

# Trim the first 5 seconds off the video
trimmed_video = video.subclip(t_start=5)

# Write the result to a file
trimmed_video.write_videofile("output.mkv", codec="libx264")
