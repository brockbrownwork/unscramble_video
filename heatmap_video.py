import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Paths
input_video_path = 'cab_ride_trimmed.mkv'   # Replace with your input video path
output_video_path = 'output_with_heatmap.mp4'  # Desired output path

# Reference pixel position (x, y)
ref_x, ref_y = 100, 100  # Example coordinates; adjust as needed

# Open the video
cap = cv2.VideoCapture(input_video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec

# Initialize VideoWriter
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

i = 0
while cap.isOpened() and i < 100:
    ret, frame = cap.read()
    i += 1
    print(i)
    if not ret:
        break

    # Ensure the reference pixel is within frame boundaries
    if ref_x >= frame_width or ref_y >= frame_height:
        print(f"Reference pixel ({ref_x}, {ref_y}) is out of frame boundaries.")
        break

    # Get the reference pixel color (in BGR format)
    ref_color = frame[ref_y, ref_x].astype(float)  # Shape: (3,)

    # Compute color similarity
    # Calculate Euclidean distance in color space
    diff = frame.astype(float) - ref_color  # Broadcasting
    distance = np.linalg.norm(diff, axis=2)  # Shape: (height, width)

    # Normalize distance to [0, 1]
    max_distance = np.sqrt(3 * (255 ** 2))  # Maximum possible distance in BGR
    normalized_distance = distance / max_distance  # Closer colors have smaller values

    # Invert distances so that similar colors have higher values
    similarity = 1 - normalized_distance  # Values between 0 and 1

    # Convert similarity to a heatmap
    heatmap = (similarity * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Blend the heatmap with the original frame
    alpha = 0.6  # Transparency factor
    overlay = cv2.addWeighted(frame, 1 - alpha, heatmap_color, alpha, 0)

    # Optionally, mark the reference pixel
    cv2.circle(overlay, (ref_x, ref_y), radius=5, color=(255, 255, 255), thickness=2)

    # Write the frame to the output video
    out.write(overlay)

    # (Optional) Display the frame for debugging
    # cv2.imshow('Heatmap Overlay', overlay)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Output video saved to {output_video_path}")
