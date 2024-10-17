import cv2
import numpy as np

class TV:
    def __init__(self, original_position, from_frame, color_series):
        """
        Initialize a TV instance.

        :param original_position: Tuple[int, int] representing (x, y) coordinates.
        :param from_frame: int, the starting frame index.
        :param color_series: NumPy array of shape (num_frames, 3) representing RGB colors.
        """
        self.original_position = original_position
        self.from_frame = from_frame
        self.color_series = color_series  # Shape: (num_frames, 3)

    def __getitem__(self, frame_index):
        """
        Retrieve the color at a specific frame index.

        :param frame_index: int, index relative to from_frame.
        :return: Tuple[int, int, int] representing the RGB color.
        """
        if frame_index < 0 or frame_index >= len(self.color_series):
            raise IndexError("Frame index out of range.")
        return tuple(self.color_series[frame_index])

    def __len__(self):
        """
        Return the number of frames in the color series.

        :return: int
        """
        return len(self.color_series)

    def __repr__(self):
        return (f"TV(original_position={self.original_position}, "
                f"from_frame={self.from_frame}, "
                f"num_frames={len(self.color_series)})")

class VideoProcessor:
    def __init__(self, video_path):
        """
        Initialize the VideoProcessor by loading the video.

        :param video_path: str, path to the video file.
        """
        self.video_path = video_path
        self.frames = []
        self.load_video()

    def load_video(self):
        """
        Load all frames from the video into a NumPy array.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {self.video_path}")

        frame_list = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_list.append(frame_rgb)

        cap.release()

        if not frame_list:
            raise ValueError("No frames were loaded from the video.")

        # Convert list of frames to a 4D NumPy array for efficient storage
        self.frames = np.stack(frame_list)  # Shape: (num_frames, height, width, 3)
        print(f"Loaded {len(self.frames)} frames with resolution {self.frames.shape[1:3]}.")

    def create_tv(self, position, from_frame=0):
        """
        Create a TV instance for a specific pixel position.

        :param position: Tuple[int, int] representing (x, y) coordinates.
        :param from_frame: int, the starting frame index.
        :return: TV instance.
        """
        x, y = position
        num_frames, height, width, _ = self.frames.shape

        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"Position {position} is out of bounds for frame size ({width}, {height}).")

        if not (0 <= from_frame < num_frames):
            raise ValueError(f"from_frame {from_frame} is out of bounds for number of frames {num_frames}.")

        # Extract the color series for the given position starting from from_frame
        color_series = self.frames[from_frame:, y, x, :]  # Shape: (num_frames - from_frame, 3)

        return TV(original_position=position, from_frame=from_frame, color_series=color_series)

    def create_tvs(self, positions, from_frame=0):
        """
        Create multiple TV instances for a list of pixel positions.

        :param positions: List[Tuple[int, int]] representing (x, y) coordinates.
        :param from_frame: int, the starting frame index for all TVs.
        :return: List[TV] instances.
        """
        return [self.create_tv(pos, from_frame) for pos in positions]

def main():
    # Example usage
    video_path = 'cab_ride_trimmed.mkv'  # Replace with your video file path
    processor = VideoProcessor(video_path)

    # Define pixel positions you want to track
    pixel_positions = [
        (50, 50),
        (100, 100),
        (150, 150),
        # Add more positions as needed
    ]

    # Create TV instances for these positions starting from frame 0
    tvs = processor.create_tvs(pixel_positions, from_frame=0)

    # Example: Accessing color series
    for tv in tvs:
        print(tv)
        # Get color at frame index 10 relative to from_frame
        try:
            color = tv[10]
            print(f"Color at frame {tv.from_frame + 10} for position {tv.original_position}: {color}")
        except IndexError:
            print(f"Frame index 10 is out of range for TV at position {tv.original_position}.")

    # Example: Iterate over all TVs and their color series
    for tv in tvs:
        print(f"TV at position {tv.original_position} starting from frame {tv.from_frame}:")
        for i, color in enumerate(tv.color_series):
            frame_number = tv.from_frame + i
            print(f"  Frame {frame_number}: {color}")

if __name__ == "__main__":
    main()
