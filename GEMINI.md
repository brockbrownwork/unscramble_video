# Project Overview

This project is a "video unscrambler" that aims to reconstruct a coherent video from a collection of scrambled frames. It appears to be a data-heavy project that leverages computer vision and machine learning techniques to achieve this.

The core idea is to treat each pixel in the video as a "TV" that has a color value for each frame. By analyzing the time-series data of these "TVs" (i.e., how the color of each pixel changes over time), the project attempts to find similarities and relationships between frames to reconstruct the original video sequence.

## Key Technologies

*   **Python:** The project is primarily written in Python.
*   **Jupyter Notebooks:** The main logic and experiments are developed and documented in `.ipynb` files.
*   **OpenCV (`cv2`):** Used for video processing tasks like reading frames from a video file.
*   **NumPy:** Used for efficient numerical operations on multi-dimensional arrays, which represent video frames and other data.
*   **UMAP:** A dimensionality reduction technique used to find relationships and patterns in the high-dimensional data of the video frames.
*   **scikit-learn:** A machine learning library that is likely used for clustering or other analysis of the frame data.
*   **quadrilateral-fitter:** A specialized library for fitting irregular quadrilaterals, which suggests that the project might be dealing with perspective distortions in the video.
*   **Manim:** A library for creating mathematical animations, used for creating visualizations for the project.

## Directory Structure

*   **.ipynb_checkpoints/:** Checkpoints for the jupyter notebooks.
*   **media/:** Contains images, videos, and other assets used in the project.
*   **videos/:** Contains video files used in the project.
*   **__pycache__/:** Cache for python files.
*   **analyze_video.ipynb, unscramble.ipynb, etc.:** Jupyter Notebooks containing the core logic of the project.
*   **tvs.py:** Defines the `TV` and `VideoProcessor` classes, which are central to the project's approach.
*   **requirements.txt:** Lists the Python dependencies of the project.
*   **.gif, .png files:** These are likely outputs of the analysis and unscrambling process, used for visualization.
*   **.npy files:** NumPy data files, which likely store intermediate data like scrambled frames or similarity matrices.

## How to Run the Project

The project seems to be primarily based on Jupyter Notebooks. To run the project, you would need to have a Python environment with all the dependencies listed in `requirements.txt` installed.

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
    Then, you can open and run the notebooks (e.g., `unscramble.ipynb`) to see the video unscrambling process in action.

## Development Conventions

The project follows a typical structure for a data science project in Python. The use of Jupyter Notebooks allows for interactive development and visualization of the results. The code is reasonably well-structured, with classes like `TV` and `VideoProcessor` encapsulating the core logic.

## unscramble.py Breakdown

The `unscramble.py` script follows a multi-step process to reconstruct the video.

1.  **Load Video Frames**: Random frames are extracted from the video file `cab_ride_trimmed.mkv`.
2.  **Preview Frames**: A small sample of the extracted frames is displayed to provide a visual check.
3.  **Create "TVs"**: Each pixel's color variation across all frames is treated as a time series, or a "TV". These are flattened and stored in a list.
4.  **Run UMAP**: The UMAP dimensionality reduction algorithm is applied to the "TVs" to generate a 2D coordinate for each one.
5.  **Format Data for Animation**: The "TV" data is converted back into a list of colors, preparing it for visualization.
6.  **Create Animation**: An animated GIF is generated where each point represents a "TV" at its UMAP-assigned coordinate, with its color changing over time according to its original pixel data.
7.  **Apply Perspective Correction**: A `QuadrilateralFitter` is used to correct the perspective of the UMAP output, and a new, corrected animation is created.
