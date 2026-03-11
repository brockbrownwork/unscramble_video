# OOP for Video Puzzles

<img align="left" src="shuffle_animation.gif" width="75%" />

This document outlines some specifications for some objects that are useful for solving **video puzzles**. **Video puzzles** are like regular jigsaw puzzles, except instead of putting the pieces together to form a non-moving image, you are forming a looping video. Assume all of the pieces are square and that each pixel of the video is one piece of the puzzle, see above animation for intuition. *These puzzle pieces will be referred to as **pixels** throughout this document.* Each **pixel** is treated as a time series of RGB values as a Numpy array in the form of **T x C**. Note: instead of loading each timestep for each **pixel**, which may be memory intensive, we will define which **pixels** we want to load and which timesteps to load. Each **pixel** has a unique ID, assigned in **shuffled_indices**.

(Actually, we should just use HDF5; this will WAY simplify the process of extraction and organization)

## PixelBag

<img align="left" src="pixel_bag_animation.gif" width="50%" />

A bag of **pixels**. This object can be iterated or indexed, and it will yield a random **pixel** from the Video Puzzle in the form of ``(pixel_id, pixel_color_time_series)``.

Since videos can be quite large, instead of loading the entire video into memory we will extract each frame of the video into a directory as a *.bmp* file and load them into memory as needed. It will use OpenCV to load the images into memory.

### Initializing parameters:

- **file_name** (default = "cab_ride_trimmed.mkv"): The name of the input video that we want to turn into a **video puzzle**. If the the directory of extracted frames doesn't already exist or there aren't any images in the directory, create the directory then extract the frames into **file_name**_frames folder *(i.e.: **file_name** = example.mp4 should load all frames into the example_frames directory)*, load them using ``ffmpeg -i example.mp4 example_frames/$filename%06d.bmp``
- **stride** (default = 1): number of frames to skip when iterating along loading into memory, if stride = 1 then use all frames between **start_frame** and **end_frame**.
- **start_frame** (default = 1): the video frame to start loading into memory
- **end_frame** (default = last frame of video): the frame to stop loading into memory
- **crop_percentage** (default = 100): This will crop the video down to the middle portion by some percentage of the width of the video.
- **seed** (default = 42): a random seed that will be used to shuffle and flatten the **pixels** using Numpy advanced indexing

### **Attributes**:

- **loaded_frames**: Numpy array of frames that are loaded into memory, if **stride** is 2, **start_frame** is 3, and **end_frame** is 10: ``[2, 4, 6, 8, 10]``
- **shuffled_indices**: A one-dimenional Numpy array that is used to shuffle the **pixels**, generated with the **seed** parameter as its random seed.
- **original_positions**: A Numpy array which is a list of the original 2D positions of the **pixels**, i.e.: ``[(0, 0), (42, 1337), …]``. In the example, the original position of **pixel** ID ```1``` is ``(42, 1337)``.
- **loaded_pixel_ids**: These are the IDs of the **pixels** that are currently stored in memory.

### Methods:

- **load( frame_indices, pixel_ids)**: 

  Parameters:

  - **frame_indices**
  - **pixel_ids**

## PuzzleBoard

This is a two-dimensional grid that you can tack **pixels** onto.

(should this be a numpy array? or should it be set up so that it's infinite?)

## Pinwheel

This object is meant to be used for circular type constructions that start from the center and construct layer by layer. The procedure for constructing a pinwheel is as follows:

- Pick a random pixel from a **PixelBag**.
- Gather the top **n** most similar **pixels**.
- Use the top **n** most similar **pixels** to construct the layers around it.
