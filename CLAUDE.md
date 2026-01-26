# CLAUDE.md

## Project Overview

**unscramble_video** is an experimental video analysis and visualization tool that reconstructs scrambled video frames by analyzing pixel color sequences over time using UMAP dimensionality reduction.

This project is based on a thought problem. Consider that you have a gigantic wall of CRT TVs stacked in a 2d array. Each TV is only capable of displaying one color on its screen at any given moment. You can play a video back on the TV wall where each TV is a pixel position of the video. If you knock down the wall of TVs, can you figure out where the TVs go again? And what kinds of videos can you use that will work for putting the TVs back in their proper place? For example, if the video is just 30 seconds of white and black blinking it would be impossible to figure out where they go. The idea is that TVs with similar color series will have strong tendencies to belong next to each other as long as the video is rich in color variety.

The thought was based on neuroscience originally. There was an experiment where they rewired ferrets' ocular nerve to the auditory cortex and vise-versa and they were still able to learn how to see. Therefore, the positions of "pixels" in animal visual fields is learned and not hard coded. A similar experiment was done with deafferentation in macaques by Edward Taub. It's based on the notion of topological tendencies in the brain; if you tap the index finger, and tap the ring finger, the region that lights up when the middle finger is tapped will live between the index and ring finger regions that light up. If you rewire the nerve from one finger to another, this topological tendency will be relearned.

## Tech Stack

- **Language:** Python
- **Core Libraries:** opencv-python, numpy, umap-learn, matplotlib, Pillow, tqdm, fastdtw, scipy
- **External Tools:** ffmpeg (via subprocess)
- **Development:** Jupyter Notebooks for experimentation

## Project Structure

```
unscramble_video/
├── unscramble.py          # Main script (Euclidean distance)
├── unscramble_dtw.py      # DTW experiment (Dynamic Time Warping distance)
├── videos/
│   └── stitch.py          # Video concatenation utility
├── *.ipynb                # Experimental notebooks
├── *.mkv                  # Input video files
├── *.gif                  # Output animations
└── *.npy                  # Cached numpy arrays
```

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run main script (Euclidean distance)
python unscramble.py

# Run DTW experiment (slower but captures temporal shifts)
python unscramble_dtw.py

# Stitch videos
cd videos && python stitch.py
```

## Key Concepts

- **TVs**: Pixel color time-series (each pixel position across all frames)
- **Pipeline**: Video → frames → TVs → UMAP embedding → RGB visualization → animation
- **DTW (Dynamic Time Warping)**: Distance metric that accounts for time-shifts between sequences, useful when edges/objects move across adjacent pixels
- **Stride**: Frame skip interval for capturing longer-term temporal patterns without increasing computation

## Coding Conventions

- Lowercase with underscores for functions/variables
- Descriptive function names with docstrings
- Use `tqdm` for progress bars on long operations
- Hardcoded parameters at script start (e.g., `number_of_frames`, `n_neighbors`)
- Scripts may contain `# In[X]:` cell markers (converted from Jupyter)

## Git Conventions

- Commit prefixes: `feat:`, `chore:`, `fix:`, `docs:`
- Keep notebooks and generated media out of git (see .gitignore)

## Ideas

- Find nearest neighbors in 9x9 fashion, and take the 9x9 and attach it to four other 9x9s, making it 18x18, and so on so that you're dealing with chunks instead of trying to compare distance between each pixel all the time. Maybe take a step back as a refinement stage, like removing pixels randomly and filling it back in with the best fit based on the heuristic (probably DTW).

- Consider that the fovea is densely packed, and so you get more information at the center of your vision than the outside. There are patterns that exist of hexagonal packing that can be infinitely densely packed at the center and grow gradually to a fixed size on the borders. Escher demonstrated this, but some finite version of that idea might not be a bad way to try to organize them.

- Consider an experiment where you take a video then just randomize a few pixel positions and then figure out which ones are wrong and put them back, and figure out what number you'd have to raise that to in order to start seeing serious trouble.

- Try the thing you did where you demonstrated that the Euclidean distance metric has some real tendencies to actual TV distance, except with DTW side by side to show if DTW is actually a better metric or not.
