# CLAUDE.md

## Project Overview

**unscramble_video** is an experimental video analysis and visualization tool that reconstructs scrambled video frames by analyzing pixel color sequences over time using UMAP dimensionality reduction.

This project is based on a thought problem. Consider that you have a gigantic wall of CRT TVs stacked in a 2d array. Each TV is only capable of displaying one color on its screen at any given moment. You can play a video back on the TV wall where each TV is a pixel position of the video. If you knock down the wall of TVs, can you figure out where the TVs go again? And what kinds of videos can you use that will work for putting the TVs back in their proper place? For example, if the video is just 30 seconds of white and black blinking it would be impossible to figure out where they go. The idea is that TVs with similar color series will have strong tendencies to belong next to each other as long as the video is rich in color variety.

The thought was based on neuroscience originally. There was an experiment where they rewired ferrets' ocular nerve to the auditory cortex and vise-versa and they were still able to learn how to see. Therefore, the positions of "pixels" in animal visual fields is learned and not hard coded. A similar experiment was done with deafferentation in macaques by Edward Taub. It's based on the notion of topological tendencies in the brain; if you tap the index finger, and tap the ring finger, the region that lights up when the middle finger is tapped will live between the index and ring finger regions that light up. If you rewire the nerve from one finger to another, this topological tendency will be relearned.

## Tech Stack

- **Language:** Python
- **Core Libraries:** opencv-python, numpy, umap-learn, matplotlib, Pillow, tqdm
- **External Tools:** ffmpeg (via subprocess)
- **Development:** Jupyter Notebooks for experimentation

## Project Structure

```
unscramble_video/
├── unscramble.py          # Main production script
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

# Run main script
python unscramble.py

# Stitch videos
cd videos && python stitch.py
```

## Key Concepts

- **TVs**: Pixel color time-series (each pixel position across all frames)
- **Pipeline**: Video → frames → TVs → UMAP embedding → RGB visualization → animation

## Coding Conventions

- Lowercase with underscores for functions/variables
- Descriptive function names with docstrings
- Use `tqdm` for progress bars on long operations
- Hardcoded parameters at script start (e.g., `number_of_frames`, `n_neighbors`)
- Scripts may contain `# In[X]:` cell markers (converted from Jupyter)

## Git Conventions

- Commit prefixes: `feat:`, `chore:`, `fix:`, `docs:`
- Keep notebooks and generated media out of git (see .gitignore)
