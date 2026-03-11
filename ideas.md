# Ideas and Resources

This document contains some relevant thoughts and resources to the Video Unscrambling Problem.

<figure style="display: table">
  <video src="C:\Users\Brock\Documents\code\unscramble_video\bfs_output\comparison.mp4" controls autoplay loop muted></video>
  <figcaption style="display: table-caption; caption-side: bottom"><em>Original video (top) vs. BFS reconstruction attempt (bottom). There are clear boundaries that can be detected with the sobel algorithm, and many of these edges are shared with the other frames. Theoretically if the video is in its original permutation, the heatmap of the edges should be roughly uniform in color.</em></figcaption>
</figure>
<figure style="display: table">
  <img src="C:\Users\Brock\Documents\code\unscramble_video\bfs_output\common_edges\common_edges_heatmap.png" alt="common edges heatmap" />
  <figcaption style="display: table-caption; caption-side: bottom"><em>Heatmap of persistent edges detected across all frames of the BFS reconstruction of the above video. Brighter regions indicate edges that appear consistently across many frames. The non-uniform distribution reveals boundary artifacts from misplaced pixels.</em></figcaption>
</figure>


## Ideas

- Use HDF5 to store the video as a Numpy array, then do slicing instead of keeping a bunch of separate image files!
  - …Then after you've grabbed the top 1000 top similar pixels, use higher framerates (lower stride) to get finer grain information, repeat the process over and over (why didn't I think of this earlier?????)

- Use motion ghosting to improve the neighbor dissonance metrics
  - Use the original metric to determine general neighborhood, then scrub through motion ghosting for instances of movement. Use instances where motion is available and color of the motion streak is relatively uniform without being *too* uniform.
- Use normalization before taking distance so that it considers each channel equally if the distribution of the colors is different. Compare non-normalized result to normalized result in the compare_metrics_gui. Maybe this will have a similar result to Mahalanobis distance? This may help especially since for example in the train footage you a high concentration in the green channel?
- When constructing pinwheels, use a greater number of frames, and make sure that there is high entropy in the frames that you sample. It should also be normalized maybe just for those specific selected pixels to construct the pinwheel so that it becomes more sensitive to the other channels?2



### Improving Neighbor Dissonance metric with Motion Ghosting



## Interesting and relevant resources:

### [Motion aftereffect](https://en.wikipedia.org/wiki/Motion_aftereffect)

https://www.youtube.com/watch?v=AJIQ7aSFpt8&list=PLp7LEd9YN4TM0uD98EPQxbQqWjzWJ7nSc&index=7

This is a type of illusion that will warp the visual field, even after you've looked away from the illusion. The commonly held explanation is that neurons coding a particular movement get exhausted, and when that motion suddenly disappears the movement will appear to flow the other way. I think this might be the strongest evidence that motion has a lot to do with how the pixels in our visual field get organized.

- MAE Applet (radial, non-nausea inducing): https://michaelbach.de/ot/mot-adapt/

### [Primary Visual Cortex](https://www.sciencedirect.com/topics/computer-science/primary-visual-cortex#:~:text=The%20primary%20visual%20cortex%2C%20also,visual%20data%20compaction%20and%20abstraction.)

![elements of movement color differences v1](C:\Users\Brock\Documents\code\unscramble_video\elements of movement color differences v1.jpg)

- #### [Direction selectivity](https://www.youtube.com/watch?v=ePP0G7FJGPI) (Nancy Kanwisher)

  - Certain neurons fire off when movement in a particular direction starts happening, and when movement happens in the opposite direction the same neuron will remain inhibited.

- #### [Visual Thinking for Design](https://jpaulgibson.synology.me/~jpaulgibson/TSP/Teaching/Teaching-ReadingMaterial/Ware10.pdf) (see chapter 2)

### [Assembling Jigsaw Puzzles](https://cvgl.stanford.edu/teaching/cs231a_winter1415/prev/projects/JigsawPuzzle.pdf)

- Found Mahalanobis Distance in this paper, which outperforms Summed Color Distance by a significant margin, producing higher sphericity for circular clusters.
- This mentions a technique that they used before they tried Mahalanobis where they used sobel edge detection to see if two puzzle pieces fit together properly or not. A similar principle applies in the Video Unscrambling Problem because you can very clearly see edges where patches of pixels don't belong together.



### [Temporal dependence Mahalanobis distance for anomaly detection in multivariate spacecraft telemetry series](https://drive.google.com/file/d/1Jexcezank6e0ziD84SxcMzdd4lCjWwbb/view?usp=drive_link)

- Contains a temporal variant of Mahalanobis Distance; ~~I haven't tried this yet because the paper is behind a paywall, I will try it when I get access to the paper~~ A neuroscientist on Reddit was nice enough to send me the PDF, I've placed it on Google Drive. There is not yet a software implementation of it.

### [World's Hardest Jigsaw Puzzle vs. Puzzle Machine](https://www.youtube.com/watch?v=WsPHBD5NsS0)

- Implements Locality-sensitive hashing in order to quickly organize similar puzzle piece edges together, allowing for nearest neighbor search without doing comparisons. Definitely worth looking into, but I think it might be possible that K-D Trees with Flattened Euclidean might be better for pruning in this case because it's possible for members of different bins to be similar, especially if you have gradual changes between the members. [Beyond Locality-Sensitive Hashing](https://www.youtube.com/watch?v=a8Aaap9maB0) may be a helpful resources to pursue, there might be a better method hidden in here.

### [Solving Square Jigsaw Puzzles with Loop Constraints](https://faculty.cc.gatech.edu/~hays/papers/puzzle_eccv14.pdf)

- Pertains to solving jigsaw puzzles with square pieces where the rotation of the pieces is not given, so it's a very similar problem.

### [Beyond Locality-Sensitive Hashing](https://www.youtube.com/watch?v=a8Aaap9maB0)

### [How Super Resolution Works](https://www.youtube.com/watch?v=KULkSwLk62I&t=249s)

- If you can properly arrange a lower resolution version of the image, it may be possible to use Super Resolution to help reconstruct the rest of the image.