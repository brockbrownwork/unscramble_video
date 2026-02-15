# Ideas

This document contains some relevant thoughts and resources to the Video Unscrambling Problem.

## Interesting and relevant resources:

### [Assembling Jigsaw Puzzles](https://cvgl.stanford.edu/teaching/cs231a_winter1415/prev/projects/JigsawPuzzle.pdf)

- Found Mahalanobis Distance in this paper, which outperforms Summed Color Distance by a significant margin, producing higher sphericity for circular clusters.
- This mentions a technique that they used before they tried Mahalanobis where they used sobel edge detection to see if two puzzle pieces fit together properly or not. A similar principle applies in the Video Unscrambling Problem because you can very clearly see edges where patches of pixels don't belong together.

<table>
  <tr>
    <td><img src="C:\Users\Brock\Documents\code\unscramble_video\visible_edges_example.png" alt="visible_edges_example"></td>
    <td><img src="C:\Users\Brock\Documents\code\unscramble_video\bfs_output\common_edges\common_edges_heatmap.png" alt="common_edges_heatmap"></td>
  </tr>
</table>

*Pictured: one frame of an unscrambling attempt, along with a sobel heatmap representing edges of all frames. There are clear boundaries that can be detected with the sobel algorithm, and many of these edges are shared with the other frames.*

### [Temporal dependence Mahalanobis distance for anomaly detection in multivariate spacecraft telemetry series](https://pubmed.ncbi.nlm.nih.gov/37331907/)

- Contains a temporal variant of Mahalanobis Distance; I haven't tried this yet because the paper is behind a paywall, I will try it when I get access to the paper.

### [World's Hardest Jigsaw Puzzle vs. Puzzle Machine](https://www.youtube.com/watch?v=WsPHBD5NsS0)

- Implements Locality-sensitive hashing in order to quickly organize similar puzzle piece edges together, allowing for nearest neighbor search without doing comparisons. Definitely worth looking into, but I think it might be possible that K-D Trees with Flattened Euclidean might be better for pruning in this case because it's possible for members of different bins to be similar, especially if you have gradual changes between the members. [Beyond Locality-Sensitive Hashing](https://www.youtube.com/watch?v=a8Aaap9maB0) may be a helpful resources to pursue, there might be a better method hidden in here.

### [Solving Square Jigsaw Puzzles with Loop Constraints](https://faculty.cc.gatech.edu/~hays/papers/puzzle_eccv14.pdf)

- Pertains to solving jigsaw puzzles with square pieces where the rotation of the pieces is not given. 

### [Beyond Locality-Sensitive Hashing](https://www.youtube.com/watch?v=a8Aaap9maB0)