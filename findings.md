# Findings

## Naïve Unscrambling

<img src="bfs_original.png" width="500"/>
*Fig. 1: The unscrambled first frame*

<img src="bfs_result.png" width="500"/>
*Fig. 2: Unscrambling attempt of the first frame using a greedy BFS type search*

I attempted this form of greedy BFS style search gradually building layers around a randomly selected visual field pixel (fig. 2). It looks like it's trying to roughly place the correct textures together, focusing sharply on local location optimization without any notion of global location optimization. It appears to have a roughly symmetrical pattern, with a square in the middle, and another square in the middle of that with rays shooting out of the corners. I think what it did is it just placed pixels that were similar to each other together until it ran out of pixels that were like that and had to pick something else. This phenomenon is emphasized by the clear boundaries of the center square. The rearrangement is clumpy and stringy, to the point where the visual field pixels that represent the sky don't even know that they're supposed to go together, likewise with the landscape.

The lesson learned here is that small errors will propagate into large ones; the neighbor dissonance map must provide an absolutely clean signal or an approach like this will not work.

## Evaluating Neighbor Dissonance Metrics

