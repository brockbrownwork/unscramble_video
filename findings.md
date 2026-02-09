# Findings

## Naïve Unscrambling

<table>
<tr>
<td><img src="bfs_original.png" width="500"/></td>
<td><img src="bfs_result.png" width="500"/></td>
</tr>
<tr>
<td><em>Fig. 1: The unscrambled first frame</em></td>
<td><em>Fig. 2: Unscrambling attempt of the first frame using a greedy BFS type search</em></td>
</tr>
</table>

I attempted a form of greedy BFS style search gradually building layers around a randomly selected **visual field pixel** *(fig. 2)*. It looks like it's trying to roughly place the correct textures together, focusing sharply on local location optimization without any notion of global location optimization. It appears to have a roughly symmetrical pattern, with a square in the middle, and another square in the middle of that with rays shooting out of the corners. I think what it did is it just placed pixels that were similar to each other together until it ran out of pixels that were like that and had to pick something else. This phenomenon is emphasized by the clear boundaries of the center square. The rearrangement is clumpy and stringy, at some points taking on the square formation and at other points it is meandering with rivers shooting out from the middle. The visual field pixels that represent the sky don't even know that they're supposed to go together, likewise with the landscape, and so the visual field pixels become interspersed. Most of the brighter portions of the landscape have been pushed to the left and right edges.

The lesson learned here is that *small errors will propagate into large ones*; the **neighbor dissonance map** must provide an absolutely clean signal or an approach like this will not work.

## Evaluating Neighbor Dissonance Metrics

The neighbor dissonance metric that works best for locating the most similar visual field pixels so far is Euclidean distance; it is better as distinguishing unshuffled positions using just the neighbor dissonance metric alone than other methods. More specifically, to calculate the dissonance between two visual field pixels:

Take the difference between the two visual field pixels' color series across all 3 RGB channels and all T frames, then compute the standard Euclidean distance: square every difference, sum them all, and take the square root. This treats each pixel's color history as a single vector in 3×T dimensional space. Think of considering each color in the color series as a point in three dimensional space, then take the sum of the distances for each point in the series.

For example, with T=3 frames and two pixels A and B:

1. **Pixel A:** Frame 1 = (255, 0, 0), Frame 2 = (200, 10, 0), Frame 3 = (180, 20, 10)

   **Pixel B:** Frame 1 = (250, 5, 0), Frame 2 = (190, 15, 5), Frame 3 = (170, 25, 15)

2. **Differences:** Frame 1: (5, −5, 0), Frame 2: (10, −5, −5), Frame 3: (10, −5, −5)

3. **Sum of the squared differences:** 25 + 25 + 0 + 100 + 25 + 25 + 100 + 25 + 25 = 350

4. **Euclidean Distance:** √350 ≈ 18.7

   And so, 18.7 is the **dissonance** between **visual field pixel A** and **visual field pixel B**.

### Flattened Euclidean vs Sum-of-Per-Frame Euclidean

The metric described above is called **Flattened Euclidean** — it treats each pixel's entire color history as a single vector in 3×T dimensional space and computes one Euclidean distance.

An alternative metric is **Sum-of-Per-Frame Euclidean**: instead of flattening all channels and frames into one vector, compute the Euclidean distance between the two pixels *independently for each frame*, then sum those per-frame distances.

Using the same example (T=3):

1. **Frame 1 distance:** √(5² + 5² + 0²) = √50 ≈ 7.07
2. **Frame 2 distance:** √(10² + 5² + 5²) = √150 ≈ 12.25
3. **Frame 3 distance:** √(10² + 5² + 5²) = √150 ≈ 12.25
4. **Sum-of-Per-Frame distance:** 7.07 + 12.25 + 12.25 = **31.57**

The key difference: Flattened Euclidean allows a large difference in one frame to be partially cancelled by small differences elsewhere (via the square root over the total sum of squares). Sum-of-Per-Frame treats each frame independently, so a single frame with a large color jump contributes its full weight — it penalizes *any* frame-level disagreement more heavily.

### Inscribed Circle Coverage

To evaluate how spatially coherent a metric's top-N selections are, we measure **inscribed circle coverage**. After selecting the top-N least dissonant pixels relative to a clicked pixel:

1. Find the **furthest** top-N pixel from the clicked pixel (Euclidean distance in image space)
2. Draw a circle of that radius centered on the clicked pixel
3. Count how many of the top-N pixels fall within the circle vs the total pixels in the circle

A high coverage percentage means the top-N pixels are tightly clustered around the clicked pixel in image space — the metric is spatially coherent. A low percentage means the top-N pixels are scattered with gaps, requiring a large circle to contain them all.

### Average Spatial Distance

Another way to compare metrics: for increasing values of top-N, plot the **average spatial distance** (in pixels) from the clicked pixel to the top-N least dissonant pixels. A better metric will have a lower average spatial distance at each N, meaning its closest matches (by color-series distance) are also physically closer in the image. This curve reveals how quickly spatial coherence degrades as we include more pixels.

### Circle Quality

Inscribed circle coverage alone can be misleading — a metric could achieve high fill rate by selecting pixels that are all very close to the clicked pixel (tiny radius, dense fill) or by having a huge radius that happens to contain most of the image. Neither extreme is what we want. **Circle Quality** combines two factors into a single score using the geometric mean:

- **Compactness** = 1 − radius / max_diagonal — measures how tight the circle is relative to the image size. A small radius (top-N pixels are nearby) gives high compactness; a radius spanning the whole image gives ~0.
- **Fill rate** = top_N / (π × radius²) — measures how densely the top-N pixels pack the circle. A solid disc of selected pixels gives fill rate ~1; scattered outliers that inflate the radius give a low fill rate.

**Circle Quality = √(compactness × fill_rate) × 100**

The geometric mean ensures *both* components must be good for a high score. A tiny radius with sparse fill gets penalised, and so does a huge radius with dense fill. This makes Circle Quality a single number that captures the ideal: top-N pixels forming a tight, solid disc around the clicked pixel — exactly what a spatially coherent metric should produce.

### Temporal Gradient Distance

Instead of comparing absolute pixel colors across time, the **Temporal Gradient** metric compares frame-to-frame color *changes* (temporal derivatives). The intuition is that neighboring pixels undergo correlated changes from the same lighting shifts and motion, even when their absolute colors differ (e.g. at object boundaries). For each pair of pixels, compute `np.diff` along the time axis to get the frame-to-frame deltas, then measure the sum-of-per-frame Euclidean distance on those deltas.

In practice, temporal gradient's circle quality tends to fall between flattened Euclidean and sum-of-per-frame Euclidean — it doesn't consistently beat either of the raw-color metrics. The derivative signal is noisier than the absolute color signal, which likely limits its standalone performance.
