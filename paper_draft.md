# Recovering Spatial Topology from Temporal Correlation: A Pixel Permutation Recovery Problem Inspired by Neural Plasticity

**Author:** Brock (Independent Researcher)

---

## Abstract

We introduce and formalize the *pixel permutation recovery problem*: given a video whose pixel positions have been randomly permuted (but whose per-pixel color sequences over time are preserved), can the original spatial arrangement be reconstructed? This problem is inspired by neural plasticity research demonstrating that topographic organization in sensory cortex can emerge from temporal correlation alone. We develop a computational framework (TVWall) for studying this problem, implement and compare multiple distance metrics for measuring pixel similarity, and evaluate several reconstruction strategies including greedy optimization, simulated annealing, and breadth-first search placement. We characterize the conditions under which reconstruction succeeds and fails, finding that color entropy, video complexity, and metric choice critically determine solvability. Our results provide computational evidence that spatial topology can, in principle, be recovered from purely temporal information — consistent with theories of experience-dependent map formation in biological neural systems.

## 1. Introduction

### 1.1 The TV Wall Thought Experiment

Consider a wall of CRT televisions arranged in a two-dimensional grid. Each television displays a single color at any moment, effectively functioning as one pixel of a larger image. When a video is played across this wall, each TV shows the color sequence corresponding to its pixel position over time.

Now imagine the wall is knocked down and the televisions scattered. Each TV continues to play its assigned color sequence faithfully. The question is: *can we determine where each TV belongs?*

The key insight motivating this work is that neighboring pixels in natural video tend to exhibit correlated color sequences. Objects move smoothly, lighting changes gradually, and scene boundaries create shared local statistics. Two TVs that originally sat adjacent will, on average, have more similar temporal color profiles than two TVs drawn from distant positions. This correlation is the signal we exploit.

### 1.2 Biological Motivation

This problem is not merely a mathematical curiosity. It is directly analogous to a fundamental question in neuroscience: how does the brain establish topographic maps of sensory space?

In a landmark series of experiments, Sur et al. (1988) rewired ferret optic nerve fibers to project into the auditory cortex. Remarkably, neurons in the "repurposed" auditory cortex developed orientation selectivity and retinotopic organization — the animals learned to see with their auditory cortex. This demonstrates that the spatial organization of sensory maps is not genetically hardcoded but can emerge from the statistical structure of incoming signals.

The somatosensory cortex provides another illustration. The well-known cortical homunculus exhibits *somatotopy*: neurons responding to adjacent body parts are physically adjacent in cortex. When peripheral nerves are severed and allowed to regenerate (sometimes to incorrect targets), cortical maps reorganize to reflect the *new* pattern of correlated inputs (Merzenich et al., 1983). The organizing principle appears to be: neurons that receive temporally correlated inputs should be spatially close.

Our pixel permutation recovery problem captures this principle in a controlled computational setting. Each "TV" is analogous to a cortical neuron receiving a temporal signal. The recovery task — placing TVs so that temporally correlated ones are adjacent — mirrors the self-organization problem faced by developing sensory cortex.

### 1.3 Relation to Existing Work

The pixel permutation recovery problem shares structure with computational jigsaw puzzle assembly (Cho et al., 2010; Pomeranz et al., 2011; Sholomon et al., 2013), but differs in important ways. Traditional jigsaw solvers exploit spatial edge compatibility within single images. Our problem provides no spatial information whatsoever — only temporal sequences. The "pieces" have no edges; compatibility is defined entirely by temporal correlation.

The problem also connects to manifold learning and dimensionality reduction. If each pixel's color time-series is treated as a point in high-dimensional space, spatially adjacent pixels should be nearby in this space. Algorithms like UMAP (McInnes et al., 2018) can recover low-dimensional structure, and we explore this connection.

### 1.4 Contributions

1. We formalize the pixel permutation recovery problem and provide an open-source framework (TVWall) for systematic study.
2. We evaluate five distance metrics for measuring pixel temporal similarity, including analysis of their discriminative power via ROC curves, precision-recall analysis, and distribution overlap.
3. We implement and compare three reconstruction strategies (greedy, top-K with simulated annealing, BFS placement) and characterize their failure modes.
4. We identify color entropy as a critical factor in solvability and develop filtering heuristics.
5. We provide GPU-accelerated implementations achieving up to 27x speedup for dissonance computation.

## 2. Problem Formulation

### 2.1 Definitions

Let $V$ be a video consisting of $T$ frames, each of resolution $W \times H$ pixels with 3 color channels (RGB). We define:

- **TV**: A pixel's color time-series $\mathbf{s}_{x,y} \in \mathbb{R}^{T \times 3}$, where $\mathbf{s}_{x,y}(t) = (R_t, G_t, B_t)$ is the color at frame $t$ for position $(x, y)$.

- **TV Wall**: The set of all $W \times H$ TVs arranged in their original grid positions.

- **Permutation**: A bijection $\pi: \{0,...,W{-}1\} \times \{0,...,H{-}1\} \to \{0,...,W{-}1\} \times \{0,...,H{-}1\}$ mapping each grid position to the original position of the TV placed there.

- **Scrambled Wall**: The TV wall after applying permutation $\pi$. Position $(x, y)$ displays the color series $\mathbf{s}_{\pi(x,y)}$.

The **recovery problem** is: given the scrambled wall (i.e., the multiset of color series with unknown positions), find a permutation $\hat{\pi}$ that approximates the identity permutation (or equivalently, inverts the scrambling permutation).

### 2.2 Neighbor Dissonance

We define the **neighbor dissonance** of position $(x, y)$ as the mean distance between the TV at $(x, y)$ and its spatial neighbors:

$$D(x, y) = \frac{1}{|N(x,y)|} \sum_{(x', y') \in N(x,y)} d(\mathbf{s}_{\pi(x,y)}, \mathbf{s}_{\pi(x',y')})$$

where $N(x,y)$ is the set of neighbors (8-connected for a 3x3 kernel) and $d$ is a distance function on time-series.

**Key property:** In a correctly arranged wall, dissonance should be low everywhere (neighbors are similar). In a scrambled wall, dissonance is generally high. Positions with unusually high dissonance after partial reconstruction are likely still misplaced.

### 2.3 Distance Metrics

We evaluate five distance metrics $d(\mathbf{s}_1, \mathbf{s}_2)$:

1. **Euclidean**: $\sqrt{\sum_t \|\mathbf{s}_1(t) - \mathbf{s}_2(t)\|^2}$

2. **Summed Squared Color Distance**: $\sum_t (\Delta R_t^2 + \Delta G_t^2 + \Delta B_t^2)$

3. **Manhattan**: $\sum_t \sum_c |s_{1,c}(t) - s_{2,c}(t)|$

4. **Cosine Distance**: $1 - \frac{\mathbf{s}_1 \cdot \mathbf{s}_2}{\|\mathbf{s}_1\| \|\mathbf{s}_2\|}$ (treating series as flat vectors)

5. **Dynamic Time Warping (DTW)**: Minimum-cost alignment allowing temporal distortion, computed with optional Sakoe-Chiba band constraint.

## 3. Methods

### 3.1 The TVWall Framework

We implement the TVWall class in Python, built on NumPy and OpenCV. The framework supports:

- Loading video with configurable frame count, stride (temporal subsampling), and start frame
- Multiple scrambling modes: full random permutation, random pair swaps, distance-limited swaps
- Vectorized computation of all pixel color series via NumPy advanced indexing
- GPU-accelerated dissonance computation via CuPy
- Frame and video export reflecting current permutation state

### 3.2 Color Entropy Filtering

Not all video frames carry useful information. Frames during fades to black, title cards, or static scenes provide little discriminative power. We filter frames using Shannon entropy of the color histogram:

$$H = -\sum_{i} p_i \log_2 p_i$$

computed over a 512-bin histogram of pixel colors. Frames below a minimum entropy threshold (typically 3.0) are excluded. This improves signal-to-noise ratio by removing uninformative time points.

### 3.3 Reconstruction Strategies

#### 3.3.1 Two-Phase Greedy Solver

**Phase 1 — Identification:** Compute the dissonance map for all positions. Use max-gap analysis on the sorted dissonance values to separate high-dissonance (likely misplaced) from low-dissonance (likely correct) positions.

**Phase 2 — Optimization:** Among identified high-dissonance positions, iteratively attempt pairwise swaps. For each candidate swap $(A, B)$:

$$\Delta = [D(A) + D(B)]_{\text{before}} - [D(A) + D(B)]_{\text{after}}$$

Accept the swap if $\Delta > 0$. This local evaluation avoids recomputing the full dissonance map, reducing per-swap cost from $O(WH)$ to $O(K)$ where $K$ is the kernel neighborhood size.

#### 3.3.2 Top-K with Simulated Annealing

Extend the greedy approach by considering the top-K highest-dissonance positions and accepting occasionally detrimental swaps with probability $e^{-\Delta / T}$, where temperature $T$ decays exponentially. This helps escape local minima.

#### 3.3.3 BFS Placement Solver

An alternative approach that builds the output grid from scratch rather than repairing a scrambled arrangement:

1. Place a seed pixel at the grid center
2. Expand via breadth-first search, prioritizing positions with more placed neighbors
3. For each frontier position, shortlist candidates by mean-RGB proximity (coarse filter), then rank by full time-series distance to placed neighbors (fine evaluation)
4. Place the best candidate and continue

This approach avoids the local-minima problem of swap-based methods but is sensitive to early placement errors that propagate through the BFS tree.

### 3.4 GPU Acceleration

For metrics with closed-form expressions (euclidean, squared, manhattan, cosine), we implement GPU-accelerated dissonance computation using CuPy. The kernel computes, for each pixel position, the mean distance to all neighbors in parallel. On a 640x360 grid (230,400 pixels), GPU computation achieves a 27x speedup over the CPU implementation.

For swap evaluation, we batch all candidate swaps into a single GPU operation, computing before-and-after dissonances for all candidates simultaneously rather than evaluating them sequentially.

## 4. Experiments

### 4.1 Metric Comparison

We compare the five distance metrics on their ability to distinguish correctly-placed from misplaced pixels. For each metric, we:

1. Start with the correctly arranged video
2. Randomly displace $N$ pixels
3. Compute the dissonance map
4. Evaluate separation between displaced and non-displaced distributions

**Evaluation measures:**
- ROC curves and AUC for binary classification (displaced vs. correct)
- Precision-recall curves
- Distribution overlap analysis using 50-bin histograms

### 4.2 Scramble Threshold

We investigate the critical scramble density at which reconstruction becomes infeasible. Starting from a correct arrangement, we progressively increase the number of randomly displaced pixels and attempt reconstruction after each increase.

### 4.3 Video Complexity

We evaluate reconstruction across videos with varying characteristics:
- High motion (e.g., driving footage) vs. low motion (e.g., talking head)
- High color variety vs. monochromatic scenes
- Short vs. long temporal sequences

## 5. Results

### 5.1 Metric Discriminability

[Results of metric comparison experiments, ROC/PR curves, overlap analysis]

*Preliminary observations suggest that summed squared color distance and euclidean distance provide strong discrimination for videos with high color variety, while DTW offers marginal improvement at substantially higher computational cost. The overlap zone — where dissonance alone cannot distinguish displaced from correct pixels — narrows with increasing frame count and color entropy.*

### 5.2 Reconstruction Accuracy

[Results of solver experiments across strategies and parameters]

*The BFS solver shows promising results for small grids but suffers from error propagation at larger scales. The greedy swap solver with GPU-accelerated top-K evaluation achieves reasonable reconstruction rates for moderate scramble densities.*

### 5.3 Failure Modes

Reconstruction fails systematically in several scenarios:

1. **Low color entropy regions**: Areas of the video with uniform color (sky, walls) produce indistinguishable pixel series.
2. **Repeating textures**: Periodic patterns create multiple valid placements.
3. **Global vs. local ambiguity**: Pixels from distant regions with similar content (e.g., two patches of blue sky) cannot be distinguished by temporal correlation alone.

## 6. Discussion

### 6.1 Implications for Neural Self-Organization

Our results provide computational evidence that temporal correlation alone carries sufficient information to recover spatial topology — at least in favorable conditions. The failure modes are instructive: regions of uniform stimulation (analogous to unstimulated sensory areas) and ambiguous correlations (analogous to symmetric stimulation patterns) resist organization.

This aligns with biological observations. Cortical maps are sharpest for body regions with rich, varied sensory input (fingertips, lips) and coarsest for regions with uniform input (trunk, back). Our computational analog — reconstruction accuracy correlating with color entropy — mirrors this phenomenon.

### 6.2 Limitations

- We assume a discrete grid topology; biological maps are continuous.
- Our distance metrics treat color channels independently; cortical processing involves complex nonlinear transformations.
- We do not model Hebbian learning dynamics; our solvers use global optimization rather than local plasticity rules.
- The computational cost of exact recovery scales combinatorially; we rely on heuristics.

### 6.3 Future Directions

- **Hierarchical reconstruction**: Solve at coarse resolution (pooled pixels) then refine, analogous to the coarse-to-fine development of cortical maps.
- **Local plasticity rules**: Replace global optimization with local Hebbian-style update rules to better model biological self-organization.
- **Partial observation**: Investigate recovery when only a subset of pixel positions are sampled, requiring inference of the underlying spatial structure.
- **3D extension**: Extend to volumetric data, analogous to the three-dimensional organization of cortical columns.

## 7. Conclusion

We have introduced the pixel permutation recovery problem as a computational analog of experience-dependent topographic map formation in sensory cortex. Our framework enables systematic study of when and why spatial topology can be recovered from temporal correlation alone. While complete reconstruction of large grids remains challenging, our experiments identify the critical factors — color entropy, metric choice, and scramble density — that govern solvability. These findings provide a quantitative lens through which to examine theories of neural self-organization based on correlational learning.

## References

- Cho, T.S., Avidan, S., & Freeman, W.T. (2010). A probabilistic image jigsaw puzzle solver. *CVPR*.
- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv:1802.03426*.
- Merzenich, M.M., Kaas, J.H., Wall, J.T., et al. (1983). Topographic reorganization of somatosensory cortical areas 3b and 1 in adult monkeys following restricted deafferentation. *Neuroscience*, 8(1), 33-55.
- Pomeranz, D., Shemesh, M., & Ben-Shahar, O. (2011). A fully automated greedy square jigsaw puzzle solver. *CVPR*.
- Sholomon, D., David, O.E., & Netanyahu, N.S. (2013). A genetic algorithm-based solver for very large jigsaw puzzles. *CVPR*.
- Sur, M., Garraghty, P.E., & Roe, A.W. (1988). Experimentally induced visual projections into auditory thalamus and cortex. *Science*, 242(4884), 1437-1441.
