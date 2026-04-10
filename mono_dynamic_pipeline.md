# Monocular Video Static/Dynamic Classification Pipeline
## Engineering Implementation Plan

---

## 1. System Overview

### What this builds

A modular research prototype that takes a single raw monocular video and produces:
- Per-frame depth maps (metric-scale estimates from Depth Anything 3)
- Camera pose trajectory (initialized from geometry, refined iteratively)
- Soft per-pixel dynamic probability maps `P_dyn[t] ∈ [0,1]`
- Temporally consistent masks for dynamic instances (via SAM 2)
- Tracked static 3D points and tracked dynamic regions over time

### Core insight (from Shape of Motion)

The key difficulty in monocular dynamic video is that a moving camera observing a static scene and a static camera observing a moving object can produce indistinguishable optical flow. To disambiguate, you need to:
1. Estimate camera motion from static background regions
2. Use those poses to compute what flow *should* look like under the static-world assumption
3. Treat residuals from that prediction as evidence of actual object motion

This pipeline does exactly that, in a geometrically sound way.

### What this is NOT

- Not a clone of Shape of Motion (which uses neural scene representations)
- Not production-ready (research prototype)
- Not real-time (offline processing)
- Not a fake "depth difference" pipeline

---

## 2. Version-1 Pipeline

### High-level data flow

```
Raw video (H×W frames, 24-60 fps)
    │
    ├─► Depth Anything 3 ──────────────────► D_t  [H×W float, per frame]
    │
    ├─► RAFT optical flow ─────────────────► F_{t→t+1}  [H×W×2 float]
    │
    ├─► Pose initialization (E-matrix/PnP) ► R_t, t_t  [3×3, 3×1 per frame]
    │
    ├─► Reprojection residuals ────────────► E_t  [H×W float, motion evidence]
    │
    ├─► Dynamic seed extraction ───────────► M^seed_t  [sparse binary masks]
    │
    ├─► SAM 2 propagation ─────────────────► M^sam_t  [H×W binary, per frame]
    │
    ├─► Soft mask fusion ──────────────────► P_dyn_t  [H×W∈[0,1], per frame]
    │
    ├─► Static-only pose refinement ───────► R_t*, t_t*  [refined poses]
    │
    ├─► Static point tracking ────────────► static_tracks  [point trajectories]
    └─► Dynamic region tracking ──────────► dynamic_tracks  [instance trajectories]
```

### V1 design principles

- Every approximation is explicit and documented
- Each stage saves debug visualizations to disk
- No stage is a black box — every intermediate can be inspected
- Fail loudly rather than silently produce garbage

---

## 3. Repository Structure

```
monocular_dynamic/
│
├── configs/
│   └── default.yaml            # All hyperparameters in one place
│
├── data/
│   ├── video_loader.py         # VideoLoader class, frame sampling
│   └── frame_buffer.py         # FrameBuffer: indexed access, caching
│
├── depth/
│   ├── depth_anything.py       # DepthAnythingV3 wrapper
│   └── depth_utils.py          # scale normalization, hole filling
│
├── geometry/
│   ├── camera.py               # CameraIntrinsics, projection/backprojection
│   ├── pose_init.py            # PoseInitializer: E-matrix, PnP
│   ├── reprojection.py         # Warper: depth-guided frame warping
│   ├── residuals.py            # ResidualComputer: photometric + geometric
│   └── bundle_adjust.py        # BundleAdjuster: static-pixel BA (optional v1)
│
├── segmentation/
│   ├── sam2_wrapper.py         # SAM2VideoPredictor wrapper
│   └── seed_extractor.py       # DynamicSeedExtractor: seeds from residuals
│
├── flow/
│   ├── raft_wrapper.py         # RAFT optical flow wrapper
│   └── flow_utils.py           # flow visualization, consistency checks
│
├── fusion/
│   ├── soft_mask_fusion.py     # SoftMaskFusion: combine E_t, SAM, flow
│   └── temporal_smoother.py    # TemporalSmoother: EMA / CRF over time
│
├── tracking/
│   ├── static_tracker.py       # StaticTracker: 3D reprojection consistency
│   └── dynamic_tracker.py      # DynamicTracker: mask + sparse support pts
│
├── visualization/
│   ├── debug_writer.py         # DebugWriter: saves frames, overlays, videos
│   └── vis_utils.py            # colormap, overlay, arrow drawing utilities
│
├── pipeline/
│   ├── runner.py               # PipelineRunner: orchestrates all stages
│   └── stage_base.py           # PipelineStage base class
│
├── scripts/
│   ├── run_pipeline.py         # Main entrypoint
│   ├── run_depth_only.py       # Debug: just run depth
│   ├── run_pose_init.py        # Debug: just run pose init
│   └── visualize_residuals.py  # Debug: visualize reprojection error
│
└── tests/
    ├── test_geometry.py
    ├── test_reprojection.py
    └── test_fusion.py
```

---

## 4. Step-by-Step Implementation Plan

### Milestone 0: Project skeleton and video loading

**Goal**: Load a video, extract frames, save them, confirm everything works.

```python
# data/video_loader.py

class VideoLoader:
    def __init__(self, video_path: str, target_fps: float = None, 
                 max_frames: int = None, resize: tuple = None):
        """
        Args:
            target_fps: if set, subsample video to this fps
            resize: (H, W) — resize all frames for faster processing
        """
        pass

    def __len__(self) -> int:
        """Number of frames after subsampling."""
        pass

    def __getitem__(self, idx: int) -> dict:
        """Returns: {frame: np.ndarray[H,W,3], timestamp: float, frame_id: int}"""
        pass

    def get_fps(self) -> float:
        pass
```

**Debug output**: Save a grid of 16 sampled frames to `debug/frames_sample.jpg`.

**Failure modes**:
- Video codec issues: use `ffmpeg -i video.mp4 frames/%06d.jpg` as fallback
- Memory: do not load all frames at once; use lazy loading with cv2.VideoCapture

---

### Milestone 1: Depth estimation (Depth Anything 3)

**Goal**: Run Depth Anything 3 on every frame and save depth maps as float32 `.npy`.

**What Depth Anything 3 provides**:
- Relative depth (affine-invariant): up to unknown scale `s` and shift `t`
- Metric depth (if metric model is used): absolute scale in meters
- For v1, use the metric variant if available. If not, use relative + align scale to adjacent frames via RANSAC-median.

**Important**: Depth Anything 3 does NOT provide camera poses. It only gives geometry priors.

```python
# depth/depth_anything.py

class DepthAnythingV3:
    def __init__(self, model_size: str = "vitl", metric: bool = True,
                 device: str = "cuda"):
        """
        model_size: "vits", "vitb", "vitl"
        metric: use metric depth variant (requires specific checkpoint)
        """
        pass

    @torch.no_grad()
    def infer_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Args:
            frame_rgb: [H, W, 3] uint8
        Returns:
            depth: [H, W] float32, in meters (if metric) or relative units
        """
        pass

    @torch.no_grad()
    def infer_batch(self, frames: list[np.ndarray], batch_size: int = 4) -> list[np.ndarray]:
        """Batch inference for efficiency."""
        pass
```

**Depth scale normalization** (for relative depth):

```python
# depth/depth_utils.py

def align_depth_scale_to_reference(depth_t: np.ndarray, depth_ref: np.ndarray,
                                    mask: np.ndarray = None) -> tuple[float, float]:
    """
    Estimate scale s and shift b such that s * depth_t + b ≈ depth_ref.
    Uses least-squares on valid (non-zero, non-infinite) pixels.
    APPROXIMATION: assumes reference frame has correct scale.
    Returns: (scale, shift)
    """
    pass

def fill_depth_holes(depth: np.ndarray) -> np.ndarray:
    """
    Fill zero/nan regions via inpainting or nearest-neighbor.
    Needed because DA3 sometimes produces holes at specular surfaces.
    """
    pass
```

**Debug outputs**:
- `debug/depth/frame_{t:05d}_depth.png` — colormap (turbo or plasma)
- `debug/depth/frame_{t:05d}_depth_overlay.jpg` — depth overlaid on RGB
- Save min/max/mean depth stats to CSV for inspection

**Failure modes**:
- Textureless surfaces → noisy or flat depth. Flag frames where depth variance < threshold.
- Temporal flickering: DA3 runs per-frame independently, depths can jump. Apply simple temporal smoothing (EMA with α=0.3) as v1 approximation.

---

### Milestone 2: Initial camera pose estimation

**Goal**: Get a first estimate of `R_t, t_t` for every frame, up to scale.

**Geometrically correct approach**: Use essential matrix decomposition on matched keypoints.

```python
# geometry/pose_init.py

class PoseInitializer:
    def __init__(self, intrinsics: CameraIntrinsics, method: str = "essential"):
        """
        method: "essential" (geometric, preferred) or "homography" (fallback for planar scenes)
        """
        pass

    def estimate_pairwise_pose(self, frame_t: np.ndarray, frame_t1: np.ndarray,
                                depth_t: np.ndarray = None) -> PosePair:
        """
        1. Detect SIFT/SuperPoint keypoints in both frames
        2. Match with FLANN or LightGlue
        3. Compute essential matrix E with RANSAC
        4. Decompose E into 4 candidate (R, t) pairs
        5. Cheirality check: use depth_t to select correct candidate
        
        Returns: PosePair(R, t, inlier_mask, num_inliers)
        
        NOTE: t is only up to scale unless metric depth is used.
        """
        pass

    def estimate_pose_sequence(self, frames: list, depths: list) -> list[Pose]:
        """
        Chain pairwise poses: T_t = T_{t-1} @ T_{t→t-1}
        Accumulates drift. For v1, this is acceptable.
        For v2, use COLMAP or full BA.
        """
        pass

    def refine_pose_pnp(self, frame_t: np.ndarray, depth_t: np.ndarray,
                         static_mask: np.ndarray, intrinsics: CameraIntrinsics,
                         prev_pose: Pose) -> Pose:
        """
        PnP refinement using only static pixels.
        - Backproject static pixels from frame t-1 using depth and prev_pose
        - Find 2D matches in frame t
        - cv2.solvePnPRansac on (3D, 2D) correspondences
        """
        pass
```

**CameraIntrinsics**: If intrinsics are unknown (raw video), estimate from EXIF, or use `fx = fy = 0.7 * max(H, W)` as a reasonable prior. This is an approximation.

**Debug outputs**:
- `debug/poses/pose_trajectory.png` — top-down view of camera path
- `debug/poses/frame_{t:05d}_matches.jpg` — keypoint matches with inlier overlay
- Log number of inliers per frame pair; flag pairs with <50 inliers

**Failure modes**:
- Pure rotation (no translation): essential matrix degenerate. Detect via inlier count + translation magnitude.
- Forward motion: depths of all points similar → ill-conditioned. Fallback to homography.
- Scale drift: accumulates over long sequences. For v1, acceptable; v2 uses loop closure.

---

### Milestone 3: Optical flow (RAFT)

**Goal**: Dense optical flow `F_{t→t+1}` for every consecutive frame pair.

```python
# flow/raft_wrapper.py

class RAFTFlow:
    def __init__(self, model_path: str, device: str = "cuda", iters: int = 20):
        pass

    @torch.no_grad()
    def compute_flow(self, frame_t: np.ndarray, frame_t1: np.ndarray) -> np.ndarray:
        """
        Returns: flow [H, W, 2] in pixels (dx, dy)
        """
        pass

    @torch.no_grad()
    def compute_bidirectional_flow(self, frame_t, frame_t1) -> tuple:
        """
        Returns: (flow_fwd [H,W,2], flow_bwd [H,W,2])
        Used for forward-backward consistency check.
        """
        pass
```

**Forward-backward consistency**:

```python
# flow/flow_utils.py

def forward_backward_consistency_mask(flow_fwd: np.ndarray, 
                                       flow_bwd: np.ndarray,
                                       threshold: float = 1.0) -> np.ndarray:
    """
    For each pixel p, warp p forward to p' = p + F_fwd(p),
    then warp back via F_bwd(p') and check |p - warpback| < threshold.
    
    Returns: consistency_mask [H, W] bool — True = consistent (reliable flow)
    """
    pass
```

**Debug outputs**:
- `debug/flow/frame_{t:05d}_flow.png` — HSV-encoded flow (color = direction, brightness = magnitude)
- `debug/flow/frame_{t:05d}_consistency.png` — binary consistency mask

---

### Milestone 4: Reprojection residuals (core geometric stage)

**This is the most important stage. Read section 5 for full math.**

```python
# geometry/reprojection.py

class Warper:
    def __init__(self, intrinsics: CameraIntrinsics):
        pass

    def backproject(self, depth_t: np.ndarray) -> np.ndarray:
        """
        For each pixel (u, v) with depth d:
            X = (u - cx) * d / fx
            Y = (v - cy) * d / fy
            Z = d
        Returns: points3D [H, W, 3]
        """
        pass

    def transform_points(self, points3D: np.ndarray, R: np.ndarray, 
                          t: np.ndarray) -> np.ndarray:
        """
        Apply rigid transform: P' = R @ P + t
        Returns: transformed_points [H, W, 3]
        """
        pass

    def project(self, points3D: np.ndarray) -> tuple:
        """
        Project 3D points to 2D:
            u = fx * X/Z + cx
            v = fy * Y/Z + cy
        Returns: (coords_2d [H,W,2], valid_mask [H,W] bool)
        valid_mask = True where Z > 0 and projected point inside image
        """
        pass

    def warp_frame(self, frame_t: np.ndarray, depth_t: np.ndarray,
                    R_rel: np.ndarray, t_rel: np.ndarray) -> tuple:
        """
        Warp frame_t into the coordinate system of frame_{t+1}.
        
        Steps:
        1. Backproject pixels of frame_t using depth_t → 3D points in cam_t
        2. Transform to cam_{t+1} using R_rel, t_rel
        3. Project to get predicted pixel locations in frame_{t+1}
        4. Sample frame_t at those locations (bilinear interpolation)
        
        Returns:
            warped_frame: [H, W, 3] — frame_t warped to frame_{t+1} viewpoint
            valid_mask: [H, W] bool — pixels with valid depth and in-bounds projection
            projected_coords: [H, W, 2] — where each t pixel lands in t+1
        """
        pass
```

```python
# geometry/residuals.py

class ResidualComputer:
    def __init__(self, intrinsics: CameraIntrinsics):
        self.warper = Warper(intrinsics)

    def compute_photometric_residual(self, frame_t: np.ndarray, frame_t1: np.ndarray,
                                      depth_t: np.ndarray, R_rel: np.ndarray,
                                      t_rel: np.ndarray) -> dict:
        """
        Under the static-world assumption, if we warp frame_t into frame_{t+1},
        the reprojected pixels should match frame_{t+1}.
        
        High residual = pixel violates static assumption = likely dynamic.
        
        Returns:
            photometric_error: [H, W] — |warped_t - frame_{t+1}| in luminance
            geometric_error: [H, W] — |projected_coord - flow_coord| if flow available
            valid_mask: [H, W] bool
            warped_frame: [H, W, 3] for debug visualization
        """
        pass

    def compute_flow_vs_rigid_residual(self, flow_observed: np.ndarray,
                                        depth_t: np.ndarray, R_rel: np.ndarray,
                                        t_rel: np.ndarray) -> np.ndarray:
        """
        GEOMETRIC RESIDUAL (preferred over photometric for robustness):
        
        The predicted flow under rigid motion is:
            F_rigid(u,v) = project(R @ backproject(u,v,d) + t) - (u,v)
        
        The observed flow is F_raft(u,v).
        
        Residual = |F_observed - F_rigid|
        
        High residual = pixel moves differently from what camera motion predicts
                      = strong evidence of independent object motion.
        
        This is GEOMETRICALLY CORRECT and is the preferred approach.
        
        Returns: flow_residual [H, W] — magnitude of motion inconsistency
        """
        pass

    def build_motion_confidence_map(self, flow_residual: np.ndarray,
                                     valid_mask: np.ndarray,
                                     sigma: float = 5.0) -> np.ndarray:
        """
        Normalize residuals into a probability-like confidence:
            confidence = 1 - exp(-residual^2 / (2 * sigma^2))
        
        Gaussian smoothing to reduce noise.
        Returns: motion_confidence [H, W] in [0, 1]
        """
        pass
```

**Debug outputs**:
- `debug/residuals/frame_{t:05d}_photometric.png` — heatmap of photometric error
- `debug/residuals/frame_{t:05d}_flow_residual.png` — heatmap of flow vs rigid residual
- `debug/residuals/frame_{t:05d}_warped.jpg` — warped frame for visual inspection
- `debug/residuals/frame_{t:05d}_warp_comparison.jpg` — side by side: warped | target | diff

**Failure modes**:
- Low-texture regions: photometric residuals are unreliable. Use geometric (flow) residuals instead.
- Occlusions: when a pixel in frame_t is occluded in frame_{t+1}, residual will be high even for static pixels. Mitigate with forward-backward consistency mask from RAFT.
- Scale ambiguity: if depth is only relative, t_rel is only up to scale. The reprojection will be correct in direction but not magnitude. Use metric depth or depth + scale-from-flow.

---

### Milestone 5: Dynamic seed extraction

**Goal**: Find high-confidence regions that are definitely moving. These become prompts for SAM 2.

```python
# segmentation/seed_extractor.py

class DynamicSeedExtractor:
    def __init__(self, config: dict):
        """
        config keys:
            residual_threshold: float — motion_confidence > this → dynamic
            min_seed_area: int — discard tiny seeds (noise)
            max_seeds_per_frame: int — limit SAM 2 prompts
            use_flow_consistency: bool — require flow consistency
        """
        pass

    def extract_seeds(self, motion_confidence: np.ndarray,
                       flow_consistency_mask: np.ndarray = None,
                       depth: np.ndarray = None) -> list[SeedRegion]:
        """
        1. Threshold motion_confidence map
        2. Optionally AND with flow_consistency_mask
        3. Connected components → individual seed regions
        4. Filter by area, aspect ratio
        5. Compute per-region: centroid, bounding box, confidence score
        6. Return top-k seeds for SAM 2 prompting
        
        Returns: list of SeedRegion(centroid, bbox, mask, confidence)
        """
        pass

    def seeds_to_sam_prompts(self, seeds: list[SeedRegion]) -> list[dict]:
        """
        Convert seeds to SAM 2 point/box prompts.
        Prefer box prompts for connected regions.
        Returns: list of {'type': 'box'/'point', 'data': ...}
        """
        pass
```

**Debug outputs**:
- `debug/seeds/frame_{t:05d}_seeds.jpg` — RGB frame with seed regions overlaid
- `debug/seeds/frame_{t:05d}_confidence.png` — motion confidence heatmap

---

### Milestone 6: SAM 2 mask propagation

**Goal**: Turn sparse seeds into temporally consistent segmentation masks.

```python
# segmentation/sam2_wrapper.py

class SAM2VideoSegmenter:
    def __init__(self, model_cfg: str = "sam2_hiera_large.yaml",
                 checkpoint: str = "sam2_hiera_large.pt",
                 device: str = "cuda"):
        """
        Uses sam2.build_sam and SAM2VideoPredictor.
        See: https://github.com/facebookresearch/segment-anything-2
        """
        pass

    def segment_video(self, frames: list[np.ndarray],
                      seed_prompts_per_frame: dict[int, list]) -> dict[int, np.ndarray]:
        """
        Run SAM 2 video predictor:
        1. Initialize inference state with all frames
        2. For each key frame with seeds, add prompts (boxes/points/masks)
        3. Call propagate_in_video() to get masks for all frames
        
        seed_prompts_per_frame: {frame_idx: [list of prompts]}
        
        Returns: {frame_idx: mask [H, W] binary}
        
        SAM 2 handles:
        - Memory-based propagation forward and backward
        - Multi-object tracking (each seed → separate object)
        - Occlusion handling
        """
        pass

    def refine_with_new_seeds(self, inference_state, frame_idx: int,
                               new_prompts: list) -> dict:
        """
        Add new prompts mid-video (e.g., when a new object appears).
        """
        pass
```

**v1 approximation**: For very long videos, segment in overlapping windows and stitch.

**Debug outputs**:
- `debug/sam2/frame_{t:05d}_mask.png` — binary mask overlay
- `debug/sam2/masks_video.mp4` — masks rendered as a video

---

### Milestone 7: Soft mask fusion

**Goal**: Combine residual evidence + SAM masks + flow consistency into a single `P_dyn_t ∈ [0,1]`.

```python
# fusion/soft_mask_fusion.py

class SoftMaskFusion:
    def __init__(self, config: dict):
        """
        config keys:
            w_residual: float — weight for reprojection residual evidence
            w_sam: float — weight for SAM 2 mask
            w_flow: float — weight for flow magnitude anomaly
            temporal_alpha: float — EMA weight for temporal smoothing
        """
        pass

    def fuse(self, motion_confidence: np.ndarray,
             sam_mask: np.ndarray,
             flow_magnitude: np.ndarray = None,
             prev_soft_mask: np.ndarray = None) -> np.ndarray:
        """
        V1 APPROACH: weighted sum (hand-crafted, interpretable):
        
            P_dyn = sigmoid(
                w_r * normalize(motion_confidence)
                + w_s * float(sam_mask)
                + w_f * normalize(flow_magnitude_anomaly)
            )
        
        + EMA temporal smoothing:
            P_dyn_smoothed = alpha * P_dyn + (1 - alpha) * prev_P_dyn
        
        WHY HAND-CRAFTED FOR V1:
        - Each term is interpretable and debuggable
        - No training data needed
        - Easy to tune weights per-scene
        
        V2 OPTION: train a tiny 3-channel → 1 channel conv net on labeled frames.
        
        Returns: soft_mask [H, W] in [0, 1]
        """
        pass

    def hard_threshold(self, soft_mask: np.ndarray, 
                        threshold: float = 0.5) -> np.ndarray:
        """Convert soft mask to binary for downstream use."""
        pass
```

**Debug outputs**:
- `debug/fusion/frame_{t:05d}_soft_mask.png` — soft probability as heatmap
- `debug/fusion/frame_{t:05d}_contributions.png` — 3-panel: residual | SAM | flow

---

### Milestone 8: Static-only pose refinement

**Goal**: Re-estimate poses using only pixels confidently in the static background.

**Why this matters**: Initial poses (from essential matrix) may be corrupted by dynamic pixels in keypoint matching. Refining on only static pixels gives much cleaner poses.

```python
# geometry/pose_init.py

class PosePairRefiner:
    def __init__(self, intrinsics: CameraIntrinsics):
        pass

    def refine_with_static_mask(self, frame_t: np.ndarray, frame_t1: np.ndarray,
                                  depth_t: np.ndarray, static_mask: np.ndarray,
                                  initial_pose: Pose) -> Pose:
        """
        PnP-based refinement:
        1. Backproject frame_t pixels where static_mask=True using depth_t
           → 3D points P_3d [N, 3] in cam_t coordinate system
        
        2. Match those backprojected points to frame_{t+1} via:
           - Optical flow (track 2D point from t to t+1)
           - Or SIFT matching restricted to static regions
           → 2D observations P_2d_t1 [N, 2]
        
        3. cv2.solvePnPRansac(P_3d, P_2d_t1, K, distCoeffs=None,
                               flags=cv2.SOLVEPNP_ITERATIVE)
        
        4. Return refined (R, t)
        
        APPROXIMATION: Uses depth_t which may have scale ambiguity.
        GEOMETRICALLY CORRECT: PnP is the correct algorithm for known-3D to 2D.
        """
        pass
```

**V2 option**: Full bundle adjustment over a sliding window of frames, optimizing only over static pixels. Use g2o or ceres via Python bindings.

**Debug outputs**:
- `debug/poses/refined_trajectory.png` vs initial trajectory
- `debug/poses/frame_{t:05d}_static_pixels.jpg` — pixels used for refinement

---

### Milestone 9: Tracking

See Section 6 for details.

---

## 5. Geometry and Reprojection Details

### Camera model

We use the standard pinhole model:

```
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]

# Backprojection:
X = (u - cx) * d / fx
Y = (v - cy) * d / fy
Z = d

# Projection:
u = fx * X/Z + cx
v = fy * Y/Z + cy
```

### The static-world reprojection test

**Setup**: We have frame_t and frame_{t+1}, depth D_t, and camera poses T_t, T_{t+1} (4×4 homogeneous).

**Relative pose**: `T_rel = T_{t+1}^{-1} @ T_t` (transforms points from cam_t into cam_{t+1})

**For a pixel (u,v) at depth d in frame_t:**

1. Backproject to 3D: `P_t = K_inv @ [u*d, v*d, d]^T = [X, Y, d]^T`
2. Transform to cam_{t+1}: `P_{t+1} = R_rel @ P_t + t_rel`
3. Project: `[u', v', 1]^T ~ K @ P_{t+1}`
4. Predicted pixel location in frame_{t+1}: `(u', v')`

**Rigid flow prediction**:
```
F_rigid(u,v) = (u', v') - (u, v)
```

**Observed flow (RAFT)**:
```
F_observed(u,v) = RAFT(frame_t, frame_{t+1})[u,v]
```

**Motion residual**:
```
E(u,v) = |F_observed(u,v) - F_rigid(u,v)|_2
```

**Pseudocode**:
```python
# For each frame pair (t, t+1):
D_t = depth_model(frame_t)            # [H, W]
F_obs = raft(frame_t, frame_t1)       # [H, W, 2]
R_rel, t_rel = relative_pose(T_t, T_t1)

# Backproject
uu, vv = meshgrid(W, H)               # [H, W] each
X = (uu - cx) * D_t / fx
Y = (vv - cy) * D_t / fy
Z = D_t                               # all [H, W]

# Transform
X_ = R_rel[0,0]*X + R_rel[0,1]*Y + R_rel[0,2]*Z + t_rel[0]
Y_ = R_rel[1,0]*X + R_rel[1,1]*Y + R_rel[1,2]*Z + t_rel[1]
Z_ = R_rel[2,0]*X + R_rel[2,1]*Y + R_rel[2,2]*Z + t_rel[2]

# Project
u_ = fx * X_/Z_ + cx                 # [H, W]
v_ = fy * Y_/Z_ + cy

# Rigid flow
F_rigid = stack([u_ - uu, v_ - vv], axis=-1)   # [H, W, 2]

# Residual
E = norm(F_obs - F_rigid, axis=-1)   # [H, W]

# Valid mask: Z_ > 0 and (u_, v_) inside image
valid = (Z_ > 0) & (0 <= u_) & (u_ < W) & (0 <= v_) & (v_ < H)
E[~valid] = 0
```

### Occlusion masking

When a static background pixel is occluded in frame_{t+1} by a moving object, it will appear to have high residual even though it is static. You must mask these out.

**V1 approach**: Use forward-backward consistency. Pixels where F_fwd + F_bwd[warped] ≠ 0 are likely occluded or on motion boundaries. These are unreliable and should be excluded from dynamic classification.

**V2 approach**: Explicit occlusion estimation via Z-buffer comparison between frame depths.

### Scale ambiguity handling

If using relative depth (not metric):
- D_t is correct up to a per-frame scale `s_t` and shift `b_t`
- Backprojected points `P_t` are off by this factor
- The relative translation `t_rel` has scale ambiguity too
- These two ambiguities can partially cancel

**V1 approximation**: Align depth scale across frames using median of matched SIFT points' depths. This is approximate but sufficient to detect motion.

**Geometric correctness**: The rigidity residual `|F_obs - F_rigid|` is scale-invariant for the rotation component but NOT for the translation component. With wrong scale, even static pixels will have high residual. Always validate by checking that background pixels have near-zero residual.

---

## 6. Tracking Details

### Static point tracking

**Goal**: Track specific 3D points (corresponding to static scene elements) across all frames.

**Strategy**: Pure geometric tracking — no learned features needed.

```python
# tracking/static_tracker.py

class StaticTracker:
    def __init__(self, intrinsics: CameraIntrinsics, min_track_length: int = 5):
        pass

    def initialize_tracks(self, frame_t: np.ndarray, depth_t: np.ndarray,
                           static_mask: np.ndarray, n_points: int = 500) -> list[Track3D]:
        """
        1. Sample N keypoints in static regions (Harris corners or SIFT)
        2. Backproject using depth_t → 3D position in world coordinates
        3. Initialize Track3D objects
        """
        pass

    def propagate_tracks(self, tracks: list[Track3D], poses: list[Pose],
                          frame_t1: np.ndarray, depth_t1: np.ndarray) -> list[Track3D]:
        """
        For each live track with 3D position P_world:
        1. Project P_world into frame_{t+1} using T_{t+1}
        2. Get predicted pixel location (u', v')
        3. Search for matching corner/SIFT feature near (u', v')
        4. Update depth estimate from depth_t1[v', u']
        5. Update 3D position (can smooth with running average)
        6. Mark track as lost if no match found within search radius
        """
        pass
```

**Key invariant**: A correctly tracked static point should have the same world-coordinate 3D position across all frames (up to depth noise).

**Debug**: Visualize track reprojection error over time. Static tracks should have near-zero reprojection error if poses are good.

### Dynamic region tracking

**Goal**: Track individual moving instances (not just their masks) across frames, with identity persistence.

```python
# tracking/dynamic_tracker.py

class DynamicTracker:
    def __init__(self, intrinsics: CameraIntrinsics):
        self.sam2_segmenter = SAM2VideoSegmenter(...)
        pass

    def initialize_instance(self, frame_t: np.ndarray, mask_t: np.ndarray,
                              instance_id: int) -> DynamicInstance:
        """
        1. Extract sparse support points inside the mask (SIFT or Harris)
        2. Estimate instance centroid and bounding box
        3. Store SAM 2 object ID for later re-identification
        Returns: DynamicInstance(id, mask, support_points, centroid, bbox)
        """
        pass

    def propagate_instance(self, instance: DynamicInstance,
                            frame_t1: np.ndarray,
                            sam_mask_t1: np.ndarray = None,
                            flow_t: np.ndarray = None) -> DynamicInstance:
        """
        Multi-cue propagation:
        
        Cue 1 (SAM 2): use propagated SAM 2 mask directly.
        
        Cue 2 (Flow): warp support points by RAFT flow to get predicted location.
        
        Cue 3 (Consistency): check support point flow consistency.
          Points with consistent flow → reliable tracking.
          Points with inconsistent flow → on occlusion boundary, drop.
        
        Final position: weighted merge of SAM 2 mask centroid + flow-warped centroid.
        """
        pass

    def handle_occlusion(self, instance: DynamicInstance, 
                          lost_frames: int) -> bool:
        """
        If instance is occluded for <= max_lost_frames, keep it alive (predict forward).
        Otherwise, terminate the track.
        Returns: True if instance should be kept alive.
        """
        pass
```

### Merging mask tracking and sparse points

The key insight is that SAM 2 handles coarse temporal consistency (does the mask look like the same object?) while sparse support points handle fine-grained position precision (where exactly is the object center?).

```python
def merge_sam_and_flow_tracking(sam_mask: np.ndarray,
                                  flow_warped_centroid: np.ndarray,
                                  flow_confidence: float) -> np.ndarray:
    """
    If flow_confidence is high: trust flow for centroid position,
      use SAM mask for shape.
    If flow_confidence is low: trust SAM mask centroid.
    
    Output: refined instance mask and centroid.
    """
    pass
```

---

## 7. Risks and Pitfalls

### Depth scale ambiguity (critical)
Relative depth from DA3 has unknown scale. If scale is wrong, even background pixels will show high reprojection error, making everything look dynamic.

**Mitigation**: Use metric depth model. Fallback: estimate scale from sparse SIFT matches using epipolar geometry.

### Degeneracy in pose estimation
Essential matrix is degenerate when:
- All matched points are coplanar → use homography instead
- Pure rotation, no translation → cannot estimate depth-based relative pose
- Very small baseline → numerically unstable

**Mitigation**: Always compute and log the translation magnitude. If below threshold, skip reprojection residual for this frame pair.

### SAM 2 sensitivity to seeds
SAM 2 will happily segment whatever you give it. Noisy seeds → noisy masks.

**Mitigation**: Only prompt with seeds above high-confidence threshold. Accept that early frames may miss some dynamic objects (they'll be caught later as more frames are processed).

### Accumulating pose drift
Chained pairwise poses accumulate error. After 100+ frames, poses can be significantly off.

**Mitigation for v1**: Use short window (5-10 frames) for reprojection residuals. V2: sliding window bundle adjustment.

### Photometric residuals vs geometric residuals
Photometric (pixel color) residuals are sensitive to lighting changes, motion blur, and auto-exposure. Geometric (flow vs rigid-flow) residuals are much more robust.

**Recommendation**: Prefer geometric residuals. Use photometric only as a secondary signal.

### The chicken-and-egg problem
Better poses → better dynamic classification.
Better dynamic classification → better poses (via static-only refinement).

**V1 solution**: Run 2-3 iterations of (pose → classification → refine). This is sufficient in practice.

---

## 8. Recommended Implementation Order

```
Week 1: Foundation
    Day 1-2:  video_loader.py + basic frame visualization
    Day 3-4:  depth_anything.py + depth visualization
    Day 5:    camera.py (intrinsics, projection math) + unit tests

Week 2: Geometry
    Day 1-2:  raft_wrapper.py + flow visualization
    Day 3-4:  reprojection.py (backproject, transform, project)
              Test on synthetic data first (known R,t → residual should be ~0)
    Day 5:    pose_init.py (essential matrix, basic pose chaining)

Week 3: Residuals and Seeds
    Day 1-2:  residuals.py — implement and test flow_vs_rigid_residual
              CRITICAL: validate that static pixels have ~0 residual
    Day 3:    seed_extractor.py
    Day 4-5:  sam2_wrapper.py + mask visualization

Week 4: Fusion and Iteration
    Day 1-2:  soft_mask_fusion.py
    Day 3:    pose refinement using static masks
    Day 4-5:  iterate: (depth) → (pose) → (residuals) → (masks) → (refine poses)

Week 5: Tracking and Polish
    Day 1-3:  static_tracker.py + dynamic_tracker.py
    Day 4:    runner.py (full pipeline integration)
    Day 5:    debug outputs, timing profiling, test on 2-3 videos
```

---

## 9. Python Skeleton

```python
# ============================================================
# geometry/camera.py
# ============================================================

import numpy as np
import torch
from dataclasses import dataclass

@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    H: int
    W: int

    @classmethod
    def from_fov(cls, fov_deg: float, H: int, W: int) -> "CameraIntrinsics":
        """Initialize from field-of-view estimate."""
        fx = fy = W / (2 * np.tan(np.deg2rad(fov_deg / 2)))
        return cls(fx=fx, fy=fy, cx=W/2, cy=H/2, H=H, W=W)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]], dtype=np.float64)

    @property
    def K_inv(self) -> np.ndarray:
        return np.linalg.inv(self.K)


@dataclass
class Pose:
    R: np.ndarray   # [3, 3] rotation matrix
    t: np.ndarray   # [3] translation vector
    frame_id: int

    @property
    def T(self) -> np.ndarray:
        """4×4 homogeneous transform."""
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    @classmethod
    def identity(cls, frame_id: int = 0) -> "Pose":
        return cls(R=np.eye(3), t=np.zeros(3), frame_id=frame_id)

    def relative_to(self, other: "Pose") -> "Pose":
        """Return pose of self relative to other (T_rel = T_other^{-1} @ T_self)."""
        T_rel = np.linalg.inv(other.T) @ self.T
        return Pose(R=T_rel[:3,:3], t=T_rel[:3,3], frame_id=self.frame_id)


# ============================================================
# geometry/reprojection.py
# ============================================================

class Warper:
    def __init__(self, intrinsics: CameraIntrinsics):
        self.K = intrinsics
        H, W = intrinsics.H, intrinsics.W
        uu, vv = np.meshgrid(np.arange(W), np.arange(H))
        self.uu = uu.astype(np.float32)  # [H, W]
        self.vv = vv.astype(np.float32)  # [H, W]

    def backproject(self, depth: np.ndarray) -> np.ndarray:
        """
        depth: [H, W] float32
        Returns: points3D [H, W, 3]
        """
        X = (self.uu - self.K.cx) * depth / self.K.fx
        Y = (self.vv - self.K.cy) * depth / self.K.fy
        Z = depth
        return np.stack([X, Y, Z], axis=-1)  # [H, W, 3]

    def transform_points(self, points3D: np.ndarray,
                          R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        points3D: [H, W, 3]
        R: [3, 3], t: [3]
        Returns: [H, W, 3]
        """
        # Reshape to [N, 3], apply R@p + t, reshape back
        H, W = points3D.shape[:2]
        pts = points3D.reshape(-1, 3).T   # [3, N]
        pts_transformed = (R @ pts) + t[:, None]  # [3, N]
        return pts_transformed.T.reshape(H, W, 3)

    def project(self, points3D: np.ndarray) -> tuple:
        """
        points3D: [H, W, 3]
        Returns: (coords2D [H,W,2], valid_mask [H,W] bool)
        """
        X, Y, Z = points3D[..., 0], points3D[..., 1], points3D[..., 2]
        eps = 1e-6
        valid = Z > eps
        Z_safe = np.where(valid, Z, eps)
        u = self.K.fx * X / Z_safe + self.K.cx
        v = self.K.fy * Y / Z_safe + self.K.cy
        in_bounds = (u >= 0) & (u < self.K.W) & (v >= 0) & (v < self.K.H)
        valid = valid & in_bounds
        return np.stack([u, v], axis=-1), valid

    def compute_rigid_flow(self, depth: np.ndarray,
                            R: np.ndarray, t: np.ndarray) -> tuple:
        """
        Compute the optical flow expected under rigid camera motion.
        depth: [H, W]
        R: [3,3], t: [3] (relative pose from t to t+1)
        Returns: (rigid_flow [H,W,2], valid_mask [H,W] bool)
        """
        pts3D = self.backproject(depth)
        pts3D_t1 = self.transform_points(pts3D, R, t)
        coords_t1, valid = self.project(pts3D_t1)
        rigid_flow = coords_t1 - np.stack([self.uu, self.vv], axis=-1)
        return rigid_flow, valid


# ============================================================
# geometry/residuals.py
# ============================================================

class ResidualComputer:
    def __init__(self, intrinsics: CameraIntrinsics):
        self.warper = Warper(intrinsics)

    def flow_vs_rigid_residual(self, observed_flow: np.ndarray,
                                depth_t: np.ndarray,
                                R_rel: np.ndarray,
                                t_rel: np.ndarray,
                                valid_flow_mask: np.ndarray = None) -> np.ndarray:
        """
        Core function: compute |F_observed - F_rigid| per pixel.
        
        observed_flow: [H, W, 2]
        depth_t: [H, W]
        R_rel, t_rel: relative pose from frame t to t+1
        valid_flow_mask: [H, W] bool — reliable flow pixels (from FB consistency)
        
        Returns: residual [H, W] float32
        """
        rigid_flow, valid_proj = self.warper.compute_rigid_flow(depth_t, R_rel, t_rel)
        
        diff = observed_flow - rigid_flow                  # [H, W, 2]
        residual = np.linalg.norm(diff, axis=-1)          # [H, W]
        
        # Zero out invalid regions
        residual[~valid_proj] = 0.0
        if valid_flow_mask is not None:
            residual[~valid_flow_mask] = 0.0
        
        return residual

    def residual_to_motion_confidence(self, residual: np.ndarray,
                                       sigma: float = 3.0,
                                       smooth_sigma: float = 5.0) -> np.ndarray:
        """
        Convert raw residual to soft dynamic probability.
        Uses Gaussian-based conversion: high residual → high dynamic probability.
        
        Returns: confidence [H, W] in [0, 1]
        """
        from scipy.ndimage import gaussian_filter
        confidence = 1.0 - np.exp(-(residual ** 2) / (2 * sigma ** 2))
        if smooth_sigma > 0:
            confidence = gaussian_filter(confidence, sigma=smooth_sigma)
        return confidence.clip(0, 1)


# ============================================================
# segmentation/seed_extractor.py
# ============================================================

from dataclasses import dataclass

@dataclass
class SeedRegion:
    centroid: tuple         # (x, y) in pixel coords
    bbox: tuple             # (x1, y1, x2, y2)
    mask: np.ndarray        # [H, W] binary
    confidence: float       # mean motion confidence in region

class DynamicSeedExtractor:
    def __init__(self, residual_threshold: float = 0.6,
                 min_area: int = 200, max_seeds: int = 10):
        self.threshold = residual_threshold
        self.min_area = min_area
        self.max_seeds = max_seeds

    def extract(self, motion_confidence: np.ndarray,
                 flow_valid_mask: np.ndarray = None) -> list:
        """
        Returns list of SeedRegion, sorted by confidence descending.
        """
        from scipy import ndimage
        
        binary = (motion_confidence > self.threshold).astype(np.uint8)
        if flow_valid_mask is not None:
            binary = binary & flow_valid_mask.astype(np.uint8)
        
        # Morphological cleanup
        struct = ndimage.generate_binary_structure(2, 2)
        binary = ndimage.binary_closing(binary, structure=struct, iterations=3)
        binary = ndimage.binary_opening(binary, structure=struct, iterations=2)
        
        # Connected components
        labeled, n_components = ndimage.label(binary)
        seeds = []
        for i in range(1, n_components + 1):
            comp_mask = labeled == i
            area = comp_mask.sum()
            if area < self.min_area:
                continue
            ys, xs = np.where(comp_mask)
            centroid = (int(xs.mean()), int(ys.mean()))
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            conf = float(motion_confidence[comp_mask].mean())
            seeds.append(SeedRegion(centroid=centroid, bbox=bbox,
                                     mask=comp_mask, confidence=conf))
        
        seeds.sort(key=lambda s: s.confidence, reverse=True)
        return seeds[:self.max_seeds]


# ============================================================
# fusion/soft_mask_fusion.py
# ============================================================

class SoftMaskFusion:
    def __init__(self, w_residual: float = 1.0, w_sam: float = 2.0,
                 w_flow: float = 0.5, temporal_alpha: float = 0.3):
        self.w_residual = w_residual
        self.w_sam = w_sam
        self.w_flow = w_flow
        self.alpha = temporal_alpha

    def fuse(self, motion_confidence: np.ndarray,
             sam_mask: np.ndarray,
             flow_magnitude: np.ndarray = None,
             prev_soft_mask: np.ndarray = None) -> np.ndarray:
        """
        Weighted combination → sigmoid → optional EMA smoothing.
        All inputs should be in [0, 1].
        """
        score = self.w_residual * motion_confidence + self.w_sam * sam_mask.astype(float)
        
        if flow_magnitude is not None:
            # Normalize flow magnitude anomaly
            flow_norm = np.clip(flow_magnitude / (flow_magnitude.max() + 1e-6), 0, 1)
            score += self.w_flow * flow_norm
        
        # Sigmoid to get probability
        soft_mask = 1.0 / (1.0 + np.exp(-score + 1.5))  # offset centers threshold
        
        # Temporal EMA
        if prev_soft_mask is not None:
            soft_mask = self.alpha * soft_mask + (1 - self.alpha) * prev_soft_mask
        
        return soft_mask.clip(0, 1)


# ============================================================
# pipeline/runner.py
# ============================================================

class PipelineRunner:
    def __init__(self, config_path: str):
        """Load config and initialize all modules."""
        import yaml
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self._init_modules()

    def _init_modules(self):
        """Initialize: VideoLoader, DepthAnythingV3, RAFTFlow, PoseInitializer,
        ResidualComputer, DynamicSeedExtractor, SAM2VideoSegmenter,
        SoftMaskFusion, StaticTracker, DynamicTracker, DebugWriter."""
        raise NotImplementedError

    def run(self, video_path: str, output_dir: str):
        """
        Main pipeline:
        
        1. Load video frames
        2. Run depth estimation on all frames
        3. Run optical flow on all consecutive pairs
        4. Initialize poses
        5. Compute reprojection residuals
        6. Extract dynamic seeds
        7. Run SAM 2 mask propagation
        8. Fuse into soft dynamic masks
        9. Refine poses using static masks
        10. Re-run residuals with refined poses (optional 2nd iteration)
        11. Run tracking (static + dynamic)
        12. Save all outputs and debug visualizations
        """
        raise NotImplementedError

    def _run_single_iteration(self, frames, depths, flows, poses):
        """
        One pass of: residuals → seeds → SAM → fusion → refine poses.
        Can be called multiple times to iterate.
        """
        raise NotImplementedError


# ============================================================
# visualization/debug_writer.py
# ============================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

class DebugWriter:
    def __init__(self, output_dir: str, enabled: bool = True):
        self.output_dir = output_dir
        self.enabled = enabled
        os.makedirs(output_dir, exist_ok=True)

    def save_depth(self, depth: np.ndarray, frame_id: int):
        """Save colorized depth map."""
        pass

    def save_flow(self, flow: np.ndarray, frame_id: int):
        """Save HSV-encoded optical flow."""
        pass

    def save_residual(self, residual: np.ndarray, frame_id: int):
        """Save motion confidence as heatmap."""
        pass

    def save_seeds(self, frame_rgb: np.ndarray, seeds: list, frame_id: int):
        """Save RGB frame with seed regions overlaid."""
        pass

    def save_soft_mask(self, soft_mask: np.ndarray, frame_rgb: np.ndarray,
                        frame_id: int):
        """Save soft mask overlay on RGB."""
        pass

    def save_comparison(self, frames: list, titles: list, filename: str):
        """Save side-by-side comparison of multiple images."""
        fig, axes = plt.subplots(1, len(frames), figsize=(5*len(frames), 5))
        for ax, frame, title in zip(axes, frames, titles):
            ax.imshow(frame, cmap='gray' if frame.ndim == 2 else None)
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=100, bbox_inches='tight')
        plt.close()

    def render_tracks(self, frame_rgb: np.ndarray, tracks: list, frame_id: int):
        """Draw tracked points/regions on frame."""
        pass
```

---

## Library Recommendations

| Library | Use | Priority |
|---|---|---|
| `depth-anything-v2` (HuggingFace) | Depth estimation | Essential v1 |
| `segment-anything-2` (Meta) | Dynamic mask propagation | Essential v1 |
| RAFT (`torch_optical_flow`) | Dense optical flow | Essential v1 |
| OpenCV | Keypoint matching, essential matrix, PnP | Essential v1 |
| scipy | Connected components, Gaussian filter | Essential v1 |
| PyTorch | All neural forward passes | Essential v1 |
| matplotlib | Debug visualizations | Essential v1 |
| COLMAP (pycol) | Better pose initialization | Optional v1 |
| SuperPoint + LightGlue | Better keypoint matching | Optional v1 |
| g2o / Ceres (pyg2o) | Bundle adjustment | V2 |
| `open3d` | 3D point cloud visualization | Optional v1 |

---

## What is geometrically correct vs approximated

| Component | Status |
|---|---|
| Backprojection + rigid flow formula | Geometrically correct |
| Flow vs rigid-flow residual as motion evidence | Geometrically correct |
| PnP for pose from 2D-3D correspondences | Geometrically correct |
| Essential matrix decomposition | Geometrically correct |
| Depth scale alignment via median | Approximation (v1) |
| Chained pairwise poses (no BA) | Approximation (v1) |
| EMA temporal smoothing of depths | Approximation (v1) |
| Soft mask fusion weights (hand-tuned) | Engineering choice |
| FOV estimation from video width | Approximation (v1) |