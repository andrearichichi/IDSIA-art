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
    ├─► Forward-backward consistency ──────► C_valid_t  [H×W bool, flow reliability]
    │
    ├─► Pose initialization (E-matrix/PnP) ► R_t, t_t  [3×3, 3×1 coarse poses]
    │
    ├─► Bootstrap static prior ────────────► M_static^0  [H×W soft static confidence]
    │   (essential inliers + flow consistency + conservative low-motion prior)
    │
    ├─► SLIDING WINDOW pose refinement ────► R_t*, t_t*  [locally refined poses]
    │   (5-10 frame local BA or PnP refinement on static-only pixels)
    │
    ├─► Occlusion mask computation ────────► M^occluded_t  [H×W bool, visibility]
    │   (forward-backward consistency + optional depth-based Z-buffer)
    │
    ├─► Scale-invariant residual ──────────► E_t(α_t)  [H×W scale-aligned residual]
    │   (estimate α_t on reliable static pixels after occlusion masking)
    │
    ├─► Track-level motion reasoning ──────► Track coords + motion scores
    │   (RAFT trajectories → temporal aggregation → static/dynamic clustering)
    │
    ├─► Dynamic seed extraction ───────────► M^seed_t  [seed regions from motion clusters]
    │
    ├─► SAM 2 propagation ─────────────────► M^sam_t  [H×W binary, per frame]
    │   (SAM refines track-derived seeds, not primary motion source)
    │
    ├─► Soft mask fusion (normalized) ─────► P_dyn_t  [H×W∈[0,1], per frame]
    │   (track-aggregated residual + confidence gating + SAM refinement)
    │
    ├─► Static point tracking ────────────► static_tracks  [point trajectories]
    └─► Dynamic region tracking ──────────► dynamic_tracks  [instance trajectories]
```

### V1 design principles

- Every approximation is explicit and documented
- Each stage saves debug visualizations to disk
- No stage is a black box — every intermediate can be inspected
- Fail loudly rather than silently produce garbage
- **Scale-invariant where possible**: depth ambiguity handled via per-frame scale alignment
- **Track-level aggregation**: motion reasoning operates on short temporal trajectories, not individual pixels
- **Explicit occlusion handling**: discard occluded pixels before computing residuals
- **Failure-aware**: detect and skip degenerate cases (pure rotation, low translation, etc.)
- **SAM as refinement, not discovery**: track clusters are primary motion signal; SAM refines boundaries

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
│   └── frame_buffer.py         # Optional cached frame access for long videos / random access
│
├── depth/
│   ├── depth_anything.py       # DepthAnythingV3 wrapper
│   └── depth_utils.py          # scale normalization, hole filling
│
├── geometry/
│   ├── camera.py               # CameraIntrinsics, projection/backprojection
│   ├── pose_init.py            # PoseInitializer: E-matrix, PnP
│   ├── pose_refinement.py      # SlidingWindowPoseRefiner: local BA, static-only refinement
│   ├── reprojection.py         # Warper: depth-guided frame warping
│   ├── residuals.py            # ResidualComputer: photometric + geometric
│   ├── scale_alignment.py      # ScaleInvariantResidualComputer: per-frame scale via least-squares
│   ├── occlusion.py            # OcclusionMaskComputer: forward-backward + depth-based visibility
│   └── bundle_adjust.py        # BundleAdjuster: static-pixel BA (optional v1)
│
├── segmentation/
│   ├── sam2_wrapper.py         # SAM2VideoPredictor wrapper
│   ├── seed_extractor.py       # DynamicSeedExtractor: seeds from track clusters (not residuals)
│   └── track_clustering.py     # TrackLevelMotionAnalyzer: aggregate trajectory motion into motion scores
│
├── flow/
│   ├── raft_wrapper.py         # RAFT optical flow wrapper
│   ├── flow_utils.py           # flow visualization, consistency checks, forward-backward masking
│   └── trajectory_builder.py   # TrajectoryBuilder: short-term (2-5 frame) flow-based trajectories
│
├── fusion/
│   ├── soft_mask_fusion.py     # SoftMaskFusion: combine E_t, SAM, flow
│   └── temporal_smoother.py    # Optional EMA / temporal regularization utilities used by fusion
│
├── tracking/
│   ├── static_tracker.py       # StaticTracker: 3D reprojection consistency
│   └── dynamic_tracker.py      # DynamicTracker: mask + sparse support pts
│
├── visualization/
│   ├── debug_writer.py         # DebugWriter: saves frames, overlays, videos
│   └── vis_utils.py            # colormap, overlay, arrow drawing utilities
│
├── sfm/
│   └── colmap_bridge.py        # Optional COLMAP / external SfM import for stronger initialization
│
├── pipeline/
│   ├── failure_detection.py    # FailureDetector: degenerate cases, scale reliability, drift checks
│   └── runner.py               # PipelineRunner: orchestrates all stages and iterative refinement
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
- `frame_buffer.py` is optional and only needed when repeated random access / windowed caching becomes important

---

### Milestone 1: Depth estimation (Depth Anything 3)

**Goal**: Run Depth Anything 3 on every frame and save depth maps as float32 `.npy`.

**What Depth Anything 3 provides**:
- Relative depth (affine-invariant): up to unknown scale `s` and shift `t`
- Metric depth (if metric model is used): absolute scale in meters
- For v1, use the metric variant if available. If not, keep relative depth as-is; do not force global consistency here. The main residual stage will estimate a per-frame scale factor during motion reasoning.

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

**Optional depth preprocessing** (for relative depth, mainly for visualization / temporal regularization, not as the main scale-handling mechanism):

```python
# depth/depth_utils.py

def align_depth_scale_to_reference(depth_t: np.ndarray, depth_ref: np.ndarray,
                                    mask: np.ndarray = None) -> tuple[float, float]:
    """
    Estimate scale s and shift b such that s * depth_t + b ≈ depth_ref.
    Uses least-squares on valid (non-zero, non-infinite) pixels.
    This is only an optional preprocessing heuristic for smoothing or visualization;
    the main pipeline should still rely on the scale-invariant residual stage.
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
        Build a coarse initial pose sequence via pairwise chaining:
            T_t = T_{t-1} @ T_{t→t-1}

        IMPORTANT:
        - This stage is only an initialization step.
        - The chained poses are NOT considered final because drift accumulates quickly.
        - All downstream residual computation should use sliding-window refined poses,
          not raw chained poses.

        For stronger initialization, an optional external COLMAP pass can be used before local refinement.
        COLMAP is NOT required for v1, but it is the preferred stronger initializer when available.
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

**CameraIntrinsics**: If intrinsics are unknown, prefer EXIF metadata, device calibration, or COLMAP focal refinement. Use `fx = fy = 0.7 * max(H, W)` only as a rough fallback for debugging or coarse initialization, not as a trusted final calibration.

**Debug outputs**:
- `debug/poses/pose_trajectory.png` — top-down view of camera path
- `debug/poses/frame_{t:05d}_matches.jpg` — keypoint matches with inlier overlay
- Log number of inliers per frame pair; flag pairs with <50 inliers

**Failure modes**:
- Pure rotation (no translation): essential matrix degenerate. Detect via inlier count + translation magnitude.
- Forward motion: depths of all points similar → ill-conditioned. Fallback to homography.
- Scale / pose drift: raw chained poses accumulate error over long sequences. Treat them as coarse initialization only and run sliding-window refinement before using them in residual computation.

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
                                       threshold: float | None = None) -> np.ndarray:
    """
    For each pixel p, warp p forward to p' = p + F_fwd(p),
    then warp back via F_bwd(p') and check |p - warpback| < threshold.

    If threshold is None, set it adaptively from image resolution
    rather than using a fixed 1-pixel rule for every video.

    Returns: consistency_mask [H, W] bool — True = consistent (reliable flow)
    """
    pass
```

**Debug outputs**:
- `debug/flow/frame_{t:05d}_flow.png` — HSV-encoded flow (color = direction, brightness = magnitude)
- `debug/flow/frame_{t:05d}_consistency.png` — binary consistency mask

---

### Milestone 3.5: Occlusion mask computation

**Goal**: Identify and mask out pixels that are occluded or on occlusion boundaries to prevent false dynamic classifications.

**Why critical**: Static pixels in frame_t that become occluded by moving objects in frame_{t+1} will have high reprojection residual despite being static. Without occlusion masking, entire background regions appear dynamic.

```python
# geometry/occlusion.py

class OcclusionMaskComputer:
    def __init__(self, consistency_threshold: float | None = None):
        """
        consistency_threshold: forward-backward consistency threshold in pixels.
        If None, choose it adaptively from image resolution
        (e.g. max(1.0, 0.0015 * image_diagonal)).
        """
        pass

    def compute_fb_consistency_mask(self, flow_fwd: np.ndarray,
                                      flow_bwd: np.ndarray) -> np.ndarray:
        """
        Forward-backward consistency check.
        For each pixel p in frame_t:
            1. Warp forward: p' = p + flow_fwd(p)
            2. Sample flow_bwd at p': F_bwd_warped = bilinear(flow_bwd, p')
            3. Warp back: p_back = p' + F_bwd_warped
            4. Check: |p - p_back| < threshold
        
        Returns: consistency_mask [H, W] bool
                 True = consistent (reliable), False = inconsistent (occluded /boundary)
        """
        pass

    def compute_depth_based_visibility(self, depth_t: np.ndarray,
                                        depth_t1: np.ndarray,
                                        R_rel: np.ndarray, t_rel: np.ndarray,
                                        intrinsics: CameraIntrinsics,
                                        z_buffer_threshold: float = 0.05) -> np.ndarray:
        """
        Depth-based occlusion detection (optional, more expensive).
        
        For each pixel in frame_t:
        1. Project 3D point to frame_{t+1} using depth_t + relative pose
        2. Get depth from frame_{t+1} depth map at projected location
        3. Compare: if depth_t1_sampled << depth_t_reprojected, pixel is occluded
        
        Returns: visibility_mask [H, W] bool
                 True = visible, False = occluded
        """
        pass

    def combine_occlusion_signals(self, fb_consistency_mask: np.ndarray,
                                    visibility_mask: np.ndarray = None,
                                    use_depth_check: bool = False) -> np.ndarray:
        """
        Combine multiple occlusion signals.
        For v1, use FB consistency alone.
        For robustness, AND with depth-based visibility if available.
        
        Returns: final_occlusion_mask [H, W] bool
                 True = NOT occluded (safe to use), False = occluded (exclude)
        """
        pass
```

**Debug outputs**:
- `debug/occlusion/frame_{t:05d}_fb_consistency.png` — binary consistency mask
- `debug/occlusion/frame_{t:05d}_visibility.png` — depth-based visibility (if computed)
- `debug/occlusion/frame_{t:05d}_combined_mask.png` — final occlusion mask

**Key insight**: Pixels with `occlusion_mask=False` should be excluded from residual computation and dynamic seed extraction.

---

### Milestone 3.6: Sliding window pose refinement

**Goal**: Replace pure pairwise pose chaining with locally-refined poses to reduce drift.

**Why critical**: Pairwise chaining accumulates error → reprojection becomes unreliable after ~20-30 frames. Local refinement on static-only pixels provides much better poses for residual computation.

```python
# geometry/pose_refinement.py

class SlidingWindowPoseRefiner:
    def __init__(self, intrinsics: CameraIntrinsics,
                 window_size: int = 7,
                 method: str = "pnp"):
        """
        window_size: number of frames to jointly optimize (5-10 typical).
        The default 7 is simply the midpoint starting choice, not a claimed optimum.
        method: "pnp" (lightweight) or "ba" (bundle adjustment, v2)
        """
        pass

    def refine_window(self, frames: list[np.ndarray],
                       depths: list[np.ndarray],
                       static_masks: list[np.ndarray],
                       initial_poses: list[Pose],
                       ref_frame_idx: int = 0) -> list[Pose]:
        """
        Local refinement of poses within a 5-10 frame window.
        
        Approach (v1 — PnP-based):
        1. Set reference frame (usually center of window) to identity pose
        2. For each other frame in window:
           a. Backproject static pixels from ref frame using depth + ref pose
           b. Track those 3D points to current frame via optical flow or SIFT
           c. Run PnP+RANSAC to refine pose
        
        Approach (v2 — Bundle Adjustment):
        1. Set up residuals: photometric + geometric on static pixels
        2. Optimize poses + (optionally) depths using g2o or Ceres
        3. Return refined poses
        
        Returns: refined_poses [list of Pose, length = window_size]
        """
        pass

    def propagate_refined_poses(self, poses_initial: list[Pose],
                                 poses_refined: list[Pose],
                                 window_idx: int,
                                 window_size: int) -> list[Pose]:
        """
        Blend refined poses from overlapping windows.
        For frames in multiple windows, use weighted average or take best estimate.
        
        Returns: globally_refined_poses [list of Pose, length = num_frames]
        """
        pass
```

**V1 variant (simpler, still effective)**:
- Use only PnP refinement on static pixels, not full BA
- Sliding window of 5–10 frames
- Re-estimate poses once before computing residuals

**V2 variant (more robust)**:
- Full local BA over sliding windows
- Optimize jointly over: poses + (optionally) relative scale factors

**Debug outputs**:
- `debug/poses/refined_trajectory.png` vs initial trajectory
- `debug/poses/window_refinement_report.txt` — which frames improved, by how much

**Key invariant**: Refined poses should have lower reprojection error on static pixels than initial poses.

---

### Milestone 4: Scale-invariant residual computation (CRITICAL)

**Goal**: Compute depth-to-flow residuals with per-frame scale alignment to handle depth ambiguity.

**Critical issue being fixed**: Depth Anything produces relative depth (unknown scale). Raw residual computation = wrong scale → static pixels appear dynamic.

**Bootstrap for the first static estimate**:
Before the first reliable residual is available, the pipeline still needs a conservative static prior to estimate `α_t`.
Use a bootstrap mask built from the intersection of:
- high-confidence essential-matrix / homography inliers,
- forward-backward flow-consistent pixels,
- regions away from strong motion boundaries,
- optionally, low photometric reprojection error under the coarse pose.

This initial `M_static^0` is only a seed. After the first scale-aligned residual is computed, the static prior should be updated iteratively using refined poses + lower residual regions.

**Solution**: Estimate per-frame scale factor α_t that aligns rigid flow prediction to observed flow.

```python
# geometry/scale_alignment.py

class ScaleInvariantResidualComputer:
    def __init__(self, intrinsics: CameraIntrinsics):
        self.warper = Warper(intrinsics)

    def estimate_scale_from_static_region(self, observed_flow: np.ndarray,
                                            depth_t: np.ndarray,
                                            R_rel: np.ndarray, t_rel: np.ndarray,
                                            static_confidence: np.ndarray,
                                            occlusion_mask: np.ndarray | None = None,
                                            flow_reliability_mask: np.ndarray | None = None,
                                            confidence_threshold: float = 0.6) -> float:
        """
        Estimate scale factor α_t such that:
            norm(observed_flow - flow_rigid(α_t * depth_t)) is minimized
        on reliable static pixels.

        In practice, the optimization mask should be:
            reliable_static = static_confidence > threshold
                              AND occlusion_mask
                              AND flow-consistency mask
        so that α_t is never estimated on occluded or highly ambiguous regions.
        
        Algorithm:
        1. Generate coarse depth range [α_min, α_max] (e.g., [0.5, 2.0])
        2. For each candidate α in the range:
           a. Scale depth: D_scaled = α * depth_t
           b. Compute rigid flow: F_rigid(D_scaled, R_rel, t_rel)
           c. Compute residual: E_α = ||F_observed - F_rigid||
           d. Build a boolean optimization mask from static confidence, occlusion, and flow reliability
           e. Score with a true masked mean: loss_α = mean(E_α[optimization_mask]^2)
        3. Return α that minimizes loss_α
        
        Alternative (more robust): least-squares optimization.
        
        Returns: alpha_t [float], estimated scale for this frame pair
        """
        pass

    def compute_scale_aligned_residual(self, observed_flow: np.ndarray,
                                        depth_t: np.ndarray,
                                        R_rel: np.ndarray, t_rel: np.ndarray,
                                        static_confidence: np.ndarray | None = None,
                                        occlusion_mask: np.ndarray | None = None,
                                        flow_reliability_mask: np.ndarray | None = None,
                                        use_adaptive_scale: bool = True) -> dict:
        """
        Compute residuals with scale alignment.
        
        If use_adaptive_scale=True (recommended):
            Estimate α_t from high-confidence static regions using the current static prior
            E_scaled = |F_observed - F_rigid(α_t * depth_t)|
        Else:
            Use α_t = 1 (standard approach, less robust)
        
        Returns: {
            'residual': [H, W] — per-pixel motion inconsistency
            'alpha': float — estimated scale factor
            'valid_mask': [H, W] bool — pixels used for scale estimation
            'rigid_flow_scaled': [H, W, 2] — rigid flow computed from α_t * depth_t (for debugging)
        }
        
        IMPORTANT: This is scale-invariant in the sense that the residual
        magnitude is now roughly invariant to depth scale ambiguity.
        """
        pass
```

**Math formulation**:

For a pixel at depth `d`, unknown scale `α`:
```
P_3d = α * backproject(u, v, d)
P_reprojected = R @ P_3d + t
(u', v') = project(P_reprojected)
F_rigid_α = (u', v') – (u, v)

E_α = ||F_observed – F_rigid_α||^2  (per-pixel)
```

To find α, minimize on static pixels:
```
α* = argmin_α Σ_{static pixels} E_α^2
```

Then use: `E_final = ||F_observed - F_rigid(α* D_da)||`

This makes the residual approximately **scale-invariant** within the range of reasonable depth scales.

**Debug outputs**:
- `debug/residuals/scale_alignment_report.txt` — estimated α_t for each frame pair
- `debug/residuals/frame_{t:05d}_alpha_sweep.png` — loss vs α to visualize scale estimation
- `debug/residuals/frame_{t:05d}_scaled_residual.png` — heatmap of final E_α
- Flag frames where α_t is far from 1.0 (e.g., > 1.5 or < 0.7) as "depth scale uncertain"

**Key insight**: This scales away the depth ambiguity, making reprojection residuals meaningful even without metric scale.

**Module responsibility**:
- `scale_alignment.py` is the **primary pipeline path** for geometric motion residuals used by motion reasoning.
- `residuals.py` contains generic residual utilities (photometric residuals, wrappers, confidence-map helpers, debugging helpers).
- In the actual pipeline, track-level motion analysis should consume the **scale-aligned residual**, not the raw unscaled rigid residual.

---

### Milestone 4.5: Reprojection residuals (core geometric stage)

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
                                        t_rel: np.ndarray,
                                        occlusion_mask: np.ndarray = None) -> np.ndarray:
        """
        GEOMETRIC RESIDUAL (preferred over photometric for robustness):
        
        The predicted flow under rigid motion is:
            F_rigid(u,v) = project(R @ backproject(u,v,d) + t) - (u,v)
        
        The observed flow is F_raft(u,v).
        
        Residual = |F_observed - F_rigid|
        
        High residual = pixel moves differently from what camera motion predicts
                      = strong evidence of independent object motion.

        NOTE:
        In the full pipeline, this generic routine is typically wrapped by the
        scale-alignment stage so that `depth_t` is first rescaled and unreliable
        pixels are masked before the residual is consumed downstream.
        
        CRITICAL: Apply occlusion mask to exclude occluded pixels:
            residual[~occlusion_mask] = 0  (mark as unreliable)
        
        This is GEOMETRICALLY CORRECT and is the preferred approach.
        
        Returns: flow_residual [H, W] — magnitude of motion inconsistency
        """
        pass

    def build_motion_confidence_map(self, flow_residual: np.ndarray,
                                     valid_mask: np.ndarray,
                                     occlusion_mask: np.ndarray = None,
                                     sigma: float = 5.0) -> np.ndarray:
        """
        Normalize residuals into a probability-like confidence:
            confidence = 1 - exp(-residual^2 / (2 * sigma^2))
        
        Gaussian smoothing to reduce noise.
        
        IMPORTANT: Set confidence to 0 for occluded pixels:
            confidence[~occlusion_mask] = 0
        
        Returns: motion_confidence [H, W] in [0, 1]
        """
        pass
```

**Debug outputs**:
- `debug/residuals/frame_{t:05d}_photometric.png` — heatmap of photometric error
- `debug/residuals/frame_{t:05d}_flow_residual.png` — heatmap of flow vs rigid residual
- `debug/residuals/frame_{t:05d}_warped.jpg` — warped frame for visual inspection
- `debug/residuals/frame_{t:05d}_warp_comparison.jpg` — side by side: warped | target | diff
- `debug/residuals/frame_{t:05d}_occlusion_masked.png` — residuals with occluded regions zeroed

**Key application of occlusion handling**:
- Exclude occluded pixels from residual → only truly visible/dynamic pixels considered
- This dramatically reduces false positives (occluded static pixels appearing dynamic)
- Difference between "dynamic motion" and "not visible" is now explicit

**Failure modes**:
- Low-texture regions: photometric residuals are unreliable. Use geometric (flow) residuals instead.
- Occlusions: when a pixel in frame_t is occluded in frame_{t+1}, residual will be high even for static pixels. **MITIGATED** by occlusion masking in this version.
- Scale ambiguity: if depth is only relative, t_rel is only up to scale. The reprojection will be correct in direction but not magnitude. Use metric depth or depth + scale-from-flow.

---

### Milestone 5: Track-level motion reasoning (NEW: replaces pure pixel-level thresholding)

**Goal**: Aggregate per-pixel residuals into short-term motion trajectories, then cluster trajectories into static/dynamic groups. This replaces noisy pixel-level thresholding with more stable track-level reasoning.

**Why critical**: Pixel-wise residuals are noisy and unstable. A bright artifact or texture boundary can produce a high residual spike for a single pixel. Temporally aggregating motion evidence over 2-5 frames produces much more robust motion scores.

```python
# flow/trajectory_builder.py

class TrajectoryBuilder:
    def __init__(self, window_size: int = 5):
        """
        window_size: number of frames to use for trajectory aggregation
        """
        pass

    def build_trajectories(self, flows: list[np.ndarray],
                            depths: list[np.ndarray],
                            motion_scores: list[np.ndarray],
                            poses: list[Pose],
                            intrinsics: CameraIntrinsics,
                            sample_grid_spacing: int = 8) -> dict:
        """
        Build short-term (2-5 frame) motion trajectories.
        
        Algorithm:
        1. Initialize trajectory seeds at regular grid points (spacing = 8 pixels)
        2. For each trajectory seed (u, v) at frame t:
           a. Track seed forward through next window_size-1 frames using optical flow
           b. Accumulate motion residuals along trajectory
           c. Compute trajectory-level motion score = mean(residual along trajectory)
           d. Store trajectory: [frame_t] -> [frame_t+1] -> ... -> [frame_t+window_size-1]
                               with positions and per-frame motion scores
        3. Return all trajectories with aggregated motion evidence
        
        Returns: {
            'trajectories': [list of Trajectory objects],
            'per_pixel_aggregated_score': [H, W] — motion score from best trajectory through each pixel
        }
        """
        pass

    def aggregate_trajectories(self, trajectories: list) -> np.ndarray:
        """
        For each pixel, collect motion scores from all trajectories passing through it.
        Average or take max of scores.
        Returns: aggregated_score [H, W] — motion evidence per pixel from trajectories
        """
        pass


# segmentation/track_clustering.py

class TrackLevelMotionAnalyzer:
    def __init__(self, intrinsics: CameraIntrinsics):
        self.trajectory_builder = TrajectoryBuilder(window_size=5)

    def cluster_trajectories(self, trajectories: list,
                               dynamic_sigma_k: float = 3.0) -> dict:
        """
        Cluster trajectories into static and dynamic groups.
        
        Algorithm:
        1. For each trajectory, compute motion score = mean(scale-aligned residual along path)
        2. Estimate a robust static baseline using median / MAD statistics
           (v1) or a mixture model over motion scores (v2)
        3. Separate into:
           - Static cluster: trajectories near the low-motion mode
           - Dynamic candidates: trajectories significantly above the static baseline
        4. For dynamic trajectories, perform spatial clustering (e.g., DBSCAN)
           to group nearby trajectories into coherent motion groups
        
        Returns: {
            'static_trajectories': [list of Trajectory],
            'dynamic_clusters': [list of [Trajectory]], 
                               each element is a list of trajectories in same cluster
            'per_pixel_dynamic_probability': [H, W] — aggregated dynamic probability
        }
        """
        pass

```

**Stage ownership note**:
- **Milestone 5** stops at trajectory construction, temporal aggregation, and static/dynamic clustering.
- **Milestone 5.5** is the only stage that converts dynamic clusters into `SeedRegion` objects for SAM 2.
- This separation avoids duplication: clustering discovers motion groups, seed extraction turns them into promptable regions.

**Key difference from pixel-level approach**:
- **Old**: threshold(motion_confidence_pixel) > τ → connected_components → seeds
- **New**: track_trajectories → cluster_trajectories → dense_per_pixel_from_clusters → threshold → seeds

Benefits:
- Temporal coherence: a single high-residual pixel no longer creates a spurious seed
- Static cluster baseline: compute mean motion score per cluster → can set adaptive threshold
- Explicit motion grouping: all trajectories in same cluster move coherently

**Debug outputs**:
- `debug/tracking/frame_{t:05d}_trajectories.jpg` — RGB with trajectory lines overlaid
- `debug/tracking/trajectory_motion_histogram.png` — distribution of trajectory motion scores
- `debug/tracking/frame_{t:05d}_trajectory_clusters.png` — colored by cluster ID
- `debug/tracking/static_vs_dynamic_split_report.txt` — number of static/dynamic clusters per frame

**Key invariant**: Trajectories in static cluster should have motion_score ≈ 0 (within noise); trajectories in dynamic clusters should have motion_score >> noise level.

---

### Milestone 5.5: Dynamic seed extraction (updated)

**Goal**: Find high-confidence regions that are definitely moving. These become prompts for SAM 2.

**CHANGE FROM V1**: Seeds now come directly from track-level clustering (previous milestone), not from raw pixel thresholding. The extractor converts coherent motion clusters into promptable seed regions for SAM 2.

```python
# segmentation/seed_extractor.py

class DynamicSeedExtractor:
    def __init__(self, config: dict):
        """
        config keys:
            min_seed_area: int — discard tiny seeds (noise)
            max_seeds_per_frame: int — limit SAM 2 prompts
            dilation_radius: int — optional mask expansion around sparse track support
            use_cluster_hulls: bool — build seeds from convex hull / union of trajectory support
        """
        pass

    def extract_seeds(self, dynamic_clusters: list,
                       image_shape: tuple[int, int]) -> list[SeedRegion]:
        """
        Build SAM-ready seeds from coherent dynamic trajectory clusters.

        1. For each dynamic cluster, collect all trajectory samples in image space
        2. Convert sparse support into a dense support region
           (convex hull, dilated rasterization, or union of local disks)
        3. Filter tiny or degenerate clusters by area / aspect ratio
        4. Compute centroid, bounding box, binary mask, and confidence score
        5. Return top-k seeds for SAM 2 prompting

        Returns: list of SeedRegion(centroid, bbox, mask, confidence)
        """
        pass

    def seeds_to_sam_prompts(self, seeds: list[SeedRegion]) -> list[dict]:
        """
        Convert seed regions to SAM 2 point / box / mask prompts.
        Prefer box prompts + optional mask initialization when support is dense.
        Returns: list of {'type': 'box'/'point'/'mask', 'data': ...}
        """
        pass
```

**Debug outputs**:
- `debug/seeds/frame_{t:05d}_seeds.jpg` — RGB frame with seed regions overlaid
- `debug/seeds/frame_{t:05d}_confidence.png` — motion confidence heatmap

---

### Milestone 6: SAM 2 mask propagation (NOW REFINEMENT STAGE)

**Goal**: Refine track-derived seeds into precise, temporally consistent segmentation masks.

**CRITICAL ROLE CHANGE**: In this version, SAM 2 is **NOT** the primary motion discovery mechanism. Instead:
- Track clusters provide the primary motion signal
- SAM 2 refines spatial boundaries and handles edge cases

**Why this change matters**:
- SAM 2 can be confused by texture or lighting changes → not reliable for motion discovery
- Track-level reasoning is more robust and geometrically principled
- SAM 2 is excellent at boundary refinement → perfect for refinement role

```python
# segmentation/sam2_wrapper.py

class SAM2VideoSegmenter:
    def __init__(self, model_cfg: str = "sam2_hiera_large.yaml",
                 checkpoint: str = "sam2_hiera_large.pt",
                 device: str = "cuda"):
        """
        Uses sam2.build_sam and SAM2VideoPredictor.
        See: https://github.com/facebookresearch/segment-anything-2
        
        ROLE: Refinement only. Assumes seeds from track-level clustering are valid.
        """
        pass

    def segment_video(self, frames: list[np.ndarray],
                      seed_prompts_per_frame: dict[int, list],
                      initial_masks: dict[int, np.ndarray] = None) -> dict[int, np.ndarray]:
        """
        Run SAM 2 video predictor on track-derived seeds:
        1. Initialize inference state with all frames
        2. For each key frame with seeds, add prompts (boxes/points from track clusters)
        3. Call propagate_in_video() to get masks for all frames
        
        OPTIONAL: Provide initial_masks from track clustering → SAM 2 refines them
        
        seed_prompts_per_frame: {frame_idx: [list of prompts from track clusters]}
        initial_masks: {frame_idx: [H, W] binary}, if available for guided refinement
        
        Returns: {frame_idx: mask [H, W] binary}
        
        SAM 2 handles:
        - Memory-based propagation forward and backward
        - Multi-object tracking (each seed → separate object)
        - Occlusion handling via appearance memory
        - Boundary refinement (the main value SAM 2 adds)
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
- `debug/sam2/frame_{t:05d}_mask_vs_track_init.jpg` — comparison of SAM 2 refined vs track-derived
- `debug/sam2/masks_video.mp4` — masks rendered as a video

**Key insight**: SAM 2 now improves precision, not discovery. Track clusters were already correct; SAM 2 just makes boundaries cleaner.

---

### Milestone 7: Soft mask fusion (UPDATED: normalized inputs + confidence gating)

**Goal**: Combine track-aggregated residual evidence + track-clustered masks + SAM refinement into a single `P_dyn_t ∈ [0,1]` with robust fusion.

**IMPROVEMENTS IN THIS VERSION**:
- Normalize each input per-frame (not just weighted sum)
- Add confidence gating based on flow/depth reliability
- Adaptive weighting instead of fixed weights

```python
# fusion/soft_mask_fusion.py

class SoftMaskFusion:
    def __init__(self, config: dict):
        """
        config keys:
            w_residual: float — weight for track-clustered residual evidence (default: 1.0)
            w_sam: float — weight for SAM 2 refinement mask (default: 1.5)
            w_flow: float — weight for flow magnitude anomaly (default: 0.3)
            temporal_alpha: float — EMA weight for temporal smoothing (default: 0.3)
            enable_confidence_gating: bool — adaptive weighting based on signal quality (default: True)
        """
        pass

    def normalize_cue(self, cue: np.ndarray, method: str = "percentile") -> np.ndarray:
        """
        Normalize individual cue to [0, 1] range per-frame to remove scale differences.
        
        Methods:
        - 'percentile': scale by 99th percentile (robust to outliers)
        - 'max': scale by maximum value
        - 'std': scale by mean + 3*std (Gaussian normalization), then clip to [0, 1]
        
        Returns: normalized_cue [H, W] in [0, 1]
        """
        pass

    def compute_flow_confidence(self, flow: np.ndarray,
                                fb_consistency_mask: np.ndarray,
                                threshold_variance: float = 5.0) -> np.ndarray:
        """
        Estimate per-pixel confidence in optical flow.
        
        Factors:
        - FB consistency: high consistency → high confidence
        - Flow magnitude: extreme values (< 0.1 or > 100) → low confidence
        - Motion variance in neighborhood: high → motion boundary, lower confidence
        
        Returns: flow_confidence [H, W] in [0, 1]
        """
        pass

    def fuse(self, motion_confidence: np.ndarray,
             sam_mask: np.ndarray,
             flow: np.ndarray = None,
             fb_consistency_mask: np.ndarray = None,
             prev_soft_mask: np.ndarray = None) -> np.ndarray:
        """
        Robust weighted combination with adaptive gating.

        IMPORTANT:
        `motion_confidence` here is assumed to come from the scale-aligned,
        track-aggregated motion signal, not from raw single-frame pixel residuals.
        
        1. Normalize each cue:
           C_residual_norm = normalize(motion_confidence)
           C_sam_norm = normalize(float(sam_mask))
           flow_mag = ||flow||_2
           C_flow_norm = normalize(flow_mag) if flow provided
        
        2. Compute confidence gates:
           conf_flow = compute_flow_confidence(flow, fb_consistency_mask)
           conf_sam = 1.0 (or could be learned uncertainty)
        
        3. Gated fusion:
           score = w_r * C_residual_norm
                 + w_s * C_sam_norm
                 + w_f * C_flow_norm * conf_flow  (downweight low-confidence flow)
        
        4. Threshold and smooth:
           P_dyn = sigmoid(score - 1.5)  (centers threshold at score ≈ 1.5)
           P_dyn_smooth = EMA(P_dyn, prev_P_dyn, alpha=0.3)
        
        Returns: soft_mask [H, W] in [0, 1]
        """
        pass

    def hard_threshold(self, soft_mask: np.ndarray,
                        threshold: float = 0.5) -> np.ndarray:
        """Convert soft mask to binary for downstream use."""
        pass
```

**Key improvements**:
- **Normalization**: removes scale differences between cues → fairer contribution
- **Confidence gating**: low-confidence flow doesn't corrupt fusion → more robust
- **Adaptive weighting**: can adjust based on scene (e.g., high flow → lower weight)
- NO neural network: stays analytical and interpretable
- `temporal_smoother.py` can host the EMA / optional temporal regularization used here

**Debug outputs**:
- `debug/fusion/frame_{t:05d}_soft_mask.png` — soft probability as heatmap
- `debug/fusion/frame_{t:05d}_contributions.png` — 4-panel: residual | SAM | flow | final
- `debug/fusion/frame_{t:05d}_confidence_gates.png` — flow confidence map
- `debug/fusion/normalization_report.txt` — per-cue min/max/mean per frame

---

### Milestone 7.5: Failure-aware logic and safeguards (NEW)

**Goal**: Detect and handle degenerate cases where pipeline components fail or give unreliable outputs.

**Why critical**: These cases occur commonly in real videos (pure rotation, low motion, etc.) and can silently produce garbage if not detected.

```python
# ============================================================
# pipeline/failure_detection.py
# ============================================================

class FailureDetector:
    def __init__(self):
        pass

    def check_pure_rotation_case(self, R_rel: np.ndarray, t_rel: np.ndarray,
                                  translation_threshold: float = 0.01) -> bool:
        """
        Detect pure rotation (no translation).
        Returns: True if translation magnitude < threshold
        
        Action: Skip or downweight depth-based rigid residual for this frame pair.
        Reason: With t ≈ 0, image motion is dominated by rotation and becomes nearly depth-invariant,
                so scale estimation and translation-based motion disambiguation are not informative.
        """
        pass

    def check_low_flow_confidence(self, fb_consistency_mask: np.ndarray,
                                   confidence_threshold: float = 0.5) -> bool:
        """
        Check if most of the image has low flow confidence (occluded, textureless, etc.).
        Returns: True if <50% of pixels are flow-consistent
        
        Action: Downweight flow-based residuals for this frame.
        Reason: Low-quality flow will corrupt motion detection.
        """
        pass

    def check_depth_uncertainty(self, depth: np.ndarray,
                                 variance_threshold: float = 0.05) -> bool:
        """
        Check if depth map has very low variance (nearly flat).
        Returns: True if coefficient of variation < threshold
        
        Action: Mark frame as "unreliable depth".
        Reason: Flat depth = nearly frontal-parallel scene. Geometry becomes ill-conditioned.
        """
        pass

    def check_pose_drift(self, static_reprojection_errors: np.ndarray,
                         error_threshold: float = 1.5) -> bool:
        """
        Check whether pose quality on static regions has degraded.
        Returns: True if the static reprojection error or overlap disagreement exceeds threshold.
        
        Action: Trigger pose refinement window.
        Reason: true drift should be detected from worsening geometric consistency,
                not from total camera displacement since the start of the sequence.
        """
        pass

    def check_low_inlier_ratio(self, num_inliers: int, total_matches: int,
                                inlier_threshold: float = 0.3) -> bool:
        """
        Check if pose estimation had low inlier ratio.
        Returns: True if inlier_ratio < threshold
        
        Action: Mark pose as unreliable; fallback to previous pose or skip frame.
        Reason: Mismatches or dynamic pixels corrupted pose estimation.
        """
        pass

    def check_scale_estimation_reliability(self, alpha_sweep_loss: np.ndarray,
                                           flatness_threshold: float = 1e-3) -> bool:
        """
        Check whether the scale-estimation objective is unreliable.
        Returns: True if the loss-vs-alpha curve is flat, weakly identified, or clearly multi-modal.

        Action: Mark the frame pair as scale-uncertain and skip or strongly downweight
                the depth-based rigid residual for that pair.
        Reason: if alpha is not identifiable, the geometric residual is not trustworthy.
        """
        pass

    def report_failures(self) -> dict:
        """
        Generate report of failure events:
        { 'frame_id': [...], 'failure_type': [...], 'severity': [...] }
        """
        pass
```

**Failure mode checklist**:

| Failure Case | Detection | Action | Severity |
|---|---|---|---|
| Pure rotation / rotation-dominated pair | t_mag < ε | Skip or downweight depth-based rigid residual | HIGH |
| Low flow confidence | <50% consistent | Downweight flow signal | HIGH |
| Flat depth (frontal) | low depth variance | Mark frame unreliable | MEDIUM |
| Pose drift accumulation | cumulative t change > τ | Trigger refinement | MEDIUM |
| Low inlier count | inlier_ratio < 0.3 | Use fallback pose | HIGH |
| Scale estimation unreliable | flat / multimodal alpha loss | Skip or downweight rigid residual | HIGH |
| Occlusion-heavy frame | >50% occluded | Use conservative thresholds | MEDIUM |

**Debug outputs**:
- `debug/failures/failure_report.txt` — per-frame failure flags
- `debug/failures/pose_drift_plot.png` — cumulative pose drift over time
- `debug/failures/inlier_histogram.png` — inlier ratios per frame pair

### Milestone 7.6: Iterative refinement schedule (explicit)

**Goal**: Make the pose ↔ residual ↔ motion loop explicit in the implementation plan.

**Recommended v1 schedule**:
1. **Iteration 0**: coarse poses + bootstrap static prior + occlusion mask + first scale estimate
2. **Iteration 1**: compute scale-aligned residuals → trajectories → clusters → seeds → SAM refinement → fused masks
3. **Pose update**: refine poses using confident static regions from the fused result
4. **Iteration 2**: recompute occlusion, scale-aligned residuals, and trajectories with refined poses
5. Stop after 2–3 iterations or earlier if residual statistics / masks stabilize

**Failure-aware trigger points**:
- run `check_low_inlier_ratio()` right after pairwise pose estimation
- run `check_pose_drift()` on static-region reprojection errors before each new residual pass; if triggered, launch local pose refinement before continuing
- run `check_low_flow_confidence()` and `check_depth_uncertainty()` before trusting the geometric residual for that pair
- run a `check_scale_estimation_reliability()` test after the alpha sweep; if unreliable, skip or downweight that pair in fusion

This iteration loop is part of `PipelineRunner.run()` and is not just an optional note.

---

### Milestone 8: Static-only pose refinement (already integrated)

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

This subsection defines the **generic rigid residual before scale alignment**.
The actual pipeline uses the scale-aligned variant described later as the primary motion signal.

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

**Application in pipeline**: `residual[~occlusion_mask] = 0` before computing motion confidence.

### Scale-invariant residual formulation (CRITICAL FIX)

This is the **main geometric residual used by the pipeline**. The generic rigid residual in `residuals.py` is useful for debugging and visualization, but motion reasoning should be driven by the scale-aligned version described here.

**Problem**: Depth from Depth Anything is per-frame and has unknown absolute scale. If scale is wrong, even static pixels have high residual → false dynamic classification.

**Solution**: Estimate per-frame scale factor α_t that best explains the motion.

**Mathematical derivation**:

Suppose true depth is `D_true = α * D_da` where `D_da` is from Depth Anything.

For a pixel at depth `α * d`, the backprojected 3D point is:
```
P = α * backproject(u, v, d)
```

After transformation:
```
P' = R @ P + t = α * R @ backproject(u, v, d) + t
```

The projected flow depends on α:
```
F_rigid(α) = project(α * R @ backproject(u, v, d) + t) - (u, v)
```

**Key insight**: For high-confidence static pixels, the best α should make `F_rigid(α) ≈ F_observed`.

To find α:
```
α* = argmin_α Σ_{static pixels} ||F_observed - F_rigid(α)||^2
```

This can be solved via:
1. Grid search over [0.5, 2.0] (typical range for relative depth)
2. Least-squares optimization (more precise)

For the first iteration, optimize only on a conservative bootstrap static prior. In later iterations, reuse the updated low-residual static mask from the previous pass.

Once α* is found, use it to rescale depth before recomputing the rigid flow:
```
D_scaled = α* * D_da
E_final = ||F_observed - F_rigid(D_scaled)||
```

This makes the residual **approximately scale-invariant** within reasonable depth ranges.

**Algorithm in code**:

```python
# Estimate scale
alpha = estimate_scale_from_static_region(
    flow_obs, depth, R_rel, t_rel, 
    static_confidence_map,
    confidence_threshold=0.6
)

# Compute scale-aligned residual
F_rigid = compute_rigid_flow(alpha * depth, R_rel, t_rel)
E = norm(F_obs - F_rigid, axis=-1)
```

**Validation**: Plot loss vs α. The loss should have a clear minimum at the true α*, not be flat. If flat, depth scale is unreliable for this frame pair.

**Failure case**: If loss curve is flat or multi-modal, flag frame as "scale uncertain". Use conservative residual threshold for that frame, or skip flow-based motion detection.

### Scale ambiguity handling (older approach, superseded by scale-invariant formulation)

If using relative depth (not metric):
- D_t is correct up to a per-frame scale `s_t` and shift `b_t`
- Backprojected points `P_t` are off by this factor
- The relative translation `t_rel` has scale ambiguity too
- These two ambiguities can partially cancel

**Legacy V1 approximation**: Align depth scale across frames using median of matched SIFT points' depths. This is approximate but sufficient to detect motion.

**Geometric correctness**: The rigidity residual `|F_obs - F_rigid|` is scale-invariant for the rotation component but NOT for the translation component. With wrong scale, even static pixels will have high residual. 

**THIS IS NOW HANDLED** by the scale-invariant formulation above. Static pixels should still have near-zero residual after scale alignment.

---

## 6. Tracking and Motion Analysis Details

### Track-level motion reasoning (CORE NEW COMPONENT)

**Purpose**: Aggregate pixel-level residuals into coherent motion trajectories, then cluster into static/dynamic groups. This replaces noisy pixel thresholding with geometrically-grounded motion grouping.

**Key insight**: A single noisy residual spike at one pixel is unreliable. But a consistent trajectory through multiple frames with coherent motion is strong evidence of actual object motion.

**Algorithm**:

1. **Trajectory initialization**: Place seeds on regular grid (e.g., 8×8 pixel spacing)
2. **Trajectory propagation** (forward through 2-5 frames):
   ```
   For each trajectory seed:
       For frame t to t+window_size:
           Use optical flow to track seed position forward
           Record motion residual at each step
           Accumulate: trajectory_motion_score = mean(residuals_along_path)
   ```
3. **Motion score aggregation**: Collect all trajectories passing through each pixel
4. **Clustering**: 
   - Static cluster: trajectory_motion_score < τ_static
   - Dynamic clusters: trajectory_motion_score > τ_dynamic (spatial clustering of trajectories)
5. **Hand-off to seed extraction**: pass dynamic clusters to Milestone 5.5, which converts them into `SeedRegion` objects for SAM 2

**Advantages over pixel-level thresholding**:
- Temporal coherence built in: spurious spikes are diluted
- Spatial grouping: nearby trajectories with similar motion → same object
- Adaptive threshold: static cluster baseline enables relative threshold
- Explicit distinction: "dynamic" vs "not visible"

**Mathematical formulation**:

For trajectory t = (p_t⁰, p_t¹, …, p_t^n), where each p_t^i is the pixel location at frame i:

```
motion_score(t) = (1/n) Σ_i E_aligned(p_t^i)
```

High motion_score → object is truly moving.
Low motion_score → pixel is static (or noise).
Here `E_aligned` denotes the scale-aligned residual actually used by the pipeline:
`E_aligned = ||F_observed - F_rigid(α_t D_t)||`. 

To separate static from dynamic motion in v1, use a robust threshold from the motion-score distribution, for example:
```
μ_static = median(motion_scores)
σ_static = 1.4826 * MAD(motion_scores)
τ_dynamic = μ_static + k * σ_static   # e.g. k = 2.5 or 3.0
```
A percentile-based rule can still be used as a fallback, but MAD-based thresholding is usually more stable across videos. In v2, replace this with a learned or probabilistic two-mode separation (for example, a mixture model over motion scores).

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

**Note on scale factors**: `α_t` is estimated per frame pair for the residual stage and should not be treated as a single persistent global scene scale. For static tracking, use the locally refined poses and the current frame depth; if relative depth is used, any scale compensation should be applied locally to the relevant transition, not carried forward as one global alpha.

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

**Mitigation**: Use metric depth when possible. Otherwise rely on per-frame scale-aligned residuals, and explicitly detect scale-uncertain pairs when the alpha objective is flat or multi-modal. Sparse SIFT/epipolar cues can still be used as an auxiliary fallback.

### Degeneracy in pose estimation
Essential matrix is degenerate when:
- All matched points are coplanar → use homography instead
- Pure rotation, no translation → cannot estimate depth-based relative pose
- Very small baseline → numerically unstable

**Mitigation**: Always compute and log the translation magnitude. If below threshold, treat the frame pair as rotation-dominated and skip or downweight depth-based rigid residuals for that pair.

### SAM 2 sensitivity to seeds
SAM 2 will happily segment whatever you give it. Noisy seeds → noisy masks.

**Mitigation**: Only prompt with seeds above high-confidence threshold. Accept that early frames may miss some dynamic objects (they'll be caught later as more frames are processed).

### Accumulating pose drift
Chained pairwise poses accumulate error. After 100+ frames, poses can be significantly off.

**Mitigation for v1**: Never use raw chained poses directly for final residuals. Use them only to initialize a short sliding-window refinement stage (5-10 frames). V2 adds stronger local or global BA.

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

Week 2: Motion + Geometry Foundations
    Day 1-2:  raft_wrapper.py + bidirectional flow visualization
    Day 3:    flow_utils.py + forward-backward consistency checks
    Day 4-5:  reprojection.py (backproject, transform, project)
              Test on synthetic data first (known R,t → residual should be ~0)

Week 3: Pose Initialization and Local Refinement
    Day 1-2:  pose_init.py (essential matrix / homography fallback)
    Day 3-4:  pose_refinement.py (sliding-window PnP refinement on static pixels)
    Day 5:    failure_detection.py for low inliers / pure rotation / drift safeguards

Week 4: Bootstrap Static Prior + Scale-Aware Residuals + Occlusion
    Day 1:    bootstrap conservative static prior from pose inliers + flow consistency
    Day 2-3:  occlusion.py (FB consistency first, depth visibility optional)
    Day 4-5:  scale_alignment.py + scale-invariant rigid residual
              validate that static pixels have low residual after scale alignment

Week 5: Track-Level Motion + Seed Generation
    Day 1-2:  trajectory_builder.py + trajectory aggregation
    Day 3:    track_clustering.py + static/dynamic clustering
    Day 4:    seed_extractor.py (cluster-to-seed conversion)
    Day 5:    sam2_wrapper.py as refinement over track-derived seeds

Week 6: Fusion, Tracking, and Polish
    Day 1-2:  soft_mask_fusion.py + confidence gating
    Day 3:    iterate: pose → residuals → trajectories → masks → pose refinement
    Day 4:    static_tracker.py + dynamic_tracker.py
    Day 5:    runner.py, debug outputs, timing profiling, test on 2-3 videos
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
    def __init__(self, min_area: int = 200, max_seeds: int = 10,
                 dilation_radius: int = 5):
        self.min_area = min_area
        self.max_seeds = max_seeds
        self.dilation_radius = dilation_radius

    def extract(self, dynamic_clusters: list, image_shape: tuple[int, int]) -> list:
        """
        Convert coherent dynamic trajectory clusters into SeedRegion objects.
        Each cluster is converted into a dense support mask (for example via
        rasterization + dilation or a convex hull), then summarized by bbox,
        centroid, and confidence.
        """
        seeds = []
        H, W = image_shape
        for cluster in dynamic_clusters:
            # cluster is assumed to contain trajectory samples in image coordinates
            mask = np.zeros((H, W), dtype=bool)
            confidence_values = []
            for traj in cluster:
                for x, y in traj.points:
                    x_i, y_i = int(round(x)), int(round(y))
                    if 0 <= x_i < W and 0 <= y_i < H:
                        mask[y_i, x_i] = True
                confidence_values.append(getattr(traj, 'motion_score', 0.0))

            if mask.sum() == 0:
                continue

            ys, xs = np.where(mask)
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
            if area < self.min_area:
                continue

            centroid = (int(xs.mean()), int(ys.mean()))
            confidence = float(np.mean(confidence_values)) if confidence_values else 0.0
            seeds.append(SeedRegion(centroid=centroid, bbox=bbox,
                                    mask=mask, confidence=confidence))

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
# geometry/scale_alignment.py
# ============================================================

class ScaleInvariantResidualComputer:
    """Estimates per-frame depth scale factor for scale-invariant residuals."""
    
    def __init__(self, intrinsics: CameraIntrinsics):
        self.warper = Warper(intrinsics)
    
    def estimate_scale_from_static_region(self, observed_flow: np.ndarray,
                      depth_t: np.ndarray,
                      R_rel: np.ndarray, t_rel: np.ndarray,
                      static_confidence: np.ndarray,
                      occlusion_mask: np.ndarray | None = None,
                      flow_reliability_mask: np.ndarray | None = None,
                      confidence_threshold: float = 0.6,
                      alpha_range: tuple = (0.5, 2.0),
                      num_samples: int = 20) -> float:
        """
        Estimate scale factor α* that minimizes residual on reliable static pixels.
        Uses grid search over alpha_range with num_samples and a true masked mean over
        static / visible / flow-reliable pixels.
        Returns: optimal_alpha (float)
        """
        pass

    def compute_scale_aligned_residual(self, observed_flow: np.ndarray,
                                      depth_t: np.ndarray,
                                      R_rel: np.ndarray, t_rel: np.ndarray,
                                      static_confidence: np.ndarray | None = None,
                                      occlusion_mask: np.ndarray | None = None,
                                      flow_reliability_mask: np.ndarray | None = None) -> dict:
        """Compute the residual after depth rescaling and masking unreliable pixels."""
        pass


# ============================================================
# geometry/occlusion.py
# ============================================================

class OcclusionMaskComputer:
    """Computes occlusion masks from forward-backward consistency and depth."""
    
    def __init__(self, intrinsics: CameraIntrinsics):
        self.intrinsics = intrinsics
    
    def forward_backward_consistency(self, flow_fwd: np.ndarray,
                                      flow_bwd: np.ndarray,
                                      threshold: float | None = None) -> np.ndarray:
        """
        Compute forward-backward consistency mask using a resolution-aware threshold when None.
        Returns: mask [H, W] bool, True = not occluded
        """
        pass
    
    def depth_based_visibility(self, depth_t: np.ndarray,
                               depth_t1: np.ndarray,
                               R_rel: np.ndarray, t_rel: np.ndarray,
                               z_threshold: float = 0.05) -> np.ndarray:
        """
        Compute visibility mask via depth comparison.
        Returns: mask [H, W] bool, True = visible
        """
        pass


# ============================================================
# geometry/pose_refinement.py
# ============================================================

class SlidingWindowPoseRefiner:
    """Refines poses using sliding window local bundle adjustment or PnP."""
    
    def __init__(self, intrinsics: CameraIntrinsics,
                 window_size: int = 7,
                 method: str = "pnp"):
        self.intrinsics = intrinsics
        self.window_size = window_size
        self.method = method
    
    def refine_window(self, frames: list, depths: list, static_masks: list,
                     initial_poses: list, ref_frame_idx: int = 0) -> list:
        """
        Refine poses within a sliding window using PnP on static pixels.
        Returns: refined_poses (list of Pose)
        """
        pass


# ============================================================
# flow/trajectory_builder.py
# ============================================================

class TrajectoryBuilder:
    """Builds short-term flow trajectories for track-level motion analysis."""
    
    def __init__(self, intrinsics: CameraIntrinsics, window_size: int = 5):
        self.intrinsics = intrinsics
        self.window_size = window_size
    
    def build_trajectories(self, flows: list, depths: list,
                          motion_scores: list, poses: list,
                          sample_grid_spacing: int = 8) -> dict:
        """
        Build trajectories from optical flow.
        Returns: {'trajectories': [...], 'per_pixel_aggregated_score': [...]}
        """
        pass


class TrackLevelMotionAnalyzer:
    """Clusters trajectories into static/dynamic groups. Seed generation happens in DynamicSeedExtractor."""
    
    def __init__(self, intrinsics: CameraIntrinsics):
        self.intrinsics = intrinsics
        self.trajectory_builder = TrajectoryBuilder(intrinsics)
    
    def cluster_trajectories(self, trajectories: list,
                            dynamic_sigma_k: float = 3.0) -> dict:
        """
        Cluster trajectories into static and dynamic clusters using a robust static baseline.
        Returns: clustering result with per-pixel motion probabilities
        """
        pass
    


# ============================================================
# segmentation/track_clustering.py

class TrackLevelMotionAnalyzer:
    """Clusters trajectories into static/dynamic groups. Seed generation happens in DynamicSeedExtractor."""
    
    def __init__(self, intrinsics: CameraIntrinsics):
        self.intrinsics = intrinsics
        self.trajectory_builder = TrajectoryBuilder(intrinsics)
    
    def cluster_trajectories(self, trajectories: list,
                            dynamic_sigma_k: float = 3.0) -> dict:
        """
        Cluster trajectories into static and dynamic clusters using a robust static baseline.
        Returns: clustering result with per-pixel motion probabilities
        """
        pass


# ============================================================
# pipeline/failure_detection.py
# ============================================================

class FailureDetector:
    """Detects degenerate cases and pipeline failures."""
    
    def __init__(self):
        self.failures = {}
    
    def check_pure_rotation_case(self, t_rel: np.ndarray,
                           threshold: float = 0.01) -> bool:
        """Detect a rotation-dominated pair where translation is too small to make
        depth-based rigid residuals informative."""
        return np.linalg.norm(t_rel) < threshold
    
    def check_low_flow_confidence(self, fb_consistency_mask: np.ndarray,
                                  threshold: float = 0.5) -> bool:
        """Check if most of image has low flow confidence."""
        return fb_consistency_mask.mean() < threshold
    
    def check_depth_uncertainty(self, depth: np.ndarray,
                               threshold: float = 0.05) -> bool:
        """Check if depth has very low variance."""
        valid = depth > 0
        if valid.sum() < 10:
            return True
        cv = np.std(depth[valid]) / (np.mean(depth[valid]) + 1e-6)
        return cv < threshold

    def check_pose_drift(self, static_reprojection_errors: np.ndarray,
                         error_threshold: float = 1.5) -> bool:
        """Trigger refinement when static-region reprojection error becomes too large."""
        return np.mean(static_reprojection_errors) > error_threshold

    def check_low_inlier_ratio(self, num_inliers: int, total_matches: int,
                               threshold: float = 0.3) -> bool:
        """Return True when the pose-estimation inlier ratio is too low."""
        return total_matches <= 0 or (num_inliers / max(total_matches, 1)) < threshold

    def check_scale_estimation_reliability(self, alpha_sweep_loss: np.ndarray,
                                           flatness_threshold: float = 1e-3) -> bool:
        """Return True when the alpha objective is too flat or ambiguous to trust."""
        return (alpha_sweep_loss.max() - alpha_sweep_loss.min()) < flatness_threshold


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
        ResidualComputer, ScaleInvariantResidualComputer, OcclusionMaskComputer,
        TrackLevelMotionAnalyzer, DynamicSeedExtractor,
        SAM2VideoSegmenter, SoftMaskFusion, SlidingWindowPoseRefiner, FailureDetector,
        StaticTracker, DynamicTracker, DebugWriter."""
        raise NotImplementedError

    def run(self, video_path: str, output_dir: str):
        """
        Main pipeline (UPDATED):
        
        1. Load video frames
        2. Run depth estimation on all frames
        3. Run optical flow on all consecutive pairs
        4. Compute forward-backward consistency and occlusion masks
        5. Initialize poses
        6. Sliding-window pose refinement (first pass)
        7. Run failure checks on low inliers / drift / rotation-dominated pairs
        8. Compute scale-invariant residuals (per-frame scale alignment)
        9. Build and cluster short-term trajectories (track-level motion)
        10. Extract dynamic seeds from trajectory clusters
        11. Run SAM 2 mask refinement on seeds
        12. Fuse residuals + trajectories + SAM into soft dynamic masks
        13. If masks / residuals improved, run another pose-refinement iteration
        14. Re-check pose drift before any new residual pass
        15. Run tracking (static + dynamic instances)
        16. Save all outputs and debug visualizations
        """
        raise NotImplementedError

    def _run_single_iteration(self, frames, depths, flows, poses):
        """
        One pass of: bootstrap/update static prior → occlusion → scale → residuals → trajectories → seeds → SAM → fusion → refine poses.
        This function is called for Iteration 1, Iteration 2, and optionally Iteration 3.
        Before each call, use FailureDetector.check_pose_drift(static_reprojection_errors)
        to decide whether a new local pose-refinement window is required.
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
| Scale-aligned residual (per-frame α estimation) | **Geometrically sound (NEW)** |
| Forward-backward consistency occlusion masking | Geometrically sound |
| Track-level motion clustering | Geometrically motivated |
| PnP for pose from 2D-3D correspondences | Geometrically correct |
| Essential matrix decomposition | Geometrically correct |
| Sliding window pose refinement | Approximation (local, but sound) |
| Depth scale alignment via least-squares | Approximation (v1 fallback) |
| Chained pairwise poses (no global BA) | Approximation (used only for coarse initialization before local refinement) |
| EMA temporal smoothing | Engineering heuristic |
| Soft mask fusion (normalized + gated) | Engineering choice (robust) |
| FOV estimation from video width | Approximation (v1) |