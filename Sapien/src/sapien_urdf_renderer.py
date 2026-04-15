from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

from camera_utils import camera_pose_matrix, matrix3_to_quaternion_wxyz
from config_types import CameraSpec, ConfigValidationError, RenderJobConfig, RootPoseSpec
from io_utils import (
    ensure_output_layout,
    get_package_versions,
    save_depth_npy,
    save_depth_png_preview,
    save_mask_png,
    save_rgb_png,
    write_frame_values_csv,
    write_json,
    write_yaml,
)

LOGGER = logging.getLogger(__name__)

sapien = None
_sapien_import_error: Exception | None = None
try:
    import sapien.core as sapien  # type: ignore[assignment]
except Exception as exc_primary:  # noqa: BLE001
    try:
        import sapien as sapien  # type: ignore[assignment,no-redef]
    except Exception as exc_fallback:  # noqa: BLE001
        _sapien_import_error = exc_fallback
    else:
        _sapien_import_error = None
else:
    _sapien_import_error = None


class RenderInitializationError(RuntimeError):
    """Raised when SAPIEN renderer cannot initialize."""


class SapienURDFRenderer:
    def __init__(
        self,
        background_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        add_ground: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or LOGGER
        self.background_color = background_color
        self.add_ground = add_ground

        self.engine: Any | None = None
        self.renderer: Any | None = None
        self.scene: Any | None = None
        self.articulation: Any | None = None

        self._camera_cache: dict[str, tuple[Any, Any]] = {}
        self._active_joint_index: dict[str, int] = {}
        self._active_joint_qpos_offset: dict[str, int] = {}
        self._active_joint_obj: dict[str, Any] = {}
        self._articulation_actor_ids: set[int] = set()
        self._urdf_joint_type_map: dict[str, str] = {}

        if sapien is None:
            raise RuntimeError(
                "Failed to import SAPIEN. Install dependencies first "
                "(pip install -r requirements.txt)."
            ) from _sapien_import_error

        self._init_engine_and_scene()

    @staticmethod
    def _resolve_urdf_mesh_path(urdf_path: Path, mesh_ref: str) -> Path:
        mesh_ref = mesh_ref.strip()
        if mesh_ref.startswith("file://"):
            return Path(mesh_ref[len("file://") :]).expanduser()

        if mesh_ref.startswith("package://"):
            rel = mesh_ref[len("package://") :]
            # Try URDF directory and all ancestors as package roots.
            for base in (urdf_path.parent, *urdf_path.parent.parents):
                candidate = (base / rel).resolve()
                if candidate.exists():
                    return candidate
            return (urdf_path.parent / rel).resolve()

        raw = Path(mesh_ref).expanduser()
        if raw.is_absolute():
            return raw
        return (urdf_path.parent / raw).resolve()

    @classmethod
    def _find_missing_urdf_meshes(cls, urdf_path: Path) -> list[str]:
        """
        Parse URDF mesh references and report missing files with clear paths.
        """
        try:
            root = ET.parse(urdf_path).getroot()
        except ET.ParseError:
            # If URDF is not plain XML (e.g., unresolved xacro), skip pre-check.
            return []

        missing: list[str] = []
        seen: set[str] = set()
        for elem in root.iter():
            if not elem.tag.endswith("mesh"):
                continue

            mesh_ref = elem.attrib.get("filename") or elem.attrib.get("url")
            if not mesh_ref:
                continue

            resolved = cls._resolve_urdf_mesh_path(urdf_path, mesh_ref)
            if not resolved.exists():
                key = f"{mesh_ref} -> {resolved}"
                if key not in seen:
                    seen.add(key)
                    missing.append(key)
        return missing

    @staticmethod
    def _parse_urdf_joint_types(urdf_path: Path) -> dict[str, str]:
        mapping: dict[str, str] = {}
        try:
            root = ET.parse(urdf_path).getroot()
        except ET.ParseError:
            return mapping

        for elem in root.iter():
            if not elem.tag.endswith("joint"):
                continue
            name = (elem.attrib.get("name") or "").strip()
            joint_type = (elem.attrib.get("type") or "").strip()
            if name and joint_type:
                mapping[name] = joint_type
        return mapping

    def _get_active_joints(self) -> list[Any]:
        if self.articulation is None:
            raise RuntimeError("Articulation is not loaded")

        if hasattr(self.articulation, "get_active_joints"):
            return list(self.articulation.get_active_joints())

        if hasattr(self.articulation, "get_joints"):
            joints = list(self.articulation.get_joints())
            return [joint for joint in joints if self._safe_joint_dof(joint) > 0]

        raise RuntimeError(
            "Articulation does not expose get_active_joints() or get_joints(). "
            "Cannot discover controllable joints."
        )

    def _init_engine_and_scene(self) -> None:
        try:
            self.engine = sapien.Engine()
            self.renderer = sapien.SapienRenderer(offscreen_only=True)
            self.engine.set_renderer(self.renderer)
            self.scene = self.engine.create_scene()
        except Exception as exc:  # noqa: BLE001
            raise RenderInitializationError(
                "Failed to initialize SAPIEN rendering device. "
                "Check: (1) docker run --gpus all, (2) NVIDIA Container Toolkit installation, "
                "(3) libegl1/libxext6 installed, (4) NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute."
            ) from exc

        self.scene.set_timestep(1.0 / 240.0)
        self._configure_lighting()
        if self.add_ground:
            self.scene.add_ground(0.0)

    def _configure_lighting(self) -> None:
        self.scene.set_ambient_light([0.35, 0.35, 0.35])
        self.scene.add_directional_light([0.7, -0.7, -1.0], [1.2, 1.2, 1.2], shadow=False)
        self.scene.add_directional_light([-0.7, 0.4, -0.8], [0.7, 0.7, 0.7], shadow=False)

    @staticmethod
    def _safe_joint_dof(joint: Any) -> int:
        if hasattr(joint, "get_dof"):
            return int(joint.get_dof())
        limits = SapienURDFRenderer._safe_joint_limits(joint)
        if limits is None:
            return 1
        return int(len(limits))

    @staticmethod
    def _safe_joint_type(joint: Any) -> str:
        if hasattr(joint, "get_type"):
            jt = joint.get_type()
            return str(jt)
        return "unknown"

    def _resolve_joint_type(self, joint: Any, joint_name: str) -> str:
        raw = self._safe_joint_type(joint).strip()
        if raw and raw.lower() not in {"unknown", "none", "null"} and not raw.lstrip("-").isdigit():
            return raw
        return self._urdf_joint_type_map.get(joint_name, "unknown")

    @staticmethod
    def _safe_joint_limits(joint: Any) -> list[list[float]] | None:
        if not hasattr(joint, "get_limits"):
            return None
        try:
            limits = np.asarray(joint.get_limits(), dtype=np.float64)
        except Exception:  # noqa: BLE001
            return None
        if limits.size == 0:
            return None
        if limits.ndim == 1:
            if limits.shape[0] == 2:
                return [[float(limits[0]), float(limits[1])]]
            return [[float(v), float(v)] for v in limits.tolist()]
        if limits.ndim == 2 and limits.shape[1] == 2:
            return limits.astype(float).tolist()
        return None

    @staticmethod
    def _build_pose(position: np.ndarray, quaternion_wxyz: np.ndarray) -> Any:
        return sapien.Pose(position.astype(float).tolist(), quaternion_wxyz.astype(float).tolist())

    def _root_pose_to_sapien_pose(self, root_pose: RootPoseSpec) -> Any:
        position = np.asarray(root_pose.position, dtype=np.float64)
        quat = np.asarray(root_pose.quaternion_wxyz, dtype=np.float64)
        quat_norm = np.linalg.norm(quat)
        if quat_norm <= 1e-12:
            raise ConfigValidationError("root_pose quaternion norm must be > 0")
        quat /= quat_norm
        return self._build_pose(position, quat)

    def _camera_spec_to_sapien_pose(self, camera: CameraSpec) -> Any:
        matrix = camera_pose_matrix(camera)
        rot = matrix[:3, :3]
        pos = matrix[:3, 3]
        quat = matrix3_to_quaternion_wxyz(rot)
        return self._build_pose(pos, quat)

    def load_articulation(self, urdf_path: str, root_pose: RootPoseSpec | None = None) -> None:
        assert self.scene is not None

        urdf = Path(urdf_path).expanduser().resolve()
        if not urdf.exists():
            raise FileNotFoundError(f"URDF path does not exist: {urdf}")

        self._urdf_joint_type_map = self._parse_urdf_joint_types(urdf)

        missing_meshes = self._find_missing_urdf_meshes(urdf)
        if missing_meshes:
            preview = "\n".join(f"  - {item}" for item in missing_meshes[:10])
            extra = ""
            if len(missing_meshes) > 10:
                extra = f"\n  ... and {len(missing_meshes) - 10} more"
            raise FileNotFoundError(
                "URDF references mesh files that are missing inside the container.\n"
                "Ensure all referenced assets are mounted and keep relative paths consistent.\n"
                f"URDF: {urdf}\n"
                "Missing mesh references (URDF ref -> resolved path):\n"
                f"{preview}{extra}"
            )

        loader = self.scene.create_urdf_loader()
        if hasattr(loader, "fix_root_link"):
            loader.fix_root_link = True

        articulation = None
        load_errors: list[Exception] = []
        for loader_fn_name in ("load_kinematic", "load"):
            if hasattr(loader, loader_fn_name):
                loader_fn = getattr(loader, loader_fn_name)
                try:
                    articulation = loader_fn(str(urdf))
                except Exception as exc:  # noqa: BLE001
                    load_errors.append(exc)
                    articulation = None
                if articulation is not None:
                    break

        if articulation is None:
            details = ""
            if load_errors:
                details = f" Last error: {load_errors[-1]}"
            raise RuntimeError(
                "Failed to load URDF. Check that all mesh assets referenced by the URDF exist "
                f"and are reachable from its directory.{details}"
            )

        self.articulation = articulation
        if root_pose is not None:
            pose = self._root_pose_to_sapien_pose(root_pose)
            if hasattr(self.articulation, "set_root_pose"):
                self.articulation.set_root_pose(pose)
            elif hasattr(self.articulation, "set_pose"):
                self.articulation.set_pose(pose)

        self._refresh_joint_cache()
        self._refresh_articulation_actor_ids()
        self._camera_cache.clear()

    def _refresh_joint_cache(self) -> None:
        if self.articulation is None:
            raise RuntimeError("Articulation is not loaded")

        self._active_joint_index.clear()
        self._active_joint_qpos_offset.clear()
        self._active_joint_obj.clear()

        active_joints = self._get_active_joints()
        qpos_offset = 0
        for idx, joint in enumerate(active_joints):
            name = str(joint.get_name())
            if name in self._active_joint_index:
                raise RuntimeError(f"Duplicate active joint name detected: {name}")
            self._active_joint_index[name] = idx
            self._active_joint_qpos_offset[name] = qpos_offset
            self._active_joint_obj[name] = joint
            qpos_offset += max(self._safe_joint_dof(joint), 0)

    def _refresh_articulation_actor_ids(self) -> None:
        if self.articulation is None:
            raise RuntimeError("Articulation is not loaded")

        actor_ids: set[int] = set()
        if hasattr(self.articulation, "get_links"):
            for link in self.articulation.get_links():
                for attr in ("get_id", "get_actor_id"):
                    if hasattr(link, attr):
                        try:
                            actor_ids.add(int(getattr(link, attr)()))
                        except Exception:  # noqa: BLE001
                            pass
                for attr in ("id", "actor_id"):
                    if hasattr(link, attr):
                        try:
                            actor_ids.add(int(getattr(link, attr)))
                        except Exception:  # noqa: BLE001
                            pass

        self._articulation_actor_ids = {v for v in actor_ids if v >= 0}
        if not self._articulation_actor_ids:
            self.logger.warning(
                "Could not extract articulation actor IDs from links. Masking will fallback to segmentation>0."
            )

    def list_active_joints(self) -> list[dict[str, Any]]:
        if self.articulation is None:
            raise RuntimeError("Articulation is not loaded")

        rows: list[dict[str, Any]] = []
        for idx, joint in enumerate(self._get_active_joints()):
            name = str(joint.get_name())
            rows.append(
                {
                    "index": idx,
                    "name": name,
                    "qpos_offset": self._active_joint_qpos_offset.get(name, idx),
                    "type": self._resolve_joint_type(joint, name),
                    "dof": self._safe_joint_dof(joint),
                    "limits": self._safe_joint_limits(joint),
                }
            )
        return rows

    def _validate_joint_for_rendering(self, joint_name: str) -> dict[str, Any]:
        joints = self.list_active_joints()
        match = [row for row in joints if row["name"] == joint_name]
        if not match:
            raise ValueError(f"Joint '{joint_name}' not found among active joints")
        info = match[0]
        if int(info["dof"]) != 1:
            raise ValueError(
                f"Joint '{joint_name}' has DoF={info['dof']}. Only 1-DoF joints are supported in v1"
            )
        return info

    def set_joint_value(self, joint_name: str, value: float) -> None:
        if self.articulation is None:
            raise RuntimeError("Articulation is not loaded")
        if joint_name not in self._active_joint_index:
            raise ValueError(f"Joint '{joint_name}' not found")

        joint = self._active_joint_obj[joint_name]
        dof = self._safe_joint_dof(joint)
        if dof != 1:
            raise ValueError(f"Joint '{joint_name}' has DoF={dof}. Expected 1")

        qpos = np.asarray(self.articulation.get_qpos(), dtype=np.float64)
        idx = self._active_joint_qpos_offset[joint_name]
        if idx < 0 or idx >= qpos.shape[0]:
            raise RuntimeError(
                f"Joint index mapping is invalid for '{joint_name}' (index={idx}, qpos size={qpos.shape[0]})"
            )

        qpos[idx] = float(value)
        self.articulation.set_qpos(qpos)
        self.scene.step()

    def _create_camera_mount(self, name: str) -> Any:
        assert self.scene is not None
        builder = self.scene.create_actor_builder()
        try:
            mount = builder.build_kinematic(name=f"mount_{name}")
        except TypeError:
            mount = builder.build_kinematic()
            if hasattr(mount, "set_name"):
                mount.set_name(f"mount_{name}")
        return mount

    def _create_camera(self, camera_spec: CameraSpec) -> tuple[Any, Any]:
        assert self.scene is not None

        mount = self._create_camera_mount(camera_spec.name)
        fovy = np.deg2rad(camera_spec.fov_y_deg if camera_spec.fov_y_deg is not None else 60.0)
        camera = self.scene.add_mounted_camera(
            camera_spec.name,
            mount,
            sapien.Pose(),
            int(camera_spec.width),
            int(camera_spec.height),
            float(fovy),
            float(camera_spec.near),
            float(camera_spec.far),
        )

        if camera_spec.has_intrinsics() and hasattr(camera, "set_perspective_parameters"):
            try:
                camera.set_perspective_parameters(
                    near=float(camera_spec.near),
                    far=float(camera_spec.far),
                    fx=float(camera_spec.fx),
                    fy=float(camera_spec.fy),
                    cx=float(camera_spec.cx),
                    cy=float(camera_spec.cy),
                    skew=0.0,
                )
            except TypeError:
                camera.set_perspective_parameters(
                    float(camera_spec.near),
                    float(camera_spec.far),
                    float(camera_spec.fx),
                    float(camera_spec.fy),
                    float(camera_spec.cx),
                    float(camera_spec.cy),
                    0.0,
                )

        mount.set_pose(self._camera_spec_to_sapien_pose(camera_spec))
        return camera, mount

    def _ensure_camera(self, camera_spec: CameraSpec) -> tuple[Any, Any]:
        cached = self._camera_cache.get(camera_spec.name)
        if cached is not None:
            camera, mount = cached
            mount.set_pose(self._camera_spec_to_sapien_pose(camera_spec))
            return camera, mount

        camera, mount = self._create_camera(camera_spec)
        self._camera_cache[camera_spec.name] = (camera, mount)
        return camera, mount

    def _get_float_texture(self, camera: Any, name: str) -> np.ndarray:
        tried: list[str] = []
        for candidate in (name, name.lower(), name.upper()):
            if candidate in tried:
                continue
            tried.append(candidate)
            try:
                return np.asarray(camera.get_float_texture(candidate))
            except Exception:  # noqa: BLE001
                continue
        raise RuntimeError(f"Could not fetch float texture '{name}' from camera")

    def _get_uint32_texture(self, camera: Any, name: str) -> np.ndarray:
        tried: list[str] = []
        for candidate in (name, name.lower(), name.upper()):
            if candidate in tried:
                continue
            tried.append(candidate)
            for getter in ("get_uint32_texture", "get_float_texture"):
                if not hasattr(camera, getter):
                    continue
                try:
                    data = np.asarray(getattr(camera, getter)(candidate))
                    if getter == "get_float_texture":
                        data = data.astype(np.uint32)
                    return data
                except Exception:  # noqa: BLE001
                    continue
        raise RuntimeError(f"Could not fetch uint32 texture '{name}' from camera")

    @staticmethod
    def _extract_rgb_uint8(color_rgba: np.ndarray) -> np.ndarray:
        if color_rgba.ndim != 3 or color_rgba.shape[2] < 3:
            raise RuntimeError(f"Unexpected Color texture shape: {color_rgba.shape}")
        rgb = np.clip(color_rgba[..., :3], 0.0, 1.0)
        return (rgb * 255.0).astype(np.uint8)

    @staticmethod
    def _extract_depth_m_from_position(position_tex: np.ndarray) -> np.ndarray:
        if position_tex.ndim != 3 or position_tex.shape[2] < 3:
            raise RuntimeError(f"Unexpected Position texture shape: {position_tex.shape}")

        # In SAPIEN Position texture, camera-space z is typically negative in front of the camera.
        depth_m = -position_tex[..., 2].astype(np.float32)
        invalid = ~np.isfinite(depth_m) | (depth_m <= 0.0)
        depth_m[invalid] = 0.0
        return depth_m

    def _extract_actor_labels(self, seg_tex: np.ndarray) -> np.ndarray:
        if seg_tex.ndim == 2:
            return seg_tex.astype(np.int64)
        if seg_tex.ndim != 3:
            raise RuntimeError(f"Unexpected Segmentation texture shape: {seg_tex.shape}")

        if seg_tex.shape[2] >= 2:
            labels = seg_tex[..., 1]
            if np.any(labels > 0):
                return labels.astype(np.int64)
        return seg_tex[..., 0].astype(np.int64)

    def _extract_mask(self, actor_labels: np.ndarray) -> np.ndarray:
        if self._articulation_actor_ids:
            mask = np.isin(actor_labels, list(self._articulation_actor_ids))
            if np.any(mask):
                return (mask.astype(np.uint8) * 255)

        # Fallback when actor-id mapping is unavailable.
        return ((actor_labels > 0).astype(np.uint8) * 255)

    def render_frame(self, camera_spec: CameraSpec) -> dict[str, np.ndarray]:
        if self.articulation is None:
            raise RuntimeError("Articulation is not loaded")

        camera, mount = self._ensure_camera(camera_spec)
        mount.set_pose(self._camera_spec_to_sapien_pose(camera_spec))

        self.scene.update_render()
        camera.take_picture()

        color_rgba = self._get_float_texture(camera, "Color")
        position_tex = self._get_float_texture(camera, "Position")
        seg_tex = self._get_uint32_texture(camera, "Segmentation")

        rgb = self._extract_rgb_uint8(color_rgba)
        depth_m = self._extract_depth_m_from_position(position_tex)
        actor_labels = self._extract_actor_labels(seg_tex)
        mask = self._extract_mask(actor_labels)

        return {
            "rgb": rgb,
            "depth_m": depth_m,
            "mask": mask,
            "actor_labels": actor_labels,
        }

    def render_sequence(self, cfg: RenderJobConfig) -> None:
        cfg.validate()

        urdf_path = cfg.resolved_urdf_path()
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF path does not exist: {urdf_path}")

        self.load_articulation(str(urdf_path), root_pose=cfg.root_pose)
        joint_info = self._validate_joint_for_rendering(cfg.joint_name)

        num_frames = cfg.resolved_num_frames()
        q_values = np.linspace(float(cfg.q_start), float(cfg.q_end), num_frames, dtype=np.float64)

        camera_names = [cam.name for cam in cfg.cameras]
        dirs = ensure_output_layout(
            output_dir=cfg.resolved_output_dir(),
            camera_names=camera_names,
            save_depth_npy=cfg.save_depth_npy,
            save_depth_png=cfg.save_depth_png,
            save_video=cfg.save_video,
        )

        resolved_cfg = asdict(cfg)
        resolved_cfg["urdf_path"] = str(urdf_path)
        resolved_cfg["output_dir"] = str(cfg.resolved_output_dir())
        resolved_cfg["num_frames_resolved"] = num_frames

        metadata_dir = dirs["metadata"]
        write_yaml(resolved_cfg, metadata_dir / "job_config_resolved.yaml")
        write_json({"cameras": [cam.to_dict() for cam in cfg.cameras]}, metadata_dir / "cameras.json")
        write_json(
            {
                "selected_joint": joint_info,
                "all_active_joints": self.list_active_joints(),
            },
            metadata_dir / "joint_info.json",
        )
        frame_values = [(idx, float(v)) for idx, v in enumerate(q_values)]
        write_frame_values_csv(metadata_dir / "frame_values.csv", frame_values, cfg.joint_name)

        package_versions = get_package_versions(
            [
                "sapien",
                "numpy",
                "imageio",
                "imageio-ffmpeg",
                "opencv-python-headless",
                "Pillow",
                "PyYAML",
                "tqdm",
            ]
        )
        if hasattr(sapien, "__version__"):
            package_versions["sapien_module"] = str(sapien.__version__)
        write_json(package_versions, metadata_dir / "package_versions.json")

        if cfg.dry_run:
            self.logger.info("Dry run enabled. Metadata emitted without rendering frames.")
            return

        video_writers: dict[str, Any] = {}
        if cfg.save_video:
            fps = cfg.resolved_video_fps()
            for cam in cfg.cameras:
                out_path = dirs["video_rgb"] / f"{cam.name}.mp4"
                video_writers[cam.name] = imageio.get_writer(
                    out_path,
                    fps=float(fps),
                    codec=cfg.video_codec,
                    quality=8,
                )

        try:
            progress = tqdm(
                enumerate(q_values),
                total=num_frames,
                desc="Rendering",
                unit="frame",
            )
            for frame_idx, q in progress:
                self.set_joint_value(cfg.joint_name, float(q))
                frame_stem = f"frame_{frame_idx:06d}"

                for camera in cfg.cameras:
                    frame = self.render_frame(camera)

                    rgb_path = dirs["rgb"] / camera.name / f"{frame_stem}.png"
                    save_rgb_png(frame["rgb"], rgb_path)

                    mask_path = dirs["mask"] / camera.name / f"{frame_stem}.png"
                    save_mask_png(frame["mask"], mask_path)

                    if cfg.save_depth_npy:
                        depth_npy_path = dirs["depth_npy"] / camera.name / f"{frame_stem}.npy"
                        save_depth_npy(frame["depth_m"], depth_npy_path)

                    if cfg.save_depth_png:
                        depth_png_path = dirs["depth_png"] / camera.name / f"{frame_stem}.png"
                        save_depth_png_preview(frame["depth_m"], depth_png_path)

                    if cfg.save_video:
                        video_writers[camera.name].append_data(frame["rgb"])

        finally:
            for writer in video_writers.values():
                writer.close()
