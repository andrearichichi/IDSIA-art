#!/usr/bin/env python3
"""Render one 360-degree turntable RGB video for each URDF."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import imageio.v2 as imageio
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SAPIEN_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SAPIEN_ROOT.parent
SRC_DIR = SAPIEN_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _default_urdf_root() -> Path:
    candidates = [
        PROJECT_ROOT / "sapien_unified",
        Path.cwd() / "sapien_unified",
        Path.cwd().parent / "sapien_unified",
        Path("/workspace") / "sapien_unified",
        Path("/data/assets"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / "sapien_unified"


def _default_output_dir() -> Path:
    env_output = os.environ.get("OUTPUT_DIR")
    if env_output:
        return Path(env_output).expanduser() / "turntable_360"
    if Path("/data/output").exists():
        return Path("/data/output") / "turntable_360"
    return SAPIEN_ROOT / "outputs" / "turntable_360"


@dataclass
class MeshData:
    vertices: np.ndarray
    triangles: np.ndarray
    color_rgb: np.ndarray


def _load_sapien_backend() -> tuple[type[Any], type[Any]]:
    from config_types import CameraSpec  # noqa: PLC0415
    from sapien_urdf_renderer import SapienURDFRenderer  # noqa: PLC0415

    return CameraSpec, SapienURDFRenderer


def _parse_vec(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=np.float64)
    parts = [float(v) for v in text.split()]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 vector values, got: {text}")
    return np.asarray(parts, dtype=np.float64)


def _origin_matrix(origin_elem: ET.Element | None) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if origin_elem is None:
        return transform

    xyz = _parse_vec(origin_elem.attrib.get("xyz"), (0.0, 0.0, 0.0))
    rpy = _parse_vec(origin_elem.attrib.get("rpy"), (0.0, 0.0, 0.0))
    roll, pitch, yaw = rpy.tolist()

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    transform[:3, :3] = rz @ ry @ rx
    transform[:3, 3] = xyz
    return transform


def _child_with_tag_suffix(elem: ET.Element, suffix: str) -> ET.Element | None:
    for child in elem:
        if child.tag.endswith(suffix):
            return child
    return None


def _compute_link_world_transforms(root: ET.Element) -> dict[str, np.ndarray]:
    link_names = [
        str(link.attrib["name"])
        for link in root.iter()
        if link.tag.endswith("link") and link.attrib.get("name")
    ]
    child_links: set[str] = set()
    edges: dict[str, list[tuple[str, np.ndarray]]] = {}

    for joint in root.iter():
        if not joint.tag.endswith("joint"):
            continue
        parent_elem = _child_with_tag_suffix(joint, "parent")
        child_elem = _child_with_tag_suffix(joint, "child")
        if parent_elem is None or child_elem is None:
            continue
        parent_name = parent_elem.attrib.get("link")
        child_name = child_elem.attrib.get("link")
        if not parent_name or not child_name:
            continue
        origin_tf = _origin_matrix(_child_with_tag_suffix(joint, "origin"))
        edges.setdefault(parent_name, []).append((child_name, origin_tf))
        child_links.add(child_name)

    root_links = [name for name in link_names if name not in child_links] or link_names
    transforms: dict[str, np.ndarray] = {name: np.eye(4, dtype=np.float64) for name in root_links}
    queue = list(root_links)
    while queue:
        parent_name = queue.pop(0)
        parent_tf = transforms[parent_name]
        for child_name, joint_tf in edges.get(parent_name, []):
            if child_name in transforms:
                continue
            transforms[child_name] = parent_tf @ joint_tf
            queue.append(child_name)

    for name in link_names:
        transforms.setdefault(name, np.eye(4, dtype=np.float64))
    return transforms


def _resolve_mesh_path(urdf_path: Path, mesh_ref: str) -> Path:
    mesh_ref = mesh_ref.strip()
    if mesh_ref.startswith("file://"):
        return Path(mesh_ref[len("file://") :]).expanduser()
    if mesh_ref.startswith("package://"):
        rel = mesh_ref[len("package://") :]
        for base in (urdf_path.parent, *urdf_path.parent.parents):
            candidate = (base / rel).resolve()
            if candidate.exists():
                return candidate
        return (urdf_path.parent / rel).resolve()

    raw = Path(mesh_ref).expanduser()
    if raw.is_absolute():
        return raw
    return (urdf_path.parent / raw).resolve()


def _transform_points(vertices: np.ndarray, transform: np.ndarray, scale: np.ndarray) -> np.ndarray:
    scaled = vertices * scale
    return scaled @ transform[:3, :3].T + transform[:3, 3]


def _iter_visual_meshes(urdf_path: Path) -> Iterable[tuple[Path, np.ndarray, np.ndarray]]:
    root = ET.parse(urdf_path).getroot()
    link_transforms = _compute_link_world_transforms(root)

    for link in root.iter():
        if not link.tag.endswith("link"):
            continue
        link_name = link.attrib.get("name")
        link_tf = link_transforms.get(str(link_name), np.eye(4, dtype=np.float64))

        for visual in link:
            if not visual.tag.endswith("visual"):
                continue

            visual_tf = np.eye(4, dtype=np.float64)
            mesh_entries: list[tuple[str, np.ndarray]] = []
            for child in visual:
                if child.tag.endswith("origin"):
                    visual_tf = _origin_matrix(child)
                if not child.tag.endswith("geometry"):
                    continue
                for geom_child in child:
                    if not geom_child.tag.endswith("mesh"):
                        continue
                    mesh_ref = geom_child.attrib.get("filename") or geom_child.attrib.get("url")
                    if not mesh_ref:
                        continue
                    scale = _parse_vec(geom_child.attrib.get("scale"), (1.0, 1.0, 1.0))
                    mesh_entries.append((mesh_ref, scale))

            full_tf = link_tf @ visual_tf
            for mesh_ref, scale in mesh_entries:
                yield _resolve_mesh_path(urdf_path, mesh_ref), full_tf, scale


def _read_obj_vertices(path: Path) -> np.ndarray:
    vertices: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise ValueError(f"No OBJ vertices found in {path}")
    return np.asarray(vertices, dtype=np.float64)


def _face_index(token: str, vertex_count: int) -> int:
    raw = token.split("/")[0]
    index = int(raw)
    if index < 0:
        return vertex_count + index
    return index - 1


def _read_obj_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                continue

            if not line.startswith("f "):
                continue
            parts = line.split()[1:]
            if len(parts) < 3:
                continue
            face = [_face_index(token, len(vertices)) for token in parts]
            for idx in range(1, len(face) - 1):
                faces.append((face[0], face[idx], face[idx + 1]))

    if not vertices:
        raise ValueError(f"No OBJ vertices found in {path}")
    if not faces:
        raise ValueError(f"No OBJ faces found in {path}")
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def estimate_urdf_bounds(urdf_path: Path) -> tuple[np.ndarray, np.ndarray]:
    point_sets: list[np.ndarray] = []
    for mesh_path, transform, scale in _iter_visual_meshes(urdf_path):
        if mesh_path.suffix.lower() != ".obj":
            continue
        vertices = _read_obj_vertices(mesh_path)
        point_sets.append(_transform_points(vertices, transform, scale))

    if not point_sets:
        return np.asarray([-0.5, -0.5, -0.5]), np.asarray([0.5, 0.5, 0.5])

    points = np.concatenate(point_sets, axis=0)
    return points.min(axis=0), points.max(axis=0)


def load_urdf_meshes(urdf_path: Path, max_faces_per_mesh: int | None = None) -> list[MeshData]:
    palette = [
        np.asarray([190, 190, 184], dtype=np.float64),
        np.asarray([75, 145, 220], dtype=np.float64),
        np.asarray([219, 138, 73], dtype=np.float64),
        np.asarray([96, 166, 111], dtype=np.float64),
    ]
    meshes: list[MeshData] = []
    for idx, (mesh_path, transform, scale) in enumerate(_iter_visual_meshes(urdf_path)):
        if mesh_path.suffix.lower() != ".obj":
            continue
        vertices, triangles = _read_obj_mesh(mesh_path)
        vertices = _transform_points(vertices, transform, scale)

        if max_faces_per_mesh is not None and triangles.shape[0] > max_faces_per_mesh:
            sample_idx = np.linspace(0, triangles.shape[0] - 1, max_faces_per_mesh, dtype=np.int64)
            triangles = triangles[sample_idx]

        meshes.append(
            MeshData(
                vertices=vertices,
                triangles=triangles,
                color_rgb=palette[idx % len(palette)],
            )
        )

    if not meshes:
        raise ValueError(f"No renderable OBJ visual meshes found in {urdf_path}")
    return meshes


def find_urdfs(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    if root.is_file():
        return [root]
    return sorted(root.glob("*/mobility.urdf"))


def reset_articulation_qpos(renderer: Any) -> None:
    articulation = renderer.articulation
    if articulation is None or not hasattr(articulation, "get_qpos"):
        return

    qpos = np.asarray(articulation.get_qpos(), dtype=np.float64)
    if qpos.size == 0:
        return
    articulation.set_qpos(np.zeros_like(qpos))
    if renderer.scene is not None:
        renderer.scene.step()


def make_sapien_camera(
    camera_spec_cls: type[Any],
    name: str,
    angle_rad: float,
    center: np.ndarray,
    radius: float,
    camera_z: float,
    width: int,
    height: int,
    near: float,
    far: float,
    fov_y_deg: float,
) -> Any:
    position = [
        float(center[0] + radius * math.cos(angle_rad)),
        float(center[1] + radius * math.sin(angle_rad)),
        float(camera_z),
    ]
    return camera_spec_cls(
        name=name,
        width=width,
        height=height,
        near=near,
        far=far,
        fov_y_deg=fov_y_deg,
        position=position,
        target=[float(center[0]), float(center[1]), float(center[2])],
        up=[0.0, 0.0, 1.0],
    )


def _video_and_metadata_paths(urdf_path: Path, args: argparse.Namespace) -> tuple[str, Path, Path]:
    object_name = urdf_path.parent.name if urdf_path.name == "mobility.urdf" else urdf_path.stem
    output_root = Path(args.output_dir).expanduser().resolve()
    video_dir = output_root / "video_rgb"
    metadata_dir = output_root / "metadata"
    video_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    video_path = video_dir / f"{object_name}.mp4"
    return object_name, video_path, metadata_dir / f"{object_name}.json"


def _view_parameters(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float, float]:
    center = (bbox_min + bbox_max) * 0.5
    extent = bbox_max - bbox_min
    max_extent = max(float(np.max(extent)), 1e-6)
    diagonal = max(float(np.linalg.norm(extent)), 1e-6)
    radius = max(float(args.min_radius), diagonal * float(args.radius_scale))
    camera_z = float(center[2] + max(extent[2] * 0.35, diagonal * 0.12, args.min_height_offset))
    far = float(args.far) if args.far is not None else max(10.0, radius + diagonal * 4.0)
    return center, extent, max_extent, diagonal, radius, camera_z, far


def _turntable_angles(args: argparse.Namespace) -> np.ndarray:
    return np.linspace(
        math.radians(float(args.start_angle_deg)),
        math.radians(float(args.start_angle_deg)) + 2.0 * math.pi,
        int(args.frames),
        endpoint=bool(args.include_endpoint),
        dtype=np.float64,
    )


def render_turntable_sapien(urdf_path: Path, args: argparse.Namespace) -> Path:
    camera_spec_cls, renderer_cls = _load_sapien_backend()
    object_name, video_path, metadata_path = _video_and_metadata_paths(urdf_path, args)
    if video_path.exists() and args.skip_existing:
        print(f"[skip] {object_name}: {video_path}")
        return video_path

    bbox_min, bbox_max = estimate_urdf_bounds(urdf_path)
    center, extent, max_extent, diagonal, radius, camera_z, far = _view_parameters(bbox_min, bbox_max, args)

    renderer = renderer_cls(
        background_color=tuple(float(v) for v in args.background_color),
        add_ground=bool(args.add_ground),
    )
    renderer.load_articulation(str(urdf_path))
    reset_articulation_qpos(renderer)

    angles = _turntable_angles(args)

    writer = imageio.get_writer(
        video_path,
        fps=float(args.fps),
        codec=str(args.video_codec),
        quality=int(args.video_quality),
    )
    try:
        for frame_idx, angle in enumerate(angles):
            camera = make_sapien_camera(
                camera_spec_cls,
                name="turntable",
                angle_rad=float(angle),
                center=center,
                radius=radius,
                camera_z=camera_z,
                width=int(args.width),
                height=int(args.height),
                near=float(args.near),
                far=far,
                fov_y_deg=float(args.fov_y_deg),
            )
            frame = renderer.render_frame(camera)
            writer.append_data(frame["rgb"])
            if args.verbose:
                print(f"  frame {frame_idx + 1:04d}/{len(angles)}")
    finally:
        writer.close()

    metadata = {
        "backend": "sapien",
        "urdf_path": str(urdf_path),
        "video_path": str(video_path),
        "frames": int(args.frames),
        "fps": float(args.fps),
        "degrees": 360.0,
        "camera": asdict(
            make_sapien_camera(
                camera_spec_cls,
                name="turntable",
                angle_rad=float(angles[0]),
                center=center,
                radius=radius,
                camera_z=camera_z,
                width=int(args.width),
                height=int(args.height),
                near=float(args.near),
                far=far,
                fov_y_deg=float(args.fov_y_deg),
            )
        ),
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "bbox_extent": extent.tolist(),
        "bbox_max_extent": max_extent,
        "bbox_diagonal": diagonal,
        "radius": radius,
        "camera_z": camera_z,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    print(f"[ok] {object_name}: {video_path}")
    return video_path


def _normalize(vec: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        if fallback is None:
            raise ValueError("Cannot normalize zero-length vector")
        return fallback.astype(np.float64)
    return vec / norm


def _camera_basis(camera_pos: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = _normalize(target - camera_pos)
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = _normalize(np.cross(forward, world_up), np.asarray([1.0, 0.0, 0.0], dtype=np.float64))
    true_up = _normalize(np.cross(right, forward), world_up)
    return right, true_up, forward


def _rotate_z(points: np.ndarray, center: np.ndarray, angle: float) -> np.ndarray:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rot = np.asarray(
        [
            [cos_a, -sin_a, 0.0],
            [sin_a, cos_a, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return (points - center) @ rot.T + center


def _project_vertices(
    vertices: np.ndarray,
    camera_pos: np.ndarray,
    right: np.ndarray,
    true_up: np.ndarray,
    forward: np.ndarray,
    width: int,
    height: int,
    fov_y_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    rel = vertices - camera_pos
    cam = np.column_stack((rel @ right, rel @ true_up, rel @ forward))
    focal = (0.5 * float(height)) / math.tan(math.radians(float(fov_y_deg)) * 0.5)
    projected = np.empty((vertices.shape[0], 2), dtype=np.float64)
    projected[:, 0] = width * 0.5 + focal * (cam[:, 0] / cam[:, 2])
    projected[:, 1] = height * 0.5 - focal * (cam[:, 1] / cam[:, 2])
    return projected, cam


def _render_software_frame(
    meshes: list[MeshData],
    angle: float,
    center: np.ndarray,
    radius: float,
    camera_z: float,
    args: argparse.Namespace,
) -> np.ndarray:
    import cv2  # noqa: PLC0415

    width = int(args.width)
    height = int(args.height)
    background = np.clip(np.asarray(args.background_color, dtype=np.float64), 0.0, 1.0)
    image = np.full((height, width, 3), (background * 255.0).astype(np.uint8), dtype=np.uint8)

    camera_pos = np.asarray([center[0] + radius, center[1], camera_z], dtype=np.float64)
    right, true_up, forward = _camera_basis(camera_pos, center)
    light_dir = _normalize(np.asarray([0.25, -0.45, 0.86], dtype=np.float64))

    draw_items: list[tuple[float, np.ndarray, np.ndarray]] = []
    for mesh in meshes:
        rotated = _rotate_z(mesh.vertices, center, angle)
        projected, cam = _project_vertices(
            rotated,
            camera_pos,
            right,
            true_up,
            forward,
            width,
            height,
            float(args.fov_y_deg),
        )

        tri_vertices = rotated[mesh.triangles]
        tri_cam = cam[mesh.triangles]
        tri_projected = projected[mesh.triangles]
        valid = np.all(tri_cam[:, :, 2] > float(args.near), axis=1)
        if not np.any(valid):
            continue

        tri_vertices = tri_vertices[valid]
        tri_cam = tri_cam[valid]
        tri_projected = tri_projected[valid]
        depths = tri_cam[:, :, 2].mean(axis=1)
        normals = np.cross(
            tri_vertices[:, 1, :] - tri_vertices[:, 0, :],
            tri_vertices[:, 2, :] - tri_vertices[:, 0, :],
        )
        normal_norm = np.linalg.norm(normals, axis=1)
        valid_normals = normal_norm > 1e-12
        normals[valid_normals] /= normal_norm[valid_normals, None]
        intensity = 0.34 + 0.66 * np.abs(normals @ light_dir)
        colors = np.clip(mesh.color_rgb[None, :] * intensity[:, None], 0.0, 255.0).astype(np.uint8)

        for depth, pts, color in zip(depths, tri_projected, colors):
            if np.any(~np.isfinite(pts)):
                continue
            if (
                np.max(pts[:, 0]) < -width
                or np.min(pts[:, 0]) > 2 * width
                or np.max(pts[:, 1]) < -height
                or np.min(pts[:, 1]) > 2 * height
            ):
                continue
            draw_items.append((float(depth), np.rint(pts).astype(np.int32), color))

    for _, pts, color in sorted(draw_items, key=lambda item: item[0], reverse=True):
        cv2.fillConvexPoly(image, pts, color.tolist(), lineType=cv2.LINE_AA)

    return image


def render_turntable_software(urdf_path: Path, args: argparse.Namespace) -> Path:
    object_name, video_path, metadata_path = _video_and_metadata_paths(urdf_path, args)
    if video_path.exists() and args.skip_existing:
        print(f"[skip] {object_name}: {video_path}")
        return video_path

    meshes = load_urdf_meshes(urdf_path, max_faces_per_mesh=args.max_faces_per_mesh)
    points = np.concatenate([mesh.vertices for mesh in meshes], axis=0)
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    center, extent, max_extent, diagonal, radius, camera_z, far = _view_parameters(bbox_min, bbox_max, args)
    angles = _turntable_angles(args)

    writer = imageio.get_writer(
        video_path,
        fps=float(args.fps),
        codec=str(args.video_codec),
        quality=int(args.video_quality),
    )
    try:
        for frame_idx, angle in enumerate(angles):
            frame = _render_software_frame(meshes, float(angle), center, radius, camera_z, args)
            writer.append_data(frame)
            if args.verbose:
                print(f"  frame {frame_idx + 1:04d}/{len(angles)}")
    finally:
        writer.close()

    metadata = {
        "backend": "software",
        "urdf_path": str(urdf_path),
        "video_path": str(video_path),
        "frames": int(args.frames),
        "fps": float(args.fps),
        "degrees": 360.0,
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "bbox_extent": extent.tolist(),
        "bbox_max_extent": max_extent,
        "bbox_diagonal": diagonal,
        "radius": radius,
        "camera_z": camera_z,
        "far": far,
        "max_faces_per_mesh": args.max_faces_per_mesh,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    print(f"[ok] {object_name}: {video_path}")
    return video_path


def render_turntable(urdf_path: Path, args: argparse.Namespace) -> Path:
    if args.backend == "software":
        return render_turntable_software(urdf_path, args)
    if args.backend == "sapien":
        return render_turntable_sapien(urdf_path, args)

    try:
        return render_turntable_sapien(urdf_path, args)
    except Exception as exc:  # noqa: BLE001
        print(f"[fallback] SAPIEN unavailable for {urdf_path}: {exc}")
        return render_turntable_software(urdf_path, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a 360-degree turntable MP4 for each mobility.urdf"
    )
    parser.add_argument("--urdf-root", type=Path, default=_default_urdf_root())
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--backend", choices=["auto", "sapien", "software"], default="auto")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--near", type=float, default=0.01)
    parser.add_argument("--far", type=float, default=None)
    parser.add_argument("--fov-y-deg", type=float, default=45.0)
    parser.add_argument("--radius-scale", type=float, default=1.7)
    parser.add_argument("--min-radius", type=float, default=1.2)
    parser.add_argument("--min-height-offset", type=float, default=0.25)
    parser.add_argument("--start-angle-deg", type=float, default=0.0)
    parser.add_argument("--include-endpoint", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--video-codec", default="libx264")
    parser.add_argument("--video-quality", type=int, default=8)
    parser.add_argument("--max-faces-per-mesh", type=int, default=50000)
    parser.add_argument("--background-color", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--add-ground", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frames < 2:
        raise ValueError("--frames must be >= 2")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")

    urdfs = find_urdfs(Path(args.urdf_root))
    if not urdfs:
        raise FileNotFoundError(f"No mobility.urdf files found under {args.urdf_root}")

    print(f"Rendering {len(urdfs)} URDF turntable videos")
    print(f"URDF root: {Path(args.urdf_root).expanduser().resolve()}")
    print(f"Output: {Path(args.output_dir).expanduser().resolve()}")
    for urdf in urdfs:
        render_turntable(urdf, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
