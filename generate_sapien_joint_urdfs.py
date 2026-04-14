#!/usr/bin/env python3

import argparse
import json
import math
import shutil
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


SOURCE_STATIC_MESH = Path("textured_objs/start/start_static_rotate.ply")
SOURCE_DYNAMIC_MESH = Path("textured_objs/start/start_dynamic_rotate.ply")
OUTPUT_STATIC_MESH = Path("meshes/base_link.obj")
OUTPUT_DYNAMIC_MESH = Path("meshes/moving_link.obj")


@dataclass
class ObjectSpec:
    category: str
    object_id: str
    source_dir: Path
    motion_type: str
    axis_origin: Sequence[float]
    axis_direction: Sequence[float]
    start_value: float
    end_value: float

    @property
    def name(self) -> str:
        return f"{self.category}_{self.object_id}"

    @property
    def joint_type(self) -> str:
        if self.motion_type == "rotate":
            return "revolute"
        if self.motion_type == "translate":
            return "prismatic"
        raise ValueError(f"Unsupported motion type: {self.motion_type}")

    @property
    def delta(self) -> float:
        return self.end_value - self.start_value

    @property
    def lower(self) -> float:
        return min(0.0, self.delta)

    @property
    def upper(self) -> float:
        return max(0.0, self.delta)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("Dataset/data/sapien"))
    parser.add_argument("--output-root", type=Path, default=Path("generated_urdfs/sapien_unified"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def format_float(value: float) -> str:
    if abs(value) < 1e-12:
        value = 0.0
    text = f"{value:.10f}".rstrip("0").rstrip(".")
    return text or "0"


def format_vec(values: Sequence[float]) -> str:
    return " ".join(format_float(float(v)) for v in values)


def normalize(values: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(float(v) ** 2 for v in values))
    if norm <= 1e-12:
        raise ValueError(f"Invalid axis direction: {values}")
    return [float(v) / norm for v in values]


def read_binary_ply(path: Path) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, ...]]]:
    with path.open("rb") as handle:
        header = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Unexpected EOF in header: {path}")
            decoded = line.decode("ascii").strip()
            header.append(decoded)
            if decoded == "end_header":
                break

        if header[0] != "ply":
            raise ValueError(f"Not a PLY file: {path}")
        if "format binary_little_endian 1.0" not in header:
            raise ValueError(f"Unsupported PLY format: {path}")

        vertex_count = 0
        face_count = 0
        in_vertex = False
        vertex_property_count = 0
        face_list_ok = False

        for line in header[1:]:
            if line.startswith("element "):
                _, name, count = line.split()
                in_vertex = name == "vertex"
                if name == "vertex":
                    vertex_count = int(count)
                elif name == "face":
                    face_count = int(count)
            elif line.startswith("property ") and in_vertex:
                vertex_property_count += 1
            elif line == "property list uchar uint vertex_indices":
                face_list_ok = True

        if vertex_count <= 0 or face_count <= 0 or not face_list_ok:
            raise ValueError(f"Unsupported mesh layout: {path}")

        vertex_struct = struct.Struct("<" + "d" * vertex_property_count)
        vertices: List[Tuple[float, float, float]] = []
        for _ in range(vertex_count):
            raw = handle.read(vertex_struct.size)
            if len(raw) != vertex_struct.size:
                raise ValueError(f"Unexpected EOF in vertices: {path}")
            values = vertex_struct.unpack(raw)
            vertices.append((float(values[0]), float(values[1]), float(values[2])))

        faces: List[Tuple[int, ...]] = []
        for _ in range(face_count):
            raw_count = handle.read(1)
            if len(raw_count) != 1:
                raise ValueError(f"Unexpected EOF in face count: {path}")
            n = struct.unpack("<B", raw_count)[0]
            raw_indices = handle.read(4 * n)
            if len(raw_indices) != 4 * n:
                raise ValueError(f"Unexpected EOF in face indices: {path}")
            faces.append(tuple(int(v) for v in struct.unpack("<" + "I" * n, raw_indices)))

    return vertices, faces


def write_obj(path: Path, vertices: List[Tuple[float, float, float]], faces: List[Tuple[int, ...]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Generated from binary PLY\n")
        for x, y, z in vertices:
            handle.write(f"v {format_float(x)} {format_float(y)} {format_float(z)}\n")
        for face in faces:
            indices = " ".join(str(index + 1) for index in face)
            handle.write(f"f {indices}\n")


def load_specs(input_root: Path) -> List[ObjectSpec]:
    specs: List[ObjectSpec] = []
    for category_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        for object_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
            trans_path = object_dir / "textured_objs/trans.json"
            if not trans_path.exists():
                continue

            with trans_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)

            motion = data["input"]["motion"]
            axis = data["trans_info"]["axis"]
            motion_type = motion["type"]

            if motion_type == "rotate":
                start_value, end_value = motion["rotate"]
            elif motion_type == "translate":
                start_value, end_value = motion["translate"]
            else:
                raise ValueError(f"Unsupported motion type in {trans_path}: {motion_type}")

            spec = ObjectSpec(
                category=category_dir.name,
                object_id=object_dir.name,
                source_dir=object_dir,
                motion_type=motion_type,
                axis_origin=[float(v) for v in axis["o"]],
                axis_direction=normalize(axis["d"]),
                start_value=float(start_value),
                end_value=float(end_value),
            )

            for required in (
                spec.source_dir / SOURCE_STATIC_MESH,
                spec.source_dir / SOURCE_DYNAMIC_MESH,
                trans_path,
            ):
                if not required.exists():
                    raise FileNotFoundError(required)

            specs.append(spec)

    return specs


def build_urdf(spec: ObjectSpec) -> str:
    child_origin = [-float(v) for v in spec.axis_origin]
    lower = spec.lower
    upper = spec.upper
    if spec.joint_type == "revolute":
        lower = math.radians(lower)
        upper = math.radians(upper)

    return f"""<?xml version="1.0"?>
<robot name="{spec.name}">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{OUTPUT_STATIC_MESH.as_posix()}" scale="1 1 1"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{OUTPUT_STATIC_MESH.as_posix()}" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>
  <link name="moving_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
    <visual>
      <origin xyz="{format_vec(child_origin)}" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{OUTPUT_DYNAMIC_MESH.as_posix()}" scale="1 1 1"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="{format_vec(child_origin)}" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{OUTPUT_DYNAMIC_MESH.as_posix()}" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>
  <joint name="{spec.name}_joint" type="{spec.joint_type}">
    <parent link="base_link"/>
    <child link="moving_link"/>
    <origin xyz="{format_vec(spec.axis_origin)}" rpy="0 0 0"/>
    <axis xyz="{format_vec(spec.axis_direction)}"/>
    <limit lower="{format_float(lower)}" upper="{format_float(upper)}" effort="100" velocity="10"/>
  </joint>
</robot>
"""


def validate_object_dir(object_dir: Path) -> None:
    urdf_path = object_dir / "mobility.urdf"
    tree = ET.parse(urdf_path)
    for mesh in tree.findall(".//mesh"):
        mesh_path = object_dir / mesh.attrib["filename"]
        if not mesh_path.exists():
            raise FileNotFoundError(mesh_path)


def prepare_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(path)
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def generate(specs: List[ObjectSpec], output_root: Path, force: bool) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {"counts": {"revolute": 0, "prismatic": 0}, "objects": []}

    for spec in specs:
        object_dir = output_root / spec.name
        prepare_dir(object_dir, force)
        mesh_dir = object_dir / "meshes"
        mesh_dir.mkdir(parents=True, exist_ok=True)

        base_vertices, base_faces = read_binary_ply(spec.source_dir / SOURCE_STATIC_MESH)
        moving_vertices, moving_faces = read_binary_ply(spec.source_dir / SOURCE_DYNAMIC_MESH)
        write_obj(object_dir / OUTPUT_STATIC_MESH, base_vertices, base_faces)
        write_obj(object_dir / OUTPUT_DYNAMIC_MESH, moving_vertices, moving_faces)

        with (object_dir / "mobility.urdf").open("w", encoding="utf-8") as handle:
            handle.write(build_urdf(spec))

        metadata = {
            "name": spec.name,
            "category": spec.category,
            "object_id": spec.object_id,
            "joint_type": spec.joint_type,
            "motion_type": spec.motion_type,
            "source_dir": str(spec.source_dir),
            "mesh_dir": "meshes",
            "mesh_files": {
                "base_link": OUTPUT_STATIC_MESH.name,
                "moving_link": OUTPUT_DYNAMIC_MESH.name,
            },
            "axis_origin": list(spec.axis_origin),
            "axis_direction": list(spec.axis_direction),
            "start_value": spec.start_value,
            "end_value": spec.end_value,
            "relative_delta": spec.delta,
        }
        with (object_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")

        validate_object_dir(object_dir)
        manifest["counts"][spec.joint_type] += 1
        manifest["objects"].append(
            {
                "name": spec.name,
                "joint_type": spec.joint_type,
                "object_dir": str(object_dir),
                "urdf_path": str(object_dir / "mobility.urdf"),
            }
        )

    with (output_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    specs = load_specs(args.input_root.resolve())
    generate(specs, args.output_root.resolve(), args.force)
    print(f"Generated {len(specs)} URDFs in {args.output_root}")
    print(f"revolute={sum(1 for s in specs if s.joint_type == 'revolute')}")
    print(f"prismatic={sum(1 for s in specs if s.joint_type == 'prismatic')}")


if __name__ == "__main__":
    main()
