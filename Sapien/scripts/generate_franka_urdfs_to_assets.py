#!/usr/bin/env python3
"""
Generate URDF files from Franka xacro robot definitions and rewrite mesh paths.

What it does:
1. Copies meshes from:
   <assets>/franka_description/meshes
   to:
   <assets>/meshes
2. Generates URDF for every *.urdf.xacro under:
   <assets>/franka_description/robots/*
3. Writes URDFs directly in <assets> and rewrites mesh filenames to point to:
   <assets>/meshes
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Franka URDFs into assets and rewrite mesh paths")
    parser.add_argument(
        "--assets-dir",
        default="/home/samuelemara/Joint3DGS/Sapien/assets",
        help="Target assets directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without writing files",
    )
    return parser.parse_args()


def discover_xacro_files(robots_dir: Path) -> list[Path]:
    xacros: list[Path] = []
    for robot_dir in sorted(p for p in robots_dir.iterdir() if p.is_dir()):
        if robot_dir.name == "common":
            continue
        xacros.extend(sorted(robot_dir.rglob("*.urdf.xacro")))
    return xacros


def output_name_for_xacro(xacro_path: Path) -> str:
    name = xacro_path.name
    if name.endswith(".urdf.xacro"):
        return name[: -len(".xacro")]
    return f"{xacro_path.stem}.urdf"


def render_targets_for_xacro(xacro_path: Path) -> list[tuple[str, list[str]]]:
    """
    Return (output_file_name, extra_xacro_args) targets for a given xacro.
    """
    base_name = xacro_path.name
    default_output = output_name_for_xacro(xacro_path)

    if base_name == "fr3_duo_arm.urdf.xacro":
        return [
            ("fr3_duo_arm_left.urdf", ["arm_prefix:=left"]),
            ("fr3_duo_arm_right.urdf", ["arm_prefix:=right"]),
        ]

    return [(default_output, [])]


def make_temp_ament_prefix(franka_description_dir: Path) -> tuple[Path, dict[str, str]]:
    temp_prefix = Path(tempfile.mkdtemp(prefix="franka_ament_prefix_"))
    pkg_index = temp_prefix / "share" / "ament_index" / "resource_index" / "packages"
    pkg_index.mkdir(parents=True, exist_ok=True)
    (pkg_index / "franka_description").write_text("", encoding="utf-8")

    share_dir = temp_prefix / "share"
    share_dir.mkdir(parents=True, exist_ok=True)
    (share_dir / "franka_description").symlink_to(franka_description_dir)

    env = os.environ.copy()
    existing = env.get("AMENT_PREFIX_PATH", "")
    env["AMENT_PREFIX_PATH"] = f"{temp_prefix}:{existing}" if existing else str(temp_prefix)
    return temp_prefix, env


def run_xacro(xacro_file: Path, env: dict[str, str], extra_args: list[str]) -> str:
    cmd = ["xacro", str(xacro_file), *extra_args]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    return proc.stdout


def rewrite_mesh_paths(urdf_text: str, new_mesh_root: Path) -> str:
    old_prefix = "package://franka_description/meshes/"
    new_prefix = f"{new_mesh_root.as_posix()}/"
    return urdf_text.replace(old_prefix, new_prefix)


def main() -> int:
    args = parse_args()

    assets_dir = Path(args.assets_dir).expanduser().resolve()
    franka_description_dir = assets_dir / "franka_description"
    robots_dir = franka_description_dir / "robots"
    source_meshes = franka_description_dir / "meshes"
    target_meshes = assets_dir / "meshes"

    if not franka_description_dir.exists():
        raise FileNotFoundError(f"Missing franka_description directory: {franka_description_dir}")
    if not robots_dir.exists():
        raise FileNotFoundError(f"Missing robots directory: {robots_dir}")
    if not source_meshes.exists():
        raise FileNotFoundError(f"Missing source meshes directory: {source_meshes}")

    xacro_files = discover_xacro_files(robots_dir)
    if not xacro_files:
        raise RuntimeError(f"No *.urdf.xacro files found under {robots_dir}")

    print(f"Discovered {len(xacro_files)} URDF xacro files")

    if args.dry_run:
        print(f"[dry-run] Would copy meshes: {source_meshes} -> {target_meshes}")
    else:
        target_meshes.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_meshes, target_meshes, dirs_exist_ok=True)
        print(f"Copied meshes to: {target_meshes}")

    temp_prefix, env = make_temp_ament_prefix(franka_description_dir)
    generated: list[Path] = []
    failures: list[str] = []

    try:
        for xacro_file in xacro_files:
            for out_name, extra_args in render_targets_for_xacro(xacro_file):
                out_path = assets_dir / out_name
                if args.dry_run:
                    print(f"[dry-run] Would generate: {out_path} from {xacro_file.name} {extra_args}")
                    continue

                try:
                    urdf = run_xacro(xacro_file, env=env, extra_args=extra_args)
                    urdf = rewrite_mesh_paths(urdf, target_meshes)
                    out_path.write_text(urdf, encoding="utf-8")
                    generated.append(out_path)
                    print(f"Generated: {out_path}")
                except subprocess.CalledProcessError as exc:
                    stderr = (exc.stderr or "").strip()
                    failures.append(f"{xacro_file} {extra_args}: {stderr}")
                    print(f"FAILED: {xacro_file} {extra_args}")
                    if stderr:
                        print(stderr)

    finally:
        shutil.rmtree(temp_prefix, ignore_errors=True)

    if failures:
        print("\nCompleted with failures:")
        for fail in failures:
            print(f"- {fail}")
        return 1

    if not args.dry_run:
        print("\nDone. Generated URDF files:")
        for path in generated:
            print(f"- {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
