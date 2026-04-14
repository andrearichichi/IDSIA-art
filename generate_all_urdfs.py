#!/usr/bin/env python3
"""Generate URDF files for all 10 SAPIEN objects and collect them in one directory."""

import subprocess
import shutil
from pathlib import Path

# Define all 10 SAPIEN objects
SAPIEN_OBJECTS = [
    ("blade", "103706"),
    ("foldchair", "102255"),
    ("fridge", "10905"),
    ("laptop", "10211"),
    ("oven", "101917"),
    ("scissor", "11100"),
    ("stapler", "103111"),
    ("storage", "45135"),
    ("USB", "100109"),
    ("washer", "103776"),
]

def main():
    # Create output directory outside Dataset
    workspace_dir = Path(__file__).parent
    output_dir = workspace_dir / "urdf_collection"
    output_dir.mkdir(exist_ok=True)
    
    meshes_dir = output_dir / "meshes"
    meshes_dir.mkdir(exist_ok=True)
    
    dataset_dir = workspace_dir / "Dataset" / "data" / "sapien"
    
    print(f"Generating URDF files for all 10 SAPIEN objects...")
    print(f"Output directory: {output_dir}")
    print()
    
    for idx, (category, obj_id) in enumerate(SAPIEN_OBJECTS, 1):
        print(f"[{idx}/10] Generating URDF for {category}/{obj_id}...")
        
        object_dir = dataset_dir / category / obj_id / "textured_objs"
        
        # Check if object directory exists
        if not object_dir.exists():
            print(f"  ⚠️  Object directory not found: {object_dir}")
            print(f"  Skipping {category}/{obj_id}")
            print()
            continue
        
        # Check if trans.json exists (required for generation)
        trans_path = object_dir / "trans.json"
        if not trans_path.exists():
            print(f"  ⚠️  trans.json not found for {category}/{obj_id}")
            print(f"  Skipping {category}/{obj_id}")
            print()
            continue
        
        # Check for required pose directories
        has_start = (object_dir / "start").exists()
        has_end = (object_dir / "end").exists()
        
        if not (has_start or has_end):
            print(f"  ⚠️  No start/end directories found for {category}/{obj_id}")
            print(f"  Skipping {category}/{obj_id}")
            print()
            continue
        
        try:
            # Create temporary output directory for this object
            temp_output = workspace_dir / "temp_urdf" / f"{category}_{obj_id}"
            temp_output.mkdir(parents=True, exist_ok=True)
            
            # Call the URDF generation script
            cmd = [
                "python3",
                "generate_scissor_urdf.py",
                "--object-dir", str(object_dir),
                "--output-dir", str(temp_output),
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(workspace_dir))
            
            if result.returncode != 0:
                print(f"  ❌ Error generating URDF for {category}/{obj_id}")
                print(f"  Error output: {result.stderr}")
                print()
                continue
            
            # Copy generated files to main output directory
            robot_name = f"{category}_{obj_id}"
            
            # Copy URDF
            urdf_src = temp_output / f"{robot_name}.urdf"
            urdf_dst = output_dir / f"{robot_name}.urdf"
            if urdf_src.exists():
                shutil.copy2(urdf_src, urdf_dst)
                print(f"  ✓ URDF: {urdf_dst.name}")
            
            # Copy metadata
            metadata_src = temp_output / f"{robot_name}_metadata.json"
            metadata_dst = output_dir / f"{robot_name}_metadata.json"
            if metadata_src.exists():
                shutil.copy2(metadata_src, metadata_dst)
                print(f"  ✓ Metadata: {metadata_dst.name}")
            
            # Copy meshes
            src_meshes = temp_output / "meshes"
            if src_meshes.exists():
                for mesh_file in src_meshes.iterdir():
                    shutil.copy2(mesh_file, meshes_dir / mesh_file.name)
                print(f"  ✓ Meshes copied (2 files)")
            
            print(f"  ✓ Successfully generated for {category}/{obj_id}")
            
        except Exception as e:
            print(f"  ❌ Exception for {category}/{obj_id}: {e}")
        
        print()
    
    # Clean up temporary directory
    temp_root = workspace_dir / "temp_urdf"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    
    print("=" * 60)
    print(f"✓ URDF generation complete!")
    print(f"All URDF files collected in: {output_dir}")
    print(f"Meshes in: {meshes_dir}")
    print()
    
    # List all generated files
    urdf_files = list(output_dir.glob("*.urdf"))
    print(f"Generated {len(urdf_files)} URDF files:")
    for urdf_file in sorted(urdf_files):
        print(f"  - {urdf_file.name}")


if __name__ == "__main__":
    main()
