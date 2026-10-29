"""Decode and color the aggregated NIO LiDAR map; no per-frame LiDAR is assumed."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tarfile

import cv2
import numpy as np
from scipy.spatial import cKDTree


MAP_SUFFIX = "vis_mapping_results/bev/locmap/mega_general_point_cloud_lidar_map.elm"
GROUND_SUFFIX = "vis_mapping_results/bev/ground.pcd"


def write_ply(path, points, colors, mask):
    """Write the same vertex fields as data_processor.utils.pcd_utils.storePly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                      ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("mask", "u1")])
    vertices = np.empty(len(points), dtype=dtype)
    for axis, name in enumerate(("x", "y", "z")):
        vertices[name] = points[:, axis]
    for channel, name in enumerate(("red", "green", "blue")):
        vertices[name] = colors[:, channel]
    vertices["mask"] = np.asarray(mask).reshape(-1).astype(np.uint8)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "property uchar mask\nend_header\n")
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        vertices.tofile(handle)


def resolve_decoder(cache, adh_root, configured=None):
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise FileNotFoundError(f"LiDAR decoder is not executable: {path}")
        return path
    source = Path(__file__).parent / "vendor" / "dump_lidar_elm.cpp"
    sdk = Path(adh_root)
    digest = hashlib.sha256(source.read_bytes() + str(sdk).encode()).hexdigest()[:16]
    executable = cache / f"dump_lidar_elm_{digest}"
    if executable.exists():
        return executable
    cache.mkdir(parents=True, exist_ok=True)
    command = ["g++", "-std=c++17", "-O2", str(source), f"-I{sdk / 'include'}",
               f"-L{sdk / 'lib'}", f"-Wl,-rpath,{sdk / 'lib'}", "-ladh_map_hub",
               "-ladh_data_hub", "-ladh_utils", "-o", str(executable)]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Cannot compile the LiDAR decoder using the existing ADH SDK: "
                           + str(getattr(error, "stderr", error))) from error
    return executable


def decode_map(input_tar, cache, adh_root, decoder=None, resume=False):
    cache.mkdir(parents=True, exist_ok=True)
    archive_path = Path(input_tar).resolve()
    source_key = hashlib.sha256((str(archive_path) + str(archive_path.stat().st_size)
                                + str(archive_path.stat().st_mtime_ns)).encode()).hexdigest()[:16]
    binary = cache / f"lidar_map_{source_key}.bin"
    member_name = None
    if not (resume and binary.exists()):
        executable = resolve_decoder(cache, adh_root, decoder)
        elm = cache / f"lidar_map_{source_key}.elm"
        with tarfile.open(input_tar) as archive:
            members = [m for m in archive.getmembers() if m.isfile() and m.name.endswith(MAP_SUFFIX)]
            if len(members) != 1:
                raise ValueError(f"Expected one LiDAR map ELM, found {len(members)}")
            member_name = members[0].name
            with archive.extractfile(members[0]) as source, elm.open("wb") as target:
                shutil.copyfileobj(source, target)
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = str(Path(adh_root) / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        try:
            subprocess.run([str(executable), str(elm), str(binary)], check=True,
                           capture_output=True, text=True, env=env)
        except (OSError, subprocess.CalledProcessError) as error:
            binary.unlink(missing_ok=True)
            raise RuntimeError("LiDAR ELM decode failed: " + str(getattr(error, "stderr", error))) from error
        finally:
            elm.unlink(missing_ok=True)
    with binary.open("rb") as handle:
        if handle.read(8) != b"GSLXYZ01":
            raise ValueError("Invalid LiDAR map binary header")
        count_bytes = handle.read(8)
        if len(count_bytes) != 8:
            raise ValueError("Truncated LiDAR map binary header")
        count = struct.unpack("<Q", count_bytes)[0]
        if not count or binary.stat().st_size != 16 + count * 12:
            raise ValueError("LiDAR map binary count/size mismatch")
        points = np.fromfile(handle, dtype="<f4").reshape(-1, 3)
    return points, member_name or MAP_SUFFIX


def read_ground_pcd(input_tar):
    """Read verified ASCII ground XYZ; packed RGB is deliberately not used."""
    with tarfile.open(input_tar) as archive:
        members = [m for m in archive.getmembers() if m.isfile() and m.name.endswith(GROUND_SUFFIX)]
        if len(members) != 1:
            raise ValueError(f"Expected one ground.pcd, found {len(members)}; use --skip_ground to omit it")
        with archive.extractfile(members[0]) as handle:
            header = {}
            for line in handle:
                text = line.decode("ascii").strip()
                if not text or text.startswith("#"):
                    continue
                key, *values = text.split()
                header[key.upper()] = values
                if key.upper() == "DATA":
                    break
            if header.get("DATA") != ["ascii"]:
                raise ValueError("Only ASCII ground.pcd is supported")
            fields = header.get("FIELDS", [])
            if not all(axis in fields for axis in ("x", "y", "z")):
                raise ValueError("ground.pcd lacks XYZ fields")
            indices = tuple(fields.index(axis) for axis in ("x", "y", "z"))
            for index in indices:
                if (header.get("TYPE", [])[index] != "F"
                        or header.get("SIZE", [])[index] != "4"
                        or header.get("COUNT", ["1"] * len(fields))[index] != "1"):
                    raise ValueError("Expected scalar float32 ground XYZ fields")
            points = np.loadtxt(handle, dtype=np.float32, usecols=indices, ndmin=2)
    expected = int(header["POINTS"][0])
    if len(points) != expected or not len(points):
        raise ValueError(f"ground.pcd point count mismatch: expected {expected}, read {len(points)}")
    return points, members[0].name


def project_points(points, camera_to_world, intrinsics, width, height, near=0.2):
    """Return in-image point indices, integer pixels, and camera-Z depth."""
    camera = (points - camera_to_world[:3, 3]) @ camera_to_world[:3, :3]
    depth = camera[:, 2]
    valid = np.isfinite(camera).all(axis=1) & (depth > near)
    indices = np.flatnonzero(valid)
    camera = camera[indices]
    depth = depth[indices]
    fx, fy, cx, cy = intrinsics
    pixels = np.rint(camera[:, :2] / depth[:, None] * [fx, fy] + [cx, cy]).astype(np.int64)
    inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < width)
              & (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
    return indices[inside], pixels[inside], depth[inside]


def visible_points(indices, pixels, depths, width, height):
    """Use the nearest map point at each image pixel for visibility."""
    if not len(indices):
        return indices, pixels, depths
    flat = pixels[:, 1] * width + pixels[:, 0]
    zbuffer = np.full(width * height, np.inf, dtype=np.float32)
    np.minimum.at(zbuffer, flat, depths)
    visible = depths <= zbuffer[flat] + 1e-4
    return indices[visible], pixels[visible], depths[visible]


def excluded_pixels(output, frame, camera_id, shape):
    excluded = np.zeros(shape, dtype=bool)
    for directory in ("dynamic_mask", "sky_mask", "ego_body_mask"):
        path = output / directory / f"{frame:06d}_{camera_id}.png"
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.exists() else None
        if mask is None:
            continue
        if mask.shape != shape:
            raise ValueError(f"Mask size differs from rectified RGB: {path}")
        excluded |= mask > 0
    semantic_path = output / "semantic" / f"{frame:06d}_{camera_id}.png"
    semantic = cv2.imread(str(semantic_path), cv2.IMREAD_GRAYSCALE)
    if semantic is None or semantic.shape != shape:
        raise ValueError(f"Missing or invalid semantic image: {semantic_path}")
    excluded |= semantic == 255
    return excluded


def voxel_indices(points, size):
    if size <= 0:
        return np.arange(len(points))
    grid = np.floor(points.astype(np.float64) / size).astype(np.int64)
    _, indices = np.unique(grid, axis=0, return_index=True)
    return np.sort(indices)


def voxel_downsample(points, size):
    return points[voxel_indices(points, size)]


def process(args):
    output = Path(args.output).resolve()
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    frames = metadata["frames"]
    if args.max_frames:
        frames = frames[:args.max_frames]
    if not frames:
        raise ValueError("No converted frames available")
    cameras = metadata["cameras"]
    cache = output / "audit" / ".cache" / "map_decoder"
    points, _ = decode_map(args.input_tar, cache, args.adh_root, args.decoder, args.resume)
    ground_points = np.empty((0, 3), dtype=np.float32)
    ground_source = None
    if not args.skip_ground:
        ground_points, ground_source = read_ground_pcd(args.input_tar)
        ground_points = ground_points[np.isfinite(ground_points).all(axis=1)]
    provenance = np.concatenate((np.zeros(len(points), dtype=bool), np.ones(len(ground_points), dtype=bool)))
    points = np.concatenate((points, ground_points))
    finite = np.isfinite(points).all(axis=1)
    points, provenance = points[finite], provenance[finite]
    indices = voxel_indices(points, args.voxel_size)
    points, provenance = points[indices], provenance[indices]
    if not len(points):
        raise ValueError("Empty LiDAR map after finite/voxel filtering")
    tree = cKDTree(points[:, :2])
    colors = np.zeros((len(points), 3), dtype=np.uint8)
    best_depth = np.full(len(points), np.inf, dtype=np.float32)
    ground_usable = 0
    ground_hits = 0
    if len(ground_points):
        sample_indices = np.linspace(0, len(ground_points) - 1, min(50000, len(ground_points)), dtype=np.int64)
        ground_sample = ground_points[sample_indices]
    else:
        ground_sample = ground_points
    extrinsics = {key: np.loadtxt(output / "extrinsics" / f"{key}.txt").reshape(4, 4)
                  for key in cameras}
    selected = frames[::args.color_stride]
    if selected[-1] is not frames[-1]:
        selected.append(frames[-1])
    for frame_index, frame in enumerate(selected):
        fid = int(frame["frame"])
        pose = np.loadtxt(output / "ego_pose" / f"{fid:06d}.txt").reshape(4, 4)
        nearby = np.asarray(tree.query_ball_point(pose[:2, 3], args.radius), dtype=np.int64)
        if not len(nearby):
            continue
        for cid, view in frame["cameras"].items():
            spec = cameras[cid]
            image = cv2.imread(str(output / view["image"]), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(output / view["image"])
            h, w = image.shape[:2]
            if (w, h) != (int(spec["width"]), int(spec["height"])):
                raise ValueError(f"Image and calibration dimensions differ: {view['image']}")
            camera_pose = np.loadtxt(output / "ego_pose" / f"{fid:06d}_{cid}.txt").reshape(4, 4)
            camera_to_world = camera_pose @ extrinsics[cid]
            idx, pixels, depths = project_points(points[nearby], camera_to_world,
                                                spec["intrinsics"], w, h)
            idx, pixels, depths = visible_points(idx, pixels, depths, w, h)
            excluded = excluded_pixels(output, fid, cid, (h, w))
            if len(ground_sample) and frame_index in (0, len(selected) // 2, len(selected) - 1):
                ground_nearby = np.linalg.norm(ground_sample[:, :2] - pose[:2, 3], axis=1) <= args.radius
                gidx, gpixels, gdepths = project_points(ground_sample[ground_nearby], camera_to_world,
                                                      spec["intrinsics"], w, h)
                gidx, gpixels, gdepths = visible_points(gidx, gpixels, gdepths, w, h)
                semantic = cv2.imread(str(output / "semantic" / f"{fid:06d}_{cid}.png"), cv2.IMREAD_GRAYSCALE)
                labels = semantic[gpixels[:, 1], gpixels[:, 0]]
                usable = ~excluded[gpixels[:, 1], gpixels[:, 0]]
                ground_hits += int(np.count_nonzero((labels == 1) & usable))
                ground_usable += int(usable.sum())
            keep = ~excluded[pixels[:, 1], pixels[:, 0]]
            map_idx = nearby[idx[keep]]
            update = depths[keep] < best_depth[map_idx]
            candidate_pixels = pixels[keep][update]
            update_idx = map_idx[update]
            colors[update_idx] = image[candidate_pixels[:, 1], candidate_pixels[:, 0], ::-1]
            best_depth[update_idx] = depths[keep][update]
        print(f"Color map {frame_index + 1}/{len(selected)}: frame {fid}", flush=True)
    colored = np.isfinite(best_depth)
    ground_fraction = ground_hits / ground_usable if ground_usable else None
    if len(ground_points) and (ground_usable < 100 or ground_fraction < 0.5):
        raise ValueError("ground.pcd projection does not match semantic ground: require at least "
                         f"100 usable samples and 50% ground labels; got {ground_usable} samples, "
                         f"ground fraction {ground_fraction}")
    if not colored.any():
        raise ValueError("No usable map projections: verify map/pose coordinates and camera calibration")
    map_dir = output / "lidar" / "map"
    map_dir.mkdir(parents=True, exist_ok=True)
    write_ply(map_dir / "background.ply", points, colors, colored)
    np.savez_compressed(map_dir / "background.npz", points=points, colors=colors, mask=colored,
                        source_ground=provenance)
    for frame in frames:
        fid = int(frame["frame"])
        pose = np.loadtxt(output / "ego_pose" / f"{fid:06d}.txt").reshape(4, 4)
        nearby = np.asarray(tree.query_ball_point(pose[:2, 3], args.radius), dtype=np.int64)
        local_points = (points[nearby] - pose[:3, 3]) @ pose[:3, :3]
        write_ply(output / "lidar" / "background" / f"{fid:06d}.ply",
                  local_points, colors[nearby], colored[nearby])
    metadata.update({"geometry_source": "aggregated_lidar_map", "depth_source": "map_projection",
                     "ground_geometry_source": ground_source,
                     "actor_pointcloud_available": False})
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(points)} map points, {int(colored.sum())} colored; {len(frames)} local crops", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_tar", "--input_tar", dest="input_tar", required=True)
    parser.add_argument("--save_dir", "--output", dest="output", required=True)
    parser.add_argument("--color_stride", type=int, default=5)
    parser.add_argument("--radius", type=float, default=80.0)
    parser.add_argument("--voxel_size", type=float, default=0.1)
    parser.add_argument("--max_frames", type=int)
    parser.add_argument("--resume", action="store_true", help="Reuse source-map decoding cache")
    parser.add_argument("--skip_ground", action="store_true", help="Omit optional ground.pcd geometry (merged by default)")
    parser.add_argument("--decoder", default=os.environ.get("NIO_LIDAR_DECODER"))
    parser.add_argument("--adh_root", default=os.environ.get("NIO_ADH_ROOT", "/usr/local/aac/adh/Linux-x86_64"))
    args = parser.parse_args()
    if args.color_stride < 1 or args.radius <= 0 or args.voxel_size < 0 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("color_stride/max_frames must be positive, radius > 0, voxel_size >= 0")
    process(args)

"""
解码完整时间背景LiDAR点云地图
  → 合并 ground.pcd
  → 三维体素降采样
  → 用选定帧的所有相机进行投影着色
  → 保存完整地图
  → 生成各帧车辆附近的局部背景点云地图：缩小背景点云范围
输出：
map/background.ply：完整时间背景+地面LiDAR点云地图
map/background.npz：完整时间背景+地面LiDAR点云地图数组
background/000000.ply、000001.ply：各帧车辆附近的局部背景+地面LiDAR点云
"""
if __name__ == "__main__":
    main()
