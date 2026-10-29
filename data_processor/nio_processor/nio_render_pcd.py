"""Render NIO map and optional motion-compensated frame LiDAR with a CPU z-buffer."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def read_map(data_dir):
    path = data_dir / 'lidar/map/background.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            points = data['points'].copy()
            colors = data['colors'].copy()
            mask = data['mask'].astype(bool) if 'mask' in data else np.ones(len(points), bool)
    else:
        points, colors, mask = read_ply(data_dir / 'lidar/map/background.ply')
    if points.shape != colors.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('Map points and colors must both have shape (N, 3)')
    if mask.shape != (len(points),):
        raise ValueError('Map mask must have shape (N,)')
    finite = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
    points, colors, mask = points[finite], colors[finite], mask[finite]
    if np.issubdtype(colors.dtype, np.floating) and colors.size and colors.max() <= 1:
        colors = colors * 255
    if colors.dtype != np.uint8:
        colors = np.clip(colors, 0, 255).astype(np.uint8)
    return points.astype(np.float32, copy=False), colors, mask


def read_ply(path):
    """Read scalar vertex properties from ASCII or binary PLY without plyfile."""
    types = {'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8',
             'uchar': 'u1', 'uint8': 'u1', 'char': 'i1', 'int8': 'i1',
             'short': 'i2', 'int16': 'i2', 'ushort': 'u2', 'uint16': 'u2',
             'int': 'i4', 'int32': 'i4', 'uint': 'u4', 'uint32': 'u4'}
    with path.open('rb') as stream:
        if stream.readline().strip() != b'ply':
            raise ValueError(f'Invalid PLY: {path}')
        properties, count, fmt, element = [], None, None, None
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f'Incomplete PLY header: {path}')
            words = line.decode('ascii').strip().split()
            if not words:
                continue
            if words[0] == 'end_header':
                break
            if words[0] == 'format':
                fmt = words[1]
            elif words[0] == 'element':
                element = words[1]
                if element == 'vertex':
                    count = int(words[2])
                elif count is None and int(words[2]):
                    raise ValueError('PLY must store vertices before other nonempty elements')
            elif words[0] == 'property' and element == 'vertex':
                if words[1] == 'list':
                    raise ValueError('List-valued vertex properties are unsupported')
                properties.append((words[2], types[words[1]]))
        if count is None:
            raise ValueError('PLY has no vertex element')
        if fmt == 'ascii':
            rows = np.array([stream.readline().split() for _ in range(count)], dtype=np.float64)
            values = {name: rows[:, i] for i, (name, _) in enumerate(properties)}
        elif fmt in ('binary_little_endian', 'binary_big_endian'):
            endian = '<' if fmt == 'binary_little_endian' else '>'
            dtype = np.dtype([(name, endian + kind) for name, kind in properties])
            vertices = np.fromfile(stream, dtype=dtype, count=count)
            if len(vertices) != count:
                raise ValueError('Truncated PLY vertices')
            values = {name: vertices[name] for name, _ in properties}
        else:
            raise ValueError(f'Unsupported PLY format: {fmt}')
    points = np.stack([values[name] for name in ('x', 'y', 'z')], axis=1)
    colors = np.stack([values[name] for name in ('red', 'green', 'blue')], axis=1)
    mask = values.get('mask', np.ones(count, bool)).astype(bool)
    return points, colors, mask


def project_points(points, colors, camera_to_world, intrinsics, width, height, near, far,
                   color_mask=None):
    # Matching the float32 map avoids NumPy's very slow mixed-type matmul
    # fallback while retaining the source pointcloud's geometric precision.
    world_to_camera = np.linalg.inv(camera_to_world).astype(np.float32)
    xyz = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    visible = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > near) & (xyz[:, 2] < far)
    xyz, rgb = xyz[visible], colors[visible]
    observed = color_mask[visible] if color_mask is not None else None
    fx, fy, cx, cy = intrinsics
    u = xyz[:, 0] / xyz[:, 2] * fx + cx
    v = xyz[:, 1] / xyz[:, 2] * fy + cy
    inside = (u >= -0.5) & (u < width - 0.5) & (v >= -0.5) & (v < height - 0.5)
    xyz, rgb, u, v = xyz[inside], rgb[inside], u[inside], v[inside]
    if observed is not None:
        observed = observed[inside]
    pixel = np.floor(v + 0.5).astype(np.int64) * width + np.floor(u + 0.5).astype(np.int64)
    depth = np.full(height * width, np.inf, dtype=np.float32)
    np.minimum.at(depth, pixel, xyz[:, 2])
    # Resolve equal-depth points deterministically by their input order.
    winner = np.full(height * width, len(xyz), dtype=np.int64)
    indices = np.arange(len(xyz))
    nearest = xyz[:, 2].astype(np.float32) == depth[pixel]
    np.minimum.at(winner, pixel[nearest], indices[nearest])
    image = np.zeros((height * width, 3), np.uint8)
    occupied = np.isfinite(depth)
    image[occupied] = rgb[winner[occupied]]
    if observed is None:
        return image.reshape(height, width, 3), depth.reshape(height, width)
    colored = np.zeros(height * width, bool)
    colored[occupied] = observed[winner[occupied]]
    return image.reshape(height, width, 3), depth.reshape(height, width), colored.reshape(height, width)


def splat(image, depth, radius):
    """Spread projected samples with a circular footprint, keeping nearest Z."""
    if radius == 0:
        return image, depth
    height, width = depth.shape
    out_depth = np.full_like(depth, np.inf)
    out_image = np.zeros_like(image)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            y0, y1 = max(0, -dy), min(height, height - dy)
            x0, x1 = max(0, -dx), min(width, width - dx)
            src = np.s_[y0:y1, x0:x1]
            dst = np.s_[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
            closer = depth[src] < out_depth[dst]
            out_depth[dst][closer] = depth[src][closer]
            out_image[dst][closer] = image[src][closer]
    return out_image, out_depth


def excluded_pixels(data_dir, stem, shape, include_dynamic=True):
    excluded = np.zeros(shape, bool)
    for folder in ('dynamic_mask', 'sky_mask', 'ego_body_mask', 'semantic'):
        if folder == 'dynamic_mask' and not include_dynamic:
            continue
        path = data_dir / folder / f'{stem}.png'
        if not path.exists():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape != shape:
            raise ValueError(f'Invalid mask dimensions: {path}, expected {shape}')
        excluded |= image == 255 if folder == 'semantic' else image != 0
    return excluded


def render_scene(args):
    data_dir = Path(args.data_dir).resolve()
    metadata = json.loads((data_dir / 'metadata.json').read_text())
    cameras = metadata['cameras']
    selected = [str(cam) for cam in args.cams] if args.cams else list(cameras)
    unknown = set(selected) - set(cameras)
    if unknown:
        raise ValueError(f'Unknown camera IDs: {sorted(unknown)}')
    frames = metadata['frames'][:args.max_frames] if args.max_frames else metadata['frames']
    points, colors, color_mask = read_map(data_dir)
    if not len(points) or not np.any(color_mask):
        raise ValueError('No valid colored map points')
    frame_lidar = None
    frame_manifest = data_dir / 'lidar/frame/metadata.json'
    if metadata.get('frame_lidar_available') and not frame_manifest.is_file():
        raise FileNotFoundError(f'Frame LiDAR is marked available but {frame_manifest} is missing; '
                                'rerun the LiDAR preprocessing stage')
    if frame_manifest.is_file():
        try:
            from . import nio_get_frame_pcd as frame_module
        except ImportError:
            import nio_get_frame_pcd as frame_module
        frame_lidar = frame_module.FrameLidar(data_dir)
    extrinsics = {cam: np.loadtxt(data_dir / 'extrinsics' / f'{cam}.txt').reshape(4, 4)
                  for cam in selected}
    render_dir = data_dir / 'lidar/color_render'
    depth_dir = data_dir / 'lidar/depth'
    for folder in (render_dir, depth_dir):
        folder.mkdir(parents=True, exist_ok=True)
    tasks = [(frame, cam) for frame in frames for cam in selected if cam in frame['cameras']]
    # Rebuilding the aggregate map or changing masks/calibration must invalidate
    # existing render files, including outputs from a shorter smoke conversion.
    fingerprint = hashlib.sha256()
    map_path = data_dir / 'lidar/map/background.npz'
    if not map_path.exists():
        map_path = data_dir / 'lidar/map/background.ply'
    with map_path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            fingerprint.update(block)
    fingerprint.update(Path(__file__).read_bytes())
    fingerprint.update(json.dumps({'algorithm': 'cpu_zbuffer_frame_lidar_v3',
                       'metadata': metadata, 'cams': selected,
                       'frames': [f['frame'] for f in frames], 'radius': args.radius,
                       'near': args.near, 'far': args.far}, sort_keys=True).encode())
    if frame_lidar is not None:
        fingerprint.update(str(frame_lidar.fingerprint).encode())
        input_paths = {Path(path).resolve() for path in frame_lidar.input_paths}
        input_paths.add(Path(frame_module.__file__).resolve())
        for path in sorted(input_paths):
            fingerprint.update(str(path).encode())
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    fingerprint.update(block)
    for frame, cam in tasks:
        stem = f"{frame['frame']:06d}_{cam}"
        fingerprint.update((data_dir / 'ego_pose' / f'{stem}.txt').read_bytes())
        if frame_lidar is not None:
            image_path = data_dir / frame['cameras'][cam]['image']
            stat = image_path.stat()
            fingerprint.update(f'{image_path}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
        for folder in ('dynamic_mask', 'sky_mask', 'ego_body_mask', 'semantic'):
            path = data_dir / folder / f'{stem}.png'
            if path.is_file():
                stat = path.stat()
                fingerprint.update(f'{path}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
    for cam in selected:
        fingerprint.update((data_dir / 'extrinsics' / f'{cam}.txt').read_bytes())
    input_fingerprint = fingerprint.hexdigest()
    audit_path = data_dir / 'audit/render.json'
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    prior = json.loads(audit_path.read_text()) if audit_path.is_file() else {}
    can_resume = args.resume and prior.get('input_fingerprint') == input_fingerprint
    if args.resume and not can_resume:
        print('Map, camera, mask or render configuration changed; rebuilding guidance', flush=True)
    def render_one(task):
        frame, cam = task
        frame_id = frame['frame']
        stem = f'{frame_id:06d}_{cam}'
        rgb_path = render_dir / f'{stem}.png'
        mask_path = render_dir / f'{stem}_mask.png'
        dense_path = render_dir / f'{stem}_depth.npy'
        sparse_path = depth_dir / f'{stem}.npz'
        camera = cameras[cam]
        width, height = int(camera['width']), int(camera['height'])
        if can_resume and all(path.exists() for path in (rgb_path, mask_path, dense_path, sparse_path)):
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None and mask.shape == (height, width):
                return float(np.count_nonzero(mask) / mask.size)
        pose = np.loadtxt(data_dir / 'ego_pose' / f'{stem}.txt').reshape(4, 4)
        camera_to_world = pose @ extrinsics[cam]
        image, raw_depth, observed = project_points(points, colors, camera_to_world, camera['intrinsics'],
                                                    width, height, args.near, args.far, color_mask)
        excluded = excluded_pixels(data_dir, stem, (height, width))
        raw_depth[excluded] = np.inf
        image[excluded] = 0
        # Background guidance never fills a dynamic image region. Uncolored
        # map geometry still contributes to the shared depth and occlusion.
        color_depth = np.where(observed, raw_depth, np.inf)
        image, depth = splat(image, color_depth, args.radius)
        depth[excluded] = np.inf
        common_excluded = excluded
        if frame_lidar is not None:
            common_excluded = excluded_pixels(data_dir, stem, (height, width), include_dynamic=False)
            camera_timestamp = frame['cameras'][cam]['timestamp']
            frame_points = frame_lidar.at(frame_id, camera_timestamp)
            _, frame_depth = project_points(frame_points, np.zeros((len(frame_points), 3), np.uint8),
                                            camera_to_world, camera['intrinsics'], width, height,
                                            args.near, args.far)
            frame_depth[common_excluded] = np.inf
            frame_image = cv2.imread(str(data_dir / frame['cameras'][cam]['image']), cv2.IMREAD_COLOR)
            if frame_image is None or frame_image.shape != (height, width, 3):
                raise ValueError(f'Missing or invalid rectified RGB: {stem}')
            # Sample actual target-camera RGB at the nearest LiDAR projection;
            # scanner display colors/intensity are not photometric observations.
            frame_image = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)
            frame_image[~np.isfinite(frame_depth)] = 0
            raw_depth = np.minimum(raw_depth, frame_depth)
            frame_image, frame_color_depth = splat(frame_image, frame_depth, args.radius)
            frame_color_depth[common_excluded] = np.inf
            take_frame = frame_color_depth <= depth
            image[take_frame] = frame_image[take_frame]
            depth = np.minimum(depth, frame_color_depth)
        raw_mask = np.isfinite(raw_depth)
        np.savez_compressed(sparse_path, mask=raw_mask, value=raw_depth[raw_mask])
        mask = np.isfinite(depth) & ~common_excluded & (depth <= raw_depth + 1e-4)
        image[~mask] = 0
        depth[~mask] = 0
        if not cv2.imwrite(str(rgb_path), image[:, :, ::-1]):
            raise OSError(f'Cannot write {rgb_path}')
        if not cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255):
            raise OSError(f'Cannot write {mask_path}')
        np.save(dense_path, depth)
        return float(mask.mean())

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(render_one, tasks), 1):
            results.append(result)
            if i == 1 or i % 20 == 0 or i == len(tasks):
                print(f'Rendered {i}/{len(tasks)} images', flush=True)
    audit_path.write_text(json.dumps({'input_fingerprint': input_fingerprint}, indent=2) + '\n')
    if not results or not any(coverage > 0 for coverage in results):
        raise RuntimeError('No valid map guidance was rendered; check map and camera coordinates')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', required=True, help='Processed scene directory')
    parser.add_argument('--cams', nargs='+', help='Camera IDs, defaults to all')
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--workers', type=int, default=1, help='CPU threads; each thread uses image buffers')
    parser.add_argument('--radius', type=int, default=2, help='Circular splat radius in pixels')# 3D点半径
    parser.add_argument('--near', type=float, default=0.1)
    parser.add_argument('--far', type=float, default=120.0)
    parser.add_argument('--resume', '--skip_existing', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or not 0 <= args.radius <= 10 or not 0 < args.near < args.far:
        parser.error('Require workers >= 1, 0 <= radius <= 10, and 0 < near < far')
    if args.max_frames is not None and args.max_frames < 1:
        parser.error('max_frames must be positive')
    render_scene(args)

"""
点云渲染：读取背景地图，以及存在时的逐帧LiDAR；使用各相机拍摄时刻的位姿，并对实例运动进行补偿。
背景分支排除动态区域，逐帧LiDAR分支保留动态实例，两者均排除天空、自车和无效图像区域，并统一处理遮挡。
逐帧LiDAR使用对应相机的实际RGB着色。没有lidar/frame/metadata.json时保留仅背景地图行为。
lidar/depth/{frame}_{cam}.npz：背景地图与逐帧LiDAR统一投影后的二维深度mask+一维camera-Z深度value，作为稀疏深度约束。

lidar/color_render/{frame}_{cam}.png：点云渲染图像
lidar/color_render/{frame}_{cam}_mask.png：点云命中掩码
lidar/color_render/{frame}_{cam}_depth.npy：两路有效颜色点扩散、合并并通过统一几何遮挡检查后的二维深度图。与上面两者配套使用

"""
if __name__ == '__main__':
    main()
