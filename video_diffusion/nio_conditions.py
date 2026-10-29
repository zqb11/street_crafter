"""Render NIO lateral-view conditions from colored maps and frame LiDAR.

LiDAR receives photometric colors in the original observed camera first.
Those world points are then reprojected into the translated camera. Original
image masks classify source observations; they never mask target-view pixels.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

DEPTH_EPSILON = 1e-4
SCHEMA = 'streetcrafter.nio-novel-conditions/v1'


def _resolve(path, root):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else root/path).resolve()


def _stat(path):
    path = Path(path).resolve()
    info = path.stat()
    return {'path': str(path), 'size': info.st_size, 'mtime_ns': info.st_mtime_ns}


def _write_manifest(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _project_indices(points, camera_to_world, intrinsics, width, height, near, far):
    """Return source point IDs, flattened source pixels and camera Z."""
    inverse = np.linalg.inv(camera_to_world).astype(np.float32)
    xyz = points @ inverse[:3, :3].T + inverse[:3, 3]
    selected = np.flatnonzero(np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > near) & (xyz[:, 2] < far))
    xyz = xyz[selected]
    fx, fy, cx, cy = intrinsics
    u, v = xyz[:, 0]/xyz[:, 2]*fx+cx, xyz[:, 1]/xyz[:, 2]*fy+cy
    inside = (u >= -.5) & (u < width-.5) & (v >= -.5) & (v < height-.5)
    pixel = np.floor(v[inside]+.5).astype(np.int64)*width + np.floor(u[inside]+.5).astype(np.int64)
    return selected[inside], pixel, xyz[inside, 2]


def _read_masks(data_dir, stem, shape):
    common = np.zeros(shape, bool)
    dynamic = None
    for folder in ('dynamic_mask', 'sky_mask', 'ego_body_mask', 'semantic'):
        path = data_dir/folder/f'{stem}.png'
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.shape != shape:
            raise ValueError(f'missing or invalid source mask: {path}; expected {shape}')
        if folder == 'dynamic_mask':
            dynamic = mask != 0
        elif folder == 'semantic':
            common |= mask == 255
        else:
            common |= mask != 0
    return common, dynamic


def _source_geometry(render, map_points, map_colors, map_observed, frame_points,
                     original_rgb, common_excluded, dynamic, camera_to_world,
                     intrinsics, width, height, near, far):
    """Classify source geometry and color only source-visible frame points."""
    map_ids, map_pixels, _ = _project_indices(map_points, camera_to_world, intrinsics,
                                             width, height, near, far)
    # Remove aggregate-map samples along currently dynamic image rays, including
    # their uncolored geometry. This avoids stale map samples blocking actors.
    remove_map = np.zeros(len(map_points), bool)
    excluded = (common_excluded | dynamic).reshape(-1)
    remove_map[map_ids] = excluded[map_pixels]
    points, colors, observed = map_points[~remove_map], map_colors[~remove_map], map_observed[~remove_map]
    _, source_map_depth = render.project_points(points, colors, camera_to_world, intrinsics,
        width, height, near, far)
    frame_ids, frame_pixels, _ = _project_indices(frame_points, camera_to_world, intrinsics,
                                                 width, height, near, far)
    remove_frame = np.zeros(len(frame_points), bool)
    remove_frame[frame_ids] = common_excluded.reshape(-1)[frame_pixels]
    frame_points = frame_points[~remove_frame]
    frame_ids, frame_pixels, frame_depths = _project_indices(frame_points, camera_to_world, intrinsics,
                                                            width, height, near, far)
    source_frame_depth = np.full(width*height, np.inf, np.float32)
    np.minimum.at(source_frame_depth, frame_pixels, frame_depths)
    source_depth = np.minimum(source_frame_depth, source_map_depth.reshape(-1))
    visible = frame_depths <= source_depth[frame_pixels] + DEPTH_EPSILON
    frame_colors = np.zeros((len(frame_points), 3), np.uint8)
    frame_observed = np.zeros(len(frame_points), bool)
    frame_colors[frame_ids[visible]] = original_rgb.reshape(-1, 3)[frame_pixels[visible]]
    frame_observed[frame_ids[visible]] = True
    # Colored frame samples win exact-depth ties, followed by colored map
    # samples. Uncolored geometry remains present for occlusion at nearer Z.
    points = np.concatenate((frame_points[frame_observed], points[observed],
                             frame_points[~frame_observed], points[~observed]))
    colors = np.concatenate((frame_colors[frame_observed], colors[observed],
                             frame_colors[~frame_observed], colors[~observed]))
    observed = np.concatenate((np.ones(int(frame_observed.sum())+int(observed.sum()), bool),
                               np.zeros(int((~frame_observed).sum())+int((~observed).sum()), bool)))
    return points, colors, observed


def _render_target(render, points, colors, observed, camera_to_world, intrinsics,
                   width, height, near, far, radius):
    image, raw_depth, raw_observed = render.project_points(points, colors, camera_to_world,
        intrinsics, width, height, near, far, observed)
    color_depth = np.where(raw_observed, raw_depth, np.inf)
    image, color_depth = render.splat(image, color_depth, radius)
    # Apply the same footprint to geometry and color. A nearer uncolored sample
    # blocks a farther colored splat without inventing a photometric value.
    _, geometry_depth = render.splat(np.zeros_like(image), raw_depth, radius)
    mask = np.isfinite(color_depth) & (color_depth <= geometry_depth + DEPTH_EPSILON)
    image[~mask] = 0
    return image, mask


def prepare_conditions(config: dict, project_root: Path) -> dict[float, list[dict]]:
    project_root = Path(project_root).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    render = importlib.import_module('data_processor.nio_processor.nio_render_pcd')
    frame_module = importlib.import_module('data_processor.nio_processor.nio_get_frame_pcd')
    data_dir = _resolve(config['source_path'], project_root)
    output_dir = _resolve(config['output_dir'], project_root)/'conditions'
    metadata_path = data_dir/'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    camera_id = str(int(config.get('camera_id', 0)))
    if camera_id not in metadata['cameras']:
        raise ValueError(f'unknown NIO camera ID: {camera_id}')
    camera = metadata['cameras'][camera_id]
    width, height = int(camera['width']), int(camera['height'])
    intrinsics = np.asarray(camera['intrinsics'], dtype=np.float64)
    if intrinsics.shape != (4,) or not np.isfinite(intrinsics).all() or np.any(intrinsics[:2] <= 0):
        raise ValueError('NIO camera requires finite fx,fy,cx,cy and positive focal lengths')
    all_frames = sorted(metadata['frames'], key=lambda frame: int(frame['frame']))
    if not all_frames:
        raise ValueError('NIO scene contains no converted frames')
    interval = config.get('selected_frames', [all_frames[0]['frame'], all_frames[-1]['frame']])
    if len(interval) != 2 or any(int(value) != value for value in interval) or interval[0] > interval[1]:
        raise ValueError('selected_frames must be an inclusive [first,last] integer interval')
    frames = [frame for frame in all_frames if interval[0] <= int(frame['frame']) <= interval[1]]
    if not frames or len(frames) != int(interval[1]-interval[0]+1):
        raise ValueError('selected_frames contains unavailable NIO frames')
    if any(camera_id not in frame['cameras'] for frame in frames):
        raise ValueError('selected camera is unavailable in one or more selected frames')
    shifts = [float(value) for value in config.get('shifts', [2., 3.])]
    if not shifts or not np.isfinite(shifts).all() or len({f'{shift:.2f}' for shift in shifts}) != len(shifts):
        raise ValueError('shifts must be finite and distinct at two decimal places')
    options = config.get('render', {})
    radius = int(options.get('radius', 2))
    near, far = float(options.get('near', .1)), float(options.get('far', 120.))
    if radius < 0 or radius != options.get('radius', 2) or not np.isfinite([near, far]).all() or not 0 < near < far:
        raise ValueError('render requires integer radius >=0 and finite 0<near<far')
    resume = bool(options.get('resume', True))
    frame_lidar = frame_module.FrameLidar(data_dir)
    extrinsics_path = data_dir/'extrinsics'/f'{camera_id}.txt'
    extrinsics = np.loadtxt(extrinsics_path).reshape(4, 4)
    map_path = data_dir/'lidar/map/background.npz'
    if not map_path.exists():
        map_path = data_dir/'lidar/map/background.ply'
    paths = {metadata_path, map_path, extrinsics_path,
             *(Path(path) for path in frame_lidar.input_paths)}
    intrinsic_path = data_dir/'intrinsics'/f'{camera_id}.txt'
    if intrinsic_path.is_file():
        paths.add(intrinsic_path)
    for frame in frames:
        stem = f"{int(frame['frame']):06d}_{camera_id}"
        paths.add(data_dir/'ego_pose'/f'{stem}.txt')
        paths.add(data_dir/frame['cameras'][camera_id]['image'])
        for folder in ('dynamic_mask', 'sky_mask', 'ego_body_mask', 'semantic'):
            paths.add(data_dir/folder/f'{stem}.png')
    identity = {'schema': SCHEMA, 'config': config, 'camera': camera_id,
        'selected_frames': [int(frame['frame']) for frame in frames],
        'frame_lidar_fingerprint': frame_lidar.fingerprint,
        'inputs': [_stat(path) for path in sorted(paths)],
        'source_hashes': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__).resolve(), Path(render.__file__).resolve())},
        'translation_convention': 'ego_left_positive_y_no_rotation'}
    input_fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    records, destinations, resumable = {}, {}, {}
    for shift in shifts:
        destination = output_dir/f'shift_{shift:.2f}'
        destination.mkdir(parents=True, exist_ok=True)
        destinations[shift] = destination
        manifest_path = destination/'manifest.json'
        previous = json.loads(manifest_path.read_text()) if resume and manifest_path.is_file() else {}
        resumable[shift] = previous.get('input_fingerprint') == input_fingerprint and previous.get('shift_m') == shift
        records[shift] = []
        for frame in frames:
            frame_id, view = int(frame['frame']), frame['cameras'][camera_id]
            stem = f'{frame_id:06d}_{camera_id}'
            ego_pose = np.loadtxt(data_dir/'ego_pose'/f'{stem}.txt').reshape(4, 4)
            shifted_ego = ego_pose.copy()
            shifted_ego[:3, 3] += ego_pose[:3, 1]*shift
            fx, fy, cx, cy = intrinsics
            records[shift].append({'frame': frame_id, 'timestamp': float(view['timestamp']),
                'camera': int(camera_id), 'trajectory_id': f'shift_{shift:.2f}',
                # This K and image grid describe the rectified original-size
                # guidance, before the diffusion runner crops and resizes it.
                'c2w': (shifted_ego @ extrinsics).tolist(),
                'K': [[float(fx), 0., float(cx)], [0., float(fy), float(cy)], [0., 0., 1.]],
                'height': height, 'width': width,
                'image': str((data_dir/view['image']).resolve()),
                'guidance': str(destination/f'{stem}.png'),
                'guidance_mask': str(destination/f'{stem}_mask.png')})
    pending = []
    for position, frame in enumerate(frames):
        needed = [shift for shift in shifts if not (resumable[shift]
            and Path(records[shift][position]['guidance']).is_file()
            and Path(records[shift][position]['guidance_mask']).is_file())]
        if needed:
            pending.append((position, frame, needed))
    if pending:
        map_points, map_colors, map_observed = render.read_map(data_dir)
        if not len(map_points):
            raise ValueError('NIO scene contains no map geometry')
        for count, (position, frame, needed) in enumerate(pending, 1):
            frame_id, view = int(frame['frame']), frame['cameras'][camera_id]
            stem, timestamp = f'{frame_id:06d}_{camera_id}', float(view['timestamp'])
            ego_pose = np.loadtxt(data_dir/'ego_pose'/f'{stem}.txt').reshape(4, 4)
            camera_to_world = ego_pose @ extrinsics
            image = cv2.imread(str(data_dir/view['image']), cv2.IMREAD_COLOR)
            if image is None or image.shape != (height, width, 3):
                raise ValueError(f'missing or invalid source RGB for {stem}')
            original_rgb = image[:, :, ::-1].copy()
            common_excluded, dynamic = _read_masks(data_dir, stem, (height, width))
            frame_points = frame_lidar.at(frame_id, timestamp)
            points, colors, observed = _source_geometry(render, map_points, map_colors, map_observed,
                frame_points, original_rgb, common_excluded, dynamic, camera_to_world,
                intrinsics, width, height, near, far)
            for shift in needed:
                shifted_ego = ego_pose.copy()
                shifted_ego[:3, 3] += ego_pose[:3, 1]*shift
                target_pose = shifted_ego @ extrinsics
                guidance, mask = _render_target(render, points, colors, observed, target_pose,
                    intrinsics, width, height, near, far, radius)
                record = records[shift][position]
                if not cv2.imwrite(record['guidance'], guidance[:, :, ::-1]):
                    raise OSError(f"cannot write guidance: {record['guidance']}")
                if not cv2.imwrite(record['guidance_mask'], mask.astype(np.uint8)*255):
                    raise OSError(f"cannot write guidance mask: {record['guidance_mask']}")
            if count == 1 or count % 10 == 0 or count == len(pending):
                print(f'Rendered shifted NIO conditions {count}/{len(pending)} frames', flush=True)
    for shift in shifts:
        _write_manifest(destinations[shift]/'manifest.json', {'schema_version': SCHEMA,
            'input_fingerprint': input_fingerprint, 'shift_m': shift, 'camera_id': camera_id,
            'translation_convention': 'ego_left_positive_y_no_rotation',
            'render': {'radius': radius, 'near': near, 'far': far}, 'frames': records[shift]})
    return records
