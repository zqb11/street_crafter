"""Prepare native FALCON scans and compensate tracked motion at camera time.

PCD XYZ is transformed from the native LiDAR frame through the calibrated
LiDAR-to-ego transform and the timestamped ego pose. The remaining PCD fields
are not interpreted as RGB or as verified per-point timing information.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import pickle
import threading

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

try:
    from .nio_converter import ClipArchive
    from .nio_helpers import PoseInterpolator, nearest, parse_pose, transform, write_json
except ImportError:
    from nio_converter import ClipArchive
    from nio_helpers import PoseInterpolator, nearest, parse_pose, transform, write_json

MAX_TRACK_GAP = 0.3
MAX_EXTRAPOLATION = 0.1
MAX_SCAN_CAMERA_DELTA = 0.08
BOX_PADDING = 0.15
SCHEMA = 'streetcrafter.nio-frame-lidar/v1'


def implementation_paths():
    directory = Path(__file__).resolve().parent
    return [directory/name for name in ('nio_get_frame_pcd.py', 'nio_helpers.py', 'nio_converter.py')]


def file_signature(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_pcd_xyz(path):
    """Read scalar binary PCD fields and retain finite native XYZ only."""
    types = {('F', 4): '<f4', ('F', 8): '<f8', ('I', 1): 'i1',
             ('I', 2): '<i2', ('I', 4): '<i4', ('I', 8): '<i8',
             ('U', 1): 'u1', ('U', 2): '<u2', ('U', 4): '<u4', ('U', 8): '<u8'}
    with Path(path).open('rb') as stream:
        header = {}
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f'truncated PCD header: {path}')
            text = line.decode('ascii').strip()
            if not text or text.startswith('#'):
                continue
            key, *values = text.split()
            header[key.upper()] = values
            if key.upper() == 'DATA':
                break
        if header.get('DATA') != ['binary']:
            raise ValueError(f'expected binary PCD, got {header.get("DATA")}: {path}')
        fields = header.get('FIELDS', [])
        sizes, kinds = header.get('SIZE', []), header.get('TYPE', [])
        counts = header.get('COUNT', ['1'] * len(fields))
        if not fields or len(set(fields)) != len(fields) or not all(name in fields for name in ('x', 'y', 'z')):
            raise ValueError(f'PCD requires unique XYZ fields: {path}')
        if not len(fields) == len(sizes) == len(kinds) == len(counts):
            raise ValueError(f'PCD field metadata lengths disagree: {path}')
        dtype_fields = []
        for name, size, kind, count in zip(fields, sizes, kinds, counts):
            count = int(count)
            key = kind.upper(), int(size)
            if key not in types or count < 1 or (name in ('x', 'y', 'z') and count != 1):
                raise ValueError(f'unsupported PCD field {name}: {path}')
            dtype_fields.append((name, types[key]) if count == 1 else (name, types[key], (count,)))
        count = int(header.get('POINTS', ['0'])[0])
        width, height = int(header.get('WIDTH', ['0'])[0]), int(header.get('HEIGHT', ['1'])[0])
        if count < 1 or width * height != count:
            raise ValueError(f'PCD point count/dimensions mismatch: {path}')
        dtype = np.dtype(dtype_fields)
        remaining = Path(path).stat().st_size - stream.tell()
        if remaining != count * dtype.itemsize:
            raise ValueError(f'PCD binary size mismatch: {path}')
        records = np.fromfile(stream, dtype=dtype, count=count)
    points = np.column_stack([records[name] for name in ('x', 'y', 'z')]).astype(np.float32)
    return points[np.isfinite(points).all(axis=1)]


class TrackMotion:
    """Interpolate source annotation poses, with short bounded extrapolation."""
    def __init__(self, trajectory, max_gap=MAX_TRACK_GAP, max_extrapolation=MAX_EXTRAPOLATION):
        self.max_gap = float(max_gap)
        self.max_extrapolation = float(max_extrapolation)
        self.tracks = {}
        for identity, track in trajectory.items():
            times = np.asarray(track.get('source_timestamps', []), dtype=np.float64)
            poses = np.asarray(track.get('poses_world', []), dtype=np.float64)
            dimensions = np.asarray([track['length'], track['width'], track['height']], dtype=np.float64)
            if (not times.ndim == 1 or poses.shape != (len(times), 4, 4) or not len(times)
                    or not np.isfinite(times).all() or not np.isfinite(poses).all()
                    or not np.isfinite(dimensions).all() or np.any(dimensions <= 0)):
                raise ValueError(f'invalid source trajectory: {identity}')
            order = np.argsort(times, kind='stable')
            times, poses = times[order], poses[order]
            unique_times, unique_poses = [], []
            for timestamp, pose in zip(times, poses):
                if unique_times and abs(timestamp - unique_times[-1]) <= 1e-6:
                    if not np.allclose(pose, unique_poses[-1], atol=1e-4, rtol=0):
                        raise ValueError(f'conflicting duplicate source poses: {identity}, {timestamp}')
                    continue
                if (not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5)
                        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4)
                        or np.linalg.det(pose[:3, :3]) < 0.99):
                    raise ValueError(f'invalid rigid object pose: {identity}')
                unique_times.append(float(timestamp))
                unique_poses.append(pose)
            self.tracks[identity] = (np.asarray(unique_times), np.asarray(unique_poses), dimensions)

    def dimensions(self, identity):
        return self.tracks[identity][2]

    def sample(self, identity, timestamp):
        times, poses, _ = self.tracks[identity]
        right = int(np.searchsorted(times, timestamp))
        for index in (right, right - 1):
            if 0 <= index < len(times) and abs(times[index] - timestamp) <= 1e-6:
                return poses[index]
        if len(times) < 2:
            return None
        if timestamp < times[0]:
            if times[0] - timestamp > self.max_extrapolation + 1e-6:
                return None
            left, right = 0, 1
        elif timestamp > times[-1]:
            if timestamp - times[-1] > self.max_extrapolation + 1e-6:
                return None
            left, right = len(times) - 2, len(times) - 1
        else:
            left = right - 1
        delta = times[right] - times[left]
        if delta <= 1e-6 or delta > self.max_gap + 1e-6:
            return None
        fraction = (timestamp - times[left]) / delta
        source, target = poses[left], poses[right]
        result = np.eye(4, dtype=np.float64)
        relative = Rotation.from_matrix(source[:3, :3].T @ target[:3, :3]).as_rotvec()
        result[:3, :3] = source[:3, :3] @ Rotation.from_rotvec(relative * fraction).as_matrix()
        result[:3, 3] = source[:3, 3] + fraction * (target[:3, 3] - source[:3, 3])
        return result


def assign_tracks(points, timestamp, tracks, motion, padding):
    assigned = np.full(len(points), -1, dtype=np.int32)
    if not len(points):
        return assigned
    best = np.full(len(points), np.inf, dtype=np.float64)
    tree = cKDTree(points)
    for index, identity in enumerate(tracks):
        pose = motion.sample(identity, timestamp)
        if pose is None:
            continue
        half = motion.dimensions(identity) / 2 + padding
        # A bounding sphere cheaply limits the exact oriented-box test. The
        # small radius margin covers float32 source rotations at box corners.
        radius = np.linalg.norm(half) * (1 + 1e-6) + 1e-6
        candidates = np.asarray(tree.query_ball_point(pose[:3, 3], radius), dtype=np.int64)
        local = (points[candidates] - pose[:3, 3]) @ pose[:3, :3]
        inside = np.all(np.abs(local) <= half, axis=1)
        selected = candidates[inside]
        score = np.sum((local[inside] / half) ** 2, axis=1)
        closer = score < best[selected]
        selected = selected[closer]
        assigned[selected] = index
        best[selected] = score[closer]
    return assigned


def read_geometry_source(raw_tar):
    source = ClipArchive(raw_tar)
    try:
        calibration_member, raw = source.select('/calib/optimized_calibration.json')
        calibration = json.loads(raw)
        lidars = [item for item in calibration.get('lidars', []) if str(item.get('name', '')).lower() == 'falcon']
        if len(lidars) != 1:
            raise ValueError('optimized calibration must contain exactly one FALCON LiDAR')
        extrinsic = lidars[0]['extrinsic_param']
        lidar_to_ego = transform(extrinsic['rotation'], extrinsic['translation'])
        if not np.isfinite(lidar_to_ego).all():
            raise ValueError('nonfinite FALCON extrinsic')
        pose_member, raw = source.select('/odom/pose.txt', '/vis_mapping_results/')
        records = [record for line in raw.decode().splitlines() if (record := parse_pose(line)) is not None]
        return PoseInterpolator(records), lidar_to_ego, calibration_member, pose_member
    finally:
        source.close()


def update_scene_metadata(path, scene, tracks_present):
    scene.update({'frame_lidar_available': True, 'actor_pointcloud_available': bool(tracks_present),
                  'frame_lidar_metadata': 'lidar/frame/metadata.json'})
    write_json(path, scene)


def process(args):
    data_dir = Path(args.data_dir).resolve()
    scene_path, trajectory_path = data_dir/'metadata.json', data_dir/'track'/'trajectory.pkl'
    scene = json.loads(scene_path.read_text())
    frames = scene['frames']
    if not frames:
        raise ValueError('converted scene contains no frames')
    raw_dir = Path(args.raw_lidar_dir).resolve() if args.raw_lidar_dir else data_dir/'raw_lidar'/data_dir.name/'output'/'FALCON'
    scan_paths = sorted(raw_dir.glob('*.pcd'))
    if not scan_paths:
        raise FileNotFoundError(f'no FALCON PCD scans in {raw_dir}')
    scans = {}
    for path in scan_paths:
        try:
            timestamp = float(path.stem)
        except ValueError as error:
            raise ValueError(f'PCD filename must be a timestamp in seconds: {path}') from error
        if not np.isfinite(timestamp) or timestamp in scans:
            raise ValueError(f'invalid or duplicate FALCON timestamp: {path}')
        scans[timestamp] = path
    frame_views, selected, max_difference = {}, {}, 0.
    for frame in frames:
        scan_path, timestamp = nearest(scans, float(frame['timestamp']))
        delta = max(abs(float(view['timestamp']) - timestamp) for view in frame['cameras'].values())
        if delta > MAX_SCAN_CAMERA_DELTA + 1e-6:
            raise ValueError(f"frame {frame['frame']} scan/camera timestamp gap {delta:.6f}s exceeds {MAX_SCAN_CAMERA_DELTA}s")
        if str(int(frame['frame'])) in frame_views:
            raise ValueError(f"duplicate converted frame: {frame['frame']}")
        frame_views[str(int(frame['frame']))] = {'path': f'lidar/frame/{scan_path.stem}.npz',
            'timestamp': timestamp, 'max_camera_delta_s': delta}
        selected[timestamp] = scan_path
        max_difference = max(max_difference, delta)
    inputs = {'raw_tar': file_signature(args.raw_tar), 'trajectory': file_signature(trajectory_path),
              'scans': [file_signature(path) for path in scan_paths],
              'frames': [{'frame': f['frame'], 'timestamp': f['timestamp'],
                          'cameras': {key: view['timestamp'] for key, view in f['cameras'].items()}} for f in frames],
              'algorithm': {'schema': SCHEMA,
                  'sources': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in implementation_paths()},
                  'box_padding_m': BOX_PADDING, 'max_track_gap_s': MAX_TRACK_GAP,
                  'max_extrapolation_s': MAX_EXTRAPOLATION, 'max_scan_camera_delta_s': MAX_SCAN_CAMERA_DELTA}}
    input_fingerprint = fingerprint(inputs)
    output = data_dir/'lidar'/'frame'
    manifest_path = output/'metadata.json'
    if args.resume and manifest_path.is_file():
        old = json.loads(manifest_path.read_text())
        caches = {view['path'] for view in frame_views.values()}
        if (old.get('schema_version') == SCHEMA and old.get('input_fingerprint') == input_fingerprint
                and old.get('frames') == frame_views and all((data_dir/path).is_file() for path in caches)):
            update_scene_metadata(scene_path, scene, old.get('assigned_point_count', 0) > 0)
            print(f'Reusing {len(caches)} FALCON scan caches', flush=True)
            return
    with trajectory_path.open('rb') as stream:
        trajectory = pickle.load(stream)
    motion = TrackMotion(trajectory)
    tracks = sorted(trajectory)
    ego_poses, lidar_to_ego, calibration_member, pose_member = read_geometry_source(args.raw_tar)
    output.mkdir(parents=True, exist_ok=True)
    # An interrupted rebuild must not leave a usable manifest for mixed caches.
    manifest_path.unlink(missing_ok=True)
    point_count, assigned_count = 0, 0
    for count, (timestamp, path) in enumerate(sorted(selected.items()), 1):
        native = read_pcd_xyz(path)
        lidar_to_world = ego_poses(timestamp) @ lidar_to_ego
        points = (native @ lidar_to_world[:3, :3].T + lidar_to_world[:3, 3]).astype(np.float32)
        assigned = assign_tracks(points, timestamp, tracks, motion, BOX_PADDING)
        target = output/f'{path.stem}.npz'
        temporary = target.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, points=points, track_index=assigned, timestamp=np.float64(timestamp))
        temporary.replace(target)
        point_count += len(points)
        assigned_count += int(np.count_nonzero(assigned >= 0))
        if count == 1 or count % 10 == 0 or count == len(selected):
            print(f'Prepared FALCON scans {count}/{len(selected)}', flush=True)
    manifest = {'schema_version': SCHEMA, 'input_fingerprint': input_fingerprint,
        'input_files': {'raw_tar': inputs['raw_tar'], 'trajectory': inputs['trajectory']},
        'raw_lidar_dir': str(raw_dir), 'frames': frame_views, 'tracks': tracks,
        'coordinate_frame': 'source_world', 'scan_coordinate_frame': 'native_falcon',
        'calibration_member': calibration_member, 'pose_member': pose_member,
        'lidar_to_ego': lidar_to_ego.tolist(), 'point_count': point_count,
        'assigned_point_count': assigned_count, 'num_scans': len(selected),
        'box_padding_m': BOX_PADDING, 'max_track_gap_s': MAX_TRACK_GAP,
        'max_extrapolation_s': MAX_EXTRAPOLATION,
        'max_scan_camera_delta_s': MAX_SCAN_CAMERA_DELTA,
        'actual_max_scan_camera_delta_s': max_difference,
        'per_point_motion_compensation': False, 'pcd_auxiliary_fields_used': False}
    write_json(manifest_path, manifest)
    update_scene_metadata(scene_path, scene, assigned_count > 0)
    print(f'Saved {len(selected)} scans with {assigned_count}/{point_count} tracked points', flush=True)


class FrameLidar:
    """Thread-safe four-scan LRU with per-call tracked-object compensation."""
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir).resolve()
        self.manifest_path = self.data_dir/'lidar'/'frame'/'metadata.json'
        self.metadata = json.loads(self.manifest_path.read_text())
        if self.metadata.get('schema_version') != SCHEMA:
            raise ValueError(f'unsupported frame LiDAR schema: {self.manifest_path}')
        trajectory_path = self.data_dir/'track'/'trajectory.pkl'
        if self.metadata['input_files']['trajectory'] != file_signature(trajectory_path):
            raise ValueError('trajectory changed; regenerate the frame LiDAR cache')
        with trajectory_path.open('rb') as stream:
            trajectory = pickle.load(stream)
        self.motion = TrackMotion(trajectory, self.metadata['max_track_gap_s'], self.metadata['max_extrapolation_s'])
        self.tracks = self.metadata['tracks']
        if len(set(self.tracks)) != len(self.tracks) or set(self.tracks) != set(trajectory):
            raise ValueError('frame LiDAR track table differs from trajectory')
        self.fingerprint = self.metadata['input_fingerprint']
        paths = sorted({view['path'] for view in self.metadata['frames'].values()})
        self.input_paths = [*implementation_paths(), self.manifest_path, trajectory_path,
                            *(self.data_dir/path for path in paths)]
        self._cache = OrderedDict()
        self._lock = threading.RLock()

    def _scan(self, path):
        with self._lock:
            if path in self._cache:
                self._cache.move_to_end(path)
                return self._cache[path]
            with np.load(self.data_dir/path, allow_pickle=False) as data:
                points = np.asarray(data['points'], dtype=np.float32)
                indices = np.asarray(data['track_index'], dtype=np.int32)
                timestamp = float(data['timestamp'])
            if (points.ndim != 2 or points.shape[1] != 3 or indices.shape != (len(points),)
                    or not np.isfinite(points).all() or not np.isfinite(timestamp)
                    or np.any(indices < -1) or np.any(indices >= len(self.tracks))):
                raise ValueError(f'invalid frame LiDAR cache: {path}')
            value = points, indices, timestamp
            self._cache[path] = value
            while len(self._cache) > 4:
                self._cache.popitem(last=False)
            return value

    def at(self, frame_id, camera_timestamp):
        frame_key = str(int(frame_id))
        if frame_key not in self.metadata['frames']:
            raise KeyError(f'no frame LiDAR for frame {frame_id}')
        view = self.metadata['frames'][frame_key]
        camera_timestamp = float(camera_timestamp)
        if (not np.isfinite(camera_timestamp)
                or abs(camera_timestamp - view['timestamp']) > self.metadata['max_scan_camera_delta_s'] + 1e-6):
            raise ValueError(f'camera timestamp too far from frame {frame_id} scan')
        points, indices, scan_timestamp = self._scan(view['path'])
        if abs(scan_timestamp - view['timestamp']) > 1e-6:
            raise ValueError(f'frame LiDAR scan timestamp mismatch: {view["path"]}')
        result = points.copy()
        keep = np.ones(len(points), dtype=bool)
        for index in np.unique(indices[indices >= 0]):
            selected = indices == index
            identity = self.tracks[index]
            source = self.motion.sample(identity, scan_timestamp)
            target = self.motion.sample(identity, camera_timestamp)
            if source is None or target is None:
                keep[selected] = False
                continue
            local = (points[selected] - source[:3, 3]) @ source[:3, :3]
            result[selected] = local @ target[:3, :3].T + target[:3, 3]
        return result[keep]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--raw_tar', required=True)
    parser.add_argument('--raw_lidar_dir')
    parser.add_argument('--resume', action='store_true')
    process(parser.parse_args())

# 转换逐帧FALCON点云，按目标真实时间匹配轨迹
if __name__ == '__main__':
    main()
