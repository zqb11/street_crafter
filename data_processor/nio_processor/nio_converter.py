"""Convert a NIO processed-clip tar into StreetCrafter preprocessing inputs.

Reads individual tar members without extracting the archive. CPU workers only
write disjoint frame files; calibration/poses/track metadata are serialized by
the main thread. No dependencies are installed or compiled automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import re
import struct
import subprocess
import tarfile
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

try:
    from .nio_helpers import (CAMERA_ALIASES, CAMERA_IDS, LABEL_NAMES, PoseInterpolator,
                              box_corners, nearest, parse_pose, rectification, transform, write_json)
except ImportError:
    from nio_helpers import (CAMERA_ALIASES, CAMERA_IDS, LABEL_NAMES, PoseInterpolator,
                             box_corners, nearest, parse_pose, rectification, transform, write_json)

ASSET = re.compile(r'(?:^|/)camera_full/([A-Z0-9]+)(_seg|_mono_depth)?/([0-9]+(?:\.[0-9]+)?)\.(jpg|jpeg|png|elm)$', re.I)
DEFAULT_DECODER = '/home/nio/.cache/drivegs/depth_decoder/5505691d8d97178a/dump_depth_elm'
DEPTH_HEADER = struct.Struct('<8sIIQI')
DEPTH_RECORD = np.dtype([('id', '<i8'), ('value', '<f4', (3,))])


class ClipArchive:
    def __init__(self, path):
        self.archive = tarfile.open(path, 'r:')
        self.members = {member.name: member for member in self.archive.getmembers() if member.isfile()}
        self.lock = threading.Lock()

    def read(self, name):
        with self.lock:
            stream = self.archive.extractfile(self.members[name])
            if stream is None:
                raise OSError(f'unreadable tar member: {name}')
            with stream:
                return stream.read()

    def select(self, suffix, preferred=None):
        candidates = sorted(name for name in self.members if name.endswith(suffix))
        if preferred:
            selected = [name for name in candidates if preferred in name]
            if selected:
                candidates = selected
        if not candidates:
            raise FileNotFoundError(f'no source member ends with {suffix}')
        raw = self.read(candidates[0])
        if any(self.read(name) != raw for name in candidates[1:]):
            raise ValueError(f'conflicting source members: {candidates}')
        return candidates[0], raw

    def close(self):
        self.archive.close()


def build_index(source, max_semantic_delta):
    assets = {}
    for name in source.members:
        match = ASSET.search(name)
        if match:
            camera, suffix, timestamp, extension = match.groups()
            kind = {None: 'rgb', '_seg': 'semantic', '_mono_depth': 'depth'}[suffix]
            key, timestamp = (camera.upper(), kind), float(timestamp)
            if timestamp in assets.setdefault(key, {}):
                raise ValueError(f'duplicate asset for {key}, {timestamp}')
            assets[key][timestamp] = name
    _, raw = source.select('/odom/pose.txt', '/vis_mapping_results/')
    records = [record for line in raw.decode().splitlines() if (record := parse_pose(line)) is not None]
    interpolator = PoseInterpolator(records)
    frames = []
    for record_index, (pose, timestamps) in enumerate(records):
        if any(name not in timestamps for name in ['Body', *CAMERA_IDS]):
            continue
        reason = None
        views = {}
        for camera, camera_id in CAMERA_IDS.items():
            timestamp = timestamps[camera]
            if timestamp not in assets.get((camera, 'rgb'), {}):
                reason = f'missing RGB {camera}'; break
            if timestamp < interpolator.times[0] or timestamp > interpolator.times[-1]:
                reason = f'outside pose range {camera}'; break
            semantic, semantic_timestamp = nearest(assets.get((camera, 'semantic'), {}), timestamp)
            if abs(semantic_timestamp - timestamp) > max_semantic_delta:
                reason = f'semantic timestamp difference {camera}'; break
            views[str(camera_id)] = {'timestamp': timestamp,
                'image': f'images/{len(frames):06d}_{camera_id}.png',
                'source_image': assets[(camera, 'rgb')][timestamp],
                'semantic_timestamp': semantic_timestamp, 'source_semantic': semantic,
                'source_depth': assets.get((camera, 'depth'), {}).get(timestamp)}
        if not reason:
            frames.append({'frame': len(frames), 'timestamp': timestamps['Body'],
                           'source_record_index': record_index, 'cameras': views})
    if not frames:
        raise ValueError('no complete synchronized frames survived matching')
    return frames, interpolator


def decode_depth(source, name, timestamp, decoder):
    with tempfile.TemporaryDirectory(prefix='nio-depth-') as temporary:
        elm, dump = Path(temporary)/'depth.elm', Path(temporary)/'depth.bin'
        elm.write_bytes(source.read(name))
        subprocess.run([decoder, str(elm), str(dump)], check=True, capture_output=True)
        raw = dump.read_bytes()
    if len(raw) < DEPTH_HEADER.size:
        raise ValueError(f'truncated depth dump: {name}')
    magic, width, height, count, aux_count = DEPTH_HEADER.unpack_from(raw)
    offset = DEPTH_HEADER.size + aux_count*8
    if magic != b'GSBAUX01' or not width or not height or len(raw) != offset + count*DEPTH_RECORD.itemsize:
        raise ValueError(f'invalid depth dump: {name}')
    aux = np.frombuffer(raw, '<f8', count=aux_count, offset=DEPTH_HEADER.size)
    if len(aux) and (not np.isfinite(aux).all() or abs(aux[0]-timestamp) > 1e-6):
        raise ValueError(f'depth timestamp mismatch: {name}')
    records = np.frombuffer(raw, DEPTH_RECORD, count=count, offset=offset)
    ids, values = records['id'], records['value'][:, 0]
    if np.any((ids < 0) | (ids >= width*height)) or np.unique(ids).size != len(ids):
        raise ValueError(f'invalid/duplicate depth pixel indices: {name}')
    selected = np.isfinite(values) & (values > 1e-6)
    depth, valid = np.zeros((height, width), np.float32), np.zeros((height, width), bool)
    depth.flat[ids[selected]], valid.flat[ids[selected]] = values[selected], True
    return depth, valid, {'timestamp_s': timestamp, 'source_member': name,
         'width': width, 'height': height, 'raw_count': count, 'valid_count': int(valid.sum()),
         'rejected_value_count': int((~selected).sum()), 'aux_data': aux.tolist()}


def write_png(path, raster):
    if not cv2.imwrite(str(path), raster, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        raise OSError(f'cannot write {path}')


def materialize_frame(frame, *, source, output, maps, cameras, decoder, resume, previous_depth):
    depth_info = {}
    for camera_id, view in frame['cameras'].items():
        stem = f"{frame['frame']:06d}_{camera_id}"
        names = ['images', 'semantic', 'dynamic_mask', 'sky_mask', 'ego_body_mask', 'ground_mask']
        paths = {kind: output/kind/f'{stem}.png' for kind in names}
        if not resume or not all(path.is_file() for path in paths.values()):
            # Pillow uses the existing JPEG decoder efficiently on this host;
            # OpenCV remains responsible for rectification and PNG encoding.
            with Image.open(io.BytesIO(source.read(view['source_image']))) as image:
                rgb = np.asarray(image.convert('RGB'))[:, :, ::-1].copy()
            calibration = cameras[camera_id]['native_calibration']['intrinsic_param']
            if rgb.shape[:2] != (calibration['camera_height'], calibration['camera_width']):
                raise ValueError(f'native RGB dimensions disagree with calibration: {stem}')
            with Image.open(io.BytesIO(source.read(view['source_semantic']))) as image:
                labels = np.asarray(image)
            if labels.ndim != 2 or labels.dtype != np.uint8:
                raise ValueError('semantic PNG must contain uint8 palette indices')
            labels = np.where(labels < len(LABEL_NAMES), labels, 255).astype(np.uint8)
            u, v, valid = maps[camera_id]# 畸变原图
            rectified_rgb = cv2.remap(rgb, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)# 从原图采样生成去畸变的针孔图
            # Semantic pixel coordinates are expressed on a differently sized
            # native grid; preserve the producer's scaled principal point convention.
            su = u * (labels.shape[1]/rgb.shape[1])
            sv = v * (labels.shape[0]/rgb.shape[0])
            semantic = cv2.remap(labels, su, sv, cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=255)
            semantic[~valid] = 255
            write_png(paths['images'], rectified_rgb)# 保存去畸变图像
            write_png(paths['semantic'], semantic)
            for kind, selected in {'dynamic_mask': np.isin(semantic, [6, 24]),
                  'sky_mask': semantic == 23, 'ego_body_mask': semantic == 22,
                  'ground_mask': semantic == 1}.items():
                write_png(paths[kind], selected.astype(np.uint8)*255)
        if view['source_depth'] and decoder:
            depth_path, mask_path = output/'mono_depth'/f'{stem}.npy', output/'mono_depth_mask'/f'{stem}.png'
            if resume and depth_path.is_file() and mask_path.is_file() and stem in previous_depth:
                depth_info[stem] = previous_depth[stem]
            else:
                depth, valid, stats = decode_depth(source, view['source_depth'], view['timestamp'], decoder)
                np.save(depth_path, depth, allow_pickle=False)
                write_png(mask_path, valid.astype(np.uint8)*255)
                depth_info[stem] = stats
    return depth_info


def convert_tracks(source, frames, poses, output, cameras, extrinsics, tolerance):
    _, raw = source.select('retrack_gt_dlb360_bev.json')
    payload = json.loads(raw)
    if payload.get('coordinate_type') != 'world':
        raise ValueError('DLB360 track coordinates must be world')
    observations = {}
    annotation_timestamps = set()
    for source_frame in payload['frame_sequence']:
        # Several 15 Hz camera annotations can share a 10 Hz FALCON time.
        # Validate uniqueness using camera time rather than FALCON time.
        annotation_timestamp = float(source_frame['FW']['timestamp'])
        if annotation_timestamp in annotation_timestamps:
            raise ValueError('duplicate annotated FW timestamp')
        annotation_timestamps.add(annotation_timestamp)
        for obj in source_frame.get('od_gt', []):
            identity = f"{obj['type']}:{obj['instance_id']}"
            timestamp = float(obj['attr']['timestamp'])
            observations.setdefault(identity, {})[timestamp] = obj
    info, visible, trajectory, object_ids = {}, {}, {}, {}
    for frame in frames:
        key, timestamp = f"{frame['frame']:06d}", frame['timestamp']
        ego = poses(timestamp)
        info[key], visible[key] = {}, {int(cam): [] for cam in cameras}
        camera_worlds = {cam_id: poses(frame['cameras'][cam_id]['timestamp']) @ extrinsics[cam_id]
                         for cam_id in cameras}
        for identity, samples in observations.items():
            obj, obj_timestamp = nearest(samples, timestamp)
            if abs(obj_timestamp-timestamp) > tolerance:
                continue
            attr, box = obj['attr'], obj['3d_info']
            width, height, length = map(float, attr['size'])
            if min(width, height, length) <= 0:
                continue
            yaw, pitch, roll = map(float, box['orientation'])
            world_pose = transform(Rotation.from_euler('ZYX', [yaw, pitch, roll]).as_matrix(), box['center_point'])
            vehicle_pose = np.linalg.inv(ego) @ world_pose
            label = {'vehicle_rect': 'vehicle', 'pedestrian_rect': 'pedestrian',
                     'cyclist_rect': 'cyclist'}.get(obj['type'], 'misc')
            velocity = np.asarray(box.get('velocity', [0, 0, 0]))
            speed = float(np.linalg.norm(velocity[:2]))
            converted = {'height': height, 'width': width, 'length': length,
                'center_x': float(vehicle_pose[0, 3]), 'center_y': float(vehicle_pose[1, 3]),
                'center_z': float(vehicle_pose[2, 3]),
                'heading': float(np.arctan2(vehicle_pose[1, 0], vehicle_pose[0, 0])),
                'label': label, 'speed': speed, 'timestamp': timestamp,
                'source_timestamp': obj_timestamp}
            info[key][identity] = {'lidar_box': converted, 'camera_box': converted,
                                  'geometry_source': 'annotation_without_frame_lidar'}
            object_ids.setdefault(identity, len(object_ids))
            corners = box_corners(world_pose, [length, width, height])
            for cam_id, camera in cameras.items():
                camera_world = camera_worlds[cam_id]
                points = (corners-camera_world[:3, 3]) @ camera_world[:3, :3]
                fx, fy, cx, cy = camera['intrinsics']
                front = points[:, 2] > 1e-6
                if not front.any():
                    continue
                uv = points[front, :2]/points[front, 2:3]*[fx, fy]+[cx, cy]
                # Bounding rectangle intersection also catches boxes spanning
                # the image even when all projected corners lie outside.
                if uv[:, 0].max() >= 0 and uv[:, 1].max() >= 0 and uv[:, 0].min() < camera['width'] and uv[:, 1].min() < camera['height']:
                    visible[key][int(cam_id)].append(identity)
            track = trajectory.setdefault(identity, {'label': label, 'frames': [], 'timestamps': [],
                'source_timestamps': [], 'poses_vehicle': [], 'poses_world': [], 'speeds': [], 'dims': [],
                'symmetric': label != 'pedestrian', 'deformable': label in ('pedestrian', 'cyclist')})
            for field, value in [('frames', frame['frame']), ('timestamps', timestamp), ('source_timestamps', obj_timestamp),
                    ('poses_vehicle', vehicle_pose), ('poses_world', world_pose), ('speeds', speed),
                    ('dims', [height, width, length])]:
                track[field].append(value)
    for track in trajectory.values():
        track['height'], track['width'], track['length'] = np.max(track.pop('dims'), axis=0).tolist()
        for field in ['poses_vehicle', 'poses_world']:
            track[field] = np.asarray(track[field], dtype=np.float32)
        positions = track['poses_world'][:, :3, 3]
        dynamic = np.any(np.std(positions, axis=0) > .5) or np.linalg.norm(positions[0]-positions[-1]) > 2
        track['stationary'] = not bool(dynamic)
    for filename, value in [('track_info.pkl', info), ('track_camera_visible.pkl', visible), ('trajectory.pkl', trajectory)]:
        with (output/'track'/filename).open('wb') as stream:
            pickle.dump(value, stream)
    write_json(output/'track'/'track_ids.json', object_ids)

def convert(args):
    if args.max_width < 1 or args.workers < 1 or args.max_semantic_delta < 0 or args.track_tolerance < 0:
        raise ValueError('invalid dimensions, workers or timestamp tolerance')
    cv2.setNumThreads(1)
    output = Path(args.save_dir).resolve()
    for directory in ['images', 'intrinsics', 'extrinsics', 'ego_pose', 'semantic', 'dynamic_mask',
                      'sky_mask', 'ego_body_mask', 'ground_mask', 'mono_depth', 'mono_depth_mask', 'track']:
        (output/directory).mkdir(parents=True, exist_ok=True)
    source = ClipArchive(args.raw_tar)
    try:
        calibration_name, calibration_raw = source.select('/calib/optimized_calibration.json')
        calibrations = {c['name']: c for c in json.loads(calibration_raw)['cameras']}
        frames, poses = build_index(source, args.max_semantic_delta)
        if args.max_frames is not None:
            if args.max_frames < 1:
                raise ValueError('--max_frames must be positive')
            frames = frames[:args.max_frames]
        config = {'max_width': args.max_width, 'svc_fov': args.svc_fov, 'adas_fov': args.adas_fov,
                  'max_semantic_delta': args.max_semantic_delta, 'track_tolerance': args.track_tolerance}
        metadata_path = output/'metadata.json'
        # Reject incompatible resume options before changing any calibration,
        # poses, image files or metadata in the existing scene.
        if args.resume and metadata_path.is_file():
            old = json.loads(metadata_path.read_text())
            if old.get('conversion_options') != config or old.get('source_tar') != str(Path(args.raw_tar).resolve()):
                raise ValueError('--resume output was created with different conversion options/input')
            if old.get('num_frames', 0) > len(frames):
                raise ValueError('--resume cannot reduce frame count; use a fresh output directory')
        maps, cameras, extrinsics = {}, {}, {}
        for name, camera_id in CAMERA_IDS.items():
            key, calibration = str(camera_id), calibrations[CAMERA_ALIASES[name]]
            k, u, v, valid, details = rectification(calibration, args.max_width, args.svc_fov, args.adas_fov)
            maps[key] = u, v, valid
            cameras[key] = {'name': name, 'calibration_name': CAMERA_ALIASES[name],
                'width': u.shape[1], 'height': u.shape[0], 'intrinsics': [float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])],
                'native_calibration': calibration, 'rectification': details}
            ext = calibration['extrinsic_param']
            extrinsics[key] = transform(ext['rotation'], ext['translation'])
            np.savetxt(output/'intrinsics'/f'{camera_id}.txt', cameras[key]['intrinsics'])
            np.savetxt(output/'extrinsics'/f'{camera_id}.txt', extrinsics[key])
        metadata = {'schema_version': 'streetcrafter.nio/v1', 'source_tar': str(Path(args.raw_tar).resolve()),
            'sequence_id': output.name, 'calibration_member': calibration_name,
            'calibration_sha256': hashlib.sha256(calibration_raw).hexdigest(),
            'coordinate_conventions': {'camera': 'opencv_right_down_forward', 'extrinsics': 'camera_to_ego',
                'ego_pose': 'ego_to_world', 'world_origin': 'source_unchanged', 'translation_units': 'm'},
            'num_frames': len(frames), 'cameras': cameras, 'frames': frames,
            'semantic_labels': dict(enumerate(LABEL_NAMES)), 'semantic_ignore_index': 255,
            'geometry_source': 'aggregated_lidar_map', 'depth_source': 'map_projection',
            'actor_pointcloud_available': False, 'conversion_options': config, 'conversion_complete': False}
        write_json(metadata_path, metadata)
        timestamps = {}
        for frame in frames:
            np.savetxt(output/'ego_pose'/f"{frame['frame']:06d}.txt", poses(frame['timestamp']))
            for camera_id, view in frame['cameras'].items():
                stem = f"{frame['frame']:06d}_{camera_id}"
                np.savetxt(output/'ego_pose'/f'{stem}.txt', poses(view['timestamp']))
                timestamps[stem] = view['timestamp']
        write_json(output/'timestamps.json', timestamps)
        write_json(output/'images'/'timestamps.json', timestamps)
        convert_tracks(source, frames, poses, output, cameras, extrinsics, args.track_tolerance)
        decoder = None if args.skip_mono_depth else args.depth_decoder
        if decoder and (not Path(decoder).is_file() or not os.access(decoder, os.X_OK)):
            raise FileNotFoundError(f'depth decoder unavailable: {decoder}; provide --depth_decoder or --skip_mono_depth')
        old_depth_path = output/'mono_depth'/'metadata.json'
        previous_depth = json.loads(old_depth_path.read_text()).get('views', {}) if args.resume and old_depth_path.exists() else {}
        depth_views = {}
        with ThreadPoolExecutor(max_workers=args.workers) as workers:
            results = workers.map(lambda frame: materialize_frame(frame, source=source, output=output,
                maps=maps, cameras=cameras, decoder=decoder, resume=args.resume, previous_depth=previous_depth), frames)
            for count, result in enumerate(results, 1):
                depth_views.update(result)
                if count == 1 or count % 10 == 0 or count == len(frames):
                    print(f'Converted {count}/{len(frames)} synchronized frames', flush=True)
        depth_metadata = {'units': 'm', 'convention': 'camera_z',
            'image_grid': 'native_scaled_unrectified', 'invalid_value': 0, 'valid_min_exclusive_m': 1e-6,
            'used_for_geometry': False, 'decoder': decoder, 'views': depth_views}
        write_json(old_depth_path, depth_metadata)
        metadata['conversion_complete'] = True
        write_json(metadata_path, metadata)
        print(f"NIO conversion complete: {len(frames)} frames, {len(cameras)} cameras -> {output}", flush=True)
    finally:
        source.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw_tar', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--max_width', type=int, default=1920)
    parser.add_argument('--max_frames', type=int)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--svc_fov', type=float, default=120., help='SVC output pinhole horizontal FOV in degrees')
    parser.add_argument('--adas_fov', type=float, help='ADAS pinhole FOV override; default uses scaled native K')
    parser.add_argument('--max_semantic_delta', type=float, default=.04, help='maximum semantic/RGB timestamp difference in seconds')
    parser.add_argument('--track_tolerance', type=float, default=.035)
    parser.add_argument('--depth_decoder', default=DEFAULT_DECODER)
    parser.add_argument('--skip_mono_depth', action='store_true')
    convert(parser.parse_args())

"""
先建立同步帧索引，再转换图像、内外参、位姿、掩码和轨迹，输出：
metadata.json: 处理结果的格式说明，包括cameras-每路相机的参数、frames-逐帧数据索引、坐标系、语义、几何来源、转换参数和状态
timestamps.json: 每一张图像的时间戳，单位：秒
images: 0-FW 1-FN 2-FL 3-FR 4-RL 5-RR 6-RN 7-SFW 8-SLW 9-SRW 10-SRCW
intrinsics
extrinsics: 相机到自车的刚性变换矩阵
ego_pose: 自车位姿和每个相机曝光时的自车位姿
semantic: 语义类别图 0-地面 1-背景 6-动态障碍物 22-行人/骑自行车的人
dynamic_mask: 动态实例掩码
sky_mask: 天空掩码
ego_body_mask: 车体掩码
ground_mask: 地面掩码(排除地面标线)
track:
    track_info.pkl: 每一帧有哪些实例以及实例框标注
    track_camera_visible.pkl: 每一帧每个相机包含哪些实例
    trajectory.pkl: 每个实例的轨迹
    track_ids.json: “实例类型: 实例ID” 到连续整数编号的映射
mono_depth: 单目预测深度图，是相机z深度
注：单目深度保留原始预测网格，没有进行去畸变，images中的 RGB 已转换为针孔视图，两者不能直接逐像素对应
lidar/depth/来自聚合地图投影，与这里的 mono_depth是不同来源的深度
mono_depth_mask: 单目深度掩码
"""
if __name__ == '__main__':
    main()
