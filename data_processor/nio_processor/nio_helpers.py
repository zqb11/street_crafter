"""Camera and timestamp conventions shared by NIO preprocessing.

Projection follows drive-gs camera/{nio,xutils,spline}.py and the NIO
OCam projector in cameraProjections.cuh. All output cameras use right/down/
forward coordinates; rectification preserves their native optical axes.
"""
from __future__ import annotations

import bisect
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

CAMERA_ALIASES = {
    'FW': 'ADAS_Front120', 'FN': 'ADAS_Front30',
    'FL': 'ADAS_FrontLeft', 'FR': 'ADAS_FrontRight',
    'RL': 'ADAS_RearLeft', 'RR': 'ADAS_RearRight', 'RN': 'ADAS_Rear',
    'SFW': 'SVC_Front', 'SLW': 'SVC_Left', 'SRW': 'SVC_Right', 'SRCW': 'SVC_Rear',
}
CAMERA_IDS = {name: index for index, name in enumerate(CAMERA_ALIASES)}
LABEL_NAMES = (
    'background', 'ground', 'white_lane', 'curb', 'arrow', 'crosswalk',
    'dynamic_obstacle', 'nature', 'sidewalk', 'big_obstacle', 'small_obstacle',
    'road_isolation', 'traffic_sign', 'yellow_lane', 'parking_area',
    'parking_yellow_lane', 'parking_white_lane', 'parking_other_lane', 'bridge',
    'stop_line', 'traffic_light', 'road_marker', 'ego_body', 'sky', 'pedestrian_cyclist',
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def transform(rotation, translation):
    result = np.eye(4, dtype=np.float64)
    value = np.asarray(rotation, dtype=np.float64)
    result[:3, :3] = Rotation.from_rotvec(value).as_matrix() if value.shape == (3,) else value
    result[:3, 3] = translation
    return result


def parse_pose(line):
    parts = line.strip().split('/')
    if not parts[0].startswith('Tw2b:'):
        return None
    values = [float(value) for value in parts[0].split(':', 1)[1].split(',')]
    if len(values) < 6:
        raise ValueError('Tw2b record needs a Rodrigues rotation and translation')
    timestamps = {}
    for part in parts[1:]:
        if ':' in part:
            name, value = part.split(':', 1)
            timestamps[name] = float(value)
    return np.linalg.inv(transform(values[:3], values[3:6])), timestamps


class PoseInterpolator:
    def __init__(self, records):
        ordered = sorted((ts['Body'], pose) for pose, ts in records if 'Body' in ts)
        self.times = np.asarray([item[0] for item in ordered], dtype=np.float64)
        self.poses = np.asarray([item[1] for item in ordered])
        if len(self.times) < 2 or np.any(np.diff(self.times) <= 0):
            raise ValueError('pose timestamps must be unique and increasing')
        # Relative seconds preserve precision for Unix timestamps.
        self.slerp = Slerp(self.times - self.times[0], Rotation.from_matrix(self.poses[:, :3, :3]))

    def __call__(self, timestamp):
        if timestamp < self.times[0] or timestamp > self.times[-1]:
            raise ValueError(f'camera timestamp outside pose range: {timestamp}')
        pose = np.eye(4)
        pose[:3, :3] = self.slerp(timestamp - self.times[0]).as_matrix()
        pose[:3, 3] = [np.interp(timestamp, self.times, self.poses[:, axis, 3]) for axis in range(3)]
        return pose


def nearest(values, timestamp):
    times = sorted(values)
    if not times:
        raise ValueError('no timestamped assets')
    index = bisect.bisect_left(times, timestamp)
    selected = min(times[max(0, index - 1):min(len(times), index + 1)],
                   key=lambda value: (abs(value - timestamp), value))
    return values[selected], selected


def _spline_value(node, theta, phi):
    """Numpy implementation of xutils' periodic quadratic Spline2."""
    if node['order'] != 3:
        raise ValueError('unsupported spline order')
    dx, dy = float(node['knot_interval_x']), float(node['knot_interval_y'])
    if abs(node['knot_start_x']) > 1e-8 or abs(node['knot_start_y'] + math.pi) > 1e-5:
        raise ValueError('unsupported spline knot origin')
    cols = round(2 * math.pi / dy)
    points = np.asarray(node['control_point_data'], dtype=np.float64).reshape(-1, cols, 2)
    rows = points.shape[0]
    maximum = (rows - 2) * dx
    phi = (phi + math.pi) % (2 * math.pi) - math.pi
    tx, ty = theta / dx, (phi + math.pi) / dy
    ix = np.clip(np.floor(tx).astype(np.int64), 0, rows - 3)
    iy = np.clip(np.floor(ty).astype(np.int64), 0, cols - 1)
    x, y = np.clip(tx - ix, 0, 1), np.clip(ty - iy, 0, 1)
    wx, wy = ((1-x)**2/2, .5+x-x*x, x*x/2), ((1-y)**2/2, .5+y-y*y, y*y/2)
    result = np.zeros((*theta.shape, 2), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            cp = points[ix+i, (iy+j) % cols].copy()
            cp[..., 1] += (iy+j >= cols) * (2*math.pi)
            result += (wx[i] * wy[j])[..., None] * cp
    return result, maximum


def project_native(calibration, rays):
    intrinsic = calibration['intrinsic_param']
    model = int(intrinsic['camera_model'])
    x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
    rho = np.hypot(x, y)
    if model == 2:
        theta = np.arcsin(np.clip(-z / np.linalg.norm(rays, axis=-1), -1, 1))
        radius = np.polynomial.polynomial.polyval(theta, intrinsic['world2cam'])
        scale = radius / np.maximum(rho, 1e-10)
        uu, vv = x * scale, y * scale
        center = intrinsic.get('center', intrinsic['principal_point'])
        # At the optical axis the fitted inverse polynomial has a small
        # residual; the defined projection is exactly the principal point.
        uu, vv = np.where(rho > 1e-10, uu, 0), np.where(rho > 1e-10, vv, 0)
        return (uu + intrinsic['affine_e'] * vv + center[0],
                intrinsic['affine_d'] * uu + intrinsic['affine_c'] * vv + center[1])
    matrix = np.asarray(intrinsic['camera_matrix'])
    coefficients = intrinsic.get('distortion_coeffcients', intrinsic.get('distortion_coefficients', []))
    node = intrinsic.get('spline_camera_model') or {}
    if node.get('is_valid') is True:
        theta, phi = np.arctan2(rho, z), np.arctan2(y, x)
        value, maximum = _spline_value(node, theta, phi)
        boundary, _ = _spline_value(node, np.asarray(maximum - 1e-10), np.asarray(0.))
        radius = np.where(theta < maximum, value[..., 0], theta * boundary[0] / maximum)
        azimuth = np.where(theta < maximum, value[..., 1], phi)
        xd, yd = radius * np.cos(azimuth), radius * np.sin(azimuth)
    elif model == 4:
        theta = np.arctan2(rho, z)
        t2 = theta * theta
        k = coefficients
        radius = theta * (1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))))
        xd, yd = x * radius / np.maximum(rho, 1e-10), y * radius / np.maximum(rho, 1e-10)
    elif model == 3:
        x, y = x / z, y / z
        k = np.pad(np.asarray(coefficients), (0, 8-len(coefficients)))
        k1, k2, p1, p2, k3, k4, k5, k6 = k
        r2 = x*x + y*y
        ratio = (1+r2*(k1+r2*(k2+r2*k3))) / (1+r2*(k4+r2*(k5+r2*k6)))
        xd = x*ratio + 2*p1*x*y + p2*(r2+2*x*x)
        yd = y*ratio + p1*(r2+2*y*y) + 2*p2*x*y
    elif model == 0:
        xd, yd = x/z, y/z
    else:
        raise ValueError(f'unsupported camera model: {model}')
    return xd*matrix[0, 0]+matrix[0, 2], yd*matrix[1, 1]+matrix[1, 2]


def rectification(calibration, max_width=1920, svc_fov=120., adas_fov=None):
    intrinsic = calibration['intrinsic_param']
    # The processed JPG raster is already the calibration's pixel grid.
    # drive-gs consumes it directly even when this sensor/ISP flag is 3;
    # applying another flip here would change the calibrated projection.
    native_width, native_height = int(intrinsic['camera_width']), int(intrinsic['camera_height'])
    scale = min(1., max_width / native_width)
    width, height = round(native_width*scale), round(native_height*scale)
    requested_fov = svc_fov if intrinsic['camera_model'] == 2 else adas_fov
    if requested_fov is None:
        k = np.asarray(intrinsic['camera_matrix'], dtype=np.float64).copy()
        k[0, :] *= width/native_width
        k[1, :] *= height/native_height
    else:
        if not 1. < requested_fov < 175.:
            raise ValueError('pinhole horizontal FOV must lie between 1 and 175 degrees')
        focal = width/(2*math.tan(math.radians(requested_fov)/2))
        k = np.asarray([[focal, 0, (width-1)/2], [0, focal, (height-1)/2], [0, 0, 1.]])
    py, px = np.indices((height, width), dtype=np.float64)# 输出图像的像素坐标
    rays = np.stack(((px-k[0, 2])/k[0, 0], (py-k[1, 2])/k[1, 1], np.ones_like(px)), axis=-1)# 针孔相机输出图像的像素射线方向
    u, v = project_native(calibration, rays)# 将射线投影到原始畸变相机，得到应该采样的原图位置
    valid = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (v >= 0) & (u < native_width-1) & (v < native_height-1)
    hfov = math.degrees(math.atan2(k[0, 2], k[0, 0]) + math.atan2(width-k[0, 2], k[0, 0]))
    info = {'output_model': 'pinhole', 'native_model': int(intrinsic['camera_model']),
            'projection': 'nio-spline2' if (intrinsic.get('spline_camera_model') or {}).get('is_valid') else
                {0: 'pinhole', 2: 'nio-ocam', 3: 'opencv-radtan', 4: 'nio-poly5'}[int(intrinsic['camera_model'])],
            'rotation_output_to_native': np.eye(3).tolist(), 'horizontal_fov_degrees': hfov,
            'requested_horizontal_fov_degrees': requested_fov,
            'policy': 'same_optical_axis_cropped_pinhole', 'valid_pixel_fraction': float(valid.mean()),
            'native_flip_and_mirror': calibration.get('flip_and_mirror', 0),
            'flip_policy': 'source_raster_as_calibrated',
            'source_width': native_width, 'source_height': native_height,
            'semantic_interpolation': 'nearest', 'image_interpolation': 'linear'}
    return k, u.astype(np.float32), v.astype(np.float32), valid, info


def box_corners(pose, dimensions):
    length, width, height = dimensions
    corners = np.asarray([[x, y, z] for x in (-length/2, length/2)
                          for y in (-width/2, width/2) for z in (-height/2, height/2)])
    return corners @ pose[:3, :3].T + pose[:3, 3]
