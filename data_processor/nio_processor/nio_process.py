"""Run NIO image, aggregate-map and per-frame LiDAR preprocessing sequentially."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw_tar', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--max_width', type=int, default=1920)
    parser.add_argument('--workers', type=int, default=4, help='CPU workers for image conversion')
    parser.add_argument('--render_workers', type=int, default=2)
    parser.add_argument('--color_stride', type=int, default=5)# 地图着色时，每隔多少帧选取一帧
    parser.add_argument('--voxel_size', type=float, default=0.1)# 点云降采样的体素边长，单位米
    parser.add_argument('--crop_radius', type=float, default=80)# 车辆附近局部地图的裁剪半径，单位米
    parser.add_argument('--splat_radius', type=int, default=2)# 渲染点在图像上扩散的半径，单位像素
    parser.add_argument('--meta_cams', nargs='+', default=['0'])# 为哪些相机生成视频训练、验证窗口
    parser.add_argument('--depth_decoder')# 未指定单目深度 ELM 解码器
    parser.add_argument('--skip_mono_depth', action='store_true')
    parser.add_argument('--raw_lidar_dir', help='FALCON PCD directory; defaults to the scene raw_lidar tree')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--start_stage', choices=['convert', 'map', 'lidar', 'render', 'meta'],
                        default='convert', help='Continue from a stage whose inputs already exist')# 从哪个阶段开始执行
    args = parser.parse_args()
    if args.workers < 1 or args.render_workers < 1:
        parser.error('Worker counts must be positive')
    scripts = Path(__file__).resolve().parent
    raw_tar = str(Path(args.raw_tar).resolve())
    output = str(Path(args.save_dir).resolve())
    env = os.environ.copy()
    # Avoid nested BLAS/OpenCV pools while retaining existing environments.
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    stages = [
        ('nio_converter.py', ['--raw_tar', raw_tar, '--save_dir', output,
                              '--max_width', str(args.max_width), '--workers', str(args.workers)]),# 先建立同步帧索引，再转换图像、内外参、位姿、掩码和轨迹
        ('nio_get_map_pcd.py', ['--raw_tar', raw_tar, '--save_dir', output,
                                '--color_stride', str(args.color_stride), '--voxel_size', str(args.voxel_size),
                                '--radius', str(args.crop_radius)]),# 解码完整时间背景LiDAR点云地图
        ('nio_get_frame_pcd.py', ['--raw_tar', raw_tar, '--data_dir', output]),# 转换逐帧FALCON点云，按目标真实时间匹配轨迹
        ('nio_render_pcd.py', ['--data_dir', output, '--workers', str(args.render_workers),
                               '--radius', str(args.splat_radius)]),# 按各相机拍摄时刻补偿目标运动，融合背景地图和逐帧点云，统一处理遮挡
        ('nio_prepare_meta.py', ['--data_dir', output, '--cams', *args.meta_cams]),
    ]# 五个阶段按照顺序执行
    if args.depth_decoder:
        stages[0][1].extend(['--depth_decoder', str(Path(args.depth_decoder).resolve())])
    if args.skip_mono_depth:
        stages[0][1].append('--skip_mono_depth')
    if args.raw_lidar_dir:
        stages[2][1].extend(['--raw_lidar_dir', str(Path(args.raw_lidar_dir).resolve())])
    first_stage = ['convert', 'map', 'lidar', 'render', 'meta'].index(args.start_stage)
    for name, options in stages[first_stage:]:
        if args.resume and name in ('nio_converter.py', 'nio_get_map_pcd.py', 'nio_get_frame_pcd.py', 'nio_render_pcd.py'):
            options.append('--resume')
        print(f'Running {name}', flush=True)
        subprocess.run([sys.executable, str(scripts / name), *options], check=True, env=env)

# 完整流程预处理
if __name__ == '__main__':
    main()
