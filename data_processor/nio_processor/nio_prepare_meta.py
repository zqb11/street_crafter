"""Build NIO scene-relative video windows without train/validation time overlap."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def prepare_meta(args):
    data_dir = Path(args.data_dir).resolve()
    metadata = json.loads((data_dir / 'metadata.json').read_text())
    cameras = [str(cam) for cam in args.cams]
    if set(cameras) - set(metadata['cameras']):
        raise ValueError('Requested camera IDs are absent from metadata')
    frames = sorted(metadata['frames'], key=lambda frame: (frame['timestamp'], frame['frame']))
    boundary = int(len(frames) * args.train_ratio)
    splits = {'train': frames[:boundary], 'val': frames[boundary:]}
    total = 0
    for split, split_frames in splits.items():
        samples = []
        for cam in cameras:
            paths = {}
            for frame in split_frames:
                frame_id = frame['frame']
                stem = f'{frame_id:06d}_{cam}'
                observation = frame['cameras'].get(cam)
                if observation is None:
                    continue
                image = Path(observation['image'])
                guidance = Path('lidar/color_render') / f'{stem}.png'
                mask = Path('lidar/color_render') / f'{stem}_mask.png'
                if not all((data_dir / path).is_file() for path in (image, guidance, mask)):
                    continue
                mask_image = cv2.imread(str(data_dir / mask), cv2.IMREAD_GRAYSCALE)
                camera = metadata['cameras'][cam]
                shape = (int(camera['height']), int(camera['width']))
                if mask_image is None or mask_image.shape != shape or not np.any(mask_image):
                    continue
                paths[frame_id] = (str(image), str(guidance), str(mask))
            # Windows stay on the original timeline: dropping invalid frames must not
            # create discontinuous clips or move validation frames into training.
            for start in range(0, len(split_frames) - args.window_size + 1, args.stride):
                window = split_frames[start:start + args.window_size]
                frame_ids = [frame['frame'] for frame in window]
                if not all(frame_id in paths for frame_id in frame_ids):
                    continue
                if any(b != a + 1 for a, b in zip(frame_ids, frame_ids[1:])):
                    continue
                records = [paths[frame_id] for frame_id in frame_ids]
                samples.append({'frames': [item[0] for item in records],
                                'guidances': [item[1] for item in records],
                                'guidances_mask': [item[2] for item in records]})
        (data_dir / f'meta_info_{split}.json').write_text(json.dumps(samples, indent=1) + '\n')
        total += len(samples)
    if total == 0:
        raise RuntimeError('No complete video windows; check input files, guidance masks and window_size')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--cams', nargs='+', default=['0'])
    parser.add_argument('--window_size', type=int, default=25)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    args = parser.parse_args()
    if args.window_size < 1 or args.stride < 1 or not 0 < args.train_ratio < 1:
        parser.error('Require positive window_size/stride and 0 < train_ratio < 1')
    prepare_meta(args)


if __name__ == '__main__':
    main()
