"""Generate continuous NIO novel-view videos from LiDAR conditions.

Each 25-slot model call contains one reference anchor and 24 target frames.
Only target frames are published; saved target PNGs and production state allow
completed windows and completed MP4 files to be reused after interruption.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def resolve_path(value, project_root):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return [str(path), stat.st_size, stat.st_mtime_ns]


def window_starts(count, target_frames, overlap):
    if count < 1 or not 0 <= overlap < target_frames:
        raise ValueError('Require nonempty sequence and 0 <= overlap < target window size')
    starts = [0]
    while starts[-1] + target_frames < count:
        starts.append(starts[-1] + target_frames - overlap)
    return starts


def load_image(path, height, width, mask=False):
    """Apply the existing runner's centered-width / bottom-height crop."""
    import torch

    with Image.open(path) as source:
        image = source.convert('L' if mask else 'RGB')
        old_width, old_height = image.size
        if old_width / old_height > width / height:
            crop_width = int(width / height * old_height)
            left = (old_width - crop_width) // 2
            image = image.crop((left, 0, left + crop_width, old_height))
        elif old_width / old_height < width / height:
            crop_height = int(height / width * old_width)
            image = image.crop((0, old_height - crop_height, old_width, old_height))
        image = image.resize((width, height), Image.Resampling.NEAREST if mask else Image.Resampling.LANCZOS)
        values = np.asarray(image, dtype=np.float32).copy() / 255.0
    if mask:
        return torch.from_numpy(values[None])
    return torch.from_numpy(values.transpose(2, 0, 1)) * 2.0 - 1.0


def save_frame(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.stem}.{os.getpid()}.tmp.png')
    Image.fromarray(values).save(temporary, format='PNG')
    temporary.replace(path)


def sequence_fingerprint(config, records, shift, project_root, reference):
    digest = hashlib.sha256()
    digest.update(json.dumps({'algorithm': 'nio_anchor24_overlap_v1', 'config': config,
                              'shift': shift}, sort_keys=True).encode())
    sources = [Path(__file__), project_root / 'video_diffusion/sample_condition.py',
               project_root / 'video_diffusion/nio_conditions.py',
               resolve_path(config['diffusion']['config_path'], project_root)]
    for path in sources:
        digest.update(str(path.resolve()).encode())
        digest.update(path.read_bytes())
    digest.update(json.dumps(file_identity(resolve_path(config['diffusion']['ckpt_path'], project_root))).encode())
    for record in [reference, *records]:
        digest.update(json.dumps({'frame': record['frame'], 'timestamp': record['timestamp'],
                                  'files': [file_identity(record[key])
                                            for key in ('image', 'guidance', 'guidance_mask')]},
                                 sort_keys=True).encode())
    return digest.hexdigest()


def load_diffusion(config, project_root):
    """Load one model, restoring CLIP directly from the existing checkpoint."""
    import torch
    from safetensors import safe_open

    from . import init_proj_path  # noqa: F401; existing vwm modules use absolute imports
    from . import sample_condition as sampling

    settings = config['diffusion']
    original_loader = sampling.load_model_from_config

    def checkpoint_loader(model_config, ckpt=None):
        params = model_config.model.params
        params.en_and_decode_n_samples_a_time = int(settings.get('decode_chunk_size', 1))
        clip_embedders = []
        for index, embedder in enumerate(params.conditioner_config.params.emb_models):
            embedding = embedder.get('params', {}).get('open_clip_embedding_config')
            if embedding is not None:
                embedding.params.version = settings.get('clip_pretrained')
                clip_embedders.append(index)
        if not clip_embedders:
            raise ValueError('Model configuration has no OpenCLIP image embedder')
        if not ckpt or Path(ckpt).suffix != '.safetensors':
            raise ValueError('NIO inference requires a safetensors checkpoint')
        model = sampling.instantiate_from_config(model_config.model)
        targets = model.state_dict(keep_vars=True)
        required_parameters = {name for name, _ in model.named_parameters()}
        clip_prefixes = tuple(f'conditioner.embedders.{index}.open_clip.model.' for index in clip_embedders)
        required_clip = {name for name in required_parameters if name.startswith(clip_prefixes)}
        if not required_clip:
            raise ValueError('Instantiated model has no OpenCLIP parameters')
        # Stream CPU tensors into the CPU model. No second checkpoint/model copy
        # is sent to CUDA, and no separate OpenCLIP pretrained download is needed.
        with safe_open(str(ckpt), framework='pt', device='cpu') as checkpoint:
            keys = set(checkpoint.keys())
            missing_clip = sorted(required_clip - keys)
            if missing_clip:
                raise RuntimeError(f'Checkpoint misses {len(missing_clip)} required CLIP weights: '
                                   + ', '.join(missing_clip[:5]))
            missing_parameters = sorted(required_parameters - keys)
            if missing_parameters:
                raise RuntimeError(f'Checkpoint misses {len(missing_parameters)} model parameters: '
                                   + ', '.join(missing_parameters[:5]))
            with torch.no_grad():
                for name in keys & targets.keys():
                    value = checkpoint.get_tensor(name)
                    if value.shape != targets[name].shape:
                        raise ValueError(f'Checkpoint shape mismatch for {name}: '
                                         f'{tuple(value.shape)} != {tuple(targets[name].shape)}')
                    targets[name].copy_(value)
            missing = sorted(set(targets) - keys)
            unexpected = keys - targets.keys()
        del targets
        if missing or unexpected:
            print(f'Checkpoint: {len(missing)} other missing and {len(unexpected)} unexpected keys', flush=True)
        model.requires_grad_(False)
        model.eval()
        return sampling.initial_model_load(model)

    sampling.load_model_from_config = checkpoint_loader
    try:
        runner = sampling.VideoDiffusionModel(
            config_path=str(resolve_path(settings['config_path'], project_root)),
            ckpt_path=str(resolve_path(settings['ckpt_path'], project_root)),
            height=int(settings.get('height', 576)), width=int(settings.get('width', 1024)),
            n_steps=int(settings.get('n_steps', 50)), cfg_scale=float(settings.get('cfg_scale', 2.5)),
            seed=int(settings.get('seed', 23)), low_vram=bool(settings.get('low_vram', False)))
    finally:
        sampling.load_model_from_config = original_loader
    if runner.num_frames != 25:
        raise ValueError(f'NIO runner expects the 25-slot checkpoint, got {runner.num_frames}')
    return runner

# 老师一次按25帧组织输入，0帧是参考图像，1-24是24个时间视角的渲染图像
def build_batch(records, start, committed, frame_paths, reference, height, width, target_frames=24):
    import torch

    indices = list(range(start, min(start + target_frames, len(records))))
    padded = indices + [indices[-1]] * (target_frames - len(indices))
    if start == 0:
        anchor = reference
        anchor_image = load_image(anchor['image'], height, width)# 读取去畸变的参考图
    else:
        anchor = records[start - 1]
        anchor_image = load_image(frame_paths[start - 1], height, width)
    guides = [load_image(anchor['guidance'], height, width)]
    masks = [load_image(anchor['guidance_mask'], height, width, mask=True)]
    for index in padded:
        guides.append(load_image(records[index]['guidance'], height, width))
        masks.append(load_image(records[index]['guidance_mask'], height, width, mask=True))
    images = anchor_image[None].repeat(target_frames + 1, 1, 1, 1)
    cond_indices = [0]
    for slot, index in enumerate(indices, 1):
        if index < committed:
            images[slot] = load_image(frame_paths[index], height, width)
            cond_indices.append(slot)
    batch = {'img_seq': images, 'guide_seq': torch.stack(guides),
             'guide_mask_seq': torch.stack(masks), 'training_free_guidance': False}
    return batch, cond_indices, indices


def encode_video(path, frame_paths, fps):
    import imageio.v2 as imageio

    temporary = path.with_name(f'.{path.stem}.{os.getpid()}.tmp.mp4')
    try:
        with imageio.get_writer(str(temporary), format='FFMPEG', fps=fps, codec='libx264',
                                pixelformat='yuv420p', macro_block_size=1,
                                ffmpeg_params=['-movflags', '+faststart']) as writer:
            for frame_path in frame_paths:
                with Image.open(frame_path) as image:
                    writer.append_data(np.asarray(image.convert('RGB')))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run(config, project_root):
    from .nio_conditions import prepare_conditions

    settings = config['diffusion']
    height, width = int(settings.get('height', 576)), int(settings.get('width', 1024))
    overlap = int(settings.get('window_overlap', 4))
    if height < 64 or width < 64 or height % 64 or width % 64:
        raise ValueError('Diffusion height and width must be positive multiples of 64')
    if not 0 <= overlap < 24 or int(settings.get('decode_chunk_size', 1)) < 1:
        raise ValueError('Require 0 <= window_overlap < 24 and positive decode_chunk_size')
    fps = float(config.get('fps', 15))
    if fps <= 0:
        raise ValueError('fps must be positive')
    output = resolve_path(config['output_dir'], project_root)
    source = resolve_path(config['source_path'], project_root)
    output.mkdir(parents=True, exist_ok=True)
    conditions = prepare_conditions(config, project_root)
    resume = bool(config.get('render', {}).get('resume', True))
    runner, outputs = None, []
    for shift in config['shifts']:
        shift = float(shift)
        records = conditions[shift]
        if not records:
            raise ValueError(f'No novel-view conditions for shift {shift}')
        selected = list(map(int, config['selected_frames']))
        if [record['frame'] for record in records] != list(range(selected[0], selected[1] + 1)):
            raise ValueError(f'Novel-view records must cover the selected inclusive frame range: {shift}')
        first, camera = records[0], str(config.get('camera_id', 0))
        stem = f"{first['frame']:06d}_{camera}"
        reference = {**first, 'guidance': str(source / 'lidar/color_render' / f'{stem}.png'),
                     'guidance_mask': str(source / 'lidar/color_render' / f'{stem}_mask.png')}
        fingerprint = sequence_fingerprint(config, records, shift, project_root, reference)
        name = f'diffusion_novel_{shift:.1f}'
        video = output / f'{name}.mp4'
        state_path = output / 'state' / f'{name}.json'
        frame_dir = output / 'frames' / f'shift_{shift:.1f}'
        frame_paths = [frame_dir / f"{record['frame']:06d}.png" for record in records]
        starts = window_starts(len(records), 24, overlap)
        state = json.loads(state_path.read_text()) if resume and state_path.is_file() else {}
        if state.get('input_fingerprint') != fingerprint:
            state = {'input_fingerprint': fingerprint, 'completed_windows': 0, 'committed_frames': 0,
                     'video_complete': False, 'frames': len(records), 'height': height, 'width': width}
            atomic_json(state_path, state)
        if state.get('video_complete') and video.is_file() and video.stat().st_size == state.get('video_size'):
            print(f'Reusing {video}', flush=True)
            outputs.append(str(video))
            continue
        completed = int(state['completed_windows'])
        committed = int(state['committed_frames'])
        expected = min(len(records), starts[completed - 1] + 24) if 0 < completed <= len(starts) else 0
        if committed != expected or not 0 <= completed <= len(starts) or not all(
                path.is_file() for path in frame_paths[:committed]):
            completed, committed = 0, 0
            state.update(completed_windows=0, committed_frames=0, video_complete=False)
            atomic_json(state_path, state)
        for window, start in enumerate(starts):
            if window < completed:
                continue
            if runner is None:
                runner = load_diffusion(config, project_root)
            batch, cond_indices, indices = build_batch(records, start, committed, frame_paths,
                                                       reference, height, width)
            print(f'Shift {shift:.1f}: window {window + 1}/{len(starts)}, '
                  f'frames {records[indices[0]]["frame"]}..{records[indices[-1]]["frame"]}', flush=True)
            samples = runner.forward(batch, scale=1.0, cond_indices=cond_indices)# 调用老师
            values = (samples.detach().float().cpu().clamp(0, 1) * 255).round().byte().numpy()
            if values.shape != (25, 3, height, width):
                raise ValueError(f'Unexpected diffusion output shape: {values.shape}')
            # Anchor is slot 0. Existing overlap frames remain byte-for-byte
            # unchanged; only newly generated, real (unpadded) target slots save.
            for slot, index in enumerate(indices, 1):
                if index >= committed:
                    save_frame(frame_paths[index], values[slot].transpose(1, 2, 0))# 保存生成的新视角图像
            committed = max(committed, indices[-1] + 1)
            state.update(completed_windows=window + 1, committed_frames=committed, video_complete=False)
            atomic_json(state_path, state)
            del samples, values, batch
        encode_video(video, frame_paths, fps)
        state.update(video_complete=True, video_size=video.stat().st_size)
        atomic_json(state_path, state)
        outputs.append(str(video))
        print(f'Saved {video} ({len(records)} target frames at {fps:g} fps)', flush=True)
    return outputs


def main():
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config = OmegaConf.to_container(OmegaConf.load(resolve_path(args.config, project_root)), resolve=True)
    if not isinstance(config, dict):
        parser.error('config must contain a mapping')
    run(config, project_root)


if __name__ == '__main__':
    main()
