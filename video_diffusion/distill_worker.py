"""Resident diffusion teacher; JSONL control messages never contain CUDA tensors.

Start with ``python -m video_diffusion.distill_worker --config ...`` and set
CUDA_VISIBLE_DEVICES in the launcher before starting this interpreter. Each
request names an immutable student snapshot. A complete.json is published only
after every target frame and its metadata have been written successfully.
"""

import argparse
from collections import defaultdict
from contextlib import redirect_stdout
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np
from PIL import Image

from .sample_nio import atomic_json, load_diffusion, load_image, save_frame, window_starts


REQUEST_SCHEMA = 'streetcrafter.distill.request/v1'
RESULT_SCHEMA = 'streetcrafter.distill.result/v1'


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value, root):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def validate_config(config, project_root):
    settings = config['diffusion']
    height, width = int(settings.get('height', 576)), int(settings.get('width', 1024))
    if min(height, width) < 64 or height % 64 or width % 64:
        raise ValueError('Teacher height and width must be positive multiples of 64')
    overlap = settings.get('window_overlap', 4)
    if isinstance(overlap, bool) or not isinstance(overlap, int) or not 0 <= overlap < 24:
        raise ValueError('Teacher window_overlap must be an integer in [0,24)')
    _integer(settings.get('n_steps', 50), 'n_steps', 1)
    _integer(settings.get('decode_chunk_size', 1), 'decode_chunk_size', 1)
    model_config = _resolve(settings['config_path'], project_root)
    checkpoint = _resolve(settings['ckpt_path'], project_root)
    if not model_config.is_file() or not checkpoint.is_file():
        raise FileNotFoundError('Teacher model config/checkpoint does not exist')
    if checkpoint.suffix != '.safetensors':
        raise ValueError('Teacher checkpoint must be safetensors')
    # Model weights stay fixed for this worker lifetime; avoid rehashing a large
    # checkpoint for every refresh. Snapshot inputs are content-hashed below.
    info = checkpoint.stat()
    identity = {'settings': settings, 'model_config_sha256': _sha256(model_config),
                'checkpoint': [str(checkpoint), info.st_size, info.st_mtime_ns],
                'sources': {name: _sha256(Path(__file__).with_name(name)) for name in
                            ('distill_worker.py', 'sample_nio.py', 'sample_condition.py')}}
    return height, width, overlap, identity


def _array(path, shape, name):
    values = np.load(path, allow_pickle=False, mmap_mode='r')
    if values.shape != shape or not np.issubdtype(values.dtype, np.floating):
        raise ValueError(f'{name} must be a floating NPY array with shape {shape}: {path}')
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError(f'{name} must contain finite values in [0,1]: {path}')
    return values


def read_request(request_path, height, width, teacher_identity):
    """Validate CPU inputs and bind their actual bytes to one request digest."""
    request_path = Path(request_path).resolve()
    request = json.loads(request_path.read_text())
    if request.get('schema') != REQUEST_SCHEMA:
        raise ValueError(f'Request schema must be {REQUEST_SCHEMA}')
    for name in ('run_id', 'snapshot_id'):
        if not isinstance(request.get(name), str) or not request[name]:
            raise ValueError(f'{name} must be a nonempty string')
    for name in ('round_id', 'event_step', 'completed_steps', 'seed'):
        _integer(request.get(name), name)
    scale = _finite(request.get('scale'), 'scale')
    if not 0 < scale <= 1:
        raise ValueError('scale must lie in (0,1]')
    frames = request.get('frames')
    references = request.get('references')
    if references is None and 'reference' in request:
        references = [request['reference']]
    if not isinstance(frames, list) or not frames or not isinstance(references, list) or not references:
        raise ValueError('Request needs nonempty frames and references lists')
    request['references'] = references
    files, frame_keys = {}, set()

    def bind(record, name, image=False):
        path = _resolve(record[name], request_path.parent)
        if not path.is_file():
            raise FileNotFoundError(path)
        record[name] = str(path)
        if str(path) not in files:
            files[str(path)] = _sha256(path)
        if image:
            with Image.open(path) as source:
                source.verify()
        return path

    for frame in frames:
        _integer(frame.get('frame'), 'frame')
        camera = frame.get('camera')
        if isinstance(camera, bool) or not isinstance(camera, (int, str)) or str(camera) == '' or (isinstance(camera, int) and camera < 0):
            raise ValueError('camera must be a nonempty string or a nonnegative integer')
        _finite(frame.get('timestamp'), 'timestamp')
        if not isinstance(frame.get('trajectory_id'), str) or not frame['trajectory_id']:
            raise ValueError('Each frame needs a nonempty trajectory_id')
        key = (frame['trajectory_id'], str(camera), frame['frame'])
        if key in frame_keys:
            raise ValueError(f'Duplicate target frame: {key}')
        frame_keys.add(key)
        if frame.get('height') != height or frame.get('width') != width:
            raise ValueError('Student arrays and target camera must already use the teacher grid')
        c2w, intrinsic = np.asarray(frame.get('c2w'), dtype=np.float64), np.asarray(frame.get('K'), dtype=np.float64)
        if c2w.shape != (4, 4) or not np.isfinite(c2w).all() or not np.allclose(c2w[3], [0, 0, 0, 1]):
            raise ValueError('Target c2w must be a finite homogeneous 4x4 matrix')
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all() or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
            raise ValueError('Target K must be a finite 3x3 matrix with positive focal lengths')
        _array(bind(frame, 'student_rgb'), (height, width, 3), 'student_rgb')
        _array(bind(frame, 'student_opacity'), (height, width), 'student_opacity')
        bind(frame, 'guide_rgb', image=True)
        bind(frame, 'guide_mask', image=True)
    for reference in references:
        _finite(reference.get('timestamp'), 'reference timestamp')
        for name in ('image', 'guide_rgb', 'guide_mask'):
            bind(reference, name, image=True)
    if not isinstance(request.get('output_dir'), str) or not request['output_dir']:
        raise ValueError('output_dir must be a nonempty path')
    request['output_dir'] = str(_resolve(request['output_dir'], request_path.parent))
    expected = request.pop('request_sha256', None)
    digest = hashlib.sha256(_canonical({'request': request, 'files': files, 'teacher': teacher_identity})).hexdigest()
    if expected is not None and expected != digest:
        raise ValueError('Request digest does not match its actual inputs')
    return request, files, digest


def build_distillation_batch(frames, indices, reference, height, width):
    """Real anchor + 24 current-student frames; padding is never published."""
    import torch

    padded = indices + [indices[-1]] * (24 - len(indices))
    images = [load_image(reference['image'], height, width)]
    opacity = [torch.ones((1, height, width), dtype=torch.float32)]
    guides = [load_image(reference['guide_rgb'], height, width)]
    masks = [load_image(reference['guide_mask'], height, width, mask=True)]
    for index in padded:
        frame = frames[index]
        rgb = np.array(_array(frame['student_rgb'], (height, width, 3), 'student_rgb'), dtype=np.float32)
        alpha = np.array(_array(frame['student_opacity'], (height, width), 'student_opacity'), dtype=np.float32)
        images.append(torch.from_numpy(rgb.transpose(2, 0, 1)) * 2.0 - 1.0)
        opacity.append(torch.from_numpy(alpha[None]))
        guides.append(load_image(frame['guide_rgb'], height, width))
        masks.append(load_image(frame['guide_mask'], height, width, mask=True))
    return {'img_seq': torch.stack(images), 'img_mask_seq': torch.stack(opacity),
            'guide_seq': torch.stack(guides), 'guide_mask_seq': torch.stack(masks),
            'training_free_guidance': True}


class DistillationWorker:
    def __init__(self, config, project_root, runner=None):
        self.height, self.width, self.overlap, self.identity = validate_config(config, project_root)
        self.n_steps = int(config['diffusion'].get('n_steps', 50))
        # low_vram is intentionally taken from the existing NIO configuration.
        self.runner = runner if runner is not None else load_diffusion(config, project_root)
        self.rounds = {}

    def process(self, request_path):
        request_file_digest = _sha256(request_path)
        request, files, digest = read_request(request_path, self.height, self.width, self.identity)
        if int(self.n_steps * request['scale']) < 1:
            raise ValueError('scale would result in zero denoising steps')
        round_key = (request['run_id'], request['round_id'])
        if round_key in self.rounds and self.rounds[round_key] != digest:
            raise ValueError('Round ID was already bound to different inputs')
        output = Path(request['output_dir'])
        complete_path, binding_path = output / 'complete.json', output / 'request_identity.json'
        if binding_path.is_file():
            if json.loads(binding_path.read_text()).get('request_sha256') != digest:
                raise ValueError('Output directory is already bound to a different request digest')
        if complete_path.is_file():
            complete = json.loads(complete_path.read_text())
            if complete.get('request_sha256') != digest:
                raise ValueError('Completed round has a different request digest')
            for frame in complete['frames']:
                if _sha256(frame['label_png']) != frame['label_sha256']:
                    raise ValueError('Completed round contains a missing or corrupted label')
            self.rounds[round_key] = digest
            return {'status': 'complete', 'cached': True, 'manifest_path': str(complete_path),
                    'request_sha256': digest, 'round_id': request['round_id']}
        self.rounds[round_key] = digest
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(binding_path, {'schema': REQUEST_SCHEMA, 'request_sha256': digest,
                                  'run_id': request['run_id'], 'round_id': request['round_id']})
        groups = defaultdict(list)
        for frame in request['frames']:
            groups[(frame['trajectory_id'], str(frame['camera']))].append(frame)
        published = []
        for (trajectory, camera), frames in sorted(groups.items()):
            frames.sort(key=lambda frame: (frame['timestamp'], frame['frame']))
            references = [reference for reference in request['references'] if
                          reference.get('trajectory_id', trajectory) == trajectory and
                          str(reference.get('camera', camera)) == camera]
            if not references:
                raise ValueError(f'No real reference for trajectory {trajectory}, camera {camera}')
            directory = output / 'labels' / hashlib.sha256(trajectory.encode()).hexdigest()[:16]
            committed = set()
            self.runner.seed = request['seed']
            for start in window_starts(len(frames), 24, self.overlap):
                indices = list(range(start, min(start + 24, len(frames))))
                reference = min(references, key=lambda item: abs(item['timestamp'] - frames[start]['timestamp']))
                batch = build_distillation_batch(frames, indices, reference, self.height, self.width)
                samples = self.runner.forward(batch, scale=request['scale'], cond_indices=[0])
                values = samples.detach().float().cpu().numpy()
                if values.shape != (25, 3, self.height, self.width) or not np.isfinite(values).all():
                    raise ValueError(f'Teacher returned invalid output: {values.shape}')
                values = (np.clip(values, 0, 1) * 255).round().astype(np.uint8)
                for slot, index in enumerate(indices, 1):
                    # First completed window owns its overlap frames. Every
                    # window still reads the same immutable student snapshot.
                    if index in committed:
                        continue
                    frame = frames[index]
                    camera_name = camera if all(char.isalnum() or char in '_-' for char in camera) else hashlib.sha256(camera.encode()).hexdigest()[:12]
                    path = directory / f"{frame['frame']:06d}_{camera_name}.png"
                    save_frame(path, values[slot].transpose(1, 2, 0))
                    published.append({**frame, 'label_png': str(path), 'label_sha256': _sha256(path)})
                    committed.add(index)
                del samples, values, batch
            if len(committed) != len(frames):
                raise RuntimeError('Teacher failed to cover all target frames')
        # Catch a producer modifying any supposedly immutable input mid-round.
        if any(_sha256(path) != original for path, original in files.items()):
            raise ValueError('Snapshot input changed while the teacher was running')
        if _sha256(request_path) != request_file_digest:
            raise ValueError('Request JSON changed while the teacher was running')
        complete = {name: request[name] for name in
                    ('run_id', 'round_id', 'event_step', 'completed_steps', 'snapshot_id', 'scale', 'seed')}
        complete.update(schema=RESULT_SCHEMA, request_sha256=digest,
                        request_file_sha256=request_file_digest, frames=published,
                        teacher=self.identity, frame_count=len(published))
        atomic_json(complete_path, complete)
        return {'status': 'complete', 'cached': False, 'manifest_path': str(complete_path),
                'request_sha256': digest, 'round_id': request['round_id']}


def main():
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-cpu', action='store_true')
    mode.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--request', help='Optional immutable request.json to validate with --check-cpu')
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config = OmegaConf.to_container(OmegaConf.load(_resolve(args.config, project_root)), resolve=True)
    protocol = sys.stdout

    def respond(value):
        protocol.write(json.dumps(value, allow_nan=False) + '\n')
        protocol.flush()

    with redirect_stdout(sys.stderr):
        if args.prepare_only:
            from .nio_conditions import prepare_conditions

            records = prepare_conditions(config, project_root)
            respond({'status': 'prepared', 'conditions_dir': str(_resolve(config['output_dir'], project_root) / 'conditions'),
                     'frame_count': sum(len(frames) for frames in records.values())})
            return
        if args.check_cpu:
            height, width, overlap, identity = validate_config(config, project_root)
            result = {'status': 'checked', 'height': height, 'width': width, 'overlap': overlap}
            if args.request:
                request, _, digest = read_request(args.request, height, width, identity)
                result.update(request_sha256=digest, frame_count=len(request['frames']))
            respond(result)
            return
        worker = DistillationWorker(config, project_root)
        respond({'status': 'ready', 'schema': REQUEST_SCHEMA, 'height': worker.height, 'width': worker.width})
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                message = json.loads(line)
                if message.get('command') == 'shutdown':
                    respond({'status': 'stopped'})
                    return
                respond(worker.process(message['request_path']))
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                respond({'status': 'error', 'error_type': type(error).__name__, 'message': str(error)})


if __name__ == '__main__':
    main()
