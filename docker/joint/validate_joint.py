"""CPU-only joint image checks; no data, model weights or GPU are required."""

import argparse
import json
from pathlib import Path
import subprocess


BOOTSTRAP = r"""
import json, os, subprocess, sys
from pathlib import Path
assert 'stubs' not in os.environ.get('LD_LIBRARY_PATH', ''), 'runtime wrapper exposed driver stubs'
environment = os.environ.copy()
stubs = Path(environment['CUDA_HOME']) / 'lib64/stubs'
if (stubs / 'libcuda.so.1').is_file():
    environment['LD_LIBRARY_PATH'] = str(stubs) + ':' + environment['LD_LIBRARY_PATH']
raise SystemExit(subprocess.call([sys.executable, *json.loads(sys.argv[1])], env=environment))
"""


STREET_CHECK = r"""
import importlib, importlib.util, json, os, sys
from pathlib import Path
import torch
import torch.utils.cpp_extension as extension
assert sys.version_info[:2] == (3, 9), sys.version
assert torch.__version__.split('+')[0] == '2.4.0', torch.__version__
assert torch.version.cuda == '12.1', torch.version.cuda
assert sys.prefix == '/opt/conda/envs/streetcrafter', sys.prefix
assert os.environ['CC'] == '/usr/bin/gcc'
assert os.environ['TORCH_CUDA_ARCH_LIST'] == '8.0'
assert os.environ['TORCH_EXTENSIONS_DIR'] == '/opt/streetcrafter-kernels'
def forbid_compile(*args, **kwargs):
    raise AssertionError('image check unexpectedly attempted CUDA compilation')
extension.load = extension.load_inline = forbid_compile
for name in ('numpy', 'PIL', 'cv2', 'imgaug', 'scipy', 'omegaconf', 'einops',
             'pytorch_lightning', 'transformers', 'open_clip', 'clip',
             'xformers', 'sdata', 'easyvolcap', 'nvdiffrast.torch', 'gsplat'):
    importlib.import_module(name)
for name in ('simple_knn._C', 'diff_point_rasterization._C'):
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.origin, name
    assert Path(spec.origin).is_file() and '.so' in Path(spec.origin).name, spec.origin
    importlib.import_module(name)
assert importlib.util.find_spec('video_diffusion.vwm') is not None
inventory = json.loads(Path('/opt/streetcrafter-kernels/inventory.json').read_text())
assert inventory['target'] == 'sm_80'
assert len(inventory['compiled_modules']) == 2
for record in inventory['compiled_modules']:
    assert Path(record['path']).is_file(), record
    spec = importlib.util.spec_from_file_location(record['name'], record['path'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
print('StreetCrafter Python/Torch/native artifacts: OK')
"""


DRIVEGS_CHECK = r"""
import importlib, importlib.util, json, os, sys
from pathlib import Path
import torch
assert sys.version_info[:2] == (3, 11), sys.version
assert torch.__version__.split('+')[0] == '2.8.0', torch.__version__
assert torch.version.cuda == '12.8', torch.version.cuda
assert sys.prefix == '/opt/conda', sys.prefix
assert os.environ['CUDA_HOME'] == '/usr/local/cuda-12.8'
assert os.environ['DRIVEGS_ALLOW_JIT'] == '0'
for name in ('drivegs', 'drivegs_render', 'numpy', 'scipy', 'slangtorch',
             'torchmetrics', 'nvdiffrast.torch'):
    importlib.import_module(name)
assert importlib.util.find_spec('tinycudann') is not None
root = Path(os.environ['TORCH_EXTENSIONS_DIR']).resolve()
inventory = json.loads(Path(os.environ['DRIVEGS_KERNEL_INVENTORY']).read_text())
entries = inventory['kernels'] + inventory.get('auxiliary_kernels', [])
assert entries, 'empty DriveGS native inventory'
for entry in entries:
    path = Path(entry['path']).resolve()
    assert path.is_relative_to(root) and path.is_file(), path
    assert entry['cuda_architectures'] == os.environ['TORCH_CUDA_ARCH_LIST'], entry['module']
print('DriveGS Python/Torch/native inventory: OK')
"""


CONFIG_CHECK = r"""
from drivegs.execution import resolve_config_overrides
config = resolve_config_overrides('/workspace/drivegs/configs/streetcrafter_distill.container.json', [])
teacher = config.payload['training']['diffusion_distillation']
assert teacher['streetcrafter_root'] == '/workspace/street_crafter'
assert teacher['teacher_python'] == '/usr/local/bin/streetcrafter-python'
for key in ('source_path', 'conditions_dir'):
    assert teacher[key].startswith('/workspace/street_crafter/'), teacher[key]
print('Container config inheritance: OK')
"""


def run(wrapper, *arguments):
    subprocess.run([wrapper, '-c', BOOTSTRAP, json.dumps(list(arguments))], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--street-only', action='store_true')
    mode.add_argument('--joint', action='store_true')
    args = parser.parse_args()
    street = '/usr/local/bin/streetcrafter-python'
    run(street, '-c', STREET_CHECK)
    run(street, '-m', 'pip', 'check')
    run(street, '-m', 'video_diffusion.distill_worker', '--help')
    if args.street_only:
        return
    drivegs = '/usr/local/bin/drivegs-python'
    run(drivegs, '-c', DRIVEGS_CHECK)
    run(drivegs, '/tmp/precompile_train.py', '--verify-only')
    run(drivegs, '-m', 'pip', 'check')
    run(drivegs, '-m', 'drivegs.cli', '--help')
    run(drivegs, '/workspace/drivegs/train_pipeline.py', '--help')
    run(drivegs, '/workspace/drivegs/distill_pipeline.py', '--help')
    run(drivegs, '/opt/drivegs/validate_adh.py')
    run(drivegs, '-c', CONFIG_CHECK)
    fallback = Path('/workspace/street_crafter/.venv/bin/python')
    assert fallback.resolve() == Path(street).resolve(), fallback
    subprocess.run(['/usr/local/bin/joint-distill', '--help'], check=True)
    print('Joint image CPU checks: OK (A100 execution has not been tested)')


if __name__ == '__main__':
    main()
