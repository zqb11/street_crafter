"""Compile A100 extensions without allocating CUDA tensors or requiring a GPU."""
import importlib
import json
import os
from pathlib import Path

import torch
import torch.utils.cpp_extension as extension


def main():
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
    target_flags = extension._get_cuda_arch_flags()
    original_flags = extension._get_cuda_arch_flags
    # nvdiffrast 0.3.3 clears TORCH_CUDA_ARCH_LIST internally. Supply the explicit
    # build target only in this build process, without modifying vendor sources.
    extension._get_cuda_arch_flags = lambda cflags=None: target_flags
    try:
        gsplat = importlib.import_module("gsplat.cuda._backend")
        if gsplat._C is None:
            raise RuntimeError("gsplat CUDA extension was not built")
        nvdiffrast = importlib.import_module("nvdiffrast.torch.ops")
        nvdiffrast_module = nvdiffrast._get_plugin(gl=False)
        compiled_modules = [
            {"name": module.__name__, "path": module.__file__}
            for module in (gsplat._C, nvdiffrast_module)
        ]
    finally:
        extension._get_cuda_arch_flags = original_flags
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
    modules = {}
    for name in ("simple_knn._C", "diff_point_rasterization._C"):
        module = importlib.import_module(name)
        modules[name] = module.__file__
    cache = Path(os.environ["TORCH_EXTENSIONS_DIR"])
    cache.mkdir(parents=True, exist_ok=True)
    inventory = {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "target": "sm_80", "modules": modules,
        "compiled_modules": compiled_modules,
        "jit_libraries": sorted(str(path) for path in cache.rglob("*.so")),
    }
    (cache / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(json.dumps(inventory))


if __name__ == "__main__":
    main()

