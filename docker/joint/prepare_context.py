#!/usr/bin/env python3
"""Create small, reproducible build contexts without host data or credentials."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import stat
import subprocess


EXCLUDED_PARTS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache",
    ".mypy_cache", "build", "dist", "node_modules", ".ssh", ".aws",
    ".azure", ".config", ".cache", "wandb", "lightning_logs",
    "output", "outputs", "logs", "cache", "temp", "checkpoints", "ckpts",
}
EXCLUDED_SUFFIXES = {
    ".pyc", ".pyo", ".so", ".o", ".a", ".dll", ".dylib", ".whl",
    ".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".onnx", ".pkl",
    ".pem", ".key", ".p12", ".pfx", ".log", ".npy", ".npz", ".ply",
    ".mp4", ".mov", ".avi", ".tar", ".gz", ".zip",
}
SC_EXCLUDED_PREFIXES = (
    "video_diffusion/ckpts", "video_diffusion/outputs", "video_diffusion/logs",
    "video_diffusion/annos", "video_diffusion/src", "video_diffusion/data",
    "video_diffusion/conditions", "video_diffusion/results",
)
SC_TREES = (
    "configs", "scripts", "street_gaussian", "data_processor", "process",
    "video_diffusion", "submodules/sdata", "submodules/simple-knn", "docker/joint",
)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n")
    path.chmod(0o644)


def excluded(relative, project):
    parts = relative.parts
    name = relative.name.lower()
    if any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in parts):
        return True
    if name in {".env", ".envrc", ".git-credentials"} or name.startswith(".env.") or name.startswith(("id_rsa", "id_ed25519", "id_ecdsa", "id_dsa")):
        return True
    if name in {"credentials", "credentials.json", "login_server.txt", "export_env.sh", ".netrc", ".npmrc", ".pypirc"}:
        return True
    if relative.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    if project == "streetcrafter":
        value = relative.as_posix()
        if any(value == prefix or value.startswith(prefix + "/") for prefix in SC_EXCLUDED_PREFIXES):
            return True
    return False


def copy_item(root, relative, destination, project, shell_only=False):
    if excluded(relative, project):
        return
    source = root / relative
    if source.is_symlink():
        raise ValueError("Refusing source symlink: {}".format(source))
    if source.is_dir():
        for child in sorted(source.iterdir()):
            copy_item(root, relative / child.name, destination, project, shell_only)
    elif source.is_file():
        if shell_only and relative.suffix not in {".py", ".sh"}:
            return
        with source.open("rb") as stream:
            header = stream.read(4096)
        if re.search(rb"(?m)^-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----", header):
            return
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        target.chmod(stat.S_IMODE(source.stat().st_mode))
    elif source.exists():
        raise ValueError("Refusing non-regular source: {}".format(source))


def copy_streetcrafter(root, destination):
    for source in sorted(root.iterdir()):
        if source.name.endswith(".py") or source.name in {"README.md", "LICENSE", "requirements.txt", ".gitignore"}:
            copy_item(root, Path(source.name), destination, "streetcrafter")
    for tree in SC_TREES:
        relative = Path(tree)
        copy_item(root, relative, destination, "streetcrafter", shell_only=(tree == "scripts"))
    for required in ("train.py", "render.py", "requirements.txt", "video_diffusion/vwm/data/dataset.py", "submodules/sdata/setup.py", "submodules/simple-knn/setup.py"):
        if not (destination / required).is_file():
            raise ValueError("StreetCrafter source missing: {}".format(required))


def copy_drivegs(root, destination):
    ignore_file = root / "docker/Dockerfile.train.dockerignore"
    patterns = []
    for line in ignore_file.read_text().splitlines():
        line = line.strip()
        if line.startswith("!"):
            pattern = line[1:].rstrip("/")
            if pattern != "docker":
                patterns.append(pattern)
    # The existing Dockerfile's allowlist is authoritative; extra source is only
    # the distillation entry and build instructions needed for this context.
    extras = (
        "distill_pipeline.py", "docker/Dockerfile.train",
        "docker/Dockerfile.train.dockerignore",
    )
    for relative in extras:
        copy_item(root, Path(relative), destination, "drivegs")
    for pattern in patterns:
        if "*" in pattern or "?" in pattern or "[" in pattern:
            # All current recursive patterns have a corresponding directory
            # allow-rule, but process them explicitly if that changes upstream.
            prefix = pattern.split("*")[0].split("?")[0].split("[")[0].rstrip("/")
            search_root = root / prefix
            if search_root.is_dir():
                copy_item(root, Path(prefix), destination, "drivegs")
            else:
                for source in sorted(root.glob(pattern)):
                    copy_item(root, source.relative_to(root), destination, "drivegs")
        else:
            copy_item(root, Path(pattern), destination, "drivegs")
    original = destination / "docker/Dockerfile.train"
    if not original.is_file() or not (destination / "distill_pipeline.py").is_file():
        raise ValueError("DriveGS train Dockerfile or distill_pipeline.py is missing")
    dockerfile = original.read_text()
    marker = "COPY train_pipeline.py ./train_pipeline.py"
    if marker not in dockerfile:
        raise ValueError("DriveGS Dockerfile.train COPY layout changed; review joint adaptation")
    dockerfile = dockerfile.replace(marker, marker + "\nCOPY distill_pipeline.py ./distill_pipeline.py", 1)
    cmd_marker = 'CMD ["bash"]'
    if cmd_marker not in dockerfile:
        raise ValueError("DriveGS Dockerfile.train CMD layout changed; review joint adaptation")
    dockerfile = dockerfile.replace(cmd_marker, "RUN python distill_pipeline.py --help >/dev/null\n\n" + cmd_marker, 1)
    (destination / "docker/Dockerfile.joint-train").write_text(dockerfile)
    (destination / "docker/Dockerfile.joint-train").chmod(0o644)
    original_ignore = ignore_file.read_text()
    (destination / "docker/Dockerfile.joint-train.dockerignore").write_text(
        original_ignore + "\n!distill_pipeline.py\n!source-manifest.json\n"
        "!docker/Dockerfile.joint-train\n!docker/Dockerfile.joint-train.dockerignore\n"
    )
    (destination / "docker/Dockerfile.joint-train.dockerignore").chmod(0o644)


def git_metadata(root):
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode:
            raise ValueError("Cannot inspect source git metadata: {}".format(root))
        return result.stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=normal"))}


def make_manifest(root, destination):
    files = []
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            files.append({
                "path": path.relative_to(destination).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "mode": "{:04o}".format(stat.S_IMODE(path.stat().st_mode)),
            })
    manifest = {"schema_version": 1, **git_metadata(root), "source_sha256": digest(files), "files": files}
    write_json(destination / "source-manifest.json", manifest)
    return {key: value for key, value in manifest.items() if key != "files"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streetcrafter-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--drivegs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = {"streetcrafter": args.streetcrafter_root.resolve(), "drivegs": args.drivegs_root.resolve()}
    if args.output.is_symlink():
        parser.error("output must not be a symlink")
    output = args.output.resolve()
    for root in roots.values():
        if not root.is_dir():
            parser.error("source directory not found: {}".format(root))
        if output == root or root in output.parents:
            parser.error("output must be outside both source trees")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("output must be an empty directory")
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"schema_version": 1}
    try:
        for project, copy in (("streetcrafter", copy_streetcrafter), ("drivegs", copy_drivegs)):
            destination = output / project
            destination.mkdir()
            copy(roots[project], destination)
            metadata[project] = make_manifest(roots[project], destination)
    except (OSError, ValueError) as error:
        parser.exit(1, "Snapshot failed: {}\n".format(error))
    metadata["source_fingerprint"] = digest({
        project: {key: metadata[project][key] for key in ("commit", "source_sha256")}
        for project in roots
    })
    write_json(output / "source-metadata.json", metadata)
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
