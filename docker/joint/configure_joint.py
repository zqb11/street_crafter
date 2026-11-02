"""Generate the container-specific DriveGS overlay without changing source configs."""

import argparse
import json
from pathlib import Path


def create_config(drivegs_root, streetcrafter_root, output=None):
    drivegs_root = Path(drivegs_root).resolve()
    streetcrafter_root = Path(streetcrafter_root).resolve()
    source = drivegs_root / "configs/streetcrafter_distill.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    teacher = payload["training"]["diffusion_distillation"]
    old_root = teacher["streetcrafter_root"].rstrip("/")
    patch = {
        "streetcrafter_root": str(streetcrafter_root),
        "teacher_python": "/usr/local/bin/streetcrafter-python",
    }
    for key in ("source_path", "conditions_dir", "teacher_config"):
        value = teacher.get(key)
        if isinstance(value, str) and (value == old_root or value.startswith(old_root + "/")):
            patch[key] = str(streetcrafter_root) + value[len(old_root):]
    target = Path(output) if output else source.with_name("streetcrafter_distill.container.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    overlay = {
        "$extends": str(source),
        "training": {"diffusion_distillation": patch},
    }
    target.write_text(json.dumps(overlay, indent=2) + "\n", encoding="utf-8")
    print("Container config:", target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drivegs-root", default="/workspace/drivegs")
    parser.add_argument("--streetcrafter-root", default="/workspace/street_crafter")
    parser.add_argument("--output")
    args = parser.parse_args()
    create_config(args.drivegs_root, args.streetcrafter_root, args.output)


if __name__ == "__main__":
    main()
