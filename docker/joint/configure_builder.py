"""Configure only the dedicated builder, retaining its bind-mounted state."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    fallback = os.environ.get("BUILD_PROXY", "")
    proxies = {key: os.environ.get(key) or os.environ.get(key.upper()) or fallback
               for key in ("http_proxy", "https_proxy")}
    proxies["no_proxy"] = os.environ.get("no_proxy") or os.environ.get("NO_PROXY", "")
    # Keep this large dual-environment build's layers between failed attempts.
    # monitor_build.py stops the build when the dedicated partition is nearly full.
    config_text = '[worker.oci]\n  gc = false\n'
    config = args.root / "buildkitd.toml"
    config.write_text(config_text)
    signature = hashlib.sha256(json.dumps({"image": args.image, "config": config_text, **proxies}, sort_keys=True).encode()).hexdigest()
    state = args.root / "builder-config.sha256"
    exists = subprocess.run(["docker", "buildx", "inspect", args.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if exists and (not state.is_file() or state.read_text().strip() != signature):
        subprocess.run(["docker", "buildx", "rm", "--keep-state", args.name], check=True)
        exists = False
    if not exists:
        command = ["docker", "buildx", "create", "--name", args.name, "--node", args.name + "0",
                   "--driver", "docker-container", "--buildkitd-config", str(config),
                   "--driver-opt", "image=" + args.image,
                   "--driver-opt", "network=host"]
        for key, value in proxies.items():
            if value:
                field = io.StringIO()
                csv.writer(field, lineterminator="").writerow(["env.{}={}".format(key, value)])
                command.extend(["--driver-opt", field.getvalue()])
        result = subprocess.run(command, stdout=subprocess.DEVNULL)
        if result.returncode:
            raise SystemExit("Dedicated builder configuration failed")
    state.write_text(signature + "\n")
    subprocess.run(["docker", "buildx", "inspect", args.name, "--bootstrap"], check=True, stdout=subprocess.DEVNULL)
    print("Dedicated builder ready; network settings are not embedded in the image.")


if __name__ == "__main__":
    main()
