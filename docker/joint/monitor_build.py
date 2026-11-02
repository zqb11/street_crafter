"""Stop this build client before its dedicated storage filesystem fills up."""
import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--minimum-free-gb", type=int, default=5)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a build command is required")
    process = subprocess.Popen(command, start_new_session=True)
    try:
        while True:
            try:
                return process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                free = shutil.disk_usage(args.directory).free
                if free < args.minimum_free_gb * 1024 ** 3:
                    print(
                        "Stopping this build: less than {}GB free in {}. "
                        "No existing images or caches were deleted.".format(
                            args.minimum_free_gb, args.directory), file=sys.stderr, flush=True,
                    )
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    return 75
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
