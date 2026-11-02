"""Combine the existing DriveGS train stages with the joint image stages."""
import argparse
from pathlib import Path
import shlex


def compose(context):
    drive = (context / "drivegs/docker/Dockerfile.joint-train").read_text()
    street = (context / "streetcrafter/docker/joint/Dockerfile").read_text()
    lines = []
    train_stages = 0
    for line in drive.splitlines():
        if line.startswith("# syntax=") or line.startswith("ARG BASE_IMAGE="):
            continue
        if line == "FROM ${BASE_IMAGE}":
            line += " AS drivegs_train"
            train_stages += 1
        if line.startswith("COPY ") and "--from=" not in line:
            tokens = shlex.split(line)
            if any(token.startswith("--") for token in tokens[1:]):
                raise ValueError("Review changed DriveGS COPY options: " + line)
            sources = ["/" + item.lstrip("/") for item in tokens[1:-1]]
            line = "COPY --from=drivegs " + " ".join(sources + [tokens[-1]])
        lines.append(line)
    if train_stages != 1:
        raise ValueError("Review changed DriveGS train stage layout")
    marker = "FROM ${STREET_CUDA_IMAGE} AS streetcrafter"
    street = street[street.index(marker):]
    street = street.replace("FROM ${DRIVEGS_TRAIN_IMAGE} AS joint", "FROM drivegs_train AS joint", 1)
    result = "\n".join([
        "# syntax=docker/dockerfile:1",
        "ARG BASE_IMAGE",
        "ARG STREET_CUDA_IMAGE=nvidia/cuda:12.1.1-devel-ubuntu22.04",
        *lines, "", street,
    ])
    target = context / "streetcrafter/docker/joint/Dockerfile.generated"
    target.write_text(result)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    args = parser.parse_args()
    print(compose(args.context))
