#!/usr/bin/env bash
set -euo pipefail

STREETCRAFTER_ROOT="${STREETCRAFTER_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
DRIVEGS_ROOT="${DRIVEGS_ROOT:-$(dirname -- "$STREETCRAFTER_ROOT")/drive-gs}"
exec bash "$STREETCRAFTER_ROOT/docker/joint/build.sh" \
    --drivegs-root "$DRIVEGS_ROOT" \
    --build-root "${BUILD_ROOT:-/data/drivegs-zqb}" \
    --repository "${IMAGE_REPOSITORY:-adas-img.nioint.com/adsim-scenario/drivegs}" \
    --test --push "$@"
