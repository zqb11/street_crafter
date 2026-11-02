#!/usr/bin/env bash
# Build current source snapshots locally, validate, and optionally publish one image.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STREET_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
DRIVEGS_ROOT="${DRIVEGS_ROOT:-$(dirname -- "$STREET_ROOT")/drive-gs}"
REPOSITORY="${IMAGE_REPOSITORY:-adas-img.nioint.com/adsim-scenario/drivegs}"
ADH_BASE_IMAGE="${ADH_BASE_IMAGE:-adas-img.nioint.com/adsim-scenario/drivegs@sha256:8376d18408a00b782ce327d410a97bf363756a8955965798d2c3bc9ebde59a6b}"
STREET_CUDA_IMAGE="${STREET_CUDA_IMAGE:-nvidia/cuda:12.1.1-devel-ubuntu22.04}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.org/simple}"
MIN_FREE_GB="${MIN_FREE_GB:-70}"
BUILD_ROOT="${BUILD_ROOT:-/data/drivegs-zqb}"
BUILDER="${BUILDER:-drivegs-streetcrafter}"
BUILDKIT_IMAGE="${BUILDKIT_IMAGE:-moby/buildkit:v0.21.0}"
DO_TEST=false
DO_PUSH=false
PRINT_IMAGE=false

usage() {
    cat <<'EOF'
Usage: bash docker/joint/build.sh [options]
  --drivegs-root PATH      Default: sibling drive-gs checkout
  --repository NAME       Default: adas-img.nioint.com/adsim-scenario/drivegs
  --adh-base-image IMAGE  Reuse the Miniforge DriveGS base-adh (digest recommended)
  --test                  Run CPU-safe validation; never starts training
  --push                  Validate, then push only the final joint image
  --no-push               Disable push (also overrides an earlier --push)
  --build-root PATH       BuildKit state and records (default: /data/drivegs-zqb)
  --print-images          Print the joint image tag without accessing Docker
  -h, --help              Show help
Environment: ADH_BASE_IMAGE, STREET_CUDA_IMAGE, PYPI_INDEX_URL, IMAGE_REPOSITORY,
             DRIVEGS_ROOT, BUILD_ROOT, BUILDER, BUILDKIT_IMAGE, MIN_FREE_GB (70), BUILD_PROXY.
Target: linux/amd64, A100. Miniforge Python 3.11 / Torch 2.8 / CUDA 12.8 for
DriveGS; Python 3.9 / Torch 2.4 / CUDA 12.1 for StreetCrafter.
Tags contain both actual source snapshots; no Git commit or branch switch occurs.
EOF
}

while (($#)); do
    case "$1" in
        --drivegs-root|--repository|--adh-base-image|--build-root)
            (($# >= 2)) || { echo "$1 requires a value" >&2; exit 2; }
            case "$1" in
                --drivegs-root) DRIVEGS_ROOT="$2" ;;
                --repository) REPOSITORY="$2" ;;
                --adh-base-image) ADH_BASE_IMAGE="$2" ;;
                --build-root) BUILD_ROOT="$2" ;;
            esac
            shift 2 ;;
        --test) DO_TEST=true; shift ;;
        --push) DO_PUSH=true; DO_TEST=true; shift ;;
        --no-push) DO_PUSH=false; shift ;;
        --print-images) PRINT_IMAGE=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ "$REPOSITORY" != *@* && "${REPOSITORY##*/}" != *:* ]] || {
    echo 'Repository must not contain an image tag or digest' >&2; exit 2;
}
DRIVEGS_ROOT="$(cd -- "$DRIVEGS_ROOT" && pwd)"
if $PRINT_IMAGE; then
    CONTEXT="$(mktemp -d -t drivegs-streetcrafter.XXXXXXXX)"
else
    mkdir -p "$BUILD_ROOT/tmp"
    CONTEXT="$(mktemp -d "$BUILD_ROOT/tmp/drivegs-streetcrafter.XXXXXXXX")"
fi
trap 'rm -rf -- "$CONTEXT"' EXIT
python3 "$SCRIPT_DIR/prepare_context.py" \
    --streetcrafter-root "$STREET_ROOT" --drivegs-root "$DRIVEGS_ROOT" \
    --output "$CONTEXT" > /dev/null

# Bind source content and external dependency inputs into the publication tag.
read -r STREET_COMMIT DRIVEGS_COMMIT STREET_SHA DRIVEGS_SHA FINGERPRINT BASE_ID < <(
    python3 - "$CONTEXT/source-metadata.json" "$ADH_BASE_IMAGE" "$STREET_CUDA_IMAGE" "$PYPI_INDEX_URL" <<'PY'
import hashlib, json, sys
path, adh, cuda, index = sys.argv[1:]
metadata = json.load(open(path))
metadata['build_inputs'] = {'adh_base_image': adh, 'street_cuda_image': cuda, 'pypi_index_url': index}
identity = {project: {key: metadata[project][key] for key in ('commit', 'source_sha256')}
            for project in ('streetcrafter', 'drivegs')}
identity['build_inputs'] = metadata['build_inputs']
fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
metadata['build_fingerprint'] = fingerprint
with open(path, 'w') as stream:
    json.dump(metadata, stream, indent=2)
    stream.write('\n')
s, d = metadata['streetcrafter'], metadata['drivegs']
print(s['commit'], d['commit'], s['source_sha256'], d['source_sha256'], fingerprint,
      hashlib.sha256(adh.encode()).hexdigest()[:12])
PY
)
IMAGE="$REPOSITORY:joint-a100-${STREET_SHA:0:12}-${DRIVEGS_SHA:0:12}-${FINGERPRINT:0:12}"
if $PRINT_IMAGE; then printf '%s\n' "$IMAGE"; exit 0; fi

if ! docker info > /dev/null; then
    echo 'Docker is unavailable. Configure Docker socket access, then rerun this command.' >&2
    exit 1
fi
docker buildx version > /dev/null
export http_proxy="${http_proxy:-${HTTP_PROXY:-${BUILD_PROXY:-}}}"
export https_proxy="${https_proxy:-${HTTPS_PROXY:-${BUILD_PROXY:-}}}"
if [[ -n "$http_proxy" && "$http_proxy" != *://* ]]; then export http_proxy="http://$http_proxy"; fi
if [[ -n "$https_proxy" && "$https_proxy" != *://* ]]; then export https_proxy="http://$https_proxy"; fi
export no_proxy="${no_proxy:-${NO_PROXY:-}},localhost,127.0.0.1,${ADH_BASE_IMAGE%%/*},cos.ap-beijing.myqcloud.com"
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy" NO_PROXY="$no_proxy"
mkdir -p "$BUILD_ROOT/buildkit" "$BUILD_ROOT/records"
BUILD_ROOT="$(cd -- "$BUILD_ROOT" && pwd)"
FREE_KB="$(df -Pk "$BUILD_ROOT" | awk 'NR == 2 {print $4}')"
if ((FREE_KB < MIN_FREE_GB * 1024 * 1024)); then
    echo "Build storage has less than ${MIN_FREE_GB}GB free: $BUILD_ROOT" >&2
    exit 1
fi
NODE="${BUILDER}0"
VOLUME="buildx_buildkit_${NODE}_state"
if docker volume inspect "$VOLUME" >/dev/null 2>&1; then
    DEVICE="$(docker volume inspect --format '{{index .Options "device"}}' "$VOLUME")"
    [[ "$DEVICE" == "$BUILD_ROOT/buildkit" ]] || {
        echo "Builder volume uses a different directory: $DEVICE" >&2; exit 1;
    }
else
    docker volume create --driver local --opt type=none --opt o=bind \
        --opt "device=$BUILD_ROOT/buildkit" "$VOLUME" >/dev/null
fi
python3 "$SCRIPT_DIR/configure_builder.py" --name "$BUILDER" \
    --image "$BUILDKIT_IMAGE" --root "$BUILD_ROOT"
cp "$CONTEXT/source-metadata.json" "$CONTEXT/streetcrafter/source-metadata.json"
python3 "$SCRIPT_DIR/compose_dockerfile.py" --context "$CONTEXT" >/dev/null
RECORD="$BUILD_ROOT/records/${IMAGE##*:}.json"
cp "$CONTEXT/source-metadata.json" "$RECORD"
mkdir -p "$BUILD_ROOT/images"
OUTPUT=(--output "type=docker,dest=$BUILD_ROOT/images/${IMAGE##*:}.tar")
if $DO_PUSH; then OUTPUT=(--push); fi
# Validation runs inside the final Dockerfile before the exporter can publish.
python3 "$SCRIPT_DIR/monitor_build.py" --directory "$BUILD_ROOT" -- \
docker buildx build --builder "$BUILDER" --platform linux/amd64 --progress plain \
    -f "$CONTEXT/streetcrafter/docker/joint/Dockerfile.generated" \
    --build-context "drivegs=$CONTEXT/drivegs" \
    --build-arg "BASE_IMAGE=$ADH_BASE_IMAGE" \
    --build-arg "STREET_CUDA_IMAGE=$STREET_CUDA_IMAGE" \
    --build-arg "PYPI_INDEX_URL=$PYPI_INDEX_URL" \
    --build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg NO_PROXY \
    --build-arg http_proxy --build-arg https_proxy --build-arg no_proxy \
    --build-arg "STREETCRAFTER_COMMIT=$STREET_COMMIT" \
    --build-arg "DRIVEGS_COMMIT=$DRIVEGS_COMMIT" \
    --build-arg "SOURCE_COMMIT=$DRIVEGS_COMMIT" \
    --build-arg SOURCE_DIRTY=true \
    --build-arg "SOURCE_FINGERPRINT=$FINGERPRINT" \
    --metadata-file "$BUILD_ROOT/records/${IMAGE##*:}.build.json" \
    "${OUTPUT[@]}" -t "$IMAGE" "$CONTEXT/streetcrafter"
printf 'Joint image: %s\nSource fingerprint: %s\nRecord: %s\n' "$IMAGE" "$FINGERPRINT" "$RECORD"
