## Waymo Open Dataset
### Download dataset
First download the 1.4.1 version of Waymo dataset (training and validation set), please refer to [this document](https://github.com/NVlabs/EmerNeRF/blob/main/docs/NOTR.md) for more download details. 

Since it is extremely slow for the lidar sensor processing when each scene is stored in the format of `.tfrecord` file, we additionally download part of the 2.0.0 version data. For instance, you need to run the following command under the training set:

``` shell
gsutil -m cp -r \                                                  
  "gs://waymo_open_dataset_v_2_0_0/training/lidar" \
  "gs://waymo_open_dataset_v_2_0_0/training/lidar_box" \
  "gs://waymo_open_dataset_v_2_0_0/training/lidar_calibration" \
  "gs://waymo_open_dataset_v_2_0_0/training/lidar_camera_projection" \
  "gs://waymo_open_dataset_v_2_0_0/training/lidar_camera_synced_box" \
  "gs://waymo_open_dataset_v_2_0_0/training/lidar_pose" \
  "gs://waymo_open_dataset_v_2_0_0/training/vehicle_pose" \
  .
```




Install Waymo Development Toolkit
```
pip install waymo-open-dataset-tf-2-11-0==1.6.0
```

### Process dataset (except LiDAR)


``` shell
python waymo_processor/waymo_converter.py \
    --root_dir  {ROOT_DIR} \
    --save_dir  {SAVE_DIR} \
    --process_list pose calib image track dynamic
```

The processed dataset would be like:

```shell
ProjectPath/data/
  └── waymo
    └── training_set_processed/
        ├── 000/
        │  ├──images/             # Images: {frame:06d}_{cam_id}.jpg
        │  ├──ego_pose/           # Ego vehicle poses: {frame:06d}.txt
        │  ├──extrinsics/         # Camera extrinsics: {cam_id}.txt
        │  ├──intrinsics/         # Camera intrinsics: {cam_id}.txt
        │  ├──dynamic_masks/      # Coarse dynamic masks: {frame:06d}_{cam_id}.png
        │  ├──track/              # Object tracklets
        ├── 001/
        ├── ...
```

### Process dataset (LiDAR)

Convert LiDAR range image into background and actor colored pointcloud and get sparse LiDAR depth for each camera view. 

``` shell
python waymo_processor/waymo_get_lidar_pcd.py \
    --root_dir  {ROOT_DIR} \
    --save_dir  {SAVE_DIR} 
```
    
The processed LiDAR would be like:

```shell
ProjectPath/data/
└── waymo/
    └── training_set_processed/
        ├── 000/lidar/
        │  ├──actor/             # Actor pointcloud
        │  ├──background/        # Background pointcloud: {frame:06d}.ply
        │  ├──depth/             # Sparse LiDAR depth: {frame:06d}_{cam_id}.npz
        ├── 001/lidar/
        ├── ...
```

#### New feature (Predicted pointcloud)
We also provide the script `waymo_get_moge_pcd.py`
for generating predicted pointcloud with geometry foundation model such as [MoGe](https://wangrc.site/MoGePage/). However, we have 
not fully tested the effectiveness of this type of guidance.

### Render LiDAR

Render aggregated LiDAR pointcloud to image plane.

``` shell
python waymo_processor/waymo_render_lidar_pcd.py \
    --data_dir {DARA_DIR} \
    --save_dir {SAVE_DIR} \
    --delta_frames 10 \
    --cams 0, \
    --shifts 0, 
```

Parameter explanation:
- `--data_dir`: Path to the folder containing all the processed data where each sequence is in the format of scene id `xxx`.

- `--save_dir`: Path to store the rendering LiDAR condition in each sequence direcotry, default as `color_render` where the rendering condition is saved in `xxx/color_render`.

- `--delta_frames`: Number of nearby frames LiDAR point cloud to aggerate, default as 10 for 1s.

- `--cams`: Camera ids to perform LiDAR condition rendering, default as 0 for only front camera. You can set the parameter to `0 1 2 3 4` to generate surrounding LiDAR condition for all cameras.

- `--shifts`: The shifting meters for each sequence, default as 0 to only generate the LiDAR condition of input trajectory.
You can change this parameter to generate multi-trajectory LiDAR condition. For instance, `--shifts 2 3` will generate conditions for lane-change settings where the camera is laterally shifted for 2 or 3 meters to the left. Noted that the condition of novel trajectory will be automatically generated during the distillation process, you can also directly inference the video model in `render.py` script.

The rendered LiDAR pointcloud would be like:
```shell
ProjectPath/data/
└── waymo/
    └── training_set_processed/
        ├── 000/lidar/color_render
        │  ├──{frame:06d}_{cam_id}.png            # Rendered LiDAR color
        │  ├──{frame:06d}_{cam_id}_depth.npy      # Rendered LiDAR depth
        │  ├──{frame:06d}_{cam_id}_mask.png       # Rendered LiDAR mask
        ├── 001/lidar/color_render
        ├── ...
```



### Prepare meta data

Prepare training and validation meta data for video diffusion model. 
Save meta data for training and validation of streetcrafter. You can also make your own json file to test the view synthesis result.

```shell
python waymo_processor/waymo_prepare_meta.py \
    --root_dir {ROOT_DIR} \
    --split 'train' 

python waymo_processor/waymo_prepare_meta.py \
    --root_dir {ROOT_DIR} \
    --split 'val' 
```

Parameter explanation:
- `--scene_ids`: specify which sequence to be included in the json file.

- `--postfix`: specify the postfix of json file name.


After running the command, meta data will be saved in root directory with each entry in the following format:
```json
{
    "frames": [
        "training_set_processed/000/images/000000_0.png",
        "....",
        "training_set_processed/000/images/000024_0.png"
    ],
    "guidances": [
        "training_set_processed/000/lidar/color_render/000000_0.png",
        "....",
        "training_set_processed/000/lidar/color_render/000024_0.png"
    ],
    "guidances_mask": [
        "training_set_processed/000/lidar/color_render/000000_0_mask.png",
        "....",
        "training_set_processed/000/lidar/color_render/000024_0_mask.png"
    ]
}
```


The final processed dataset will be like:
```shell
ProjectPath/data/
└── waymo/
    └── training_set_processed/
        ├── 000/
        ├── ...
        ├── 797/
    └── validation_set_processed/
        ├── 000/
        ├── ...
        ├── 201/
    ├── meta_info_train.json # training json file
    ├── meta_info_val.json   # validation json file
```


### Generate sky mask 
This step is only required during the distillation process.

Install GroundingDINO following [this repo](https://github.com/IDEA-Research/GroundingDINO) and download SAM checkpoint from [this link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).

```
python waymo_processor/generate_sky_mask.py --datadir DATA_DIR --sam_checkpoint SAM_CKPT
```

## PandaSet
The processing code of PandaSet is built upon [drivestudio](https://ziyc.github.io/omnire/).

### Download dataset
Please follow Section 1, 2 of [this document](https://github.com/ziyc/drivestudio/blob/main/docs/Pandaset.md) to download the raw data and install PandaSet Development Toolkit.

### Process dataset
```shell
python pandaset_processor/pandaset_convertor.py \
    --data_root {ROOT_DIR} \
    --target_dir {SAVE_DIR} \  
    --process_keys images lidar_forward calib pose dynamic_masks objects
```

### Render LiDAR

Render aggregated LiDAR pointcloud to image plane.

``` shell
python pandaset_processor/pandaset_render_lidar_pcd.py \
    --data_dir {DARA_DIR} \
    --save_dir {SAVE_DIR} \
    --delta_frames 10 \
    --cams 0, \
    --shifts 0, 
```

### Prepare meta data
```shell
python pandaset_processor/pandaset_prepare_meta.py \
    --root_dir {ROOT_DIR} \
    --split 'train' 

python pandaset_processor/pandaset_prepare_meta.py \
    --root_dir {ROOT_DIR} \
    --split 'val' 
```

### Generate sky mask 

```
python pandaset_processor/generate_sky_mask.py --datadir DATA_DIR --sam_checkpoint SAM_CKPT
```

## NIO (NOP processed clips)

The NIO processor reads a GS Hub `processed-clip-data.tar` directly. It combines
the **aggregated LiDAR map** with per-frame FALCON scans, including dynamic objects.
All commands below run from the StreetCrafter project root. Existing NumPy,
SciPy, OpenCV and Pillow are required; no packages are installed automatically.
Decoding the map requires the existing ADH SDK and `g++`, or a prebuilt decoder
passed through `--decoder` to `nio_get_map_pcd.py`. The map decoder is compiled
under the output scene's `audit/.cache/map_decoder/` using the existing SDK.
The default SDK root is `/usr/local/aac/adh/Linux-x86_64`; override it with
`NIO_ADH_ROOT` or `--adh_root`.

### Process the complete clip

```shell
.venv/bin/python data_processor/nio_processor/nio_process.py \
    --raw_tar /home/nio/Projects/gs_rec/data/nop/efa3fa0a-4af8-4f67-846e-24116d7470d6/processed-clip-data.tar \
    --save_dir data/nio/nop/efa3fa0a-4af8-4f67-846e-24116d7470d6 \
    --depth_decoder /home/nio/.cache/drivegs/depth_decoder/5505691d8d97178a/dump_depth_elm \
    --workers 4 --render_workers 2
```

The five stages run sequentially: synchronized frame conversion, map coloring and
local crops, per-frame LiDAR conversion, CPU rendering, and video metadata. Add `--resume`
to reuse complete image outputs, the map decoding cache and converted scans. Rendering is
reused only when the map, frame scans, trajectories, RGB, cameras, masks and render configuration match.
Use `--start_stage lidar` (or `map`, `render`, `meta`) when earlier stages
are already complete. JPEG decoding uses Pillow; geometry rendering uses
float32 matrices matching the map to avoid mixed-type NumPy fallbacks.

Place per-frame binary PCD scans in
`$SCENE_DIR/raw_lidar/<sequence_id>/output/FALCON/`, or pass `--raw_lidar_dir`
to the entry point. To update an existing background-only scene, rerun the entry
point with `--start_stage lidar --resume`; this also rebuilds rendering and video metadata.

### Individual stages and their outputs

Set `RAW_TAR` and `SCENE_DIR` to your source archive and output scene:

```shell
RAW_TAR=/home/nio/Projects/gs_rec/data/nop/efa3fa0a-4af8-4f67-846e-24116d7470d6/processed-clip-data.tar
SCENE_DIR=data/nio/nop/efa3fa0a-4af8-4f67-846e-24116d7470d6

.venv/bin/python data_processor/nio_processor/nio_converter.py \
    --raw_tar "$RAW_TAR" --save_dir "$SCENE_DIR" --workers 4

.venv/bin/python data_processor/nio_processor/nio_get_map_pcd.py \
    --raw_tar "$RAW_TAR" --save_dir "$SCENE_DIR" \
    --color_stride 5 --radius 80 --voxel_size 0.1

.venv/bin/python data_processor/nio_processor/nio_get_frame_pcd.py \
    --raw_tar "$RAW_TAR" --data_dir "$SCENE_DIR" --resume

.venv/bin/python data_processor/nio_processor/nio_render_pcd.py \
    --data_dir "$SCENE_DIR" --workers 2 --radius 2

.venv/bin/python data_processor/nio_processor/nio_prepare_meta.py \
    --data_dir "$SCENE_DIR" --cams 0 --window_size 25 --stride 5 --train_ratio 0.8

```

| Stage | Output relative to the scene directory |
| --- | --- |
| Synchronization | `metadata.json`, `timestamps.json` |
| RGB and calibration | `images/{frame:06d}_{cam_id}.png`, `intrinsics/{cam_id}.txt`, `extrinsics/{cam_id}.txt` |
| Timed vehicle poses | `ego_pose/{frame:06d}.txt`, `ego_pose/{frame:06d}_{cam_id}.txt` |
| Semantic labels and masks | `semantic/`, `dynamic_mask/`, `sky_mask/`, `ego_body_mask/`, `ground_mask/` |
| Tracks | `track/{track_info.pkl,track_camera_visible.pkl,trajectory.pkl,track_ids.json}` |
| Aggregate world map | `lidar/map/background.ply`, `lidar/map/background.npz` |
| Local map in each frame's vehicle coordinates | `lidar/background/{frame:06d}.ply` |
| Per-frame world scans and object assignments | `lidar/frame/{scan_timestamp}.npz`, `lidar/frame/metadata.json` |
| Camera-Z LiDAR/map depth | `lidar/depth/{frame:06d}_{cam_id}.npz` (`mask` and one-dimensional `value`) |
| Combined guidance | `lidar/color_render/{frame:06d}_{cam_id}.png`, `_mask.png`, `_depth.npy`, `audit/render.json` |
| Optional mono prediction | `mono_depth/{frame:06d}_{cam_id}.npy`, `mono_depth/metadata.json`, `mono_depth_mask/` |
| Video windows | `meta_info_train.json`, `meta_info_val.json` |
| Resume state and decoder cache | `audit/render.json`, `audit/.cache/map_decoder/` |

Camera IDs are stable: `FW=0`, `FN=1`, `FL=2`, `FR=3`, `RL=4`, `RR=5`,
`RN=6`, `SFW=7`, `SLW=8`, `SRW=9`, `SRCW=10`. Frame counts are determined
from synchronized poses and available files. Camera models include RadTan,
Poly5, OCam, and the calibrated Spline2 used by FW. By default, output width
is capped at 1920 while preserving the source aspect ratio. ADAS uses scaled
native intrinsics; SVC uses a 120-degree pinhole view on the same optical
axis. These views crop native coverage; the parameters are recorded in
`metadata.json`. Override them with `--max_width`, `--adas_fov`, or `--svc_fov`
on the converter, using a fresh output directory when changing parameters.

`extrinsics` stores OpenCV camera-to-vehicle matrices. Both ego-pose file
patterns store vehicle-to-world matrices; the camera ID selects the vehicle
pose interpolated at that camera's capture time. World coordinates retain
the original map origin. Masks use ADH SegLabel: ground `1`, dynamic `6/24`,
sky `23`, ego body `22`; invalid semantic pixels are `255`. Semantic labels
are rectified on the RGB grid with nearest-neighbor interpolation. The
converter uses source JPEGs as calibrated; it does not apply a second flip
based on the calibration's `flip_and_mirror` field.

The map keeps uncolored geometry for depth and occlusion. Its PLY `mask`
indicates whether a point has a usable RGB observation. Sky, dynamic, ego-body
and invalid pixels are excluded from map coloring and background guidance. Two-dimensional
dynamic masks do not prove that an occluded map point is dynamic, so map
geometry is retained and this limitation is recorded. No actor pointcloud
is fabricated. The supplied `ground.pcd` supplements road geometry. It is
merged before voxel filtering and colored from the same RGB observations.
Its sample projections are checked against ground semantics; provenance is
retained as `source_ground` in the map NPZ. Use `--skip_ground` on the map
processor to omit this optional source.

Per-frame scans use the calibrated FALCON-to-vehicle transform and vehicle pose
at each scan timestamp. Points within annotated 3D boxes are assigned to tracks;
object poses are interpolated on their true `source_timestamps` to compensate
motion from scan time to each camera exposure. The full scan is retained, including
static geometry and unassigned objects. Its projected points are colored from the
corresponding rectified RGB image and combined with the map using camera-Z occlusion.
Dynamic masks exclude only the map branch; actual per-frame LiDAR returns remain
eligible inside dynamic regions. The PCD `h/s/v` fields are not treated as RGB or
assumed to encode a known per-point acquisition time. No additional per-point deskew
is applied, and objects outside LiDAR coverage can remain incomplete.

Mono depth stays on its original, unrectified grid with `units=m` and
`convention=camera_z`, as confirmed for this dataset. Dynamic geometry comes from
the measured per-frame LiDAR; mono depth is not used to generate geometry. Provide an executable
with `--depth_decoder` (the converter defaults to the existing local cached
decoder shown above), or use `--skip_mono_depth` to omit this optional stage.

Video paths are relative to the **scene directory**, which must be the data
root when consuming this JSON. The default front-camera windows contain 25
frames with stride 5. Training and validation are split by time before
windows are generated; missing guidance cannot create discontinuous windows.
This preprocessing does not register a NIO training reader in StreetCrafter
or the video model. Direct training requires a separate reader and an actor
initialization strategy appropriate to the available map.

Input validity, timestamp matching, ground alignment, nonempty rendered guidance,
and complete video windows are checked within their processing stages. The
processor does not generate standalone validation reports or preview images.
Keep `audit/render.json` and the decoder cache to support resuming work.

### Generate NIO novel-view videos

After preprocessing, run from the project root:

```shell
bash scripts/video_generation.sh configs/nio_efa3fa0a_video.yaml
```

The configuration selects FW frames 0–224 and lateral offsets of 2 m and 3 m
along the vehicle's positive Y (left) axis, without changing camera rotation.
The NIO video entry point renders new pointcloud conditions at these poses,
then runs the existing high-resolution diffusion model on one GPU. It does not
require a trained Gaussian scene. Original RGB is used as a reference anchor;
the anchor is excluded from the resulting novel-view video timeline.

Conditions, generated frames and resumable production state are saved under
`output/nio/<sequence_id>/video_generation/`. The final videos are
`diffusion_novel_2.0.mp4` and `diffusion_novel_3.0.mp4`. Rerunning the same command
reuses matching conditions and completed sampling windows. `PYTHON_BIN` can
select an existing Python environment; the default is `.venv/bin/python`.
Passing a Waymo configuration keeps the original `render.py mode diffusion` entry.

