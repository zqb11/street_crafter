# StreetCrafter — 开发约束

## 开发环境
- conda env `streetcrafter`, Python 3.9
- torch==2.4.0 / torchvision==0.19.0 / torchaudio==2.4.0，cu121（`--index-url https://download.pytorch.org/whl/cu121`）
- 依赖：`requirements.txt` + `gsplat`（`pip install "git+https://github.com/dendenxu/gsplat.git"`）+ `submodules/sdata`、`submodules/simple-knn`（均需 `pip install ./submodules/<name>`）
- 目标硬件：单张 A100/A800 80GB

## 常用运行命令
- 训练 Street Gaussian：`python train.py --config {config_path}`
- 渲染输入轨迹：`python render.py --config {config_path} mode trajectory`
- 渲染新视角：`python render.py --config {config_path} mode novel_view`
- 视频扩散模型推理：`python render.py --config {config_path} mode diffusion`
- 按 meta info 文件推理（需在 `video_diffusion/` 目录下执行）：`python sample_condition.py`
- 微调视频扩散模型（需在 `video_diffusion/` 目录下执行）：`sh training.sh`
- 数据处理流程见 `data_processor/README.md`

## 数据路径
- 场景数据：`data/waymo/{scene_id}`（对应 config 中 `source_path`）
- 扩散模型权重：`video_diffusion/ckpts/`
- 训练/渲染输出：`output/{dataset}/{exp_name}/`（如 `output/waymo/waymo_val_176/`）
- `data/`、`output*`、`video_diffusion/ckpts|outputs|logs|annos|src` 已被 `.gitignore` 排除，不进行版本管理

## 项目入口
- `train.py`：Street Gaussian 训练入口
- `render.py`：渲染/推理入口（mode: `trajectory` / `novel_view` / `diffusion`）
- `video_diffusion/sample_condition.py`、`video_diffusion/train.py`、`video_diffusion/training.sh`：视频扩散模型的推理/训练入口
- config 采用继承机制，字段 `parent_cfg` 指向父配置（见 `configs/*.yaml`），核心参数定义在 `street_gaussian/config/config.py`

## 不能修改的目录
- `submodules/sdata`、`submodules/simple-knn`：git submodule，指向外部仓库，不要直接改动内容
- `video_diffusion/vwm`：Vista 代码库的微调基座，改动需对齐上游文档（`video_diffusion/docs/`）

## 关键代码约定
- 需要断点调试时使用 `breakpoint()`（沿用 EasyVolCap 框架约定），不要引入 `pdb.set_trace()` 等其他方式
- 天空模型配置：
  - `model.nsg.include_sky=True`：天空区域大的场景，用球体外的独立 Gaussian 表示天空
  - `model.nsg.include_cube_map=False`：天空区域小的场景，天空并入背景 Gaussian
  - 二者互斥，按场景选择其中一种
- 每个场景需根据白天/夜晚正确设置 `data.white_background`，否则背景色会出错
