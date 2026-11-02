# DriveGS + StreetCrafter 联合镜像

本流程在当前主机构建和检查一个 A100 联合镜像，通过检查后推送到云端镜像仓库，不启动训练。
镜像仓库：`adas-img.nioint.com/adsim-scenario/drivegs`。

## 环境与目录

| 项目 | 源码 | Miniforge Python | Torch / CUDA |
| --- | --- | --- | --- |
| drive-gs | `/workspace/drivegs` | `/opt/conda/bin/python`，3.11.13 | 2.8.0+cu128 / 12.8.1 |
| StreetCrafter | `/workspace/street_crafter` | `/opt/conda/envs/streetcrafter/bin/python`，3.9.23 | 2.4.0+cu121 / 12.1.1 |

DriveGS 从已发布、digest 固定的 base-adh 构建当前源码的本地 train 中间镜像，
保留 ADH SDK 和训练算子。StreetCrafter 在 CUDA 12.1 / Ubuntu 22.04 构建阶段
使用固定版本、校验值的 Miniforge 创建独立环境，最后合并进 train 镜像。
两阶段均保留 CUDA 开发工具；只推送最终联合镜像。

StreetCrafter 按 A100 的 `sm_80` 编译；drive-gs 沿用 `8.0;8.9+PTX`。
容器专用依赖固定在 `docker/joint/constraints.streetcrafter.txt`，Git 依赖固定提交。
NumPy 1.26.4 / OpenCV 4.10.0.84 用于兼容 imgaug；不修改本机已有依赖。
范围包括 Gaussian 训练、渲染和视频扩散推理，不额外配置旧 Vista 的 DeepSpeed 多卡微调环境。

## 本机构建与推送

先确保当前用户可执行 `docker info`，并已登录镜像仓库：

```bash
docker login adas-img.nioint.com
```

在 StreetCrafter 根目录执行：

```bash
bash docker/joint/build.sh --test --push
```

脚本自动使用同级 `drive-gs`。可指定路径、仓库或 Miniforge base-adh：

```bash
bash docker/joint/build.sh \
  --drivegs-root /home/zqb/Projects/drive-gs \
  --repository adas-img.nioint.com/adsim-scenario/drivegs \
  --test --push
```

只查询标签用 `--print-images`；再次执行会复用专用构建器缓存。
底层 `build.sh` 通过 `--push` 发布；省略该选项则导出 Docker 镜像归档到构建目录的
`images/<镜像标签>.tar`，可用 `docker load -i` 导入。`--no-push` 可覆盖较早出现的 `--push`。

默认将源码快照、BuildKit 状态和构建记录放到 `/data/drivegs-zqb`，可用 `--build-root`
或 `BUILD_ROOT` 覆盖。使用专用 docker-container 构建器及 bind volume，
不迁移 `/var/lib/docker`，不切换默认构建器，不清理其他构建缓存。
专用构建器关闭自动缓存回收，避免双环境的大体积基础层在重试时被回收；低空间监控仍会中止构建。
推送模式直接发布到仓库；本地归档模式导出到 `/data`，均不自动加载到根分区的 Docker 镜像库。
两套环境的检查在 Dockerfile 中完成，失败时不会导出成品或进入推送步骤。
可用 `BUILD_PROXY` 指定主机代理；专用构建器使用 host 网络，内部镜像仓库及其
`cos.ap-beijing.myqcloud.com` 大文件下载地址绕过代理。

独立构建器初次构建需要重新获取基础层，建议预留 70GB；默认以此为预检门槛，
可按实际缓存情况用 `MIN_FREE_GB` 覆盖。构建期间可用空间低于 5GB 时中止当前构建，
保留已完成缓存，不删除已有镜像、容器、数据卷。需要访问内部仓库、Docker Hub、GitHub、
conda-forge、PyPI 和 PyTorch wheel 源。

推送成功后的 `.build.json` 记录包含最终 digest，存放于构建目录的 `records/`。
当前项目的 `scripts/drivegs_joint_train_docker_build.sh` 是默认构建、检查并推送的便捷入口，
自动定位当前项目及同级 `drive-gs`，也可通过 `STREETCRAFTER_ROOT` 和 `DRIVEGS_ROOT` 覆盖路径：

```bash
bash /home/zqb/Projects/street_crafter/scripts/drivegs_joint_train_docker_build.sh
```

需要仅保存本地归档时，在该入口后追加 `--no-push`。

## 源码快照与交付

两个项目分别按白名单复制到临时目录，记录逐文件内容和权限摘要。
排除宿主机 `.venv`、数据、权重、输出、编译缓存及登录笔记等凭据文件。
保留 `video_diffusion/vwm/data` 源码和两个子模块源码；不修改子模块工作区。

脚本在临时 DriveGS 构建文件中补充 `distill_pipeline.py`，不修改其原 Dockerfile。
联合发布使用独立的内容标识标签，记录两个 Git 提交、实际打包源码 SHA256、
基础镜像和构建输入；不会自动提交 Git，也不调用原 DriveGS 脚本的 master 发布流程。
未提交的授权源代码会进入对应源码快照；不打包的文件变化不改变源码内容标识。

联合标签格式为 `joint-a100-<street摘要>-<drivegs摘要>-<构建摘要>`。
推送后记录完整 tag 和 repository digest；固定内容时使用
`adas-img.nioint.com/adsim-scenario/drivegs@sha256:...`。
镜像中 `/opt/joint/source-metadata.json` 保存双项目版本及构建输入。

### 2026-10-30 已完成的构建

已完成构建、联合 CPU 检查和仓库推送，并核对远端 digest：

```text
adas-img.nioint.com/adsim-scenario/drivegs:joint-a100-9df2652b6d6f-cca8011debfd-dca00e757a95
sha256:0c0614c0606c4f9125efa041eccf6347f8f07d64a5f2c6c3b94858ff95b7c6f0
```

本次构建记录：
`/data/drivegs-zqb/records/joint-a100-9df2652b6d6f-cca8011debfd-dca00e757a95.build.json`。
未执行 GPU 训练或 A100 实机验证。

## 入口与检查边界

`streetcrafter-python` 和 `drivegs-python` 隔离解释器、CUDA、动态库、编译器和缓存。
StreetCrafter 的 `.venv/bin/python` 是容器专用兼容入口，不来自宿主机。
`joint-distill` 使用生成的 `streetcrafter_distill.container.json`，覆盖本机绝对路径和
学生 CUDA 12.6 默认路径；不修改原始场景配置或业务训练代码。

`--test` 检查两套环境、依赖一致性、入口帮助、ADH 解码器及编译产物，
不加载数据和模型权重、不启动训练、不验证 A100 上实际执行结果。
CPU 验证期间可临时加载驱动 stub；运行入口不会将 stub 放入动态库搜索路径。
部分算子保留运行时 JIT 能力，两个项目分别使用独立缓存目录。

现有联合蒸馏要求学生和教师使用不同 GPU；单镜像不改变此调度约束。
数据、扩散权重和输出目录需由后续运行环境提供。
