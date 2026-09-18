from street_gaussian.models.street_gaussian_model import StreetGaussianModel
from street_gaussian.datasets.dataset import Dataset
from street_gaussian.pointcloud_processor import getPointCloudProcessor
from street_gaussian.models.scene import Scene
from street_gaussian.config import cfg
from video_diffusion.sample_condition import VideoDiffusionModel

# 创建场景
def create_scene():
    dataset = Dataset()# 相机、原始图像、场景元数据
    gaussians = StreetGaussianModel(dataset.metadata)# 待优化的sg模型架构
    if (cfg.mode == 'train' and cfg.diffusion.use_diffusion) or cfg.mode == 'diffusion':
        print("Diffusion model loaded")
        video_diffusion = VideoDiffusionModel(
            config_path=cfg.diffusion.config_path,
            ckpt_path=cfg.diffusion.ckpt_path,
            height=cfg.diffusion.height,
            width=cfg.diffusion.width
        )# sc模型：加载sc配置和预训练权重
    else:
        video_diffusion = None

    if cfg.mode == 'train' or cfg.mode == 'diffusion':
        pointcloud_processor = getPointCloudProcessor()# 初始化点云，渲染点云条件
    else:
        pointcloud_processor = None

    scene = Scene(
        gaussians=gaussians,
        dataset=dataset,
        pointcloud_processor=pointcloud_processor,
        diffusion=video_diffusion,
    )# 初始化高斯点云，准备原始视角和新视角的lidar条件

    return scene
