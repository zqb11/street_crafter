from vwm.modules.diffusionmodules.sampling import EulerEDMSampler, EulerEDMSamplerSDS
import imageio
from vwm.util import default, instantiate_from_config
import argparse
import math
import os
import numpy as np
import torch
from pytorch_lightning import seed_everything
from typing import List, Optional, Union
from einops import rearrange, repeat
from omegaconf import ListConfig, OmegaConf
from safetensors.torch import load_file as load_safetensors
from torch import autocast
import warnings
warnings.filterwarnings("ignore")

DATASET2SOURCES = {
    "NUSCENES": {
        "data_root": "data/nuscenes",
        "anno_file": "annos/nuScenes_val.json"
    },
    "IMG": {
        "data_root": "image_folder"
    }
}


def parse_args(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--dataset", type=str, default="waymo", help="dataset name")
    parser.add_argument("--save", type=str, default="outputs", help="directory to save samples")
    parser.add_argument("--n_rounds", type=int, default=1, help="number of sampling rounds")
    parser.add_argument("--n_frames", type=int, default=25, help="number of frames for each round")
    parser.add_argument("--n_conds", type=int, default=1, help="number of initial condition frames for the first round")
    parser.add_argument('--seed', type=int, default=23, help='random seed for seed_everything')
    parser.add_argument("--height", type=int, default=576, help="target height of the generated video")
    parser.add_argument("--width", type=int, default=1024, help="target width of the generated video")
    parser.add_argument("--cfg_scale", type=float, default=2.5, help="scale of the classifier-free guidance")
    parser.add_argument("--cond_aug", type=float, default=0.0, help="strength of the noise augmentation")
    parser.add_argument("--n_steps", type=int, default=50, help="number of sampling steps")
    parser.add_argument("--rand_gen", action="store_false", help="whether to generate samples randomly or sequentially")
    parser.add_argument("--low_vram", action="store_true", help="whether to save memory or not")
    parser.add_argument("--config", type=str, help="config file for the model", default='configs/inference/vista.yaml')
    parser.add_argument("--ckpt", type=str, help="checkpoint file for the model", default='ckpts/vista.safetensors')
    return parser


lowvram_mode = False


def set_lowvram_mode(mode):
    global lowvram_mode
    lowvram_mode = mode


def initial_model_load(model):
    global lowvram_mode
    if lowvram_mode:
        model.model.half()
    else:
        model.to('cuda', non_blocking=True)
    return model


def load_model(model):
    model.to('cuda', non_blocking=True)


def unload_model(model):
    global lowvram_mode
    if lowvram_mode:
        model.cpu()
        torch.cuda.empty_cache()


def load_model_from_config(config, ckpt=None):
    model = instantiate_from_config(config.model)

    if ckpt is not None:
        print(f"Loading model from {ckpt}")
        if ckpt.endswith("ckpt"):
            pl_svd = torch.load(ckpt, map_location="cpu")
            # dict contains:
            # "epoch", "global_step", "pytorch-lightning_version",
            # "state_dict", "loops", "callbacks", "optimizer_states", "lr_schedulers"
            if "global_step" in pl_svd:
                print(f"Global step: {pl_svd['global_step']}")
            svd = pl_svd["state_dict"]
        elif ckpt.endswith("safetensors"):
            svd = load_safetensors(ckpt)
        else:
            raise NotImplementedError("Please convert the checkpoint to safetensors first")

        missing, unexpected = model.load_state_dict(svd, strict=False)  # type: ignore
        if len(missing) > 0:
            print(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected keys: {unexpected}")

    model = initial_model_load(model)
    model.eval()  # type: ignore
    return model


def init_embedder_options(keys):
    # hardcoded demo settings, might undergo some changes in the future
    value_dict = dict()
    for key in keys:
        if key in ["fps_id", "fps"]:
            fps = 10
            value_dict["fps"] = fps
            value_dict["fps_id"] = fps - 1
        elif key == "motion_bucket_id":
            value_dict["motion_bucket_id"] = 127  # [0, 511]
    return value_dict


def init_sampling(sampler="EulerEDMSamplerSDS", guider="VanillaCFG", discretization="EDMDiscretization", steps=50, cfg_scale=2.5, num_frames=25):
    discretization_config = get_discretization(discretization)
    guider_config = get_guider(guider, cfg_scale, num_frames)
    sampler = get_sampler(sampler, steps, discretization_config, guider_config)
    return sampler


def get_discretization(discretization):
    if discretization == "LegacyDDPMDiscretization":
        discretization_config = {
            "target": "vwm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization"
        }
    elif discretization == "EDMDiscretization":
        discretization_config = {
            "target": "vwm.modules.diffusionmodules.discretizer.EDMDiscretization",
            "params": {
                "sigma_min": 0.002,
                "sigma_max": 700.0,
                "rho": 7.0
            }
        }
    else:
        raise NotImplementedError
    return discretization_config


def get_guider(guider="LinearPredictionGuider", cfg_scale=2.5, num_frames=25):
    if guider == "IdentityGuider":
        guider_config = {
            "target": "vwm.modules.diffusionmodules.guiders.IdentityGuider"
        }
    elif guider == "VanillaCFG":
        scale = cfg_scale

        guider_config = {
            "target": "vwm.modules.diffusionmodules.guiders.VanillaCFG",
            "params": {
                "scale": scale
            }
        }
    elif guider == "LinearPredictionGuider":
        max_scale = cfg_scale
        min_scale = 1.0

        guider_config = {
            "target": "vwm.modules.diffusionmodules.guiders.LinearPredictionGuider",
            "params": {
                "max_scale": max_scale,
                "min_scale": min_scale,
                "num_frames": num_frames
            }
        }
    elif guider == "TrianglePredictionGuider":
        max_scale = cfg_scale
        min_scale = 1.0

        guider_config = {
            "target": "vwm.modules.diffusionmodules.guiders.TrianglePredictionGuider",
            "params": {
                "max_scale": max_scale,
                "min_scale": min_scale,
                "num_frames": num_frames
            }
        }
    else:
        raise NotImplementedError
    return guider_config


def get_sampler(sampler, steps, discretization_config, guider_config):
    # if sampler == "EulerEDMSamplerDGS":
    #     s_churn = 0.0
    #     s_tmin = 0.0
    #     s_tmax = 999.0
    #     s_noise = 1.0

    #     if steps == 25:
    #         weights = np.load('video_diffusion/nvs_solver/lambda_ts_25.npy')
    #     elif steps == 50:
    #         weights = np.load('video_diffusion/nvs_solver/lambda_ts_50.npy')
    #     elif steps == 100:
    #         weights = np.load('video_diffusion/nvs_solver/lambda_ts_100.npy')
    #     else:
    #         raise ValueError(f"Unknown steps {steps}")
    #     weights = torch.from_numpy(weights).float().to('cuda', non_blocking=True)

    #     # sampler = EulerEDMSamplerSDS(
    #     sampler = EulerEDMSamplerDGS(
    #         num_steps=steps,
    #         discretization_config=discretization_config,
    #         guider_config=guider_config,
    #         s_churn=s_churn,
    #         s_tmin=s_tmin,
    #         s_tmax=s_tmax,
    #         s_noise=s_noise,
    #         verbose=False,
    #     )
    if sampler == "EulerEDMSamplerSDS":
        s_churn = 0.0
        s_tmin = 0.0
        s_tmax = 999.0
        s_noise = 1.0
        sampler = EulerEDMSamplerSDS(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            s_churn=s_churn,
            s_tmin=s_tmin,
            s_tmax=s_tmax,
            s_noise=s_noise,
        )
    else:
        raise ValueError(f"Unknown sampler {sampler}")
    return sampler


def get_batch(keys, value_dict, N: Union[List, ListConfig], device="cuda"):
    # hardcoded demo setups, might undergo some changes in the future
    batch = dict()
    batch_uc = dict()

    for key in keys:
        if key in value_dict:
            if key in ["fps", "fps_id", "motion_bucket_id", "cond_aug"]:
                batch[key] = repeat(torch.tensor([value_dict[key]]).to(device), "1 -> b", b=math.prod(N))
            elif key in ["cond_frames", "cond_frames_without_noise"]:
                batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=N[0])
            else:
                # batch[key] = value_dict[key]
                raise NotImplementedError

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def get_condition(model, value_dict, num_samples, force_uc_zero_embeddings, device):
    load_model(model.conditioner)
    batch, batch_uc = get_batch(
        list(set([x.input_key for x in model.conditioner.embedders])),
        value_dict,
        [num_samples]
    )
    c, uc = model.conditioner.get_unconditional_conditioning(
        batch,
        batch_uc=batch_uc,
        force_uc_zero_embeddings=force_uc_zero_embeddings
    )
    unload_model(model.conditioner)

    for k in c:
        if isinstance(c[k], torch.Tensor):
            c[k], uc[k] = map(lambda y: y[k][:num_samples].to(device), (c, uc))
            if c[k].shape[0] < num_samples:
                c[k] = c[k][[0]]
            if uc[k].shape[0] < num_samples:
                uc[k] = uc[k][[0]]

    # add guidance
    load_model(model.first_stage_model)
    guidance = model.encode_first_stage(value_dict['guide_frames'])
    guidance_c = dict()
    guidance_c["input"] = guidance[:num_samples].to(device)
    guidance_c["scale"] = torch.ones(num_samples, device='cuda')
    c["guidance"] = guidance_c
    guidance_uc = dict()
    guidance_uc["input"] = guidance[:num_samples].to(device)
    guidance_uc["scale"] = torch.zeros(num_samples, device='cuda')  # TODO: 0 or 1?
    uc["guidance"] = guidance_uc

    if value_dict["training_free_guidance"]:
        c['sample_guidance'] = dict()
        c['sample_guidance']['input'] = model.encode_first_stage(value_dict['img_frames'])
        guide_masks = value_dict['guide_masks_frames']  # (f, 1, h, w)
        f, _, h, w = guide_masks.shape
        guide_masks = guide_masks.reshape(f, h // 8, 8, w // 8, 8)
        guide_masks = guide_masks.permute(0, 1, 3, 2, 4).reshape(f, h // 8, w // 8, 64)
        guide_masks = guide_masks.mean(-1)  # (f, h // 8, w // 8)
        guide_masks = guide_masks < 0.2  # (f, h // 8, w // 8)
        guide_masks[..., int(h // 16):, :] = False
        c['sample_guidance']['mask'] = guide_masks[:, None, :, :]

        render_masks = value_dict['img_masks_frames']  # (f, 1, h, w)
        f, _, h, w = render_masks.shape
        render_masks = render_masks.reshape(f, h // 8, 8, w // 8, 8)
        render_masks = render_masks.permute(0, 1, 3, 2, 4).reshape(f, h // 8, w // 8, 64)
        render_masks = render_masks.mean(-1)  # (f, h // 8, w // 8)
        c['sample_guidance']['acc'] = render_masks[:, None, :, :]

        # import numpy as np
        # import imageio
        # x = (guide_masks.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        # x = np.transpose(x, (0, 2, 3, 1))
        # imageio.mimwrite('guide_masks.mp4', x, fps=10, format='mp4')
        # x = (render_masks.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        # x = np.transpose(x, (0, 2, 3, 1))
        # imageio.mimwrite('render_masks.mp4', x, fps=10, format='mp4')

        uc['sample_guidance'] = c['sample_guidance']

    return c, uc


def fill_latent(cond, length, cond_indices, device):
    latent = torch.zeros(length, *cond.shape[1:]).to(device)
    latent[cond_indices] = cond
    return latent


class VideoDiffusionModel():
    def __init__(self,
                 n_rounds: int = 1,  # number of sampling rounds
                 seed: int = 23,  # random seed for seed_everything
                 height: int = 576,  # target height of the generated video
                 width: int = 1024,  # target width of the generated video
                 cfg_scale: float = 2.5,  # scale of the classifier-free guidance
                 cond_aug: float = 0.0,  # strength of the noise augmentation
                 n_steps: int = 50,  # number of sampling steps
                 low_vram: bool = False,  # whether to save memory or not
                 config_path: str = 'configs/inference/vista.yaml',  # config file for the model
                 ckpt_path: str = 'ckpts/vista.safetensors'  # checkpoint file for the model
                 ):
        set_lowvram_mode(low_vram)
        self.config = OmegaConf.load(config_path)
        self.num_frames = self.config.model.params.num_frames
        print(f'Num frames: {self.num_frames}')

        self.model = load_model_from_config(self.config, ckpt_path)

        self.unique_keys = set([x.input_key for x in self.model.conditioner.embedders])  # type: ignore
        self.sampler: EulerEDMSamplerSDS = init_sampling(
            guider="VanillaCFG",  # Options: VanillaCFG, TrianglePredictionGuider, LinearPredictionGuider
            sampler="EulerEDMSamplerSDS",  # Options: EulerEDMSamplerSDS, EulerEDMSamplerDGS
            steps=n_steps,
            cfg_scale=cfg_scale,
            num_frames=self.num_frames
        )  # type: ignore

        self.device = "cuda"
        self.model.to(self.device, non_blocking=True)

        self.sample_height = height
        self.sample_width = width
        self.seed = seed
        self.cond_aug = cond_aug
        self.num_rounds = n_rounds

        assert self.num_rounds == 1, "Only one round of sampling is supported"

    def prepare_value_dict(self, batch):
        value_dict = init_embedder_options(self.unique_keys)
        img_seq = batch['img_seq'].to(self.device, non_blocking=True)  # [t, 3, h, w]
        img_mask_seq = batch['img_mask_seq'].to(self.device, non_blocking=True)  # [t, 1, h, w]
        guide_seq = batch['guide_seq'].to(self.device, non_blocking=True)  # [t, 3, h, w]
        guide_mask_seq = batch['guide_mask_seq'].to(self.device, non_blocking=True)  # [t, 1, h, w]

        cond_image = img_seq[0][None]  # [b, 3, h, w]
        h, w = cond_image.shape[-2:]
        assert h == self.sample_height and w == self.sample_width, f"Input image size should be ({self.sample_height}, {self.sample_width})"

        value_dict["cond_frames_without_noise"] = cond_image
        value_dict["cond_aug"] = self.cond_aug
        value_dict["cond_frames"] = cond_image + self.cond_aug * torch.randn_like(cond_image)

        # training guidance
        value_dict["guide_frames"] = guide_seq

        # training free guidance
        value_dict["guide_masks_frames"] = guide_mask_seq
        value_dict["img_masks_frames"] = img_mask_seq
        value_dict["img_frames"] = img_seq

        if 'training_free_guidance' in batch:
            value_dict["training_free_guidance"] = batch['training_free_guidance']
        else:
            value_dict["training_free_guidance"] = False

        return value_dict

    def preprocess(self, batch):
        # crop height
        img_seq = batch['img_seq']
        guide_seq = batch['guide_seq']
        if 'guide_mask_seq' not in batch:
            guide_mask_seq = torch.ones_like(guide_seq[:, :1])
        else:
            guide_mask_seq = batch['guide_mask_seq']

        if 'img_mask_seq' not in batch:
            img_mask_seq = torch.ones_like(img_seq[:, :1])
        else:
            img_mask_seq = batch['img_mask_seq']

        batch['guide_mask_seq'] = guide_mask_seq
        batch['img_mask_seq'] = img_mask_seq
    # 扩散前向
    def forward(self, batch, scale: float = 0.3, cond_indices: List[int] = [0]):
        '''
        batch (dict): input batch
        batch['img_seq'] (torch.Tensor): [t, 3, h, w]
        batch['guide_seq'] (torch.Tensor): [t, 3, h, w]
        batch['guide_mask_seq'] (torch.Tensor): [t, 1, h, w]
        '''

        # seed_everything
        seed_everything(self.seed)
        # prepraing input
        self.preprocess(batch)
        value_dict = self.prepare_value_dict(batch)
        force_uc_zero_embeddings = ["cond_frames", "cond_frames_without_noise"]

        precision_scope = autocast
        with torch.no_grad(), precision_scope(self.device), self.model.ema_scope("Sampling"):  # type: ignore
            c, uc = get_condition(self.model, value_dict, self.num_frames, force_uc_zero_embeddings, self.device)

            load_model(self.model.first_stage_model)  # type: ignore
            z = self.model.encode_first_stage(value_dict['img_frames'])  # type: ignore
            unload_model(self.model.first_stage_model)  # type: ignore

            samples_z = torch.zeros((self.num_rounds * (self.num_frames - 3) + 3, *z.shape[1:]), device='cuda')

            def denoiser(x, sigma, cond, cond_mask):
                return self.model.denoiser(self.model.model, x, sigma, cond, cond_mask)  # type: ignore

            load_model(self.model.denoiser)  # type: ignore
            load_model(self.model.model)  # type: ignore

            cond_mask = torch.zeros(self.num_frames, device='cuda')
            cond_mask[cond_indices] = 1

            noise = torch.randn_like(z)
            sample = self.sampler(
                denoiser,
                noise,
                cond=c,
                uc=uc,
                cond_frame=z,  # cond_frame will be rescaled when calling the sampler
                cond_mask=cond_mask,
                scale=scale
            )
            samples_z[:self.num_frames] = sample

            unload_model(self.model.model)
            unload_model(self.model.denoiser)

            load_model(self.model.first_stage_model)
            samples_x = self.model.decode_first_stage(samples_z)
            unload_model(self.model.first_stage_model)

            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

        return samples

def parse_guidance_path(guidance_path):
    import re

    match = re.search(r'.*/(\d+)/lidar/[^\/]+/(\d+)_(\d+).png', guidance_path) # Waymo
    # match = re.search(r'.*/(\d+)/lidar_forward/[^\/]+/(\d+)_(\d+).png', guidance_path) # Pandaset
    if match:
        scene_id, frame_id, cam_id = match.groups()
        return scene_id + "_" + frame_id + "_" + cam_id
    else:
        print("No match found")


if __name__ == "__main__":
    from vwm.data.subsets.waymo import WaymoDataset

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='./ckpts/vista_condition_high_res_mix_v2.safetensors')
    parser.add_argument('--config_path', type=str, default='./configs/inference/waymo_high_res.yaml')
    parser.add_argument('--postfix', type=str, default=None)
    
    args = parser.parse_args()
    postfix = args.postfix
    ckpt_path = args.ckpt_path
    config_path = args.config_path
    
    seed_everything(23)

    # sample_height, sample_width = 320, 576

    # dataset = WaymoDataset(target_height=sample_height, target_width=sample_width, split='val')
    # dataset_length = len(dataset)
    # num_samples = 10
    # selected_indices = random.sample(range(dataset_length), num_samples)

    # model = VideoDiffusionModel(
    #     config_path='video_diffusion/configs/inference/waymo_low_res.yaml',
    #     ckpt_path='video_diffusion/ckpts/vista_condition_low_res.safetensors',
    #     height=sample_height,
    #     width=sample_width
    # )

    # os.makedirs("video_diffusion/outputs", exist_ok=True)
    # for index in selected_indices:
    #     print(f"Sample index: {index}")
    #     batch = dataset[index]
    #     samples = model.forward(batch)  # [num_frames, 3, h, w]

    #     samples = samples.cpu().numpy()
    #     images = ((batch['img_seq'] + 1.) / 2.).cpu().numpy()
    #     guides = ((batch['guide_seq'] + 1.) / 2.).cpu().numpy()

    #     save_result = np.concatenate([samples, images, guides], axis=-1)  # [num_frames, 3, h, 3*w]
    #     save_result = np.transpose(save_result, (0, 2, 3, 1))  # [num_frames, h, 3*w, 3]
    #     save_path = f'video_diffusion/outputs/sample_{index:06d}.mp4'
    #     imageio.mimwrite(save_path, (save_result * 255).astype(np.uint8), format='mp4', fps=10) # type: ignore

    sample_height, sample_width = 576, 1024    
    dataset = WaymoDataset(target_height=sample_height, target_width=sample_width, split='val', postfix=postfix)
        
    dataset_length = len(dataset)
    selected_indices = list(range(dataset_length))
    model = VideoDiffusionModel(
        config_path=config_path,
        ckpt_path=ckpt_path,
        height=sample_height,
        width=sample_width
    )

    save_dir = "outputs_waymo"
    os.makedirs(save_dir, exist_ok=True)
    for index in selected_indices:
        print(f"Sample index: {index}")
        batch = dataset[index]
        batch['training_free_guidance'] = False

        guide_seq_path = batch['guide_seq_path']
        sample_name = parse_guidance_path(guide_seq_path[0])
        samples = model.forward(batch, scale=1.0)  # [num_frames, 3, h, w]

        samples = samples.cpu().numpy()
        images = ((batch['img_seq'] + 1.) / 2.).cpu().numpy()
        guides = ((batch['guide_seq'] + 1.) / 2.).cpu().numpy()
        save_result = np.concatenate([samples, images, guides], axis=-1)  # [num_frames, 3, h, 3*w]
        save_result = np.transpose(save_result, (0, 2, 3, 1))  # [num_frames, h, 3*w, 3]

        if postfix is not None:
            save_path = f'{save_dir}/sample_{sample_name}_{postfix}.mp4'
        else:
            save_path = f'{save_dir}/sample_{sample_name}.mp4'

        imageio.mimwrite(save_path, (save_result * 255).astype(np.uint8), format='mp4', fps=10)  # type: ignore
