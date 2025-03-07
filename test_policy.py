from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply

class DiffusionUnetImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: MultiImageObsEncoder,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes
        # 从 shape_meta 中获取动作（action）的形状信息
        action_shape = shape_meta['action']['shape']
        # 断言动作的形状为一维，确保动作向量是一维数组
        assert len(action_shape) == 1
        # 提取动作的维度大小
        # 动作向量总维度为 ​7，对应 3D 平移 + 3D 旋转四元数 + 夹爪状态
        action_dim = action_shape[0] # =>7
        # 使用观察编码器（obs_encoder）获取输出特征的维度
        # obs_feature_dim 的具体值需根据 obs_encoder 的实际输出确定
        # 如果 obs_encoder 接收多模态输入（如图像 + 位姿），则需要通过 example_obs_dict 的构造方式推断输出维度
        # 取第一个元素: 假设输出形状为 (batch_size, feature_dim, ...), 提取特征维度 feature_dim
        obs_feature_dim = obs_encoder.output_shape()[0]  # => 
        # print(f"obs_feature_dim: {obs_feature_dim}")

        # create diffusion model
        input_dim = action_dim + obs_feature_dim # => 7 + 
        global_cond_dim = None  # => None
        if obs_as_global_cond: 
            input_dim = action_dim # => 7
            global_cond_dim = obs_feature_dim * n_obs_steps # 展平后的观测特征维度
 
        # 创建条件化的 Unet 模型
        model = ConditionalUnet1D(
            input_dim=input_dim, 
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim, # 扩散步骤嵌入维度
            down_dims=down_dims, # 下采样维度
            kernel_size=kernel_size, # 卷积核大小
            n_groups=n_groups, # 分组数
            cond_predict_scale=cond_predict_scale # 是否预测缩放因子
        )

        self.obs_encoder = obs_encoder # 观测编码器
        self.model = model # 模型
        self.noise_scheduler = noise_scheduler  # 噪声调度器
        # 创建低维度掩码生成器
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim, # 动作维度
            obs_dim=0 if obs_as_global_cond else obs_feature_dim, # 观测维度
            max_n_obs_steps=n_obs_steps, # 最大观测步数
            fix_obs_steps=True, # 固定观测步数
            action_visible=False # 动作不可见
        )
        self.normalizer = LinearNormalizer()    # 线性归一化器
        self.horizon = horizon                  # 规划水平
        self.obs_feature_dim = obs_feature_dim  # 观测特征维度
        self.action_dim = action_dim            # 动作维度
        self.n_action_steps = n_action_steps    # 动作步数
        self.n_obs_steps = n_obs_steps          # 观测步数
        self.obs_as_global_cond = obs_as_global_cond    # 观测作为全局条件
        self.kwargs = kwargs                # 参数

        # 设置推理步数
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps    # 推理步数
        self.num_inference_steps = num_inference_steps  # 推理步数
    
    # ========= inference  ============
    # 条件化采样
    # 被predict_action调用
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model # 模型
        scheduler = self.noise_scheduler # 噪声调度器

        # 采样噪声轨迹
        trajectory = torch.randn(
            size=condition_data.shape, # 条件数据形状
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        # 不同的观测传递方式
        local_cond = None
        global_cond = None
        # 如果观测作为全局条件
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        # 如果观测不作为全局条件
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss