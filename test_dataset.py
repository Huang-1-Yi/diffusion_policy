from typing import Dict, List
import torch
import numpy as np
import zarr
import os
import shutil
from filelock import FileLock
from threadpoolctl import threadpool_limits
from omegaconf import OmegaConf
import tqdm

from torch.utils.data import DataLoader
import cv2
import json
import hashlib
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.real_world.real_data_conversion import real_data_to_replay_buffer
from diffusion_policy.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)


class RealPushTImageDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            n_latency_steps=0,
            use_cache=False,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            delta_action=False,
        ):
        assert os.path.isdir(dataset_path)
        
        replay_buffer = None
        if use_cache:
            # fingerprint shape_meta
            shape_meta_json = json.dumps(OmegaConf.to_container(shape_meta), sort_keys=True)
            shape_meta_hash = hashlib.md5(shape_meta_json.encode('utf-8')).hexdigest()
            cache_zarr_path = os.path.join(dataset_path, shape_meta_hash + '.zarr.zip')
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print('Cache does not exist. Creating!')
                        replay_buffer = _get_replay_buffer(
                            dataset_path=dataset_path,
                            shape_meta=shape_meta,
                            store=zarr.MemoryStore()
                        )
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = _get_replay_buffer(
                dataset_path=dataset_path,
                shape_meta=shape_meta,
                store=zarr.MemoryStore()
            )
        
        if delta_action:
            # replace action as relative to previous frame
            actions = replay_buffer['action'][:]
            # support positions only at this time
            assert actions.shape[1] <= 3
            actions_diff = np.zeros_like(actions)
            episode_ends = replay_buffer.episode_ends[:]
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i-1]
                end = episode_ends[i]
                # delta action is the difference between previous desired position and the current
                # it should be scheduled at the previous timestep for the current timestep
                # to ensure consistency with positional mode
                actions_diff[start+1:end] = np.diff(actions[start:end], axis=0)
            replay_buffer['action'][:] = actions_diff

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        
        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon+n_latency_steps,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.val_mask = val_mask
        self.horizon = horizon
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon+self.n_latency_steps,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_mask
            )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer['action'])
        
        # obs
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key])
        
        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key][T_slice],-1,1
                ).astype(np.float32) / 255.
            # T,C,H,W
            # save ram
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            # save ram
            del data[key]
        
        action = data['action'].astype(np.float32)
        # handle latency by dropping first n_latency_steps action
        # observations are already taken care of by T_slice
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action)
        }
        return torch_data

def zarr_resize_index_last_dim(zarr_arr, idxs):
    actions = zarr_arr[:]
    actions = actions[...,idxs] #保留前两个维度的数据
    zarr_arr.resize(zarr_arr.shape[:-1] + (len(idxs),))#zarr_arr.shape为(21655,6)，这里拼接的操作为(21655,)+(,2)
    #将其变形为(21655,2)
    zarr_arr[:] = actions
    return zarr_arr
#zarr_arr是action的21655*6维的矩阵，这里经过处理后只保留前两列的数据


def _get_replay_buffer(dataset_path, shape_meta, store):
    # parse shape meta
    rgb_keys = list()
    lowdim_keys = list()
    out_resolutions = dict()
    lowdim_shapes = dict()
    obs_shape_meta = shape_meta['obs']
    #obs_shape_meta为{'camera':xxxx,'camera':xxxx,'robot_eef_pose':{shape:,type:}}
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = tuple(attr.get('shape'))
        if type == 'rgb':
            rgb_keys.append(key)
            c,h,w = shape
            out_resolutions[key] = (w,h) #320,240
        elif type == 'low_dim':
            lowdim_keys.append(key)
            lowdim_shapes[key] = tuple(shape)
            if 'pose' in key:
                assert tuple(shape) in [(2,),(6,)]
    # 打印调试信息
    print(f"Lowdim keys detected: {lowdim_keys}")
    #lowdim_keys{'robot_eef_pose':(2,)}
    action_shape = tuple(shape_meta['action']['shape']) #这里action_shape为2
    # assert action_shape in [(2,),(6,)]
    if action_shape not in [(2,), (3,), (6,)]:
        raise ValueError(f"Unsupported action shape: {action_shape}")   

    # load data
    cv2.setNumThreads(1)
    with threadpool_limits(1):
        replay_buffer = real_data_to_replay_buffer(
            dataset_path=dataset_path,
            out_store=store,
            out_resolutions=out_resolutions,
            lowdim_keys=lowdim_keys + ['action'],
            image_keys=rgb_keys
        )
    # replay_buffer = {
    # 'camera_0': <zarr.Array> (n_steps, height, width, 3),  
    # 'camera_1': <zarr.Array> (n_steps, height, width, 3),  
    # 'action': <zarr.Array> (n_steps, action_dim),           
    # ...
    # }包含相机提取的帧，机械臂的状态信息,即replay_buffer.zarr和videos文件夹的内容


    
    # transform lowdim dimensions
    if action_shape == (2,):
        # 2D action space, only controls X and Y
        zarr_arr = replay_buffer['action']
        zarr_resize_index_last_dim(zarr_arr, idxs=[0,1])
        #只保留zarr_arr的前两个维度(Xy),(21655,2)
    for key, shape in lowdim_shapes.items():
        if 'pose' in key and shape == (2,):
            # only take X and Y
            zarr_arr = replay_buffer[key]
            zarr_resize_index_last_dim(zarr_arr, idxs=[0,1])
    #将zarr_arr里面存储的键为'robot_eef_pose'的维度及所有状态的维度降为2
    return replay_buffer


shape_meta = OmegaConf.create({
    'obs': {
        'camera_0': {
            'shape': [3, 240, 320],
            'type': 'rgb'
        },
        'camera_1': {
            'shape': [3, 240, 320],
            'type': 'rgb'
        },
        'robot_eef_pose': {
            'shape': [2],
            'type': 'low_dim'
        },
        'stage':{
            'shape': [1],
            'type': 'low_dim'
        },
        'gripper': {
            'shape': [1],
            'type': 'low_dim'
        }},
    'action': {
        'shape': [3]
    }
})

# de_shape_meta = {
#     'obs': {
#         'camera_0': {
#             'shape': [3, 240, 320],
#             'type': 'rgb'
#         },
#         'camera_1': {
#             'shape': [3, 240, 320],
#             'type': 'rgb'
#         },
#         'robot_eef_pose': {
#             'shape': [2],
#             'type': 'low_dim'
#         },
#         # 'stage':{
#         #     'shape': [1],
#         #     'type': 'low_dim'
#         # }
#     },
#     'action': {
#         'shape': [2]
#     }
#     }

print(111)
dataset_path = "data/demo_test_02253"  # 数据集路径
print(222)
params = {
    "shape_meta": shape_meta,
    "dataset_path": dataset_path,
    "horizon": 16,  # 预测步数
    "pad_before": 1,  # 前向填充步数
    "pad_after": 7,   # 后向填充步数
    "n_obs_steps": 2,  # 取前3个观察步骤
    "use_cache": True,  # 是否缓存数据
    "val_ratio": 0.1,  # 验证集比例
    "max_train_episodes": 1000,  # 最大训练episode数
    "delta_action": False,  # 是否使用相对动作
}

# 创建训练集和验证集
try:
    train_dataset = RealPushTImageDataset(**params)
# val_dataset = train_dataset.get_validation_dataset()
except Exception as e:
    print(f"Failed to create dataset: {e}")
    raise

# replay_buffer = train_dataset.replay_buffer

# 使用 DataLoader 加载数据（根据GPU调整 batch_size 和 workers）
batch_size = 64
num_workers = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size,num_workers=8, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# workspace 中train训练部分代码
# with tqdm.tqdm(train_dataloader, desc=f"Training epoch ", 
#         leave=False, mininterval=1.0) as tepoch:
#     for batch_idx, batch in enumerate(tepoch):
#         #print(f"batch index: {batch_idx}")
#         #print(f"batch:{batch}")
#         #print('--------')
        
#         obs_dic = batch['obs']
#         for key,value in obs_dic.items():
#             print(f"key: {key}, shape: {value.shape}")

# for x in train_loader:
#     data, data_id = x
#     print('data: ',data)
#     print('data_id: ',data_id)

# 修改后的数据检查部分
for data in train_loader:
    # 正确访问观测数据
    obs = data['obs']
    action = data['action']
    
    # 提取具体观测值
    robot_eef_pose = obs['robot_eef_pose']
    stage = obs['stage']
    gripper = obs['gripper']
    
    # 打印形状和数值
    print("=== Batch Data Summary ===")
    print(f"robot_eef_pose: shape {robot_eef_pose.shape} | dtype {robot_eef_pose.dtype}")
    print(f"stage: shape {stage.shape} | dtype {stage.dtype}")
    print(f"gripper: shape {gripper.shape} | dtype {gripper.dtype}")
    print(f"action: shape {action.shape} | dtype {action.dtype}")
    
    # 打印第一个样本的数值
    print("\n=== First Sample Values ===")
    print("robot_eef_pose[0]:", robot_eef_pose[0].cpu().numpy())
    print("stage[0]:", stage[0].cpu().numpy())
    print("gripper[0]:", gripper[0].cpu().numpy())
    print("action[0]:", action[0].cpu().numpy())
    # break

# if len(replay_buffer) == 0:
#     print("Replay buffer is empty.")
# else:
#     first_step_data = replay_buffer[0]
#     print("Keys in first time step:")
#     for key in first_step_data:
#         print(f"- Key: `{key}`")
#         print(f"  Shape: {first_step_data[key].shape}")

#   batch_size: 64
#   num_workers: 8
#   shuffle: True
#   pin_memory: True
#   persistent_workers: True
