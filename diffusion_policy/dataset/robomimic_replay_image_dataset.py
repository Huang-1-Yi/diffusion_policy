from typing import Dict, List
import torch
import numpy as np
import h5py
from tqdm import tqdm
import zarr
import os
import shutil
import copy
import json
import hashlib
from filelock import FileLock
from threadpoolctl import threadpool_limits
import concurrent.futures
import multiprocessing
from omegaconf import OmegaConf
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset, LinearNormalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)
register_codecs()

# 考虑将n_demo＜len(demos)时的情况，修改RobomimicReplayImageDataset类，使其支持加载部分数据集。
# demos = file['data']
# len(demos)

# 加载 robomimic 数据集，支持归一化、缓存、数据采样 
class RobomimicReplayImageDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,   # 数据的形状元信息，包括 obs（观测）和 action（动作）
            dataset_path: str,
            horizon=1,          # 采样的时间跨度（影响序列长度）
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,   # 指定观测步数（若为 None，则使用全部）
            abs_action=False,   # 是否使用绝对动作坐标系（默认 False）
            rotation_rep='rotation_6d', # ignored when abs_action=False 旋转表示方式（rotation_6d）
            use_legacy_normalizer=False,
            use_cache=False,    # 是否使用 zarr 格式的缓存，加速数据加载
            seed=42,
            val_ratio=0.0,
            n_demo=100          # 修改 添加的参数，额外的  n_demo=len(demos) 个数
        ):
        self.n_demo = n_demo    # 修改
        # robomimic 采用 axis_angle 旋转表示，该代码将其转换为 rotation_6d 格式，便于模型训练
        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep=rotation_rep)

        replay_buffer = None
        # 如果启用 use_cache=True，那么数据加载后会转换成 zarr 格式，并存入缓存文件 dataset.zarr.zip，加快后续训练
        if use_cache:
            cache_zarr_path = dataset_path + f'.{n_demo}.' + '.zarr.zip'        # 修改，重命名包括数量名
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):         # FileLock 确保多个进程不会同时访问缓存，防止数据损坏
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print('Cache does not exist. Creating!')
                        # store = zarr.DirectoryStore(cache_zarr_path)
                        replay_buffer = _convert_robomimic_to_replay(
                            store=zarr.MemoryStore(), 
                            shape_meta=shape_meta, 
                            dataset_path=dataset_path, 
                            abs_action=abs_action, 
                            rotation_transformer=rotation_transformer,
                            n_demo=n_demo)       # 修改
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
            replay_buffer = _convert_robomimic_to_replay(
                store=zarr.MemoryStore(), 
                shape_meta=shape_meta, 
                dataset_path=dataset_path, 
                abs_action=abs_action, 
                rotation_transformer=rotation_transformer,
                n_demo=n_demo)       # 修改
        
        # 遍历 shape_meta，把RGB图像数据 和 低维传感器数据 分类
        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        
        # for key in rgb_keys:
        #     replay_buffer[key].compressor.numthreads=1
        # 训练/验证数据拆分
        key_first_k = dict()# 存储每个观测数据（如 RGB 图像和低维数据）的处理方式
        if n_obs_steps is not None:# 检查观测步数的数量。如果 n_obs_steps 被指定了，那么只取前 n_obs_steps 个观测数据
            # only take first k obs from images
            # 选择图像数据中的前 k 个观测数据，k 由 n_obs_steps 确定
            # 遍历拼接后的所有键（包含 RGB 图像和低维数据的键），并将每个键的值设置为 n_obs_steps
            for key in rgb_keys + lowdim_keys:# 假设 rgb_keys 和 lowdim_keys 分别是包含 RGB 图像和低维数据键名（如动作或其他传感器数据）的列表（或其他可迭代对象）。+ 操作符将这两个列表拼接在一起
                key_first_k[key] = n_obs_steps
            """
            如果 rgb_keys = ['image_1', 'image_2'] 和 lowdim_keys = ['action', 'velocity']，
            且 n_obs_steps = 3，那么最终 key_first_k 会是：
            {
            'image_1': 3,
            'image_2': 3,
            'action': 3,
            'velocity': 3
            }
            """
        
        # 随机划分训练和验证数据
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        # 负责采样固定时间跨度的轨迹数据
        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer

    # 全部作为测试数据？
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set
    
    # 实现数据归一化
    # 计算动作的统计信息，并归一化到 [0, 1] 或 [-1, 1]，防止数据范围过大影响训练
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        # LinearNormalizer 线性标准化器，用于对数据进行线性变换，将数据标准化到某个指定范围（通常是 [-1, 1] 或 [0, 1]）
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer['action'])
        """
        从 replay_buffer 中提取动作数据（self.replay_buffer['action']）并计算其统计量，存储在 stat 中。
        array_to_stats 是一个函数，通常会计算数据的均值、标准差、最大值、最小值等统计信息。
        """
        if self.abs_action:
            """
            如果使用了 绝对动作（abs_action）：
                如果动作的均值 stat['mean'] 的最后一个维度大于 10，说明这是一个双臂任务（dual_arm），使用专门为双臂设计的标准化器 robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)。
                否则，使用单臂的标准化器 robomimic_abs_action_only_normalizer_from_stat(stat)。
                如果启用了 传统标准化器 (use_legacy_normalizer) ，则会使用 normalizer_from_stat(stat) 进一步调整。
            """
            if stat['mean'].shape[-1] > 10:
                # dual arm
                this_normalizer = robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
            else:
                this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
            
            if self.use_legacy_normalizer:
                this_normalizer = normalizer_from_stat(stat)
        else:
            # already normalized
            # 如果不使用绝对动作（abs_action 为 False），那么动作已经被归一化，
            # 使用 get_identity_normalizer_from_stat(stat)，这意味着数据将不会进行进一步的标准化。
            this_normalizer = get_identity_normalizer_from_stat(stat)
        # 将处理好的动作标准化器 this_normalizer 存储到 normalizer 字典中的 action 键下
        normalizer['action'] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            # 对 lowdim_keys 中的每个键（这些键对应低维数据，如位置信息、四元数等），提取其统计信息并进行标准化
            stat = array_to_stats(self.replay_buffer[key])
            """
            根据键的结尾（例如，'pos'、'quat'、'qpos' 等）来确定标准化方式：
            如果是位置（'pos'）或关节位置（'qpos'），使用 get_range_normalizer_from_stat(stat) 进行归一化，通常将数据归一化到 [-1, 1] 范围。
            如果是四元数（'quat'），由于四元数通常已经处于 [-1, 1] 的范围，因此使用 get_identity_normalizer_from_stat(stat) 保持原始范围。
            如果没有符合这些条件的键，抛出错误（RuntimeError）。
            """
            if key.endswith('pos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer

        # image
        """
        对于图像数据（通过 rgb_keys 指定的键），使用 get_image_range_normalizer() 进行标准化。
        通常，图像数据需要根据某些预定义规则（如将像素值归一化到 [0, 1] 或 [-1, 1] 范围）进行处理
        """
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        读取 idx 对应的轨迹序列数据
        归一化 RGB 图像 (uint8 → float32) 并转换为 PyTorch Tensor
        返回格式：
        {
            'obs': {'rgb': torch.Tensor, 'lowdim': torch.Tensor},
            'action': torch.Tensor
        }

        """
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
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }
        return torch_data

# _convert_actions 函数将原始动作转换为绝对动作
# 调整动作维度，如果是双臂机器人，则拆分 14维 数据,为 2个 7维 数据
def _convert_actions(raw_actions, abs_action, rotation_transformer):
    actions = raw_actions
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
            raw_actions = raw_actions.reshape(-1,2,7)
            is_dual_arm = True

        pos = raw_actions[...,:3]
        rot = raw_actions[...,3:6]
        gripper = raw_actions[...,6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([
            pos, rot, gripper
        ], axis=-1).astype(np.float32)
    
        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1,20)
        actions = raw_actions
    return actions

# 转换 hdf5 数据为 zarr 格式，提高读取速度
def _convert_robomimic_to_replay(store, shape_meta, dataset_path, abs_action, rotation_transformer, 
        n_workers=None, max_inflight_tasks=None, n_demo=100):       # 修改
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        shape = attr['shape']
        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            rgb_keys.append(key)
        elif type == 'low_dim':
            lowdim_keys.append(key)
    
    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)

    with h5py.File(dataset_path) as file:
        # count total steps
        demos = file['data']
        episode_ends = list()
        prev_end = 0
        for i in range(len(demos)):       # 修改
            demo = demos[f'demo_{i}']
            episode_length = demo['actions'].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
        n_steps = episode_ends[-1]
        episode_starts = [0] + episode_ends[:-1]
        _ = meta_group.array('episode_ends', episode_ends, 
            dtype=np.int64, compressor=None, overwrite=True)

        # save lowdim data
        for key in tqdm(lowdim_keys + ['action'], desc="Loading lowdim data"):
            data_key = 'obs/' + key
            if key == 'action':
                data_key = 'actions'
            this_data = list()
            for i in range(len(demos)):       # 修改
                demo = demos[f'demo_{i}']
                this_data.append(demo[data_key][:].astype(np.float32))
            this_data = np.concatenate(this_data, axis=0)
            if key == 'action':
                this_data = _convert_actions(
                    raw_actions=this_data,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer
                )
                assert this_data.shape == (n_steps,) + tuple(shape_meta['action']['shape'])
            else:
                assert this_data.shape == (n_steps,) + tuple(shape_meta['obs'][key]['shape'])
            _ = data_group.array(
                name=key,
                data=this_data,
                shape=this_data.shape,
                chunks=this_data.shape,
                compressor=None,
                dtype=this_data.dtype
            )
        
        def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
            try:
                zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                # make sure we can successfully decode
                _ = zarr_arr[zarr_idx]
                return True
            except Exception as e:
                return False
        
        with tqdm(total=n_steps*len(rgb_keys), desc="Loading image data", mininterval=1.0) as pbar:
            # one chunk per thread, therefore no synchronization needed
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = set()
                for key in rgb_keys:
                    data_key = 'obs/' + key
                    shape = tuple(shape_meta['obs'][key]['shape'])
                    c,h,w = shape
                    this_compressor = Jpeg2k(level=50)
                    img_arr = data_group.require_dataset(
                        name=key,
                        shape=(n_steps,h,w,c),
                        chunks=(1,h,w,c),
                        compressor=this_compressor,
                        dtype=np.uint8
                    )
                    for episode_idx in range(len(demos)):       # 修改
                        demo = demos[f'demo_{episode_idx}']
                        hdf5_arr = demo['obs'][key]
                        for hdf5_idx in range(hdf5_arr.shape[0]):
                            if len(futures) >= max_inflight_tasks:
                                # limit number of inflight tasks
                                completed, futures = concurrent.futures.wait(futures, 
                                    return_when=concurrent.futures.FIRST_COMPLETED)
                                for f in completed:
                                    if not f.result():
                                        raise RuntimeError('Failed to encode image!')
                                pbar.update(len(completed))

                            zarr_idx = episode_starts[episode_idx] + hdf5_idx
                            futures.add(
                                executor.submit(img_copy, 
                                    img_arr, zarr_idx, hdf5_arr, hdf5_idx))
                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError('Failed to encode image!')
                pbar.update(len(completed))

    replay_buffer = ReplayBuffer(root)
    return replay_buffer

# 从统计数据中获取归一化器
def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )
