from typing import Optional
import pathlib
import numpy as np
import time
import shutil
import math
from multiprocessing.managers import SharedMemoryManager
# from diffusion_policy.real_world.rtde_interpolation_controller import RTDEInterpolationController
from diffusion_policy.real_world.franka_interpolation_controller import FrankaInterpolationController
from diffusion_policy.real_world.multi_realsense import MultiRealsense, SingleRealsense
from diffusion_policy.real_world.video_recorder import VideoRecorder
from diffusion_policy.common.timestamp_accumulator import (
    TimestampObsAccumulator,
    TimestampActionAccumulator,
    align_timestamps
)
from diffusion_policy.real_world.multi_camera_visualizer import MultiCameraVisualizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform, optimal_row_cols)

DEFAULT_OBS_KEY_MAP = {    #默认观测键映射     建立观测键和键映射的关系
    # robot
    'ActualTCPPose': 'robot_eef_pose',    #实际TCP姿势 机器人末端执行器姿势 
    'ActualQ': 'robot_joint',             #实际Q 机器人关节
    'ActualQd': 'robot_joint_vel',        #实际Qd 机器人关节速度
    # timestamps                      
    'step_idx': 'step_idx',               #步骤索引    
    'timestamp': 'timestamp'              #时间戳
}

class RealEnvFranka:
    def __init__(self,
            # required params
            output_dir,
            robot_ip,

            verbose=False,

            # env params
            frequency=10,
            n_obs_steps=2,       #观测步数
            # obs
            obs_image_resolution=(640, 480),    #观测图像分辨率
            max_obs_buffer_size=30,              #最大观测缓冲区大小
            camera_serial_numbers=None,           #相机序列号 
            obs_key_map=DEFAULT_OBS_KEY_MAP,      #观测键映射
            obs_float32=False,                    #观测浮点数
            # this latency compensates receive_timestamp
            # all in seconds
            robot_obs_latency=0.0001,            #机器人观测延迟
            # action
            max_pos_speed=0.25,                   #最大位置速度
            max_rot_speed=0.6,                     #最大旋转速度
            # robot
            tcp_offset=0.13,                       #TCP偏移
            init_joints=False,                    #初始化关节
            # video capture params
            video_capture_fps=30,                  #视频捕获帧率
            video_capture_resolution=(1280, 720),    #视频捕获分辨率
            # saving params
            record_raw_video=True,        #记录原始视频
            thread_per_video=2,           #每个视频的线程数
            video_crf=21,                  #视频质量
            # vis params
            enable_multi_cam_vis=True,      #启用多相机可视化
            multi_cam_vis_resolution=(1280, 720),      #多相机可视化分辨率
            # shared memory
            shm_manager=None          #共享内存管理器
            ):
        assert frequency <= video_capture_fps
        output_dir = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir()
        video_dir = output_dir.joinpath('videos')
        video_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())
        replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=zarr_path, mode='a')

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
        if camera_serial_numbers is None:
            camera_serial_numbers = SingleRealsense.get_connected_devices_serial()

        color_tf = get_image_transform(
            input_res=video_capture_resolution,
            output_res=obs_image_resolution,
            # obs output rgb
            bgr_to_rgb=True)
        color_transform = color_tf
        if obs_float32:
            color_transform = lambda x: color_tf(x).astype(np.float32) / 255

        def transform(data):
            data['color'] = color_transform(data['color'])
            return data

        rw, rh, col, row = optimal_row_cols(
            n_cameras=len(camera_serial_numbers),
            in_wh_ratio=obs_image_resolution[0] / obs_image_resolution[1],
            max_resolution=multi_cam_vis_resolution
        )
        vis_color_transform = get_image_transform(
            input_res=video_capture_resolution,
            output_res=(rw, rh),
            bgr_to_rgb=False
        )

        def vis_transform(data):
            data['color'] = vis_color_transform(data['color'])
            return data

        recording_transfrom = None
        recording_fps = video_capture_fps
        recording_pix_fmt = 'bgr24'
        if not record_raw_video:
            recording_transfrom = transform
            recording_fps = frequency
            recording_pix_fmt = 'rgb24'

        video_recorder = VideoRecorder.create_h264(
            fps=recording_fps,
            codec='h264',
            input_pix_fmt=recording_pix_fmt,
            crf=video_crf,
            thread_type='FRAME',
            thread_count=thread_per_video)

        realsense = MultiRealsense(
            serial_numbers=camera_serial_numbers,
            shm_manager=shm_manager,
            resolution=video_capture_resolution,
            capture_fps=video_capture_fps,
            put_fps=video_capture_fps,
            # send every frame immediately after arrival
            # ignores put_fps
            put_downsample=False,
            record_fps=recording_fps,
            enable_color=True,

            enable_depth=True,                      # 3D保存
            enable_infrared=False,
            get_max_k=max_obs_buffer_size,
            transform=transform,
            vis_transform=vis_transform,
            recording_transform=recording_transfrom,
            video_recorder=video_recorder,
            verbose=False
        )

        multi_cam_vis = None
        if enable_multi_cam_vis:
            multi_cam_vis = MultiCameraVisualizer(
                realsense=realsense,
                row=row,
                col=col,
                rgb_to_bgr=False
            )

        # cube_diag = np.linalg.norm([1, 1, 1])
        j_init = np.array([-0.165, -0.059, 0.167, -1.693, 0.002, 1.642, 0.751])
        if not init_joints:
            j_init = None

        robot = FrankaInterpolationController(
            shm_manager=shm_manager,
            robot_ip=robot_ip,
            frequency=200,
            Kx_scale=1.0,
            Kxd_scale=np.array([2.0, 1.5, 2.0, 1.0, 1.0, 1.0]),
            joints_init=j_init,
            joints_init_duration=3.0,
            verbose=False,
            receive_latency=robot_obs_latency
        )

        self.realsense = realsense
        self.robot = robot
        
        #self.gripper = gripper
        self.verbose = verbose
        self.multi_cam_vis = multi_cam_vis
        self.video_capture_fps = video_capture_fps
        self.frequency = frequency
        self.n_obs_steps = n_obs_steps
        self.max_obs_buffer_size = max_obs_buffer_size
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.obs_key_map = obs_key_map
        # recording
        self.output_dir = output_dir
        self.video_dir = video_dir
        self.replay_buffer = replay_buffer
        # temp memory buffers
        self.last_realsense_data = None
        # recording buffers
        self.obs_accumulator = None
        self.action_accumulator = None
        self.stage_accumulator = None

        self.start_time = None

    # def gripper_open(self):
    #     self.robot.open_gripper()
    
    # def gripper_close(self):
    #     self.robot.close_gripper()

    # ======== start-stop API =============
    @property
    def is_ready(self):
        return self.realsense.is_ready and self.robot.is_ready

    def start(self, wait=True):
        self.realsense.start(wait=False)
        self.robot.start(wait=False)
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start(wait=False)
        if wait:
            self.start_wait()
    

    def stop(self, wait=True):
        self.end_episode()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop(wait=False)
        self.robot.stop(wait=False)
        self.realsense.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.realsense.start_wait()
        self.robot.start_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start_wait()

    def stop_wait(self):
        self.robot.stop_wait()
        self.realsense.stop_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop_wait()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= async env API ===========
    def get_obs(self) -> dict:
        "observation dict"
        assert self.is_ready

        # get data
        # 30 Hz, camera_receive_timestamp
        k = math.ceil(self.n_obs_steps * (self.video_capture_fps / self.frequency))
        self.last_realsense_data = self.realsense.get(
            k=k,
            out=self.last_realsense_data)

        # 125 hz, robot_receive_timestamp
        last_robot_data = self.robot.get_all_state()
        # both have more than n_obs_steps data

        # align camera obs timestamps
        dt = 1 / self.frequency
        last_timestamp = np.max([x['timestamp'][-1] for x in self.last_realsense_data.values()])
        obs_align_timestamps = last_timestamp - (np.arange(self.n_obs_steps)[::-1] * dt)

        camera_obs = dict()
        for camera_idx, value in self.last_realsense_data.items():
            
            # 打印self.last_realsense_data的内容
            # print(f"Camera {camera_idx}:")
            # for key, val in value.items():
            #     print(f"  {key}: {type(val)} with shape {np.shape(val)}")

            this_timestamps = value['timestamp']    # 获取当前相机的时间戳
            this_idxs = list()                      # 初始化索引列表
            for t in obs_align_timestamps:          # 遍历对齐的时间戳
                is_before_idxs = np.nonzero(this_timestamps < t)[0] # 找到所有小于当前时间戳的索引
                this_idx = 0                        # 初始化当前索引为0
                if len(is_before_idxs) > 0:         # 如果存在小于当前时间戳的索引
                    this_idx = is_before_idxs[-1]   # 取最后一个小于当前时间戳的索引
                this_idxs.append(this_idx)          # 将索引添加到索引列表中
            # remap key
            camera_obs[f'camera_{camera_idx}'] = value['color'][this_idxs]          # 将颜色数据映射到相机观察字典中
            camera_obs[f'camera_{camera_idx}_depth'] = value['depth'][this_idxs]    # 存储深度数据

        # align robot obs
        robot_timestamps = last_robot_data['robot_receive_timestamp']
        this_timestamps = robot_timestamps
        this_idxs = list()
        for t in obs_align_timestamps:
            is_before_idxs = np.nonzero(this_timestamps < t)[0]
            this_idx = 0
            if len(is_before_idxs) > 0:
                this_idx = is_before_idxs[-1]
            this_idxs.append(this_idx)

        robot_obs_raw = dict()
        for k, v in last_robot_data.items():
            if k in self.obs_key_map:
                robot_obs_raw[self.obs_key_map[k]] = v

        robot_obs = dict()
        for k, v in robot_obs_raw.items():
            robot_obs[k] = v[this_idxs]

        # accumulate obs
        if self.obs_accumulator is not None:
            self.obs_accumulator.put(
                robot_obs_raw,
                robot_timestamps
            )

        # return obs
        obs_data = dict(camera_obs)                     # 初始化观察数据为相机观察数据
        obs_data.update(robot_obs)                      # 更新观察数据为机器人观察数据
        obs_data['timestamp'] = obs_align_timestamps    # 设置观察数据的时间戳
        return obs_data                                 # 返回观察数据
    

    # 添加的内容gripper
    def exec_gripper_actions(self, width=None, force=10.0, speed=0.1):

        # # 调用 schedule_gripper_command 发送命令
        self.robot.schedule_gripper_command(width=width, force=force, speed=speed)

        # 打印信息（可选）
        if self.verbose:
            print(f"[exec_gripper_actions] Scheduled gripper width={width}, force={force}, speed={speed}")
        
    

    def exec_actions(self,
                     actions: np.ndarray,
                     timestamps: np.ndarray,
                     stages: Optional[np.ndarray] = None):
        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)
        if stages is None:
            stages = np.zeros_like(timestamps, dtype=np.int64)
        elif not isinstance(stages, np.ndarray):
            stages = np.array(stages, dtype=np.int64)
        
        # convert action to pose
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        print("actions[is_new]=",actions[is_new])

        new_timestamps = timestamps[is_new]
        new_stages = stages[is_new]

        # schedule waypoints
        for i in range(len(new_actions)):
            self.robot.schedule_waypoint(
                pose=new_actions[i],
                target_time=new_timestamps[i]
            )

        # record actions
        if self.action_accumulator is not None:
            self.action_accumulator.put(
                new_actions,
                new_timestamps
            )
        if self.stage_accumulator is not None:
            self.stage_accumulator.put(
                new_stages,
                new_timestamps
            )

    def get_robot_state(self):
        return self.robot.get_state()

    # recording API
    def start_episode(self, start_time=None):
        "Start recording and return first obs"
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        assert self.is_ready

        # prepare recording stuff
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        this_video_dir.mkdir(parents=True, exist_ok=True)
        n_cameras = self.realsense.n_cameras
        video_paths = list()
        for i in range(n_cameras):
            video_paths.append(
                str(this_video_dir.joinpath(f'{i}.mp4').absolute()))

        # start recording on realsense
        self.realsense.restart_put(start_time=start_time)
        self.realsense.start_recording(video_path=video_paths, start_time=start_time)

        # create accumulators
        self.obs_accumulator = TimestampObsAccumulator(
            start_time=start_time,
            dt=1 / self.frequency
        )
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1 / self.frequency
        )
        self.stage_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1 / self.frequency
        )
        print(f'Episode {episode_id} started!')

    def end_episode(self):
        "Stop recording"
        assert self.is_ready

        # stop video recorder
        self.realsense.stop_recording()

        if self.obs_accumulator is not None:
            # recording
            assert self.action_accumulator is not None
            assert self.stage_accumulator is not None

            # Since the only way to accumulate obs and action is by calling
            # get_obs and exec_actions, which will be in the same thread.
            # We don't need to worry new data come in here.
            obs_data = self.obs_accumulator.data
            obs_timestamps = self.obs_accumulator.timestamps

            actions = self.action_accumulator.actions
            action_timestamps = self.action_accumulator.timestamps
            stages = self.stage_accumulator.actions
            n_steps = min(len(obs_timestamps), len(action_timestamps))
            if n_steps > 0:
                episode = dict()
                episode['timestamp'] = obs_timestamps[:n_steps]
                episode['action'] = actions[:n_steps]
                episode['stage'] = stages[:n_steps]
                for key, value in obs_data.items():
                    episode[key] = value[:n_steps]
                self.replay_buffer.add_episode(episode, compressors='disk')
                episode_id = self.replay_buffer.n_episodes - 1
                print(f'Episode {episode_id} saved!')

            self.obs_accumulator = None
            self.action_accumulator = None
            self.stage_accumulator = None

    def drop_episode(self):
        self.end_episode()
        self.replay_buffer.drop_episode()
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        if this_video_dir.exists():
            shutil.rmtree(str(this_video_dir))
        print(f'Episode {episode_id} dropped!')
