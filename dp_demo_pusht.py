"""
    Collect demonstration for the Push-T task.
    Usage: python demo_pusht.py -o data/pusht_demo.zarr
    将鼠标悬停在蓝色圆圈附近以开始, 将T块推入绿色区域。
    如果任务成功，剧集将自动终止。
    按“Q”退出
    按“R”重试
    按住“Space”暂停
"""
import numpy as np
import click
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
import pygame

@click.command()
@click.option('-o', '--output', required=True)
@click.option('-rs', '--render_size', default=96, type=int) # 指定渲染大小，默认为 96
@click.option('-hz', '--control_hz', default=10, type=int)  # 指定控制频率，默认为 10 Hz。
def main(output, render_size, control_hz):
    # 创建重放缓冲区，模式为追加模式（'a'）create replay buffer in read-write mode
    replay_buffer = ReplayBuffer.create_from_path(output, mode='a')

    # create PushT env with keypoints
    kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()# 生成关键点管理器参数
    env = PushTKeypointsEnv(render_size=render_size, render_action=False, **kp_kwargs)# 初始化环境
    agent = env.teleop_agent()          # 获取遥操作代理
    clock = pygame.time.Clock()         # 初始化 pygame 的时钟对象，用于控制频率
    
    # episode-level while loop
    while True:
        episode = list()
        # record in seed order, starting with 0
        seed = replay_buffer.n_episodes # 获取当前缓冲区中的 episode 数量作为种子
        print(f'starting seed {seed}')

        env.seed(seed)                  # 设定种子，重置环境并获取初始观测值、信息和图像
        
        # reset env and get observations (including info and render for recording)
        obs = env.reset()
        info = env._get_info()
        img = env.render(mode='human')
        
        # 初始化控制变量 loop state
        retry = False
        pause = False
        done = False
        plan_idx = 0
        pygame.display.set_caption(f'plan_idx:{plan_idx}')
        # 内部循环处理单步操作，直到任务完成或退出step-level while loop
        # 按 SPACE 键暂停或恢复；按 R 键重新开始；按 Q 键退出程序
        while not done:
            # process keypress events
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # hold Space to pause
                        plan_idx += 1
                        pygame.display.set_caption(f'plan_idx:{plan_idx}')
                        pause = True
                    elif event.key == pygame.K_r:
                        # press "R" to retry
                        retry=True
                    elif event.key == pygame.K_q:
                        # press "Q" to exit
                        exit(0)
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_SPACE:
                        pause = False
            # 手动控制 handle control flow
            if retry:
                break
            if pause:
                continue

            # 获取代理动作并执行，如果动作不为空，则记录当前状态、关键点、动作等信息。
            act = agent.act(obs)
            if not act is None:
                # 更新环境状态并渲染图像
                # teleop started
                # state dim 2+3
                state = np.concatenate([info['pos_agent'], info['block_pose']])
                # discard unused information such as visibility mask and agent pos
                # for compatibility
                keypoint = obs.reshape(2,-1)[0].reshape(-1,2)[:9]
                data = {
                    'img': img,
                    'state': np.float32(state),
                    'keypoint': np.float32(keypoint),
                    'action': np.float32(act),
                    'n_contacts': np.float32([info['n_contacts']])
                }
                episode.append(data)
                
            # step env and render
            obs, reward, done, info = env.step(act)
            img = env.render(mode='human')
            
            
            # 规定的控制频率
            clock.tick(control_hz)
        if not retry:
            # 如果不需要重试，则将 episode 数据保存到重放缓冲区 (on disk)中
            data_dict = dict()
            for key in episode[0].keys():
                data_dict[key] = np.stack(
                    [x[key] for x in episode])
            replay_buffer.add_episode(data_dict, compressors='disk')
            print(f'saved seed {seed}')
        else:
            print(f'retry seed {seed}')


if __name__ == "__main__":
    main()
