import cv2  # 导入OpenCV库
import numpy as np  # 导入numpy库

def render_env_video(env, states, actions=None):
    """
    渲染环境视频。
    @param env 环境对象
    @param states 状态列表
    @param actions 动作列表（可选）
    @return 包含渲染图像的numpy数组
    """
    observations = states  # 将状态列表赋值给observations
    imgs = list()  # 创建一个空列表用于存储图像

    for i in range(len(observations)):
        state = observations[i]  # 获取当前状态
        env.set_state(state)  # 设置环境状态
        if i == 0:
            env.set_state(state)  # 如果是第一个状态，重复设置

        img = env.render()  # 渲染环境并获取图像

        # 绘制动作
        if actions is not None:
            action = actions[i]  # 获取当前动作
            coord = (action / 512 * 96).astype(np.int32)  # 计算动作坐标
            cv2.drawMarker(img, coord, 
                color=(255, 0, 0), markerType=cv2.MARKER_CROSS,
                markerSize=8, thickness=1)  # 在图像上绘制标记

        imgs.append(img)  # 将图像添加到列表中

    imgs = np.array(imgs)  # 将图像列表转换为numpy数组
    return imgs  # 返回包含渲染图像的numpy数组
