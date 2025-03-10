# CQU_KS 2024-7-23
# 找到某个物体
import time

import cv2


#1、归零
#2、png=拍照
#3、maskpng，dist=sam（png）
#4、id=大模型理解（prompt，maskpng）
#5、mask=dist【id】
#6、image=convert（png，mask）

from Agent.sence_understanding_agent import *
from Agent.object_part_understanding_agent import *
from Agent.object_prediction_agent import *
from atom_function.basis_atom_function import *
import global_parameters

'''
global_parameters(全局参数):
    global_scene_path: str = "/home/ks/vlm-robot/scene/global_scene"  # 全局场景图像存储路径
    objects_bbox_dist: dict = {}  # 目标检测结果字典（格式：{类别: [[x1,y1,x2,y2], ...], ...}）
    obj_flag: str = "outside"  # 物体位置标记（"outside"：开放环墋；"inside"：容器内）
    target_object_pod: dict = {}  # 目标物体的6D位姿（空间坐标+姿态）
    functions_list: list = []  # 操作列表（用于存储操作步骤）
    explorable_dict: dict = {}  # 可探索对象字典（容器名称 -> 检测框信息）
    objects_key_list: list = []  # 目标检测结果的类别列表
    pre_object_flag: list = []  # 目标预测标志列表
    explorable_objects_name: list = []  # 可探索容器名称列表
    close_drawer_pod: dict = {}  # 关抽屉的6D位姿（空间坐标+姿态）
    offset_x: float = 0  # X轴偏移量
    offset_y: float = 0  # Y轴偏移量
    offset_z: float = 0  # Z轴偏移量
    g2l_r: float = 0  # 全局坐标系到本地坐标系的旋转角度（Roll）
    g2l_p: float = 0  # 全局坐标系到本地坐标系的旋转角度（Pitch）
    g2l_y: float = 0  # 全局坐标系到本地坐标系的旋转角度（Yaw）
    local_scene_path: str = "/home/ks/vlm-robot/scene/local_scene"  # 本地场景图像存储路径
    top_view_scene_path: str = "/home/ks/vlm-robot/scene/top_view_scene"  # 俯视场景图像存储路径
'''

def find_object(robot,target_key="橙子"):#m

    import llm_robot

    # ---------------------------
    # 第一阶段：机器人归零操作
    # ---------------------------
    # 1、归零：控制机器人回到初始零位状态
    # 目的：确保机械臂处于已知的安全初始位置，消除累计误差
    # 方法：调用机器人SDK的back_zero()方法
    robot.back_zero()  # 执行机械臂回零动作

    # ---------------------------
    # 第二阶段：场景图像采集
    # ---------------------------
    # 2、拍摄场景图像
    # 功能：使用机器人搭载的摄像头捕获当前环境图像
    # 参数：global_parameters.global_scene_path - 全局配置的图像存储路径
    # 输出：将拍摄的原始图像保存到指定路径（如：/path/to/scene.png）
    robot.cap_scene(global_parameters.global_scene_path)  # 主拍摄方法
    # robot.cap_pic()  # 备选拍摄方法（当前被注释）

    # ---------------------------
    # 第三阶段：目标检测处理
    # ---------------------------
    # 3、YOLO目标检测处理
    # 步骤说明：
    # （1）构建图像文件完整路径
    # 使用全局路径配置拼接得到完整图像路径（格式：存储路径/scene.png）
    yolo_single_image_path = global_parameters.global_scene_path + "/scene.png"

    # （2）执行单帧目标检测（备选直接读取方式）
    # 如果用OpenCV直接读取可取消下行注释：
    # yolo_single_image = cv2.imread(yolo_single_image_path)

    # （3）调用YOLO检测接口
    # 输入参数：yolo_single_image_path - 待检测图像路径
    # 返回值：
    #   - objects_bbox_dist：检测结果字典（格式：{类别: [[x1,y1,x2,y2], ...], ...}）
    #   - img：叠加了检测框的可视化图像（numpy数组格式）
    global_parameters.objects_bbox_dist, img = robot.yolo_single_picture(yolo_single_image_path)

    # （4）保存检测结果图像
    # 输出路径：在全局路径下生成yolo.png（格式：存储路径/yolo.png）
    # 文件内容：包含边界框、类别标签的可视化结果图
    yolo_single_image_output_path = global_parameters.global_scene_path + "/yolo.png"
    cv2.imwrite(yolo_single_image_output_path, img)  # 使用OpenCV保存图像

    #robot.show_sth("小艺 图片:/home/ks/vlm-robot/scene/global_scene/scene.png")

    # ======================================================
    # 第四阶段：目标物体搜索与容器探索流程
    # 功能：根据目标检测结果进行决策，实现目标定位或容器内探索
    # ======================================================

    # ---------------------------
    # 4.1 目标物体搜索流程
    # ---------------------------
    # 打印当前检测到的物体列表（调试用）
    print(global_parameters.objects_bbox_dist)

    # 检查目标物体是否存在于检测结果中
    if target_key in global_parameters.objects_bbox_dist:
        # 4.1.1 发现目标物体时的处理流程
        robot.back_zero1()  # 执行特定姿态归零（可能为抓取准备位）
        
        # 重新采集场景图像（确保目标可见性）
        robot.cap_scene(global_parameters.global_scene_path)
        yolo_single_image_path = global_parameters.global_scene_path + "/scene.png"
        
        # 二次目标检测（确认目标稳定性）
        global_parameters.objects_bbox_dist, img = robot.yolo_single_picture(yolo_single_image_path)
        
        # 4.1.2 目标消失处理分支
        if target_key not in global_parameters.objects_bbox_dist:
            # 回归初始零位并重新扫描环境
            robot.back_zero()
            robot.cap_scene(global_parameters.global_scene_path)
            yolo_single_image_path = global_parameters.global_scene_path + "/scene.png"
            global_parameters.objects_bbox_dist, img = robot.yolo_single_picture(yolo_single_image_path)
            yolo_single_image_output_path = global_parameters.global_scene_path + "/yolo.png"
            cv2.imwrite(yolo_single_image_output_path, img)
        else:
            # 保存确认后的检测结果
            yolo_single_image_output_path = global_parameters.global_scene_path + "/yolo.png"
            cv2.imwrite(yolo_single_image_output_path, img)

        # 4.1.3 目标定位成功处理
        global_parameters.obj_flag = "outside"  # 标记物体在开放环境
        
        # 获取目标物体的6D位姿（空间坐标+姿态）
        global_parameters.target_object_pod = robot.get_6D_pod(
            global_parameters.objects_bbox_dist[target_key],  # 目标物体的检测框信息
            global_parameters.global_scene_path  # 可能用于深度数据关联
        )
        return  # 结束当前搜索流程

    # ---------------------------
    # 4.2 容器内探索流程
    # ---------------------------
    else:
        global_parameters.obj_flag = "inside"  # 标记物体可能在容器内
        print(global_parameters.objects_bbox_dist)  # 输出当前检测结果
        
        # 4.2.1 调用场景理解Agent寻找可探索容器
        # 输入：当前检测结果 + 目标物体名称
        # 输出：可能包含目标的容器列表（格式：["box","drawer",...]）
        explorable_objects_name = sence_agent(
            global_parameters.objects_bbox_dist,
            target_key
        )
        
        # 4.2.2 无候选容器处理
        if explorable_objects_name[0] == "None":
            explorable_objects_name[0] = None  # 标准化空值表示
            global_parameters.functions_list = []  # 清空操作列表
            return  # 终止流程

        # 4.2.3 存在候选容器的处理
        else:
            print(explorable_objects_name)
            # 构建可探索对象字典（容器名称 -> 检测框信息）
            for i in range(0, len(explorable_objects_name)):
                global_parameters.explorable_dict[explorable_objects_name[i]] = \
                    global_parameters.objects_bbox_dist[explorable_objects_name[i]]
        
        # 4.2.4 候选容器排序优化
        keys_list = list(global_parameters.explorable_dict.keys())
        
        # 获取目标物体在全局列表中的索引
        index_target = global_parameters.objects_key_list.index(target_key)
        
        # 预判标志检查（可能记录历史探索信息）
        if global_parameters.pre_object_flag[index_target] == "true":
            # 生成容器标注图像路径（示例路径需根据实际项目调整）
            draw_contain_path = "/home/ks/vlm-robot/yolov5/result_png/"+str(index_target)+".png"
            
            # 4.2.5 容器视觉标注与预测
            # 在图像上绘制候选容器框并返回ID映射字典
            id_dict = robot.draw_bboxes_on_image(
                draw_contain_path,  # 输出图像路径
                global_parameters.explorable_dict  # 待标注容器信息
            )
            print(id_dict)
            
            # 调用大模型进行目标预测（示例prompt和路径需调整）
            pre_result = object_prediction_agent(
                prompt=" ",
                path="/home/ks/vlm-robot/pic/contain.png"
            )
            
            # 可视化展示中间结果
            robot.show_sth("小艺 图片:/home/ks/vlm-robot/pic/contain.png")
            print(pre_result)
        else:
            pre_result = "None"  # 无预测结果

        # 4.2.6 预测结果处理
        if pre_result != "None":
            pre_result_id = int(pre_result)  # 转换预测结果为整数ID
            pre_result_name = id_dict[pre_result_id]  # 获取对应容器名称
            
            # 调整搜索优先级：将预测容器置顶
            if pre_result_name in keys_list:
                keys_list.remove(pre_result_name)
                keys_list.insert(0, pre_result_name)
        else:
            # 保持原有搜索顺序（无预测结果时不调整）
            pass  # 可添加默认处理逻辑


    # ======================================================
    # 第五阶段：容器遍历操作循环
    # 功能：循环处理候选容器，执行开启操作并验证内部是否存在目标
    # ======================================================

    # 5.1 主循环控制（处理所有候选容器）
    # 循环条件：候选容器列表非空时持续执行
    while len(keys_list) > 0:
        # ---------------------------
        # 步骤5.1：初始化操作
        # ---------------------------
        robot.back_zero()  # 机械臂回归零位（确保运动基准）
        
        # 重置容器关闭位姿数据（安全初始化）
        global_parameters.close_drawer_pod["position"] = [0, 0, 0]

        # ---------------------------
        # 步骤6：容器定位与移动控制
        # ---------------------------
        # 6.1 获取当前候选容器的检测框信息
        bbox = global_parameters.explorable_dict[keys_list[0]]  # 获取列表首个容器的bbox
        
        # 6.2 将检测框转换为空间位姿（6D Pose）
        # 参数说明：
        # - bbox: 边界框坐标 [x1,y1,x2,y2]
        # - global_scene_path: 可能用于关联深度信息
        pod = robot.bbox2pod(bbox, global_parameters.global_scene_path)
        
        # 6.3 转换为基坐标系下的运动参数
        base_pando = robot.target2basis(pod["position"], [0,0,0,1])  # 四元数表示姿态
        
        # 6.4 应用校准偏移量（补偿机械臂安装误差）
        base_pando[0] += global_parameters.offset_x  # X轴偏移
        base_pando[1] += global_parameters.offset_y  # Y轴偏移
        base_pando[2] += global_parameters.offset_z  # Z轴偏移
        
        # 6.5 设置末端执行器姿态（全局到局部坐标系转换参数）
        base_pando[3:] = [
            global_parameters.g2l_r,  # 横滚角
            global_parameters.g2l_p,  # 俯仰角
            global_parameters.g2l_y   # 偏航角
        ]
        
        # 6.6 执行机械臂移动操作
        robot.move_to_base_pod(base_pando)  # 移动到目标位姿

        # ---------------------------
        # 步骤7-8：局部场景感知
        # ---------------------------
        # 7.1 拍摄局部场景图像
        robot.cap_scene(global_parameters.local_scene_path)  # 保存到本地场景路径
        
        # 8.1 执行SOM（Scene Object Matching）处理
        som_image_path = global_parameters.local_scene_path + "/scene.png"
        som_image, som_bbox = robot.som(som_image_path)  # 返回处理图像和部件级检测框
        
        # 8.2 保存并展示SOM结果
        SOM_path = global_parameters.local_scene_path + "/som.png"
        cv2.imwrite(SOM_path, som_image)  # 保存可视化结果
        robot.show_sth("小艺 图片:/home/ks/vlm-robot/scene/local_scene/som.png")  # 界面展示

        # ---------------------------
        # 步骤9：部件语义理解
        # ---------------------------
        # 9.1 调用部件理解Agent（如把手识别）
        # 输入参数：
        # - keys_list[0]: 当前容器名称（如"drawer"）
        # - SOM_path: 部件检测结果图像路径
        part_id = object_part_understanding_agent(keys_list[0], SOM_path)
        
        # 9.2 部件识别失败处理
        if part_id == "None":
            del keys_list[0]  # 移出当前候选
            continue  # 跳过后续步骤进入下一循环
        
        # 9.3 调整检测框尺寸（适配机械臂操作精度）
        part_id = int(part_id)
        six_D_bbox = list(som_bbox[part_id-1])  # 索引转换（ID从1开始）
        # 尺寸补偿计算（示例参数需根据相机-机械臂标定调整）
        six_D_bbox[0] = int(six_D_bbox[0] / 1.25 / 1.325)  # 宽度补偿
        six_D_bbox[1] = int(six_D_bbox[1] / 1.25 / 1.325)  # 高度补偿
        six_D_bbox[2] = int(six_D_bbox[2] / 1.25 / 1.325)  # 深度补偿
        six_D_bbox[3] = int(six_D_bbox[3] / 1.25 / 1.325)  # 角度补偿

        # ---------------------------
        # 步骤10-11：容器开启操作
        # ---------------------------
        # 10.1 计算操作位姿
        pod = robot.bbox2pod(six_D_bbox, global_parameters.local_scene_path)
        
        # 11.1 设置末端执行器姿态（水平抓取方向）
        pod["orientation"] = [0, 0, 0, 1]  # 单位四元数表示
        
        # 11.2 执行开启操作（如拉开抽屉）
        robot.open_draw(pod)  # 传入位姿参数

        # ---------------------------
        # 步骤12-13：容器内部检测
        # ---------------------------
        # 12.1 记录关闭位姿（为后续复位准备）
        global_parameters.close_drawer_pod = robot.cap_draw()  # 获取当前容器状态
        
        # 12.2 拍摄顶部视角（用于内部目标检测）
        robot.cap_scene(global_parameters.top_view_scene_path)
        
        # 13.1 执行顶部视角目标检测
        yolo_single_image_path = global_parameters.top_view_scene_path + "/scene.png"
        # 使用专用权重进行检测（示例路径需替换为实际模型）
        top_view_bbox_dist, image = robot.yolo_single_picture(
            yolo_single_image_path,
            weights='/home/ks/vlm-robot/yolov5/yolov5-master/501/last.pt'
        )
        # 保存并展示结果
        yolo_single_image_output_path = global_parameters.top_view_scene_path + "/yolo.png"
        cv2.imwrite(yolo_single_image_output_path, image)
        robot.show_sth("小艺 图片:/home/ks/vlm-robot/scene/top_view_scene/scene.png")

        # ---------------------------
        # 步骤14：结果验证与处理
        # ---------------------------
        print(top_view_bbox_dist)
        if target_key in top_view_bbox_dist:
            # 13.2 目标找到处理
            global_parameters.target_object_pod = robot.get_6D_pod(
                top_view_bbox_dist[target_key],
                global_parameters.top_view_scene_path
            )
            break  # 退出循环
        else:
            # 14.1 目标未找到处理
            robot.close_draw(global_parameters.close_drawer_pod, 0)  # 关闭容器（0表示默认速度）
            # 复位容器记录
            global_parameters.close_drawer_pod = {
                "position": [0, 0, 0],
                "orientation": [0, 0, 0, 0],
                "d": 0  # 可能表示抽屉开合程度
            }
            del keys_list[0]  # 移除当前候选

    return  # 结束整个流程




if __name__ == '__main__':
    pass