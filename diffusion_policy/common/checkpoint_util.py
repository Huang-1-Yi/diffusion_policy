from typing import Optional, Dict  # 导入类型提示模块
import os  # 导入操作系统模块

class TopKCheckpointManager:
    def __init__(self,
            save_dir,  # 保存目录
            monitor_key: str,  # 监控键
            mode='min',  # 模式，默认为'min'
            k=1,  # 保留的检查点数量
            format_str='epoch={epoch:03d}-train_loss={train_loss:.3f}.ckpt'  # 文件名格式
        ):
        assert mode in ['max', 'min']  # 确保模式为'max'或'min'
        assert k >= 0  # 确保k值非负

        self.save_dir = save_dir  # 保存目录
        self.monitor_key = monitor_key  # 监控键
        self.mode = mode  # 模式
        self.k = k  # 保留的检查点数量
        self.format_str = format_str  # 文件名格式
        self.path_value_map = dict()  # 路径-值映射字典
    
    def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
        if self.k == 0:
            return None  # 如果k为0，不保存检查点

        value = data[self.monitor_key]  # 获取监控键对应的值
        ckpt_path = os.path.join(
            self.save_dir, self.format_str.format(**data))  # 生成检查点路径
        
        if len(self.path_value_map) < self.k:
            # 如果未达到容量
            self.path_value_map[ckpt_path] = value  # 添加到映射字典
            return ckpt_path  # 返回检查点路径
        
        # 如果已达到容量
        sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])  # 按值排序映射字典
        min_path, min_value = sorted_map[0]  # 最小值路径和最小值
        max_path, max_value = sorted_map[-1]  # 最大值路径和最大值

        delete_path = None
        if self.mode == 'max':
            if value > min_value:
                delete_path = min_path  # 如果新值大于最小值，删除最小值路径
        else:
            if value < max_value:
                delete_path = max_path  # 如果新值小于最大值，删除最大值路径

        if delete_path is None:
            return None  # 如果不需要删除，返回None
        else:
            del self.path_value_map[delete_path]  # 从映射字典中删除路径
            self.path_value_map[ckpt_path] = value  # 添加新路径和值

            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)  # 如果保存目录不存在，创建目录

            if os.path.exists(delete_path):
                os.remove(delete_path)  # 如果删除路径存在，删除文件
            return ckpt_path  # 返回新检查点路径
