from typing import List, Optional
import pathlib
import pandas as pd
import numpy as np
import numba
import click
import time
import collections
import json
import wandb
import yaml
import numbers
import scipy.ndimage as sn
from diffusion_policy.common.json_logger import read_json_log, JsonLogger
import logging

@numba.jit(nopython=True)
# 计算给定数组在指定索引处的窗口平均值 
def get_indexed_window_average(
        arr: np.ndarray, idxs: np.ndarray, window_size: int): # 定义一个函数，用于计算索引窗口内的平均值
    result = np.zeros(idxs.shape, dtype=arr.dtype) # 初始化结果数组，与索引数组形状相同
    length = arr.shape[0]                       # 获取输入数组的长度
    for i in range(len(idxs)):                  # 遍历每个索引
        idx = idxs[i]                           # 获取当前索引
        start = max(idx - window_size, 0)       # 计算窗口起始位置
        end = min(start + window_size, length)  # 计算窗口结束位置
        result[i] = np.mean(arr[start:end])     # 计算窗口内的平均值
    return result                               # 返回结果数组

# 计算日志数据框中指定关键字的多种指标
def compute_metrics(log_df: pd.DataFrame, key: str, 
        end_step: Optional[int]=None,
        k_min_loss: int=10,
        k_around_max: int=10,
        max_k_window: int=10,
        replace_slash: int=True):
    if key not in log_df:   # 如果关键字不在数据框中
        return dict()       # 返回空字典

    # 准备数据
    if end_step is not None:            # 如果指定了结束步骤
        log_df = log_df.iloc[:end_step] # 截取到指定步骤的数据
    is_key = ~pd.isnull(log_df[key])    # 检查关键字列是否为空
    is_key_idxs = is_key.index[is_key].to_numpy() # 获取非空值的索引
    if len(is_key_idxs) == 0:           # 如果没有非空值
        return dict()                   # 返回空字典

    key_data = log_df[key][is_key].to_numpy()               # 获取关键字列的非空值
    # 在将验证集添加到工作区后
    # 每个 epoch 的最后一步会进行 rollout
    # 其中报告的 train_loss 和 val_loss
    # 已经是该 epoch 的平均值
    train_loss = log_df['train_loss'][is_key].to_numpy()    # 获取训练损失的非空值
    val_loss = log_df['val_loss'][is_key].to_numpy()        # 获取验证损失的非空值

    result = dict() # 初始化结果字典

    log_key = key
    if replace_slash:
        log_key = key.replace('/', '_') # 替换关键字中的斜杠
    # 最大值
    max_value = np.max(key_data) # 计算最大值
    result['max/'+log_key] = max_value

    # k_around_max
    max_idx = np.argmax(key_data)                           # 获取最大值的索引
    end = min(max_idx + k_around_max // 2, len(key_data))   # 计算窗口结束位置
    start = max(end - k_around_max, 0)                      # 计算窗口起始位置
    k_around_max_value = np.mean(key_data[start:end])       # 计算窗口内的平均值
    result['k_around_max/'+log_key] = k_around_max_value

    # max_k_window
    k_window_value = sn.uniform_filter1d(key_data, size=max_k_window, axis=0, mode='nearest') # 计算滑动窗口平均值
    max_k_window_value = np.max(k_window_value)             # 计算滑动窗口的最大值
    result['max_k_window/'+log_key] = max_k_window_value

    # min_train_loss
    min_idx = np.argmin(train_loss)             # 获取最小训练损失的索引
    min_train_loss_value = key_data[min_idx]    # 获取对应的关键字值
    result['min_train_loss/'+log_key] = min_train_loss_value

    # min_val_loss
    min_idx = np.argmin(val_loss)               # 获取最小验证损失的索引
    min_val_loss_value = key_data[min_idx]      # 获取对应的关键字值
    result['min_val_loss/'+log_key] = min_val_loss_value

    # k_min_train_loss
    min_loss_idxs = np.argsort(train_loss)[:k_min_loss]         # 获取前k个最小训练损失的索引
    k_min_train_loss_value = np.mean(key_data[min_loss_idxs])   # 计算对应关键字值的平均值
    result['k_min_train_loss/'+log_key] = k_min_train_loss_value

    # k_min_val_loss
    min_loss_idxs = np.argsort(val_loss)[:k_min_loss]           # 获取前k个最小验证损失的索引
    k_min_val_loss_value = np.mean(key_data[min_loss_idxs])     # 计算对应关键字值的平均值
    result['k_min_val_loss/'+log_key] = k_min_val_loss_value

    # 最后一个值
    result['last/'+log_key] = key_data[-1]

    # 可视化的全局步骤
    result['metric_global_step/'+log_key] = is_key_idxs[-1]
    return result                                               # 返回结果字典

# 计算多个日志数据框中指定关键字的聚合指标
def compute_metrics_agg(
        log_dfs: List[pd.DataFrame], 
        key: str, end_step:int, 
        **kwargs):
    # 计算指标
    results = collections.defaultdict(list) # 初始化结果字典
    for log_df in log_dfs:                  # 遍历每个数据框
        result = compute_metrics(log_df, key=key, end_step=end_step, **kwargs) # 计算指标
        for k, v in result.items():         # 遍历结果
            results[k].append(v)            # 添加到结果字典
    # 聚合
    agg_result = dict()                     # 初始化聚合结果字典
    for k, v in results.items():            # 遍历结果字典
        value = np.mean(v)                  # 计算平均值
        if k.startswith('metric_global_step'):
            value = int(value)              # 转换为整数
        agg_result[k] = value               # 添加到聚合结果字典
    return agg_result                       # 返回聚合结果字典

# 使用Click解析命令行参数，初始化日志记录器和WandB，读取和处理日志文件，计算和记录指标。
@click.command()                            # 定义Click命令行接口
@click.option('--input', '-i', required=True, help='Root logging dir, contains train_* dirs') # 定义命令行选项 --input 或 -i，必需，帮助信息为“根日志目录，包含train_*目录”
@click.option('--key', '-k', multiple=True, default=['test/mean_score']) # 定义命令行选项 --key 或 -k，可多次使用，默认值为['test/mean_score']
@click.option('--interval', default=10, type=float)             # 定义命令行选项 --interval，默认值为10，类型为浮点数
@click.option('--replace_slash', default=True, type=bool)       # 定义命令行选项 --replace_slash，默认值为True，类型为布尔值
@click.option('--index_key', '-ik', default='global_step')      # 定义命令行选项 --index_key 或 -ik，默认值为'global_step'
@click.option('--use_wandb', '-w', is_flag=True, default=False) # 定义命令行选项 --use_wandb 或 -w，标志选项，默认值为False
@click.option('--project', default=None)    # 定义命令行选项 --project，默认值为None
@click.option('--name', default=None)       # 定义命令行选项 --name，默认值为None
@click.option('--id', default=None)         # 定义命令行选项 --id，默认值为None
@click.option('--group', default=None)      # 定义命令行选项 --group，默认值为None
def main(
    input,
    key,
    interval,
    replace_slash,
    index_key,
    use_wandb,
    # wandb 参数
    project,
    name,
    id,
    group):
    root_dir = pathlib.Path(input)      # 获取根目录路径
    assert root_dir.is_dir()            # 断言根目录是一个目录
    metrics_dir = root_dir.joinpath('metrics') # 获取指标目录路径
    metrics_dir.mkdir(exist_ok=True)    # 创建指标目录，如果不存在

    logging.basicConfig( # 配置日志记录
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(str(metrics_dir.joinpath("metrics.log"))), # 文件日志处理器
            logging.StreamHandler()     # 流日志处理器
        ]
    )
    
    train_dirs = list(root_dir.glob('train_*'))     # 获取所有train_*目录
    log_files = [x.joinpath('logs.json.txt') for x in train_dirs] # 获取每个目录中的日志文件路径
    logging.info("Monitor waiting for log files!")  # 打印等待日志文件消息
    while True: # 无限循环
        # 等待文件出现
        files_exist = True              # 初始化文件存在标志为True
        for log_file in log_files:      # 遍历每个日志文件路径
            if not log_file.is_file():  # 如果日志文件不存在
                files_exist = False     # 设置文件存在标志为False
        if files_exist: # 如果所有文件都存在
            break # 退出循环
        time.sleep(1.0) # 等待1秒
    logging.info("All log files ready!") # 打印所有日志文件准备就绪消息

    # 初始化路径
    metric_log_path = metrics_dir.joinpath('logs.json.txt') # 获取指标日志路径
    metric_path = metrics_dir.joinpath('metrics.json') # 获取指标路径
    config_path = root_dir.joinpath('config.yaml') # 获取配置文件路径

    # 加载配置
    config = yaml.safe_load(config_path.open('r')) # 加载配置文件

    # 初始化wandb
    wandb_run = None
    if use_wandb: # 如果使用wandb
        wandb_kwargs = config['logging']    # 获取wandb配置
        if project is not None:             # 如果指定了项目，设置项目
            wandb_kwargs['project'] = project
        if id is not None:                  # 如果指定了id，设置id
            wandb_kwargs['id'] = id
        if name is not None:                # 如果指定了名称，设置名称
            wandb_kwargs['name'] = name
        if group is not None:               # 如果指定了组，设置组
            wandb_kwargs['group'] = group
        wandb_kwargs['resume'] = True       # 设置恢复模式
        wandb_run = wandb.init(             # 初始化wandb
            dir=str(metrics_dir),
            config=config,
            # 自动恢复运行，自动加载 ID
            # 只要使用相同的目录即可。
            # 参考文档：https://docs.wandb.ai/guides/track/advanced/resuming#resuming-guidance
            **wandb_kwargs
        )
        wandb.config.update( # 更新wandb配置
            {
                "output_dir": str(root_dir),
            }
        )

    with JsonLogger(metric_log_path) as json_logger:    # 使用JsonLogger记录日志
        last_log = json_logger.get_last_log()           # 获取最后一条日志
        while True:                                     # 无限循环
            # 读取json文件
            log_dfs = [read_json_log(str(x), required_keys=key) for x in log_files]

            # 之前记录的数据点
            last_log_idx = -1
            if last_log is not None:
                last_log_idx = log_dfs[0].index[log_dfs[0][index_key] <= last_log[index_key]][-1]
            
            start_idx = last_log_idx + 1
            # 所有日志中都有数据点的最后一个索引
            end_idx = min(*[len(x) for x in log_dfs])

            # 记录每个位置
            for this_idx in range(start_idx, end_idx):
                # 计算指标
                all_metrics = dict()
                global_step = log_dfs[0]['global_step'][this_idx]
                epoch = log_dfs[0]['epoch'][this_idx]
                all_metrics['global_step'] = global_step
                all_metrics['epoch'] = epoch
                for k in key:
                    metrics = compute_metrics_agg(
                        log_dfs=log_dfs, key=k, end_step=this_idx+1,
                        replace_slash=replace_slash)
                    all_metrics.update(metrics)

                # 清理指标
                old_metrics = all_metrics
                all_metrics = dict()
                for k, v in old_metrics.items():
                    if isinstance(v, numbers.Integral):
                        all_metrics[k] = int(v)
                    elif isinstance(v, numbers.Number):
                        all_metrics[k] = float(v)
                
                has_update = all_metrics != last_log
                if has_update:
                    last_log = all_metrics
                    json_logger.log(all_metrics)

                    with metric_path.open('w') as f:
                        json.dump(all_metrics, f, sort_keys=True, indent=2)

                    if wandb_run is not None:
                        wandb_run.log(all_metrics, step=all_metrics[index_key])

                    logging.info(f"Metrics logged at step {all_metrics[index_key]}")
            
            time.sleep(interval) # 等待指定的间隔时间

if __name__ == "__main__":
    main()
