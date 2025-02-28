"""
Start local ray cluster
(robodiff)$ export CUDA_VISIBLE_DEVICES=0,1,2 # select GPUs to be managed by the ray cluster
(robodiff)$ ray start --head --num-gpus=3

Training:
python ray_train_multirun.py --config-name=train_diffusion_unet_lowdim_workspace --seeds=42,43,44 --monitor_key=test/mean_score -- logger.mode=online training.eval_first=True
"""

import os # 导入操作系统模块
import ray # 导入Ray库
import click # 导入Click库，用于命令行接口
import hydra # 导入Hydra库，用于配置管理
import yaml # 导入YAML库，用于解析YAML文件
import wandb # 导入Weights and Biases库，用于实验跟踪
import pathlib # 导入Pathlib库，用于路径操作
import collections # 导入集合模块
from pprint import pprint # 导入pprint模块用于美化打印
from omegaconf import OmegaConf # 导入OmegaConf，用于处理配置
from ray_exec import worker_fn # 从ray_exec模块导入worker_fn函数
from ray.util.placement_group import ( # 从Ray库中导入placement_group模块
    placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy # 导入PlacementGroupSchedulingStrategy，用于调度策略

OmegaConf.register_new_resolver("eval", eval, replace=True) # 注册新的解析器

# 使用Click解析命令行参数，初始化Ray集群，生成和提交多个训练任务，并监控任务进度
@click.command() # 定义Click命令
@click.option('--config-name', '-cn', required=True, type=str)  # 
@click.option('--config-dir', '-cd', default=None, type=str)    # 
@click.option('--seeds', '-s', default='42,43,44', type=str)    # 
@click.option('--monitor_key', '-k', multiple=True, default=['test/mean_score']) # 
@click.option('--ray_address', '-ra', default='auto')
@click.option('--num_cpus', '-nc', default=7, type=float)
@click.option('--num_gpus', '-ng', default=1, type=float)
@click.option('--max_retries', '-mr', default=0, type=int)
@click.option('--monitor_max_retires', default=3, type=int)
@click.option('--data_src', '-d', default='./data', type=str)
@click.option('--unbuffer_python', '-u', is_flag=True, default=False)
@click.option('--single_node', '-sn', is_flag=True, default=False, help='run all experiments on a single machine') # 定义命令行选项
@click.argument('command_args', nargs=-1, type=str)             #  
def main(config_name, config_dir, seeds, monitor_key, ray_address, 
    num_cpus, num_gpus, max_retries, monitor_max_retires,
    data_src, unbuffer_python, 
    single_node, command_args):
    # 解析参数
    seeds = [int(x) for x in seeds.split(',')] # 解析种子参数
    # 展开路径
    if data_src is not None:
        data_src = os.path.abspath(os.path.expanduser(data_src)) # 获取数据源的绝对路径

    # 初始化Hydra
    if config_dir is None:
        config_path_abs = pathlib.Path(__file__).parent.joinpath(
            'diffusion_policy','config')    # 获取默认配置路径
        config_path_rel = str(config_path_abs.relative_to(pathlib.Path.cwd())) # 获取相对路径
    else:
        config_path_rel = config_dir        # 使用用户提供的配置路径

    run_command_args = list()               # 初始化运行命令参数列表
    monitor_command_args = list()           # 初始化监控命令参数列表
    with hydra.initialize(                  # 初始化Hydra
        version_base=None, 
        config_path=config_path_rel):

        # 生成原始配置
        cfg = hydra.compose(
            config_name=config_name, 
            overrides=command_args)
        OmegaConf.resolve(cfg)              # 解析配置
    
        # 手动创建输出目录
        output_dir = pathlib.Path(cfg.multi_run.run_dir)
        output_dir.mkdir(parents=True, exist_ok=False)          # 创建目录
        config_path = output_dir.joinpath('config.yaml')        # 获取配置文件路径
        print(output_dir)

        # 保存当前配置
        yaml.dump(OmegaConf.to_container(cfg, resolve=True), 
            config_path.open('w'), default_flow_style=False)

        # 初始化WandB
        wandb_group_id = wandb.util.generate_id()               # 生成WandB组ID
        name_base = cfg.multi_run.wandb_name_base

        # 创建监控命令参数
        monitor_command_args = [
            'python',
            'multirun_metrics.py',
            '--input', str(output_dir),
            '--use_wandb',
            '--project', 'diffusion_policy_metrics',
            '--group', wandb_group_id
        ]
        for k in monitor_key: # 添加监控键
            monitor_command_args.extend([
                '--key', k
            ])

        # 生成命令参数
        run_command_args = list()
        for i, seed in enumerate(seeds):
            test_start_seed = (seed + 1) * 100000               # 计算测试开始种子
            this_output_dir = output_dir.joinpath(f'train_{i}') # 获取训练输出目录
            this_output_dir.mkdir()                             # 创建目录
            wandb_name = name_base + f'_train_{i}'              # 设置WandB名称
            wandb_run_id = wandb_group_id + f'_train_{i}'       # 设置WandB运行ID

            this_command_args = [
                'python',
                'train.py',
                '--config-name='+config_name,
                '--config-dir='+config_path_rel
            ]

            this_command_args.extend(command_args)              # 添加命令参数
            this_command_args.extend([
                f'training.seed={seed}',
                f'task.env_runner.test_start_seed={test_start_seed}',
                f'logging.name={wandb_name}',
                f'logging.id={wandb_run_id}',
                f'logging.group={wandb_group_id}',
                f'hydra.run.dir={this_output_dir}'
            ])
            run_command_args.append(this_command_args)          # 添加到运行命令参数列表

    # 初始化Ray
    root_dir = os.path.dirname(__file__)                        # 获取当前文件的目录
    runtime_env = {
        'working_dir': root_dir,
        'excludes': ['.git'],
        'pip': ['dm-control==1.0.9']
    }
    ray.init(
        address=ray_address, 
        runtime_env=runtime_env
    )
    
    # 创建训练资源
    train_resources = dict()

    train_bundle = dict(train_resources)
    train_bundle['CPU'] = num_cpus
    train_bundle['GPU'] = num_gpus

    # 创建监控资源
    monitor_resources = dict()
    monitor_resources['CPU'] = 1
    
    monitor_bundle = dict(monitor_resources)

    # 聚合资源包
    bundle = collections.defaultdict(lambda:0)
    n_train_bundles = 1
    if single_node:
        n_train_bundles = len(seeds)
    for _ in range(n_train_bundles):
        for k, v in train_bundle.items():
            bundle[k] += v
    for k, v in monitor_bundle.items():
        bundle[k] += v
    bundle = dict(bundle)

    # 创建放置组
    print("Creating placement group with resources:")
    pprint(bundle)
    pg = placement_group([bundle])

    # 运行
    task_name_map = dict()  # 初始化任务名称映射
    task_refs = list()      # 初始化任务引用列表
    for i, this_command_args in enumerate(run_command_args):
        if single_node or i == (len(run_command_args) - 1):
            print(f'Training worker {i} with placement group.')
            ray.get(pg.ready())
            print("Placement Group created!")
            worker_ray = ray.remote(worker_fn).options(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                max_retries=max_retries,
                resources=train_resources,
                retry_exceptions=True,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg)
            )
        else:
            print(f'Training worker {i} without placement group.')
            worker_ray = ray.remote(worker_fn).options(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                max_retries=max_retries,
                resources=train_resources,
                retry_exceptions=True,
            )
        task_ref = worker_ray.remote(
            this_command_args, data_src, unbuffer_python)
        task_refs.append(task_ref)                  # 添加任务引用
        task_name_map[task_ref] = f'train_{i}'      # 添加任务名称映射

    # 监控工作者总是打包在同一个节点上，和训练工作者0一样
    ray.get(pg.ready())
    monitor_worker_ray = ray.remote(worker_fn).options(
        num_cpus=1, 
        num_gpus=0,
        max_retries=monitor_max_retires,
        # resources=monitor_resources,
        retry_exceptions=True,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg)
        )
    monitor_ref = monitor_worker_ray.remote(
            monitor_command_args, data_src, unbuffer_python)
    task_name_map[monitor_ref] = 'metrics'          # 添加监控任务名称映射
    
    # 正常情况
    try:
        ready_refs = list()         # 初始化已完成任务引用列表
        rest_refs = task_refs       # 剩余任务引用列表
        while len(ready_refs) < len(task_refs):
            this_ready_refs, rest_refs = ray.wait(rest_refs, 
                num_returns=1, timeout=None, fetch_local=True)
            cancel_other_tasks = False              # 初始化取消其他任务标志
            for ref in this_ready_refs:
                task_name = task_name_map[ref]      # 获取任务名称
                try:
                    result = ray.get(ref)           # 获取任务结果
                    print(f"Task {task_name} finished with result: {result}")
                except KeyboardInterrupt as e:      # 跳过到外层捕获 skip to outer try catch 
                    raise KeyboardInterrupt
                except Exception as e:
                    print(f"Task {task_name} raised exception: {e}")
                    this_cancel_other_tasks = True
                    if isinstance(e, ray.exceptions.RayTaskError):
                        if isinstance(e.cause, ray.exceptions.TaskCancelledError):
                            this_cancel_other_tasks = False
                    cancel_other_tasks = cancel_other_tasks or this_cancel_other_tasks
                ready_refs.append(ref)
            if cancel_other_tasks:
                print('Exception! Cancelling all other tasks.')
                for _ref in rest_refs:              #  取消所有其他任务
                    ray.cancel(_ref, force=False)
        print("Training tasks done.")
        ray.cancel(monitor_ref, force=False)
    except KeyboardInterrupt:
        print('KeyboardInterrupt received in the driver.')
        # 工作者中将引发KeyboardInterrupt
        _ = [ray.cancel(x, force=False) for x in task_refs + [monitor_ref]]
        print('KeyboardInterrupt sent to workers.')
    except Exception as e:
        # 工作者将被终止
        _ = [ray.cancel(x, force=True) for x in task_refs + [monitor_ref]]
        raise e

    for ref in task_refs + [monitor_ref]:
        task_name = task_name_map[ref]
        try:
            result = ray.get(ref)
            print(f"Task {task_name} finished with result: {result}")
        except KeyboardInterrupt as e:
            # 强制杀死所有任务
            print("Force killing all workers")
            _ = [ray.cancel(x, force=True) for x in task_refs]
            ray.cancel(monitor_ref, force=True)
        except Exception as e:
            print(f"Task {task_name} raised exception: {e}")

if __name__ == "__main__": # 主程序入口
    main() # 调用主函数
