"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace -- logger.mode=online
"""
import os       # 导入os模块
import ray      # 导入ray模块
import click    # 导入click模块

# 执行子进程命令，处理数据符号链接和信号
def worker_fn(command_args, data_src=None, unbuffer_python=False, use_shell=False): # 定义worker函数
    import os           # 再次导入os模块（为子进程准备）
    import subprocess   # 导入subprocess模块
    import signal       # 导入signal模块
    import time         # 导入time模块

    # 设置数据符号链接setup data symlink
    if data_src is not None:                # 如果data_src不为空
        cwd = os.getcwd()                   # 获取当前工作目录
        src = os.path.expanduser(data_src)  # 展开用户路径
        dst = os.path.join(cwd, 'data')     # 目标路径
        try:
            os.symlink(src=src, dst=dst)    # 创建符号链接
        except FileExistsError: # 如果符号链接已存在，则无需处理
            pass

    # run command # 运行命令
    process_env = os.environ.copy() # 复制环境变量
    if unbuffer_python: # 如果unbuffer_python为真
        # disable stdout/stderr buffering for subprocess (if python)
        # to remove latency between print statement and receiving printed result
        process_env['PYTHONUNBUFFERED'] = 'TRUE'    # 禁用子进程的标准输出/错误缓冲，移除状态打印和收取结果的延迟

    # Ray工作进程隐藏了Ctrl-C信号（即SIGINT）ray worker masks out Ctrl-C signal (ie SIGINT)
    # 在这里我们为子进程取消了对该信号的限制here we unblock this signal for the child process
    def preexec_function(): # 定义预执行函数
        import signal       # 导入signal模块
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT}) # 解除SIGINT信号屏蔽

    if use_shell:           # 如果use_shell为真
        command_args = ' '.join(command_args)       # 将命令参数连接成字符串

    # 将标准输出直接传递给ray工作进程，然后传递给ray驱动程序 stdout passthrough to ray worker, which is then passed to ray driver
    process = subprocess.Popen( # 创建子进程
        args=command_args, 
        env=process_env,
        preexec_fn=preexec_function,
        shell=use_shell)

    while process.poll() is None:   # 等待子进程完成
        # 睡眠以确保监控线程可以获取全局解释器锁（GIL）,并在此处引发键盘中断异常（KeyboardInterrupt）
        try:
            time.sleep(0.01)        # 睡眠以确保监控线程能够获取GIL并在此处引发KeyboardInterrupt
        except KeyboardInterrupt:   # 捕获键盘中断
            process.send_signal(signal.SIGINT) # 向子进程发送SIGINT信号
            print('SIGINT sent to subprocess') # 打印信息
        except Exception as e:      # 捕获其他异常
            process.terminate()     # 终止子进程
            raise e                 # 重新引发异常

    if process.returncode not in (0, -2):       # 如果子进程返回码不为0或-2
        print("Failed execution!")              # 打印失败信息
        raise RuntimeError("Failed execution.") # 引发运行时错误
    return process.returncode                   # 返回子进程返回码


# 使用Click解析命令行参数，初始化Ray，提交并管理任务
@click.command() # 定义click命令
@click.option('--ray_address', '-ra', default='auto')           # 定义ray_address选项
@click.option('--num_cpus', '-nc', default=7, type=float)       # 定义num_cpus选项
@click.option('--num_gpus', '-ng', default=2, type=float)       # 定义num_gpus选项
@click.option('--max_retries', '-mr', default=0, type=int)      # 定义max_retries选项
@click.option('--data_src', '-d', default='./data', type=str)   # 定义data_src选项
@click.option('--unbuffer_python', '-u', is_flag=True, default=False) # 定义unbuffer_python选项
@click.argument('command_args', nargs=-1, type=str) # 定义命令参数
def main(ray_address, 
    num_cpus, num_gpus, max_retries, 
    data_src, unbuffer_python, 
    command_args):  # 定义main函数
    # 展开路径expand path
    if data_src is not None: # 如果data_src不为空
        data_src = os.path.abspath(os.path.expanduser(data_src)) # 获取绝对路径

    # 初始化rayinit ray
    root_dir = os.path.dirname(__file__)    # 获取当前文件的目录
    runtime_env = {                         # 定义运行时环境
        'working_dir': root_dir,
        'excludes': ['.git']
    }
    ray.init(                               # 初始化ray
        address=ray_address, 
        runtime_env=runtime_env
    )
    # 定义远程worker函数remote worker func
    worker_ray = ray.remote(worker_fn).options( # 配置远程worker函数
        num_cpus=num_cpus, 
        num_gpus=num_gpus,
        max_retries=max_retries,
        retry_exceptions=True
        )
    # 运行run
    task_ref = worker_ray.remote(command_args, data_src, unbuffer_python) # 提交任务
    
    # 正常情况
    try:             
        result = ray.get(task_ref)          # 获取任务结果
        print('Return code: ', result)      # 打印返回码
    except KeyboardInterrupt:               # 捕获键盘中断a KeyboardInterrupt will be raised in worker
        ray.cancel(task_ref, force=False)   # 取消任务
        result = ray.get(task_ref)          # 获取任务结果
        print('Return code: ', result)      # 打印返回码
    except Exception as e:                  # 捕获其他异常worker will be terminated
        ray.cancel(task_ref, force=True)    # 强制取消任务
        raise e                             # 重新引发异常

if __name__ == '__main__': 
    main()
