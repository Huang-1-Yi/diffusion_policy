import copy
import torch
from torch.nn.modules.batchnorm import _BatchNorm

class EMAModel:
    """
    模型权重的指数移动平均 Exponential Moving Average of models weights
    """

    def __init__(
        self,
        model,      # 要进行权重平均的模型
        update_after_step=0,
        inv_gamma=1.0,
        power=2 / 3,
        min_value=0.0,
        max_value=0.9999
    ):
        """
        @crowsonkb's notes on EMA Warmup关于EMA预热期的笔记:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan to train
            for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at 215.4k steps).
            如果gamma=1且power=1,则实现一个简单的平均值。
            gamma=1,power=2/3是您计划训练超过一百万步(在31.6K步时达到衰减因子0.999,在1M步时达到0.9999)的模型的良好值,
            gamma=1,power=3/4是您计划训练较少步数的模型的良好值(在10K步时达到衰减因子0.999,在215.4k步时达到0.9999)
        Args:
            inv_gamma (float): EMA预热期的倒数乘法因子。默认值:1   Inverse multiplicative factor of EMA warmup. Default: 1.
            power (float): EMA预热期的指数因子。默认值:2/   Exponential factor of EMA warmup. Default: 2/3.
            min_value (float): 最小的EMA衰减率。默认值:0    The minimum EMA decay rate. Default: 0.
        """
        # 衰减率按照 inv_gamma 和 power 的指数模式逐渐增加。这种设置旨在为模型提供稳定的初始化，使其在达到典型的稳定衰减之前保持稳定，这对大型模型训练尤其有用
        self.averaged_model = model
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)

        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        根据当前的优化步数计算指数移动平均的衰减因子 Compute the decay factor for the exponential moving average.
        """
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power

        if step <= 0:
            return 0.0

        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model):
        # 使用计算出的衰减率来更新 averaged_model 的权重，使其反映当前模型的权重
        self.decay = self.get_decay(self.optimization_step)

        # old_all_dataptrs = set()
        # for param in new_model.parameters():
        #     data_ptr = param.data_ptr()
        #     if data_ptr != 0:
        #         old_all_dataptrs.add(data_ptr)

        all_dataptrs = set()
        # 遍历模型的所有模块(层)，然后对每个模块的参数进行迭代
        for module, ema_module in zip(new_model.modules(), self.averaged_model.modules()):            
            for param, ema_param in zip(module.parameters(recurse=False), ema_module.parameters(recurse=False)):
                # iterative over immediate parameters only.
                if isinstance(param, dict):
                    raise RuntimeError('Dict parameter not supported')
                
                # data_ptr = param.data_ptr()
                # if data_ptr != 0:
                #     all_dataptrs.add(data_ptr)

                if isinstance(module, _BatchNorm):
                    # skip batchnorms
                    # BatchNorm 层处理：跳过 BatchNorm 层的权重更新，因为这些层的 EMA 可能会导致不稳定
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                elif not param.requires_grad:
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                else:
                    ema_param.mul_(self.decay)
                    # 将当前 EMA 参数乘以衰减因子，并将模型参数按1 - self.decay的权重相加
                    ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1 - self.decay)
        
        # 验证对模块进行迭代然后再对参数进行迭代与递归地对参数进行迭代是相同的
        # verify that iterating over module and then parameters is identical to parameters recursively.
        # assert old_all_dataptrs == all_dataptrs
        self.optimization_step += 1     # 递增优化步数，以更新下一次迭代的衰减率
