# utils/monitor.py
import torch
import time

class SystemMonitor:
    """
    监控显存、延迟和分布漂移。
    """
    def __init__(self, device):
        self.device = device
        self.start_time = None

    def check_memory(self, message=""):
        """检查并打印当前GPU显存使用情况"""
        allocated = torch.cuda.memory_allocated(self.device) / 1024**2
        reserved = torch.cuda.memory_reserved(self.device) / 1024**2
        print(f"{message} | GPU显存: {allocated:.2f} MB 已分配 / {reserved:.2f} MB 已预留")

    def start_timer(self):
        """开始计时"""
        self.start_time = time.time()

    def stop_timer(self, message="操作耗时"):
        """停止计时并打印耗时"""
        if self.start_time is None:
            print("错误：计时器未启动")
            return
        elapsed = (time.time() - self.start_time) * 1000 # 转换为毫秒
        print(f"{message}: {elapsed:.2f} ms")
        self.start_time = None
        return elapsed

    def check_distribution_drift(self, tensor_a, tensor_b, name=""):
        """
        使用简单的统计来检查分布漂移。
        tensor_a: 参考分布 (如，真实的SigLIP特征)
        tensor_b: 待检查分布 (如，生成的e_t)
        """
        mean_a, std_a = tensor_a.mean(), tensor_a.std()
        mean_b, std_b = tensor_b.mean(), tensor_b.std()
        
        # 计算均值和标准差的差异
        mean_diff = torch.abs(mean_a - mean_b)
        std_diff = torch.abs(std_a - std_b)
        
        print(f"分布漂移监控 '{name}': 均值差异={mean_diff:.4f}, 标准差差异={std_diff:.4f}")
        
        # 在实际应用中，可以设置阈值，当差异过大时触发警报或自动校准
        return {"mean_diff": mean_diff.item(), "std_diff": std_diff.item()}