import numpy as np

def get_uniform_indices(current_len, num_frames=6):
    """
    输入:
        current_len: 当前 buffer 里有多少帧
        num_frames: 我们需要选多少帧 (例如 6)
    输出:
        indices: 选出的帧索引列表
    """
    if current_len <= num_frames:
        # 情况 A: 历史不够 (冷启动)
        # 策略: 取所有帧，然后把第一帧复制填充到头部 (或者复制最后一帧，看你喜好)
        # 你的策略: "刚开始...把第一帧复制" -> 也就是填充头部
        # 比如 buffer=[A, B], num=6 -> indices=[0, 0, 0, 0, 0, 1] -> [A, A, A, A, A, B]
        # 或者简单的 Padding 策略: [0, 0, 0, 0, 0, 1] 对应取 buffer[0], buffer[1]
        
        # 简单实现: 也就是取现有帧的全部，剩下的用第一帧补齐
        # 但为了代码统一，我们可以直接取现有索引，不够的用0补
        indices = np.linspace(0, current_len - 1, num_frames).astype(int)
        # 注意: linspace 在小区间会自动插值，比如 len=2, num=6 -> [0, 0, 0, 1, 1, 1] (近似)
        # 这比单纯复制第一帧更好，因为它均匀利用了现有的每一帧
        return indices
    else:
        # 情况 B: 历史足够
        # 策略: 把历史平均分成 6 份，选 6 帧
        # np.linspace(start, stop, num) 刚好就是做这个的
        indices = np.linspace(0, current_len - 1, num_frames).astype(int)
        return indices