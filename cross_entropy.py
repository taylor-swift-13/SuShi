import torch



def cross_entropy(logits, targets, reduction='mean'):
    """
    手写交叉熵损失函数（数值稳定版本）

    Args:
        logits:    模型原始输出，shape (batch_size, num_classes)
        targets:   真实标签，shape (batch_size,)，值为类别索引
        reduction: 'mean' | 'sum' | 'none'

    Returns:
        reduction='mean' -> scalar
        reduction='sum'  -> scalar
        reduction='none' -> shape (batch_size,)
    """
    # 假设 `batch_size=2, num_classes=3`：
    # ```
    # logits  = [[2.0, 1.0, 0.1],    targets = [0, 2]
    #            [0.5, 2.0, 0.3]]

    # → shift 后:  [[0.0, -1.0, -1.9],
    #              [-1.5, 0.0, -1.7]]

    # → log_softmax: [[-0.41, -1.41, -2.31],
    #                [-1.87, -0.37, -2.07]]

    # → 取 [0,0] 和 [1,2]:  loss = [0.41, 2.07]

    # → mean:  1.24


    batch_size = logits.shape[0]

    # 1. 数值稳定：减去每行最大值，避免 exp 溢出
    shift = logits.max(dim=1, keepdim=True).values
    shifted_logits = logits - shift

    # 2. 计算 log_softmax
    log_sum_exp = torch.log(torch.exp(shifted_logits).sum(dim=1, keepdim=True))
    log_softmax = shifted_logits - log_sum_exp

    # 3. 取出真实类别对应的 log 概率，取负得到 loss
    loss = -log_softmax[range(batch_size), targets]

    # 4. reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"reduction 须为 'mean'/'sum'/'none'，当前为 '{reduction}'")

