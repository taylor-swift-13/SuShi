
import torch
import torch.nn.functional as F

def compute_dpo_loss(
    model_preferred_logps: torch.FloatTensor,
    model_rejected_logps: torch.FloatTensor,
    ref_preferred_logps: torch.FloatTensor,
    ref_rejected_logps: torch.FloatTensor,
    beta: float = 0.1,
):
    """
    计算标准 DPO Loss（不包含 label smoothing）。
    
    参数:
        model_preferred_logps: 当前模型对偏好回答的 log 概率
        model_rejected_logps: 当前模型对拒绝回答的 log 概率
        ref_preferred_logps: 参考模型对偏好回答的 log 概率
        ref_rejected_logps: 参考模型对拒绝回答的 log 概率
        beta: 温度参数，控制偏好强度
    """
    # 1. 当前模型的 log 概率差
    pi_logratios = model_preferred_logps - model_rejected_logps

    # 2. 参考模型的 log 概率差
    ref_logratios = ref_preferred_logps - ref_rejected_logps

    # 3. DPO logits（相对偏好优势）
    logits = pi_logratios - ref_logratios

    # 4. 标准 DPO loss
    loss = -F.logsigmoid(beta * logits)

    return loss.mean()