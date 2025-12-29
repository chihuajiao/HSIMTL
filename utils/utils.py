import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F

def Nuclear_norm(inputs):
    _, band, h, w = inputs.shape
    input = torch.reshape(inputs, (band, h*w))
    out = torch.norm(input, p='nuc')
    return out

class SparseKLloss(nn.Module):
    def __init__(self):
        super(SparseKLloss, self).__init__()

    def __call__(self, input, decay=1e-2):
        input = torch.sum(input, 0, keepdim=True)
        loss = Nuclear_norm(input)
        return decay*loss

class NonZeroClipper(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(1e-6,1)

class SumToOneLoss(nn.Module):
    def __init__(self):
        super(SumToOneLoss, self).__init__()
        self.register_buffer('one', torch.tensor(1, dtype=torch.float))
        self.loss = nn.L1Loss(size_average=False)

    def get_target_tensor(self, input):
        target_tensor = self.one
        return target_tensor.expand_as(input)

    def __call__(self, input, gamma_reg=1e-6):
        input = torch.sum(input, 1)
        target_tensor = self.get_target_tensor(input)
        loss = self.loss(input, target_tensor)
        return gamma_reg*loss
    
def compute_rmse(x_true, x_pre):
    img_w, img_h, img_c = x_true.shape
    return np.sqrt( ((x_true-x_pre)**2).sum()/(img_w*img_h*img_c) )


def extract_samples(mask, test_inds, num_samples_per_class=10,seed=2333):
    """
    从test中提取每类10个样本，生成新的mask和val_inds
    
    Args:
        mask (Tensor): 包含类别标签的张量，形状为 [1, 1, H, W]
        test_inds (Tensor): 标记标签位置的布尔张量，形状为 [1, 1, H, W]
        num_samples_per_class (int): 每类要提取的样本数量，默认为10
    
    Returns:
        new_mask (Tensor): 新的mask，只包含被选中的样本
        val_inds (Tensor): 新的验证集指示张量
    """
    random.seed(seed)
    # 获取类别标签（忽略背景类0）
    classes = torch.unique(mask)
    if 0 in classes:
        classes = classes[1:]  # 假设0是背景类，跳过
    
    # 初始化新的mask和验证集指示张量
    new_mask = torch.zeros_like(mask, dtype=torch.long)
    val_inds = torch.zeros_like(test_inds, dtype=torch.bool)


    for class_id in classes:
        
        class_positions = torch.where((mask == class_id) & test_inds)
        
        if len(class_positions[0]) < num_samples_per_class:
            print(f"类别 {class_id} 样本数量不足，跳过")
            continue
            
        # 随机选择10个样本
        selected_indices = random.sample(range(len(class_positions[0])), num_samples_per_class)
        
        # 更新新的mask和验证集指示张量
        for idx in selected_indices:
            y = class_positions[2][idx]
            x = class_positions[3][idx]
            new_mask[0, 0, y, x] = class_id
            val_inds[0, 0, y, x] = True
    
    return new_mask, val_inds

def pcgrad(grads_t1, grads_t2):
    # 确保 dotp 在和 grads_t1/grads_t2 同样的 device
    # 假设第一个非 None 的梯度在什么 device，就用那个
    device = None
    for g in grads_t1:
        if g is not None:
            device = g.device
            break
    if device is None:  # 如果 grads_t1 里全是 None，就再看 grads_t2
        for g in grads_t2:
            if g is not None:
                device = g.device
                break
    if device is None:
        # 两个梯度列表全是 None，不做任何事
        return [None]*len(grads_t1)

    dotp = torch.zeros(1, dtype=torch.float32, device=device)

    # 计算 dotp
    for g1, g2 in zip(grads_t1, grads_t2):
        if g1 is not None and g2 is not None:
            dotp += torch.sum(g1 * g2)

    if dotp < 0:
        # 计算 ||g2||^2
        norm_g2_sqr = torch.zeros(1, dtype=torch.float32, device=device)
        for g2_ in grads_t2:
            if g2_ is not None:
                norm_g2_sqr += torch.sum(g2_**2)
        norm_g2_sqr = norm_g2_sqr + 1e-12
        scale = dotp / norm_g2_sqr

        new_g1 = []
        for g1_, g2_ in zip(grads_t1, grads_t2):
            if g1_ is not None and g2_ is not None:
                proj = g1_ - scale * g2_
                new_g1.append(proj)
            else:
                new_g1.append(g1_)
        grads_t1 = new_g1

    # 合并
    final_grads = []
    for g1_, g2_ in zip(grads_t1, grads_t2):
        if g1_ is not None and g2_ is not None:
            final_grads.append((g1_ + g2_) / 2.0)
        elif g1_ is None and g2_ is not None:
            final_grads.append(g2_)
        elif g1_ is not None and g2_ is None:
            final_grads.append(g1_)
        else:
            final_grads.append(None)

    return final_grads

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, init_sigma_cls=0.1, init_sigma_unmix=1.0):
        super(MultiTaskLossWrapper, self).__init__()
        
        self.sigma_cls = nn.Parameter(torch.tensor(init_sigma_cls))
        self.sigma_unmix = nn.Parameter(torch.tensor(init_sigma_unmix))
    
    def forward(self, loss_cls, loss_unmix):
        
        weighted_loss_cls = (1.0 / (2 * self.sigma_cls ** 2)) * loss_cls + torch.log(self.sigma_cls ** 2)
        weighted_loss_unmix = (1.0 / (2 * self.sigma_unmix ** 2)) * loss_unmix + torch.log(self.sigma_unmix ** 2)
        total_loss = weighted_loss_cls + weighted_loss_unmix
        return total_loss
    

def causal_distance(alpha, gamma, method='cosine'):
    """
    alpha, gamma: [C]
    返回它们的距离(或1-相似度)
    """
    if method=='cosine':
        sim = F.cosine_similarity(alpha.unsqueeze(0), gamma.unsqueeze(0))
        return 1.0 - sim
    elif method=='l2':
        return (alpha - gamma).pow(2).mean()
    else:
        # 其他距离...
        sim = F.cosine_similarity(alpha.unsqueeze(0), gamma.unsqueeze(0))
        return 1.0 - sim
    
class DynamicWeightedLoss(nn.Module):
    def __init__(self, num_losses):
        super(DynamicWeightedLoss, self).__init__()
        
        self.log_sigma_sq = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses):
        total_loss = 0
        for i, loss in enumerate(losses):
            
            sigma_sq = torch.exp(self.log_sigma_sq[i])
            weighted_loss = loss / (2.0 * sigma_sq) + 0.5 * self.log_sigma_sq[i]
            total_loss += weighted_loss
        return total_loss
    







