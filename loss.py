import torch
import torch.nn as nn
import torch.nn.functional as F
# ----------------------------- 数据项(强监督 + 一侧 DCD) -----------------------------
class OneSidedDCD(nn.Module):
    '''
    OneSidedDCD 的 Docstring
    part 向 pred 的距离
    防止塌陷
    '''
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    def forward(self, pred, part):
        d = torch.cdist(part, pred) ** 2       # (B,M,N)
        m = torch.min(d, dim=-1)[0]            # (B,M)
        s = torch.mean(torch.exp(-self.alpha * m), dim=-1)  # (B,)
        return torch.mean(1.0 - s)


class DCD(nn.Module):
    def __init__(self, alpha=1.0, weight=0.5):
        super().__init__()
        self.alpha = alpha
        self.weight = weight  # 双向权重，确保总和为1
        
    def _one_sided_dcd(self, src, dst):
        dist = torch.cdist(src, dst) ** 2  # (B, M, N)
        min_dist = torch.min(dist, dim=-1)[0]  # (B, M)
        score = torch.mean(torch.exp(-self.alpha * min_dist), dim=-1)  # (B,)
        return 1.0 - score
    
    def forward(self, pred, target):
        loss_target2pred = self._one_sided_dcd(target, pred)  # (B,)
        loss_pred2target = self._one_sided_dcd(pred, target)  # (B,)
        
        total_loss = self.weight * loss_target2pred + (1 - self.weight) * loss_pred2target
        
        return torch.mean(total_loss)


def point_rmse(pred, gt):
    """Point-wise RMSE: sqrt(mean_i(||pred_i - gt_i||_2^2)).

    Both inputs are (B, N, 3).  Returns a scalar in millimetres.
    """
    squared_l2 = (pred.float() - gt.float()).square().sum(dim=-1)
    return torch.sqrt(squared_l2.mean())


def pointwise_huber_loss(pred, gt, beta_mm=5.0):
    """Per-point Huber loss with L2-norm error per point.

    For each point pair i:
        e_i = ||pred_i - gt_i||_2   (mm)
        loss = SmoothL1(e_i, 0, beta)

    The mean over all points is returned.
    """
    per_point_l2 = (pred.float() - gt.float()).norm(p=2, dim=-1)  # (B, N)
    return F.smooth_l1_loss(
        per_point_l2,
        torch.zeros_like(per_point_l2),
        beta=beta_mm,
        reduction='mean',
    )


# Legacy helpers kept for compatibility but no longer used as primary metrics.
def rmse(a, b):  # (B,N,3)
    return torch.sqrt(F.mse_loss(a, b))

def mse_loss(a, b):  # (B,N,3)
    return F.mse_loss(a, b)

# ----------------------------- 物理项loss(KNN-MLS 雅可比) -----------------------------
def knn_indices(xyz, k=16):
    with torch.no_grad():
        dist = torch.cdist(xyz, xyz)          # (B,N,N)
        val, idx = torch.topk(dist, k=k+1, dim=-1, largest=False)
        return idx[..., 1:]                   # (B,N,k)

def estimate_jacobian(xyz, u, idx, reg=1e-4):
    B,N,_ = xyz.shape
    k = idx.shape[-1]
    i_expand = torch.arange(N, device=xyz.device)[None, :, None].expand(B, N, k)
    Xi = xyz.gather(1, i_expand.unsqueeze(-1).expand(-1,-1,-1,3))
    Xj = xyz.gather(1, idx.unsqueeze(-1).expand(-1,-1,-1,3))
    dX = Xj - Xi
    Ui = u.gather(1, i_expand.unsqueeze(-1).expand(-1,-1,-1,3))
    Uj = u.gather(1, idx.unsqueeze(-1).expand(-1,-1,-1,3))
    dU = Uj - Ui
    AT = dX.transpose(-1, -2)                # (B,N,3,k)
    ATA = AT @ dX                             # (B,N,3,3)
    ATB = AT @ dU                             # (B,N,3,3)
    I = torch.eye(3, device=xyz.device).view(1,1,3,3)
    J = torch.linalg.solve(ATA + reg*I, ATB)  # (B,N,3,3)
    return J

def linear_elastic_energy(J, mu, lam):
    eps = 0.5 * (J + J.transpose(-1, -2))     # (B,N,3,3)
    tr = eps[...,0,0] + eps[...,1,1] + eps[...,2,2]
    fro2 = (eps**2).sum(dim=(-1,-2))
    W = mu * fro2 + 0.5 * lam * (tr**2)
    return W.mean()

def deviatoric_neo_hookean_energy(J, mu, eps=1e-6):
    """
    计算 Neo-Hookean 模型的偏(deviatoric)应变能密度
    这适用于大变形，并与一个单独的体积惩罚项(如 incompressibility_penalty)配合使用。
    J: (B, N, 3, 3) 位移雅可比
    mu: (B, 1) 拉梅第一参数 (shear modulus)
    """
    B, N, _, _ = J.shape
    I = torch.eye(3, device=J.device).view(1, 1, 3, 3)
    F = I + J  # 变形梯度 F = I + ∇u, (B, N, 3, 3)
    
    # 计算 F 的行列式: J_det = det(F)
    J_det = torch.linalg.det(F)  # (B, N)
    
    # 稳定计算，防止 J_det <= 0
    J_det_stable = torch.clamp(J_det, min=eps)
    
    # 计算 C = F^T @ F
    C = F.transpose(-1, -2) @ F  # (B, N, 3, 3)
    
    # 计算第一个不变量: I_1 = trace(C)
    I_1 = torch.diagonal(C, dim1=-2, dim2=-1).sum(-1)  # (B, N)
    
    # 计算偏(deviatoric)第一不变量: I_1_bar = J_det^(-2/3) * I_1
    I_1_bar = torch.pow(J_det_stable, -2.0 / 3.0) * I_1
    
    # Neo-Hookean 偏能量密度: Ψ_dev = (μ/2) * (I_1_bar - 3)
    # (B, N)
    W_dev = 0.5 * mu * (I_1_bar - 3.0)
    
    # 返回所有点和批次的平均能量
    return W_dev.mean()

def incompressibility_penalty(J):
    I = torch.eye(3, device=J.device).view(1,1,3,3)
    F = I + J
    detF = torch.linalg.det(F)                 # (B,N)
    return ((detF - 1.0)**2).mean()

def physics_loss(src_xyz, pred_xyz, E_kPa, nu, k=16, reg=1e-4, w_energy=1.0, w_incomp=0.5):
    u = pred_xyz - src_xyz
    idx = knn_indices(src_xyz, k=k)
    J = estimate_jacobian(src_xyz, u, idx, reg=reg)
    # E,nu -> Lamé
    mu  = E_kPa / (2.0 * (1.0 + nu))
    lam = (E_kPa * nu) / ((1.0 + nu) * (1.0 - 2.0*nu))
    mu  = mu.view(-1,1,1); lam = lam.view(-1,1,1)
    L_e = linear_elastic_energy(J, mu, lam)
    L_i = incompressibility_penalty(J)
    return w_energy * L_e + w_incomp * L_i
