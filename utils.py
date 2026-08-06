import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------- 工具函数: KNN / 采样分组 -----------------------------
def knn_point(k, xyz_src, xyz_query):
    """
    为 xyz_query 中的每个点在 xyz_src 中找 k 个最近邻
    xyz_src:   (B, N, 3)
    xyz_query: (B, M, 3)
    return:
      val: (B, M, k)  欧氏距离平方
      idx: (B, M, k)  索引(指向 xyz_src 的 N 维)
    """
    # 使用 torch.cdist(query, src) 更直观 -> (B, M, N)
    dist_sq = torch.cdist(xyz_query, xyz_src) ** 2
    val, idx = torch.topk(dist_sq, k, dim=2, largest=False)
    return val, idx  # idx 索引的是 xyz_src 的第2维(N)

def sample_and_group(xyz, pts, npoint, nsample):
    """
    PointNet++ 风格的采样+分组(简化版)
    xyz: (B, N, 3)
    pts: (B, N, C)
    npoint: 采样中心点数(允许 > N, 会有重复)
    nsample: 每个组的邻居数
    return:
      new_xyz: (B, npoint, 3)
      new_pts: (B, npoint, nsample, 2C)  [局部特征 + 中心特征]
    """
    B, N, C = pts.shape
    device = xyz.device
    # 均匀随机采样中心索引(允许重复)
    indices = torch.randint(0, N, (B, npoint), dtype=torch.long, device=device)  # (B, npoint)
    new_xyz = torch.gather(xyz, 1, indices.unsqueeze(-1).expand(-1, -1, 3))
    new_pts_centers = torch.gather(pts, 1, indices.unsqueeze(-1).expand(-1, -1, C))

    # KNN 分组: 在原始 xyz (N) 上, 为每个 center (npoint) 找 nsample 邻居
    _, idx = knn_point(nsample, xyz, new_xyz)  # idx: (B, npoint, nsample)

    # 从 pts 中按 idx 收集邻域特征
    pts_expanded = pts.unsqueeze(1).expand(-1, npoint, -1, -1)                         # (B, npoint, N, C)
    grouped_pts = torch.gather(pts_expanded, 2, idx.unsqueeze(-1).expand(-1,-1,-1,C))  # (B, npoint, nsample, C)

    # 构造局部相对特征 + 中心重复
    new_pts_centers_exp = new_pts_centers.unsqueeze(2)                                  # (B, npoint, 1, C)
    grouped_local = grouped_pts - new_pts_centers_exp                                   # (B, npoint, nsample, C)
    new_pts_centers_tiled = new_pts_centers_exp.expand(-1, -1, nsample, -1)            # (B, npoint, nsample, C)
    new_pts = torch.cat([grouped_local, new_pts_centers_tiled], dim=-1)                 # (B, npoint, nsample, 2C)
    return new_xyz, new_pts


# ----------------------------- 模型 -----------------------------
class LBR(nn.Module):
    def __init__(self, in_features, out_features, use_bias=True, leaky_alpha=0.0):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=use_bias)
        self.act = nn.LeakyReLU(leaky_alpha) if leaky_alpha > 0 else nn.ReLU()
    def forward(self, x): return self.act(self.lin(x))

class SelfAttention(nn.Module):
    def __init__(self, in_features, inter_features):
        super().__init__()
        self.w_q = nn.Linear(in_features, inter_features, bias=False)
        self.w_k = nn.Linear(in_features, inter_features, bias=False)
        self.w_v = nn.Linear(in_features, in_features,   bias=False)
        # 共享权重(可选): 保持与原实现一致
        self.w_k.weight = self.w_q.weight
        self.lbr = LBR(in_features, in_features, use_bias=True)
    def forward(self, x):  # x: (B,N,C)
        q = self.w_q(x); k = self.w_k(x); v = self.w_v(x)
        energy = torch.bmm(q, k.transpose(1, 2))                   # (B,N,N)
        attn = F.softmax(energy, dim=1)
        attn = attn / (1e-9 + attn.sum(dim=2, keepdim=True))       # L1 归一
        x_r = torch.bmm(attn, v)                                   # (B,N,C)
        x_r = self.lbr(x - x_r)
        return x + x_r

class CrossAttention(nn.Module):
    def __init__(self, enc_features, dec_features):
        super().__init__()
        self.w_q = nn.Linear(dec_features, enc_features // 4, bias=False)
        self.w_k = nn.Linear(enc_features, enc_features // 4, bias=False)
        self.w_v = nn.Linear(enc_features, dec_features, bias=False)
        self.lbr = LBR(dec_features, dec_features, use_bias=True)
    def forward(self, enc_tensor, dec_tensor):
        # enc: K/V (B, N_enc, C_enc), dec: Q (B, N_dec, C_dec)
        q = self.w_q(dec_tensor)
        k = self.w_k(enc_tensor)
        v = self.w_v(enc_tensor)
        energy = torch.bmm(q, k.transpose(1, 2))                   # (B,N_dec,N_enc)
        attn = F.softmax(energy, dim=1)
        attn = attn / (1e-9 + attn.sum(dim=2, keepdim=True))
        x_r = torch.bmm(attn, v)                                   # (B,N_dec,C_dec)
        x_r = self.lbr(dec_tensor - x_r)
        return dec_tensor + x_r


# ----------------------------- PCTEncoder(鲁棒 N) -----------------------------
class PCTEncoder(nn.Module):
    """
    简化/鲁棒: 适配任意 N
    阶段1: npoint1 = min(4096, N)(不足则允许重复采样)
    阶段2: npoint2 = min(2048, npoint1)
    """
    def __init__(self):
        super().__init__()
        self.lbr1 = LBR(3, 64, use_bias=False)
        self.lbr2 = LBR(64, 128, use_bias=False)
        self.lbr_sg1 = LBR(128*2, 512, use_bias=False)
        self.lbr_sg2 = LBR(512*2, 1024, use_bias=False)

        self.sa1 = SelfAttention(1024, 1024//4)
        self.sa2 = SelfAttention(1024, 1024//4)
        self.sa3 = SelfAttention(1024, 1024//4)
        self.sa4 = SelfAttention(1024, 1024//4)

        self.lbr_out1 = LBR(1024*5, 2048, use_bias=False, leaky_alpha=0.2)

        self.sa5 = SelfAttention(2048, 2048//4)
        self.sa6 = SelfAttention(2048, 2048//4)
        self.sa7 = SelfAttention(2048, 2048//4)
        self.sa8 = SelfAttention(2048, 2048//4)

        self.lbr_out2 = LBR(2048*4, 4096, use_bias=False, leaky_alpha=0.2)

    def forward(self, xyz):  # (B,N,3)
        B, N, _ = xyz.shape
        pts = self.lbr2(self.lbr1(xyz))                # (B,N,128)

        npoint1 = min(4096, N)
        new_xyz_1, new_feat_1 = sample_and_group(xyz, pts, npoint=npoint1, nsample=32)
        x = self.lbr_sg1(new_feat_1)                  # (B,npoint1,ns,512)
        x = torch.max(x, dim=2)[0]                    # (B,npoint1,512)

        npoint2 = min(2048, npoint1)
        new_xyz_2, new_feat_2 = sample_and_group(new_xyz_1, x, npoint=npoint2, nsample=32)
        x = self.lbr_sg2(new_feat_2)                  # (B,npoint2,ns,1024)
        x = torch.max(x, dim=2)[0]                    # (B,npoint2,1024)

        x1 = self.sa1(x); x2 = self.sa2(x1); x3 = self.sa3(x2); x4 = self.sa4(x3)
        x = torch.cat([x1, x2, x3, x4, x], dim=-1)    # (B,npoint2,1024*5)
        x = self.lbr_out1(x)                          # (B,npoint2,2048)

        x1 = self.sa5(x); x2 = self.sa6(x1); x3 = self.sa7(x2); x4 = self.sa8(x3)
        x = torch.cat([x1, x2, x3, x4], dim=-1)       # (B,npoint2,2048*4)
        x = self.lbr_out2(x)                          # (B,npoint2,4096)

        g = torch.max(x, dim=1, keepdim=True)[0]      # (B,1,4096)
        return g


# ----------------------------- 文本与数值条件 -----------------------------
class TextEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased', out_dim=4096, local_files_only=False):
        super().__init__()
        from transformers import BertModel
        self.bert_model = BertModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.bert_model.eval()
        for p in self.bert_model.parameters(): p.requires_grad = False
        self.proj = nn.Linear(self.bert_model.config.hidden_size, out_dim)
        self.act = nn.ReLU()
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.pooler_output  # (B,768)
        return self.act(self.proj(cls)).unsqueeze(1)  # (B,1,4096)

class NumericBiomechEncoder(nn.Module):
    def __init__(self, out_dim=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.ReLU(),
            nn.Linear(128, 512), nn.ReLU(),
            nn.Linear(512, out_dim), nn.ReLU()
        )
    def forward(self, E_kPa, nu):                    # (B,1) (B,1)
        x = torch.cat([E_kPa, nu], dim=1)            # (B,2)
        return self.net(x).unsqueeze(1)              # (B,1,4096)


# ----------------------------- 结构对齐版解码器 -----------------------------
class PCTDecoderAligned(nn.Module):
    def __init__(self, input_feat_dim=4096, src_in_dim=3, dec_dim=1024):
        super().__init__()
        self.src_embed = nn.Linear(src_in_dim, dec_dim, bias=False)
        self.ca1 = CrossAttention(enc_features=input_feat_dim, dec_features=dec_dim)
        self.ca2 = CrossAttention(enc_features=input_feat_dim, dec_features=dec_dim)
        self.ca3 = CrossAttention(enc_features=input_feat_dim, dec_features=dec_dim)
        self.ca4 = CrossAttention(enc_features=input_feat_dim, dec_features=dec_dim)
        self.post = nn.Sequential(
            LBR(dec_dim*5, 256, use_bias=False),
            LBR(256, 128, use_bias=False),
            LBR(128, 64, use_bias=False, leaky_alpha=0.2),
            nn.Linear(64, 3)
        )
    def forward(self, input_feats, src_xyz):         # (B,1,4096), (B,N,3)
        B, N, _ = src_xyz.shape
        m_feats = input_feats.expand(-1, N, -1)      # (B,N,4096)
        x = self.src_embed(src_xyz)                  # (B,N,1024)
        x1 = self.ca1(m_feats, x)
        x2 = self.ca2(m_feats, x1)
        x3 = self.ca3(m_feats, x2)
        x4 = self.ca4(m_feats, x3)
        x_cat = torch.cat([x1, x2, x3, x4, x], dim=-1)  # (B,N,1024*5)
        delta = self.post(x_cat)                        # (B,N,3)
        return delta