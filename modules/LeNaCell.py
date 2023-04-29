from .model_utils import LayerNorm, FCN
import torch.nn as nn
import torch.nn.functional as F
import torch, math

def focal_att(attn, bs, queryL, sourceL):
    """
      attn:[bs, 1, m]
    """
    xi = attn.unsqueeze(-1).contiguous()
    xk = attn.unsqueeze(2).contiguous()
    xk_confi = torch.sqrt(xk)
    xi = xi.view(bs*queryL, sourceL, 1)
    xk = xk.view(bs*queryL, 1, sourceL)
    xk_confi = xk_confi.view(bs*queryL, 1, sourceL)
    term1 = torch.bmm(xi, xk_confi)
    term2 = xk * xk_confi
    funcF = torch.sum(term1 - term2, dim=-1)
    funcF = funcF.view(bs, queryL, sourceL)
    fattn = torch.where(funcF > 0, torch.ones_like(attn), torch.zeros_like(attn))
    return fattn

def dym_conv1d(v_emb, k_size, dila, pads):
    
    '''
       v_emb: [bs, dim, num_r]
    '''
    bs, num_hid, num_r = v_emb.size(0), v_emb.size(1), v_emb.size(2)
    filter_fw = torch.ones(k_size).float().view(1, 1, k_size).cuda()
    x = v_emb.reshape(-1, num_r).unsqueeze(1)  # (bs*dim, 1, num_r)
    ms_v_emb = F.conv1d(x, filter_fw, dilation=dila, padding=pads).view(bs, -1, num_r)  # multi-scale v_emb
    ms_v_emb = ms_v_emb.permute(0, 2, 1)  # (bs, num_r, dim)
    return ms_v_emb

class CM(nn.Module):
    '''
       Compositionality Modeling
    '''

    def __init__(self, embed_size):
        super(CM, self).__init__()
        self.k_size = [1, 3, 3, 3, 5, 5, 5]
        self.dila = [1, 1, 2, 3, 1, 2, 3]
        self.pads = [0, 1, 2, 3, 2, 4, 6]
        self.kernel_nums = len(self.k_size)
        self.convs_fc = nn.Linear(embed_size, 1)

    def forward(self, v_emb):
        bs, num_r = v_emb.size(0), v_emb.size(1)
        x = v_emb.transpose(1, 2)
        xx = [F.relu(dym_conv1d(x, self.k_size[i], self.dila[i], self.pads[i])) for i in range(self.kernel_nums)]
        xx = torch.cat(xx, dim=1)
        xx = xx.view(bs, num_r, self.kernel_nums, -1)
        mvs_att = self.convs_fc(xx).squeeze(-1)
        mvs_att = (mvs_att).softmax(dim=-1).unsqueeze(dim=-1)
        att_x = (mvs_att * xx).sum(dim=2)
        return att_x


class HA(nn.Module):
    """
       High-order Aggregation
    """
    def __init__(self, num_hid=512, dropout=0.1, glimpse=2):
        super(HA, self).__init__()
        self.head_size = num_hid // glimpse
        self.num_hid = num_hid
        self.Ws1 = nn.Linear(num_hid, num_hid)
        self.Ws2 = nn.Linear(num_hid, num_hid)
        self.Ws3 = nn.Linear(num_hid, num_hid)
        self.fc = nn.Linear(num_hid, num_hid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, v_emb):
        bs = v_emb.size(0)
        k = self.Ws1(v_emb).view(bs, -1, self.glimpse, self.head_size).transpose(1, 2)
        q = self.Ws2(v_emb).view(bs, -1, self.glimpse, self.head_size).transpose(1, 2)
        v = self.Ws3(v_emb).view(bs, -1, self.glimpse, self.head_size).transpose(1, 2)
        atted = self.fc(self.att(v, k, q).transpose(1, 2).contiguous().view(bs, -1, self.num_hid))
        return atted

    def att(self, v, k, q):
        dk = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)
        norm_scores = self.dropout(F.softmax(scores, dim=-1))
        v_o = torch.matmul(norm_scores, v)
        return v_o

class CoAtt(nn.Module):
    def __init__(self, num_hid, dropout=0.1, glimpse=2):
        super(CoAtt, self).__init__()
        self.head_size = num_hid // glimpse
        self.num_hid = num_hid
        self.Wq = nn.Linear(num_hid, num_hid)
        self.Wv = nn.Linear(num_hid, num_hid)
        self.W = nn.Linear(num_hid, num_hid)
        self.fc = nn.Linear(num_hid, num_hid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, v_emb, q_emb):
        bs = v_emb.size(0)
        k = self.Wq(q_emb).view(bs, -1, self.glimpse, self.head_size).transpose(1, 2)
        q = self.Wv(v_emb).view(bs, -1, self.glimpse, self.head_size).transpose(1, 2)
        v = self.W(q_emb).view(bs, -1, self.glimpse, self.head_size).transpose(1, 2)
        atted, att_map = self.att(v, k, q)
        atted = self.fc(atted.transpose(1, 2).contiguous().view(bs, -1, self.num_hid))
        return atted, att_map

    def att(self, value, key, query):
        dk = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dk)
        att_map = F.softmax(scores, dim=-1)
        att = F.softmax(scores, dim=2)
        att = att.sum(1) / self.glimpse
        att_map = self.dropout(att_map)
        return torch.matmul(att_map, value), att

class MDU(nn.Module):
    def __init__(self, num_hid, dropout=0.1, alpha=8.0, gamma=0.98):
        super(MDU, self).__init__()
        self.fc = FCN(num_hid)
        self.norm1 = LayerNorm(num_hid)
        self.norm2 = LayerNorm(num_hid)
        self.dropout = nn.Dropout(dropout)
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, v, q, att_map, ru):
        mm = self.norm1(v + q)
        rg = torch.sum(att_map, dim=-1)  # [bs, m]
        ru = (ru * self.gamma + rg) if ru.sum() == 0 else ((ru * self.gamma + rg) / (1 + self.gamma)); ru = ru.unsqueeze(dim=1)  # (bs, 1, m)
        rm = focal_att(ru, v.size(0), 1, 36).squeeze(dim=1); ru = ru.squeeze(dim=1)  # (bs, m)
        ru = self.alpha * ru * rm + (1 / self.alpha) * ru * (1 - rm)
        w = self.alpha * rg * rm + (1 / self.alpha) * rg * (1 - rm)
        mm = w.unsqueeze(dim=2) * mm
        mm = self.norm2(mm + self.dropout(self.fc(mm)))
        return mm, ru


# Composed Semantic-Aware Unit
class CAU(nn.Module):

    def __init__(self, num_hid):
        super(CAU, self).__init__()
        self.cm = CM(num_hid)
        self.ha = HA(num_hid)

    def forward(self, x):
        cm_x = self.cm(x)
        ha_x = self.ha(cm_x)
        return ha_x

# Focal Attention Unit
class FAU(nn.Module):

    def __init__(self, num_hid):
        super(FAU, self).__init__()
        self.coat = CoAtt(num_hid)
        self.mdu = MDU(num_hid)

    def forward(self, v, q, ru):
        g_q, att_map = self.coat(v, q)
        mdu_x = self.mdu(v, g_q, att_map, ru)
        return mdu_x

class LENACell(nn.Module):
    def __init__(self, num_hid, dropout=0.1):
        super(LENACell, self).__init__()
        self.fau = FAU(num_hid)
        self.cau = CAU(num_hid)
        self.fc = nn.Linear(num_hid, num_hid)
        self.norm = LayerNorm(num_hid)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, v, q, ru):
        v_f, r = self.fau(v, q, ru) + v
        v_c = self.cau(v_f) + v_f
        return v_c, ru
