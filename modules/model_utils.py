import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self, num_hid, dropout=0.1):
        super(FCN, self).__init__()
        self.mlp = MLP(in_dim=num_hid, mid_dim=4*num_hid, out_dim=num_hid, dropout_r=dropout, use_relu=True)
    def forward(self, x):
        return self.mlp(x)


class SA(nn.Module):
    def __init__(self, num_hid, dropout=0.1):
        super(SA, self).__init__()
        self.mhatt = MHAtt(num_hid)
        self.mlp = FCN(num_hid, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = LayerNorm(num_hid)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(num_hid)

    def forward(self, x):
        x_tmp = self.mhatt(x, x, x)
        x = self.norm1(x + self.dropout1(x_tmp))
        x = self.norm2(x + self.dropout2(self.mlp(x)))
        return x


class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu
        self.linear = nn.Linear(in_size, out_size)
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)
        if self.use_relu:
            x = self.relu(x)
        if self.dropout_r > 0:
            x = self.dropout(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()
        self.fc1 = FC(in_dim, mid_dim, dropout_r=dropout_r, use_relu=use_relu)
        self.fc2 = nn.Linear(mid_dim, out_dim)

    def forward(self, x):
        return self.fc2(self.fc1(x))

class Att(nn.Module):
    def __init__(self, num_hid=512, out_dim=1, dropout=0.1):
        super(Att, self).__init__()
        self.mlp = MLP(in_dim=num_hid, mid_dim=num_hid, out_dim=out_dim, dropout_r=dropout, use_relu=True)
        self.linear = nn.Linear(num_hid, 2 * num_hid)

    def forward(self, x):
        att = self.mlp(x)
        att = F.softmax(att, dim=1)
        x_atted = torch.sum(att * x, dim=1)
        x_atted = self.linear(x_atted)
        return x_atted


class MHAtt(nn.Module):
    def __init__(self, num_hid, dropout=0.1, glimpse=2):
        super(MHAtt, self).__init__()
        self.head_size = num_hid // glimpse
        self.num_hid = num_hid
        self.linear_v = nn.Linear(num_hid, num_hid)
        self.linear_k = nn.Linear(num_hid, num_hid)
        self.linear_q = nn.Linear(num_hid, num_hid)
        self.linear_merge = nn.Linear(num_hid, num_hid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, v, k, q):
        bs = q.size(0)
        v = self.linear_v(v).view(bs, -1, self.glimpse, self.head_size).transpose(1, 2)
        k = self.linear_k(k).view(bs, -1, self.glimpse, self.head_size).transpose(1, 2)
        q = self.linear_q(q).view(bs, -1, self.glimpse, self.head_size).transpose(1, 2)
        atted = self.att(v, k, q)
        atted = atted.transpose(1, 2).contiguous().view(bs, -1, self.num_hid)
        atted = self.linear_merge(atted)
        return atted

    def att(self, value, key, query):
        dk = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dk)  # [bs, glimpse, num_feat, q_len]
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)  # [bs, glimpse, num_feat, q_len]
        return torch.matmul(att_map, value)

class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def expand_tensor(tensor, size, dim=1):
    if size == 1 or tensor is None:
        return tensor
    tensor = tensor.unsqueeze(dim)
    tensor = tensor.expand(list(tensor.shape[:dim]) + [size] + list(tensor.shape[dim + 1:])).contiguous()
    tensor = tensor.view(list(tensor.shape[:dim - 1]) + [-1] + list(tensor.shape[dim + 1:]))
    return tensor


def load_ids(path):
    with open(path, 'r') as fid:
        lines = [int(line.strip()) for line in fid]
    return lines


def load_lines(path):
    with open(path, 'r') as fid:
        lines = [line.strip() for line in fid]
    return lines


def load_vocab(path):
    vocab = ['.']
    with open(path, 'r') as fid:
        for line in fid:
            vocab.append(line.strip())
    return vocab


def clip_gradient(optimizer, model, grad_clip_type, grad_clip):
    if grad_clip_type == 'Clamp':
        for group in optimizer.param_groups:
            for param in group['params']:
                param.grad.data.clamp_(-grad_clip, grad_clip)
    elif grad_clip_type == 'Norm':
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    else:
        raise NotImplementedError


def decode_sequence(vocab, seq):
    N, T = seq.size()
    sents = []
    for n in range(N):
        words = []
        for t in range(T):
            ix = seq[n, t]
            if ix == 0:
                break
            words.append(vocab[ix])
        sent = ' '.join(words)
        sents.append(sent)
    return sents


def fill_with_neg_inf(t):
    return t.float().fill_(float(-1e9)).type_as(t)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def bmul(inputs1, inputs2):
    b = inputs1.size()[0]
    m = []
    for i in range(b):
        m.append(inputs1[i] * inputs2[i])
    outputs = torch.stack(m, dim=0)
    return outputs


def split_filepath(filename):
    absname = os.path.abspath(filename)
    dirname, basename = os.path.split(absname)
    split_tmp = basename.rsplit('.', maxsplit=1)
    if len(split_tmp) == 2:
        rootname, extname = split_tmp
    elif len(split_tmp) == 1:
        rootname = split_tmp[0]
        extname = None
    else:
        raise ValueError("programming error!")
    return dirname, rootname, extname


class SimpleClassifier(nn.Module):
    def __init__(self, in_features, out_features, seed=None, p=None, af=None, dim=None):
        super(SimpleClassifier, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.p = p
        self.af = af
        self.dim = dim
        if seed:
            torch.manual_seed(seed)
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x):
        if x.size()[-1] != self.in_features:
            raise ValueError(
                '[error] putils.Linear(%s, %s): last dimension of input(%s) should equal to in_features(%s)' %
                (self.in_features, self.out_features, x.size(-1), self.in_features))
        if self.p:
            x = F.dropout(x, p=self.p, training=self.training)
        x = self.linear(x)
        if self.af:
            if self.af == 'softmax':
                x = getattr(F, self.af)(x, dim=self.dim)
            else:
                x = getattr(F, self.af)(x)
        return x

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)
