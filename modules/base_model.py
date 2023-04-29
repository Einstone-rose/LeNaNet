import torch
import torch.nn as nn
from .model_utils import LayerNorm, SA, Att
from modules.language_model import WordEmbedding, QuestionEmbedding, ClassEmbedding
from .LeNaCell import LENACell

class LenaNet(nn.Module):
    def __init__(self, w_emb, q_emb, c_emb, v_enc, q_enc, att_v, att_q, lenaCells, classifier, norm):
        super(LenaNet, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.c_emb = c_emb
        self.att_v = att_v
        self.att_q = att_q
        self.v_enc = v_enc
        self.q_enc = q_enc
        self.lenaCells = lenaCells
        self.classifier = classifier
        self.norm = norm

    def forward(self, v, b, c, attr, q, labels):
        """
           v: (bs, num_r, dv)
           c: (bs, num_r, dc)
           q: (bs, q_len)
        """
        # Visual and Textual Encoder
        c_emb = self.c_emb(c)
        v = torch.cat([v, c_emb], dim=-1)
        v_emb = self.v_enc(v)
        w_emb = self.w_emb(q)
        q_emb = self.q_emb.forward_all(w_emb)
        for enc in self.q_enc:
            q_emb = enc(q_emb)

        # LeNa Network
        ru = torch.zeros(v_emb.size(0), v_emb.size(1)).cuda()
        for lenaCell in self.lenaCells:
            v_emb, ru = lenaCell(v_emb, q_emb, ru)

        # Answer Reasoning
        q_feat = self.att_q(q_emb)
        v_feat = self.att_v(v_emb)
        co_feat = self.norm(q_feat + v_feat)
        logits = self.classifier(co_feat)
        return logits


def build_lena(dataset, arg):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, arg.op)
    q_emb = QuestionEmbedding(300 if 'c' not in arg.op else 600, arg.num_hid, 1, False, .0)
    c_emb = ClassEmbedding(1600, 300, .0)
    v_enc = nn.Linear(4*arg.num_hid + 300, arg.num_hid)
    q_enc = nn.ModuleList([SA(arg.num_hid) for _ in range(arg.T)])
    lenaCells = nn.ModuleList([LENACell(arg.num_hid) for _ in range(arg.T)])
    att_v = Att(arg.num_hid)
    att_q = Att(arg.num_hid)
    classifier = nn.Linear(2*arg.num_hid, dataset.num_ans_candidates)
    norm = LayerNorm(2*arg.num_hid)
    return LenaNet(w_emb, q_emb, c_emb, v_enc, q_enc, att_v, att_q, lenaCells, classifier, norm)



