import os
import json

import torch
import torch.nn as nn
from tqdm import tqdm
import utils.utils as utils
import utils.config as config


def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    _, log_index = logits.max(dim=1, keepdim=True)
    scores = labels.gather(dim=1, index=log_index)
    return scores


def saved_for_eval(dataloader, results, question_ids, answer_preds):
    """ Save as a format accepted by the evaluation server. """
    _, answer_ids = answer_preds.max(dim=1)
    answers = [dataloader.dataset.label2ans[i] for i in answer_ids]
    for q, a in zip(question_ids, answers):
        entry = {
            'question_id': q.item(),
            'answer': a,
        }
        results.append(entry)
    return results


def train(model, optim, train_loader, tracker):
    loader = tqdm(train_loader, ncols=0)
    loss_trk = tracker.track('loss', tracker.MovingMeanMonitor(momentum=0.99))
    acc_trk = tracker.track('acc', tracker.MovingMeanMonitor(momentum=0.99))

    for v, b, c, attr, q, a, q_id in loader:
        v = v.cuda()
        b = b.cuda()
        q = q.cuda()
        a = a.cuda()
        c = c.cuda()
        attr = attr.cuda()

        pred  =  model(v, b, c, attr, q, a)
        loss = instance_bce_with_logits(pred, a)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optim.step()
        optim.zero_grad()
        batch_score = compute_score_with_logits(pred, a.data)

        fmt = '{:.4f}'.format
        loss_trk.append(loss.item())
        acc_trk.append(batch_score.mean())
        loader.set_postfix(loss=fmt(loss_trk.mean.value), acc=fmt(acc_trk.mean.value))


def evaluate(model, dataloader, epoch=0, write=False):
    score = 0
    upper_bound = 0
    results = [] # saving for evaluation
    for v, b, c, attr, q, a, q_id in tqdm(dataloader, leave=False):
        v = v.cuda()
        b = b.cuda()
        q = q.cuda()
        c = c.cuda()
        attr = attr.cuda()

        pred  = model(v, b, c, attr, q, None)
        if write:
            results = saved_for_eval(dataloader, results, q_id, pred)
        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score
        upper_bound += (a.max(1)[0]).sum()

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)

    if write:
        print("saving prediction results to disk...")
        result_file = 'vqa_{}_{}_{}_{}_results.json'.format(config.task, config.test_split, config.version, epoch)
        with open(result_file, 'w') as fd:
            json.dump(results, fd)
    return score, upper_bound
