import os
import argparse
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('.')
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import utils.utils as utils
import utils.config as config
from train import train, evaluate
import modules.base_model as base_model
from utils.dataset import Dictionary, VQAFeatureDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=13)
    parser.add_argument('--num_hid', type=int, default=512)
    parser.add_argument('--model', type=str, default='lena')
    parser.add_argument('--name', type=str, default='lena_model.pth')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--tfidf', action='store_true', help='tfidf word embedding?')
    parser.add_argument('--op', type=str, default='')
    parser.add_argument('--gamma', type=int, default=8, help='glimpse')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--test', dest='test_only', action='store_true')
    parser.add_argument('--T_ctrl', type=int, default=8)
    parser.add_argument("--gpu", type=str, default='0', help='gpu card ID')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seed = 1111
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    save_path = os.path.join('logs', args.name)
    if args.test_only:
        args.resume = True
    if args.resume and not args.name:
        raise ValueError("Resuming requires folder name!")
    if args.resume:
        logs = torch.load(save_path)
        print("loading logs from {}".format(save_path))

    # Datasets
    dictionary = Dictionary.load_from_file(config.dict_path)
    
    if args.test_only:
        eval_dset = VQAFeatureDataset('test', dictionary)
    else:
        train_dset = VQAFeatureDataset('train', dictionary)
        eval_dset = VQAFeatureDataset('val', dictionary)

    if config.train_set == 'train+val' and not args.test_only:
        train_dset = train_dset + eval_dset
        eval_dset = VQAFeatureDataset('test', dictionary)

    # Model
    constructor = 'build_{}'.format(args.model)
    model = getattr(base_model, constructor)(train_dset, args).cuda()
    tfidf = None
    weights = None
    model.w_emb.init_embedding(config.glove_embed_path, tfidf, weights)
    model.c_emb.init_embedding(config.class_embed_path)
    model.attr_emb.init_embedding(config.attr_embed_path)
    model = nn.DataParallel(model).cuda()

    # Optimizer
    lr_default = 1e-4
    lr_decay_step = 2
    lr_decay_rate = .2
    lr_decay_epochs = range(10, 20, lr_decay_step)
    gradual_warmup_steps = [0.25 * lr_default, 0.5 * lr_default, 0.75 * lr_default, lr_default]
    optim = torch.optim.Adam(list(filter(lambda p: p.requires_grad, model.parameters())), lr=lr_default, betas=(0.9, 0.999), eps=1e-8)

    eval_score, best_val_score, start_epoch, best_epoch = 0.0, 0.0, 0, 0
    tracker = utils.Tracker()
    if args.resume:
        model.load_state_dict(logs['model_state'])
        optim.load_state_dict(logs['optim_state'])
        start_epoch = logs['epoch']
        best_val_score = logs['best_val_score']

    # Start to train
    eval_loader = DataLoader(eval_dset, args.batch_size, shuffle=False, num_workers=4)
    if not args.test_only:
        train_loader = DataLoader(train_dset, args.batch_size, shuffle=True, num_workers=4)
        for epoch in range(start_epoch, args.epochs):
            print("training epoch {:03d}".format(epoch))

            if epoch < len(gradual_warmup_steps):
                optim.param_groups[0]['lr'] = gradual_warmup_steps[epoch]
            elif epoch in lr_decay_epochs:
                optim.param_groups[0]['lr'] *= lr_decay_rate
            train(model, optim, train_loader, tracker)
            
            if not (config.train_set == 'train+val' and epoch in range(args.epochs-3)):
                write = True if config.train_set == 'train+val' else False
                print("validating after epoch {:03d}".format(epoch))
                model.train(False)
                eval_score, bound = evaluate(model, eval_loader, epoch, write=write)
                model.train(True)
                print("eval score: {:.2f} ({:.2f})\n".format(100 * eval_score, 100 * bound))
            results = {
                'epoch': epoch+1,
                'best_val_score': best_val_score,
                'model_state': model.state_dict(),
                'optim_state': optim.state_dict()
            }
            torch.save(results, save_path)
            if eval_score > best_val_score:
                best_val_score = eval_score
                best_epoch = epoch
        print("best accuracy {:.2f} on epoch {:03d}".format(100 * best_val_score, best_epoch))
    else:
        evaluate(model, eval_loader, write=True)
