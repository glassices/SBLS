import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import multiprocessing

import network
import logging

import os
import argparse
import hashlib
import lmdb
import pickle
import time
import sys
import math
import queue
import threading
import random
import numpy
import glob

from graphviz import Digraph

from utils import progress_bar, adjust_learning_rate, make_log_dir
from dataset import MyDataset, collate_fn
from utils import push_down, push_up, insert, remove

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AlphaPlan')
    parser.add_argument('--rng-seed', default=58, type=int, help='random number generator')
    parser.add_argument('--num-workers', default=2, type=int, help='zzz')
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    parser.add_argument('--weight-decay', default=0.0001, type=float, help='weight decay (default=1e-4)')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size per GPU')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate per sample')
    parser.add_argument('--num-epochs', default=90, type=int, help='')
    parser.add_argument('--tag', default=None, type=str, help='tags for multiple runs, e.g., lr')
    parser.add_argument('--warm-up-step', default=1000, type=int)
    parser.add_argument('--save-every', default=1, type=int)
    parser.add_argument('--batch-per-epoch', default=1000, type=int)

    args = parser.parse_args()

    torch.autograd.set_detect_anomaly(True)

    log_dir = make_log_dir(args.tag)

    logFormatter = logging.Formatter("%(asctime)s  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)

    fileHandler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    rootLogger.info(sys.argv)

    writer = SummaryWriter(log_dir = os.path.join(log_dir, 'tb'))


    torch.manual_seed(args.rng_seed)
    torch.cuda.manual_seed(args.rng_seed)
    torch.backends.cudnn.benchmark = True

    net = network.Network(hidden_dim = 256, nblock = 20)
    epoch = 0
    train_steps = 0
    if args.resume:
        rootLogger.info('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.resume)
        epoch = checkpoint['epoch_done']
        net.load_state_dict(checkpoint['model_state_dict'])
        train_steps = checkpoint['train_steps']

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net.cuda()

    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr = args.lr, momentum = 0.9, weight_decay = args.weight_decay)
    
    for epoch in range(epoch, args.num_epochs):
        rootLogger.info("=> start epoch {}...".format(epoch + 1))

        dataset = MyDataset()
        dataloader = DataLoader(dataset, batch_size = args.batch_size,
                                shuffle = False, drop_last = False,
                                collate_fn = collate_fn,
                                num_workers = args.num_workers, pin_memory = False)

        train_loss = 0.0
        net.train()
        for batch_idx, (data, label, ns, mask) in enumerate(dataloader):
            if batch_idx == args.batch_per_epoch:
                break

            if train_steps < args.warm_up_step:
                adjust_learning_rate(optimizer, args.lr * (train_steps + 1) / args.warm_up_step)

            if use_cuda:
                data = data.cuda()
                label = label.cuda()

            optimizer.zero_grad()
            output = net(data, ns)
            output = torch.sigmoid(output)
            assert (output[~mask] == 0.5).all()
            loss = criterion(output[mask], label[mask])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            progress_bar(batch_idx, args.batch_per_epoch, 'Loss :%.4f'
                    % (train_loss / (batch_idx + 1),))
            train_steps += 1

        writer.add_scalar('train/loss', train_loss / args.batch_per_epoch, epoch + 1)

        if (epoch + 1) % args.save_every == 0:
            rootLogger.info("==> Saving checkpoint")
            state = {
                'epoch_done': epoch + 1,
                'model_state_dict': net.state_dict(),
                'train_steps': train_steps,
            }

            ckpt_dir = os.path.join(log_dir, 'ckpt')
            if not os.path.isdir(ckpt_dir):
                os.mkdir(ckpt_dir)
            torch.save(state, os.path.join(ckpt_dir, '{}'.format(epoch + 1)))
            torch.save(state, os.path.join(ckpt_dir, 'latest'))


