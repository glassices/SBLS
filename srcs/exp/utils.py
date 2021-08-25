'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch


# specify tag if you want to run the same experiment multiple times, e.g., with different lr
def make_log_dir(tag = None):
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(exp_dir, os.pardir))
    proj_dir = os.path.abspath(os.path.join(src_dir, os.pardir))
    log_dir = os.path.join(proj_dir, 'logs')
    if not os.path.exists(os.path.join(log_dir)):
        os.mkdir(log_dir)
    exp_name = os.path.basename(exp_dir)
    log_dir = os.path.join(log_dir, exp_name)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    link_path = os.path.join(exp_dir, 'logs')
    if os.path.islink(link_path):
        os.unlink(link_path)
    os.symlink(log_dir, link_path)
    if tag is None:
        log_dir = os.path.join(log_dir, 'default')
    else:
        log_dir = os.path.join(log_dir, 'tag_{}'.format(tag))
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    return log_dir


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

def push_down(heap, i):
    while (i << 1) + 1 < len(heap):
        m = (i << 1) + 1
        if m + 1 < len(heap) and heap[m + 1].key < heap[m].key:
            m = m + 1
        if heap[m].key < heap[i].key:
            heap[m], heap[i] = heap[i], heap[m]
            heap[i].status = i
            heap[m].status = m
            i = m
        else:
            return

def push_up(heap, i):
    while i:
        m = i - 1 >> 1
        if heap[i].key < heap[m].key:
            heap[i], heap[m] = heap[m], heap[i]
            heap[i].status = i
            heap[m].status = m
            i = m
        else:
            return

def insert(heap, node):
    node.status = len(heap)
    heap.append(node)
    push_up(heap, len(heap) - 1)

def remove(heap, i):
    node = heap[i]
    if i + 1 == len(heap):
        heap.pop()
    else:
        heap[i] = heap.pop()
        heap[i].status = i
        if heap[i].key < node.key:
            push_up(heap, i)
        else:
            push_down(heap, i)
    node.status = -1
    return node

if __name__ == '__main__':
    make_log_dir()

