import torch
import torch.nn as nn

import numpy
import random
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self):
        cases = open('data', 'r').read().strip().split('\n\n')
        self.dt = dict()
        self.arrs = []
        for case in cases:
            m = [[int(e) for e in row.split()] for row in case.split('\n')]
            n = len(m[0])
            arr = numpy.empty((n, n), dtype = numpy.int32)
            arr[0] = range(n)
            arr[1:] = numpy.array(m, dtype = numpy.int32)
            self.arrs.append(arr)
            if n not in self.dt:
                self.dt[n] = []
            self.dt[n].append(arr)

    def __len__(self):
        return 1000000000000000000

    def __getitem__(self, index):
        a = random.choice(self.arrs)
        n = a.shape[0]
        indices = random.sample([(i, j) for i in range(n) for j in range(n)], random.randint(1, n * n))
        indices.sort()
        removed = numpy.zeros((n, n), dtype = bool)
        for index in indices:
            removed[index] = True
        valid_a = a[~removed]
        
        data = torch.zeros(2, n, n, n).float()
        for i in range(n):
            for j in range(n):
                if not removed[i, j]:
                    data[0, a[i, j], i, j] = 1.0
        data[1] = 1.0
        label = numpy.zeros((n, n, n), dtype = numpy.float32)
        for par in self.dt[n]:
            perm = [None] * n
            revs = [None] * n
            ok = True
            valid_par = par[~removed]
            empty_par = par[removed]
            for u, v in zip(valid_a, valid_par):
                if perm[v] is None:
                    if revs[u] is not None:
                        ok = False
                        break
                    perm[v] = u
                    revs[u] = v
                elif perm[v] != u:
                    ok = False
                    break

            if not ok:
                continue

            free = [i for i in range(n) if perm[i] is None]
            for v, (x, y) in zip(empty_par, indices):
                if perm[v] is None:
                    label[free, x, y] = 1.0
                else:
                    label[perm[v], x, y] = 1.0

        return data, torch.from_numpy(label)

if __name__ == '__main__':
    random.seed(58)
    dataset = MyDataset()
    dataloader = DataLoader(dataset, batch_size = 10)
    for i in range(50):
        data, label = dataset[i]
        u, v = round(data[0].sum().item()), round(label.sum().item())
        if u + v != 36:
            print(data.sum(), u, v)
    print(dataset[0])

