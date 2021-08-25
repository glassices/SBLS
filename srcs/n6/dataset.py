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
            arr = numpy.array(m, dtype = numpy.int32)
            self.arrs.append(arr)
            if arr.shape[1] not in self.dt:
                self.dt[arr.shape[1]] = []
            self.dt[arr.shape[1]].append(arr)

    def __len__(self):
        return 1000000000000000000

    def __getitem__(self, index):
        arr = random.choice(self.arrs)
        n = arr.shape[1]
        a = numpy.zeros((n, n), dtype = numpy.int32)
        a[0] = range(n)
        a[1:] = arr
        indices = random.sample([(i, j) for i in range(n) for j in range(n)], random.randint(1, n * n))
        for index in indices:
            a[index] = -1
        
        data = torch.zeros(2, n, n, n).float()
        for i in range(n):
            for j in range(n):
                if a[i, j] != -1:
                    data[0, a[i, j], i, j] = 1.0
        data[1] = 1
        label = torch.zeros(n, n, n).float()
        for par in self.dt[n]:
            perm = dict()
            ok = True
            for i in range(n):
                for j in range(n):
                    if a[i, j] == -1:
                        continue
                    u = j if i == 0 else par[i - 1, j]
                    if u not in perm:
                        perm[u] = a[i, j]
                    else:
                        if perm[u] != a[i, j]:
                            ok = False
                            break
                if not ok:
                    break

            if not ok or len(set(perm.values())) != len(perm):
                continue
            free = [i for i in range(n) if i not in perm]
            for i in range(n):
                for j in range(n):
                    if a[i, j] != -1:
                        continue
                    u = j if i == 0 else par[i - 1, j]
                    if u not in perm:
                        label[free, i, j] = 1.0
                    else:
                        label[perm[u], i, j] = 1.0

        return data, label

if __name__ == '__main__':
    dataset = MyDataset()
    dataloader = DataLoader(dataset, batch_size = 10)
    for i in range(100):
        a = dataset[i]
        print(i)

