import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, hidden_dim, nhead = 8):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead)
        self.ln = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size = 3, padding = 1, bias = False)
        self.bn = nn.BatchNorm2d(hidden_dim)

    def forward(self, x):
        # x[ND][B][C][NH][NW]
        identity = x

        x = x.permute(0, 1, 3, 4, 2)
        # X[ND][B][NH][NW][C]
        shape = x.shape
        x = x.flatten(1, 3)
        # x[ND][B * NH * NW][C]
        x = self.ln(self.self_attn(x, x, x)[0])
        x = x.view(shape)
        x = self.relu(x)

        x = x.permute(0, 1, 4, 2, 3)
        # x[ND][B][C][NH][NW]
        x = x.flatten(0, 1)
        # x[ND * B][C][NH][NW]
        x = self.conv(x)
        x = self.bn(x)
        
        x = x.view(identity.shape) + identity
        x = self.relu(x)
        return x


class Network(nn.Module):
    def __init__(self, hidden_dim, nblock):
        super().__init__()

        self.conv1 = nn.Conv2d(2, hidden_dim, kernel_size = 3, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(inplace = True)

        self.blocks = nn.ModuleList([BasicBlock(hidden_dim) for _ in range(nblock)])

        self.conv_last = nn.Conv2d(hidden_dim, 1, kernel_size = 1, bias = True)

    def forward(self, x):
        # x[B][2][ND][NH][NW]
        # ns[B] of n1, n2, ..., nb
        x = x.permute(2, 0, 1, 3, 4)
        # x[ND][B][C][NH][NW]

        N, B = x.shape[0], x.shape[1]

        x = x.flatten(0, 1)
        # x[ND, B][C][NH][NW]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.view(N, B, -1, N, N)

        for block in self.blocks:
            x = block(x)

        # x[ND][B][C][NH][NW]
        x = self.conv_last(x.flatten(0, 1))
        # x[ND * B][1][NH][NW]
        x = x.view(N, B, N, N).transpose(0, 1)
        # x[B][ND][NH][NW]
        return x

if __name__ == '__main__':
    net = Network(128, 10)

    import random
    B = 128
    ns = []
    for i in range(B):
        ns.append(random.randint(2, 10))
    N = max(ns)
    input = torch.zeros(B, 2, N, N, N)
    for i, n in enumerate(ns):
        input[i, :, :n, :n, :n].normal_()

    output = net(input)
    from IPython import embed; embed()

