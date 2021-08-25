import torch
import torch.nn as nn

class MaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)

    def forward(self, x, bn_mask):
        # x[ND * B][C][NH][NW]
        # bn_mask[ND * B][NH][NW]
        x = self.conv(x)
        x = x * bn_mask[:, None, :, :]
        return x


class MaskedBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x, bn_mask):
        # float x[ND * B][C][NH][NW]
        # float bn_mask[ND * B][NH][NW]

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            n = bn_mask.sum()
            mean = x.sum([0, 2, 3]) / n
            # use biased var in train
            var = x.square().sum([0, 2, 3]) / n
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        x = (x - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            x = x * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        x = x * bn_mask[:, None, :, :]

        return x


class MaskedSelfAttention(nn.Module):
    def __init__(self, hidden_dim, nhead):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead)

    def forward(self, x, attn_key_padding, attn_mask):
        # x[ND][B * NH * NW][C]
        # bool attn_key_padding[B * NH * NW][ND] for key_padding_mask
        # float attn_mask[ND][B * NH * NW]
        x = self.self_attn(x, x, x, key_padding_mask = attn_key_padding)[0]
        x = x * attn_mask[:, :, None]

        return x


class MaskedLayerNorm(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x, attn_mask):
        # x[ND][B * NH * NW][C]
        # attn_mask[ND][B * NH * NW]
        x = self.ln(x)
        x = x * attn_mask[:, :, None]

        return x

class BasicBlock(nn.Module):
    def __init__(self, hidden_dim, nhead = 8):
        super().__init__()

        self.self_attn = MaskedSelfAttention(hidden_dim, nhead)
        self.ln = MaskedLayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.conv = MaskedConv2d(hidden_dim, hidden_dim, kernel_size = 3, padding = 1, bias = False)
        self.bn = MaskedBatchNorm2d(hidden_dim)

    def forward(self, x, bn_mask, attn_key_padding, attn_mask):
        # x[ND][B][C][NH][NW]
        identity = x

        x = x.permute(0, 1, 3, 4, 2)
        # X[ND][B][NH][NW][C]
        shape = x.shape
        x = x.flatten(1, 3)
        # x[ND][B * NH * NW][C]
        x = self.ln(self.self_attn(x, attn_key_padding, attn_mask), attn_mask)
        x = x.view(shape)
        x = self.relu(x)

        x = x.permute(0, 1, 4, 2, 3)
        # x[ND][B][C][NH][NW]
        x = x.flatten(0, 1)
        # x[ND * B][C][NH][NW]
        x = self.conv(x, bn_mask)
        x = self.bn(x, bn_mask)
        
        x = x.view(identity.shape) + identity
        x = self.relu(x)
        return x


class Network(nn.Module):
    def __init__(self, hidden_dim, nblock):
        super().__init__()

        self.conv1 = MaskedConv2d(2, hidden_dim, kernel_size = 3, padding = 1, bias = False)
        self.bn1 = MaskedBatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(inplace = True)

        self.blocks = nn.ModuleList([BasicBlock(hidden_dim) for _ in range(nblock)])

        self.conv_last = MaskedConv2d(hidden_dim, 1, kernel_size = 1, bias = True)

    def forward(self, x, ns):
        # x[B][2][ND][NH][NW]
        # ns[B] of n1, n2, ..., nb
        x = x.permute(2, 0, 1, 3, 4)
        # x[ND][B][C][NH][NW]

        N, B = x.shape[0], x.shape[1]
        assert B == len(ns)
        raw_mask = torch.zeros(N, B, N, N)
        for i, n in enumerate(ns):
            raw_mask[:n, i, :n, :n] = 1.0
        raw_mask = raw_mask.to(x.device)
        # bn_mask[ND * B][NH][NW]
        bn_mask = raw_mask.flatten(0, 1)
        # attn_mask[ND][B * NH * NW]
        attn_mask = raw_mask.flatten(1)
        # bool attn_key_padding[B * NH * NW][ND] for key_padding_mask
        attn_key_padding = torch.zeros(B, N * N, N).bool().to(x.device)
        for i, n in enumerate(ns):
            attn_key_padding[i, :, n:] = True
        attn_key_padding = attn_key_padding.flatten(0, 1)


        x = x.flatten(0, 1)
        # x[ND, B][C][NH][NW]
        x = self.conv1(x, bn_mask)
        x = self.bn1(x, bn_mask)
        x = self.relu(x)
        x = x.view(N, B, -1, N, N)

        for block in self.blocks:
            x = block(x, bn_mask, attn_key_padding, attn_mask)

        # x[ND][B][C][NH][NW]
        x = self.conv_last(x.flatten(0, 1), bn_mask)
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

    output = net(input, ns)
    from IPython import embed; embed()

