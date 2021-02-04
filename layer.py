import torch
import torch.nn as nn
import torch.nn.functional as F

"""
ref
https://github.com/timomernick/pytorch-capsule
"""
class SecondaryCapsuleLayer(nn.Module):

    def __init__(self, in_channels, in_dim, out_channels, out_dim, device):
        super(SecondaryCapsuleLayer, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_dim = out_dim
        self.W = torch.nn.Parameter(torch.randn(1, 1, in_channels, out_channels, in_dim, out_dim), requires_grad=True)

    @staticmethod
    def squash(s):
        mag_sq = torch.sum(s ** 2, dim=3, keepdim=True)
        mag = torch.sqrt(mag_sq) + 1e-11
        a_j = mag_sq / (1.0 + mag_sq)
        s = a_j * (s / mag)
        return s, a_j

    def forward(self, x, number_of_nodes):
        input_shape = x.size()  # b x n x c x d
        batch_size = input_shape[0]
        n = input_shape[1]

        x = x.unsqueeze(3).repeat(1, 1, 1, self.out_channels, 1).unsqueeze(4)
        w = self.W.repeat(batch_size, n, 1, 1, 1, 1)    # b x n x ci x co x di x do
        u_hat = torch.matmul(x, w).squeeze(4)  # b x n x ci x co x d
        u_hat = u_hat.reshape(batch_size, n * self.in_channels, self.out_channels, self.out_dim)   # b x n*ci x co x d

        b_ij = torch.zeros(batch_size, n * self.in_channels, self.out_channels, 1, device=self.device)  # b x n*ci x co x 1

        num_iterations = 3
        v_j = None
        a_j = None
        for _ in range(num_iterations):
            c_ij = F.softmax(b_ij, dim=1)
            # c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)
            # print(number_of_nodes.shape)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)   # b x 1 x co x d
            v_j, a_j = SecondaryCapsuleLayer.squash(s_j)  # b x 1 x co x d

            v_j1 = torch.cat([v_j] * n * self.in_channels, dim=1)  # b x n*ci x co x d
            u_vj1 = torch.sum(u_hat * v_j1, dim=3, keepdim=True)  # b x n*ci x co x 1
            b_ij = b_ij + u_vj1

        return v_j, a_j


class GRUCellMod(nn.Module):
    """
    ref https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
    """
    def __init__(self, input_dim, hidden_dim):
        super(GRUCellMod, self).__init__()
        self.W_ir = nn.Linear(input_dim, hidden_dim)
        self.W_hr = nn.Linear(hidden_dim, hidden_dim)
        self.W_in = nn.Linear(input_dim, hidden_dim)
        self.W_hn = nn.Linear(hidden_dim, hidden_dim)
        self.W_iz = nn.Linear(input_dim, hidden_dim)
        self.W_hz = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, inp, ht_1):
        r_t = torch.sigmoid(self.W_ir(inp) + self.W_hr(ht_1))
        z_t = torch.sigmoid(self.W_iz(inp) + self.W_hz(ht_1))
        n_t = F.tanh(self.W_in(inp) + r_t * self.W_hn(ht_1))
        h_t = z_t * n_t + (1 - z_t) * inp
        # h_t = z_t * inp + (1 - z_t) * ht_1
        return h_t


class GCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCN, self).__init__()

        self.linear1 = nn.Linear(input_dim, output_dim)
        self.eps = nn.Parameter(torch.randn(1), requires_grad=True)
        self.gru = GRUCellMod(output_dim, output_dim)

    def forward(self, adj, x):
        h = self.linear1(x)
        h1 = torch.bmm(adj, h)
        h = self.gru(h, h1)

        return h
