import argparse
import pickle
import time

import networkx as nx
import torch
import numpy as np
from torch import optim
from sklearn.metrics import accuracy_score, f1_score, classification_report

import build_graph
from model import Model
from util import normalize_adj

start_time = time.time()


def get_dataset(batch_graph):
    max_node_num = 0
    for g in batch_graph:
        max_node_num = max(max_node_num, g.adj.shape[0])
    adj_ls = []
    feats = []
    labels = []
    masks = []
    reconstructs = []
    for g in batch_graph:
        adj = g.adj
        num_nodes = adj.shape[0]
        adj_mat_temp = np.zeros((max_node_num, max_node_num))
        adj_mat_temp[:num_nodes, :num_nodes] = adj
        adj_ls.append(adj_mat_temp)
        masks.append([1] * num_nodes + [0] * (max_node_num - num_nodes))
        feats.append(g.node_features + [0] * (max_node_num - num_nodes))
        labels.append(g.label)
        reconstructs.append(g.recon)

    adj_ls = np.array(adj_ls)

    return adj_ls, feats, labels, reconstructs, masks


def train(args, model, optimizer, graphs):
    model.train()
    # labels = []
    # preds = []

    train_size = len(graphs)
    idx_train = np.random.permutation(train_size)
    loss_accum = 0
    loss_accum_recon = 0
    loss_accum_margin = 0
    for i in range(0, train_size, args.batch_size):
        selected_idx = idx_train[i:i + args.batch_size]
        batch_graph = [graphs[idx] for idx in selected_idx]
        adj_mats, feats, label, reconstructs, masks = get_dataset(batch_graph)
        class_capsule_output, loss, margin_loss, reconstruction_loss, label, pred = model(
            adj_mats, feats, label, reconstructs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # labels.append(label.detach())
        # preds.append(pred.detach())
        loss_accum += loss.detach().cpu().item()
        loss_accum_recon += reconstruction_loss.detach().cpu().item()
        loss_accum_margin += margin_loss.detach().cpu().item()
    # labels = torch.cat(labels)
    # preds = torch.cat(preds)
    print('loss recon', loss_accum_recon, 'margin :', loss_accum_margin)
    return loss_accum


def test(args, model, graphs, split):
    minibatch_size = args.batch_size
    model.eval()
    labels = []
    preds = []

    full_idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = full_idx[i:i + minibatch_size]
        if len(sampled_idx) == 0:
            continue
        batch_graph = [graphs[j] for j in sampled_idx]
        adj_mats, feats, label, reconstructs, masks = get_dataset(batch_graph)
        with torch.no_grad():
            class_capsule_output, loss, margin_loss, reconstruction_loss, label, pred = model(
                adj_mats, feats, label, reconstructs, masks)
        labels.append(label.detach().cpu())
        preds.append(pred.detach().cpu())
    labels = torch.cat(labels)
    preds = torch.cat(preds)
    accuracy = accuracy_score(labels.numpy(), preds.numpy())
    # print(split, 'accuracy', accuracy)
    return accuracy


class S2VGraph(object):
    def __init__(self, g, label, word_freq=None, node_features=None, positions=None, adj=None, recon=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        # self.word_freq = word_freq
        self.word_freq1 = word_freq[0]
        self.word_freq2 = word_freq[1]
        self.node_features = node_features
        self.positions = positions
        # self.node_features = row_norm(self.node_features)
        self.edge_mat = 0
        self.edges_weights = []
        self.adj = adj
        self.recon = recon


def create_gaph(args):
    ls_adj, feature_list, word_freq_list, y, y_hot, train_size, word_vectors, positions_list = build_graph.build_graph(
        config=args.configfile)
    vocab_size = len(word_vectors)
    word_vectors = torch.from_numpy(word_vectors).float()

    g_list = []
    max_words = 0
    for i, adj in enumerate(ls_adj):
        adj = normalize_adj(adj)
        g = nx.from_scipy_sparse_matrix(adj)
        lb = y[i]
        feat = feature_list[i]
        recon = [0] * vocab_size
        for f in feat:
            recon[f] += 1
        m_recon = max(recon)
        for f in range(len(recon)):
            recon[f] /= m_recon
        # if frequency_as_feature:
        #     feat = np.concatenate((feat, word_freq_list[i].toarray()), axis=1)
        #     # feat = feat * word_freq_list[i].toarray()
        if i == 10:
            print(word_freq_list[i])
        # s = sum(word_freq_list[i])
        # # s = 1
        # wf = [el / s for el in word_freq_list[i]]
        s = sum(word_freq_list[i][0])
        # s = 1
        wf1 = [el / s for el in word_freq_list[i][0]]
        s = sum(word_freq_list[i][1])
        # s = 1
        wf2 = [el / s for el in word_freq_list[i][1]]
        wf = (wf1, wf2)

        g_list.append(S2VGraph(g, lb, node_features=feat, word_freq=wf,
                               positions=positions_list[i], adj=adj.todense(), recon=recon))
        for ar in positions_list[i]:
            max_words = max(max_words, max(ar))

    max_words += 1

    zero_edges = 0
    for g in g_list:
        # edges = [list(pair) for pair in g.g.edges()]
        # edges_w = [w['weight'] for i, j, w in g.g.edges(data=True)]
        # edges.extend([[i, j] for j, i in edges])
        # edges_w.extend([w for w in edges_w])
        edges = []
        edges_w = []
        for i, j, wt in g.g.edges(data=True):
            w = wt['weight']
            edges.append([i, j])
            edges_w.append(w)
            if i != j:
                edges.append([j, i])
                edges_w.append(w)

        if len(edges) == 0:
            print('zero edge : ', len(g.g))
            zero_edges += 1
            edges = [[0, 0]]
            edges_w = [0]
        g.edge_mat = torch.tensor(edges).long().transpose(0, 1)
        g.edges_weights = torch.tensor(edges_w).float()
    print('total zero edge graphs : ', zero_edges)
    return g_list, len(set(y)), train_size, word_vectors, max_words


def main():
    parser = argparse.ArgumentParser("GraphClassification")
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--configfile', type=str, default="mr", help='configuration file')
    parser.add_argument("--epochs", type=int, default=3000, help="epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="batch_size")
    parser.add_argument("--iterations", type=int, default=3, help="number of iterations of dynamic routing")
    parser.add_argument("--seed", type=int, default=0, help="Initial random seed")
    parser.add_argument("--node_embedding_size", default=16, type=int,  help="subgraph embedding size to be learnt")
    parser.add_argument("--graph_embedding_size", default=16, type=int, help="graph embedding size to be learnt")
    parser.add_argument("--num_gcn_channels", default=2, type=int, help="Number of channels at each layer")
    parser.add_argument("--num_gcn_layers", default=4, type=int, help="Number of GCN layers")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate to optimize the loss function")
    parser.add_argument("--decay_step", default=20000, type=float, help="Learning rate decay step")
    parser.add_argument("--lambda_val", default=0.5, type=float, help="Lambda factor for margin loss")
    parser.add_argument("--noise", default=0.3, type=float, help="dropout applied in input data")
    parser.add_argument("--Attention", default=True, type=bool, help="If use Attention module")
    parser.add_argument("--reg_scale", default=0.1, type=float, help="Regualar scale (reconstruction loss)")
    parser.add_argument("--coordinate", default=False, type=bool, help="If use Location record")
    parser.add_argument("--layer_depth", type=int, default=5, help="number of iterations of dynamic routing")
    parser.add_argument("--layer_width", type=int, default=2, help="number of iterations of dynamic routing")
    parser.add_argument("--num_graph_capsules", type=int, default=64, help="number of iterations of dynamic routing")
    args = parser.parse_args()

    print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print('device : ', device, flush=True)

    all_graphs, num_classes, train_size, word_vectors, max_words = create_gaph(args)

    val_size = train_size // 10
    train_size -= val_size

    train_graph = all_graphs[:train_size]
    val_graph = all_graphs[train_size:train_size+val_size]
    test_graph = all_graphs[train_size+val_size:]

    model = Model(args, num_classes, word_vectors, len(word_vectors), device).to(device)
    optimizer = optim.Adam(model.parameters(), args.lr)
    print(model)

    max_val_acc = 0
    max_test_acc = 0
    max_eopch = 0
    for epoch in range(1, args.epochs+1):
        loss_accum = train(args, model, optimizer, train_graph)
        print('Epoch : ', epoch, 'loss training: ', loss_accum, 'Time : ', int(time.time() - start_time))

        train_acc = test(args, model, train_graph, 'train')
        val_acc = test(args, model, val_graph, 'val')
        test_acc = test(args, model, test_graph, 'test')
        print("accuracy train: %f val: %f test: %f" % (train_acc, val_acc, test_acc), flush=True)

        if max_val_acc <= val_acc:
            max_val_acc = val_acc
            max_test_acc = test_acc
            max_eopch = epoch
        print('max val :', max_val_acc, 'test :', max_test_acc, 'epoch :', max_eopch)
        print('', flush=True)


if __name__ == '__main__':
    main()
