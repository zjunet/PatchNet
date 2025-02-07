import argparse
import time
from itertools import repeat

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models import GNN
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool

#from datasets import molecule_graphcl_dataset


class graphcl(nn.Module):

    def __init__(self, gnn, emb_dim, device):
        super(graphcl, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(
            nn.Linear(2*emb_dim, 2*emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2*emb_dim, 2*emb_dim))
        self.device = device

    def forward_cl(self, x, edge_index, edge_attr, batch):
        #print(x)
        #print(edge_index)
        #print(edge_attr)
        x = self.gnn(x, edge_index, edge_attr)
        #print(x.shape)
        #print(batch.shape)
        x = self.pool(x, batch)
        #print(x.size())
        x = self.projection_head(x)
        return x

    def loss_cl(self, x1, x2):
        assert(x1.shape[0] == x2.shape[0])
        assert(x1.shape[0]>1)
        T = 0.1
        batch, _ = x1.size()
        #print(x1.size())
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        #print(x1_abs)
        #print(x2_abs)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / \
                     torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch), range(batch)]
        sum_other = torch.tensor([0.0]*batch).to(self.device)
        for i in range(batch):
            for j in range(batch):
                if i != j :
                    sum_other[i] += sim_matrix[i][j]
        #print(pos_sim)
        #print(sim_matrix)
        loss = pos_sim / sum_other
        #print(sim_matrix.sum(dim=1))
        loss = - torch.log(loss).mean()
        #print(loss)
        return loss


def train(loader, model, optimizer, device, gamma_joao):

    model.train()
    train_loss_accum = 0

    for step, (_, batch1, batch2) in enumerate(loader):
        # _, batch1, batch2 = batch
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)

        # pdb.set_trace()
        x1 = model.forward_cl(batch1.x, batch1.edge_index,
                              batch1.edge_attr, batch1.batch)
        x2 = model.forward_cl(batch2.x, batch2.edge_index,
                              batch2.edge_attr, batch2.batch)
        loss = model.loss_cl(x1, x2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_accum += float(loss.detach().cpu().item())

    # joao
    aug_prob = loader.dataset.aug_prob
    loss_aug = np.zeros(25)
    for n in range(25):
        _aug_prob = np.zeros(25)
        _aug_prob[n] = 1
        loader.dataset.set_augProb(_aug_prob)
        # for efficiency, we only use around 10% of data to estimate the loss
        count, count_stop = 0, len(loader.dataset) // (loader.batch_size * 10) + 1

        with torch.no_grad():
            for step, (_, batch1, batch2) in enumerate(loader):
                # _, batch1, batch2 = batch
                batch1 = batch1.to(device)
                batch2 = batch2.to(device)

                x1 = model.forward_cl(batch1.x, batch1.edge_index,
                                      batch1.edge_attr, batch1.batch)
                x2 = model.forward_cl(batch2.x, batch2.edge_index,
                                      batch2.edge_attr, batch2.batch)
                loss = model.loss_cl(x1, x2)
                loss_aug[n] += loss.item()
                count += 1
                if count == count_stop:
                    break
        loss_aug[n] /= count

    # view selection, projected gradient descent,
    # reference: https://arxiv.org/abs/1906.03563
    beta = 1
    gamma = gamma_joao

    b = aug_prob + beta * (loss_aug - gamma * (aug_prob - 1 / 25))
    mu_min, mu_max = b.min() - 1 / 25, b.max() - 1 / 25
    mu = (mu_min + mu_max) / 2

    # bisection method
    while abs(np.maximum(b - mu, 0).sum() - 1) > 1e-2:
        if np.maximum(b - mu, 0).sum() > 1:
            mu_min = mu
        else:
            mu_max = mu
        mu = (mu_min + mu_max) / 2

    aug_prob = np.maximum(b - mu, 0)
    aug_prob /= aug_prob.sum()

    return train_loss_accum / (step + 1), aug_prob


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='JOAO')
    parser.add_argument('--device', type=int, default=0, help='gpu')
    parser.add_argument('--batch_size', type=int, default=256, help='batch')
    parser.add_argument('--decay', type=float, default=0, help='weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--JK', type=str, default="last",
                        choices=['last', 'sum', 'max', 'concat'],
                        help='how the node features across layers are combined.')
    parser.add_argument('--gnn_type', type=str, default="gin", help='gnn model type')
    parser.add_argument('--dropout_ratio', type=float, default=0, help='dropout ratio')
    parser.add_argument('--emb_dim', type=int, default=300, help='embedding dimensions')
    parser.add_argument('--dataset', type=str, default=None, help='root dir of dataset')
    parser.add_argument('--num_layer', type=int, default=5, help='message passing layers')
    # parser.add_argument('--seed', type=int, default=0, help="Seed for splitting dataset")
    parser.add_argument('--output_model_file', type=str, default='', help='model save path')
    parser.add_argument('--num_workers', type=int, default=8, help='workers for dataset loading')

    parser.add_argument('--aug_mode', type=str, default='sample')
    parser.add_argument('--aug_strength', type=float, default=0.2)

    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--output_model_dir', type=str, default='')
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) \
        if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    if 'GEOM' in args.dataset:
        dataset = Dataset_graphcl('../datasets/{}/'.format(args.dataset), dataset=args.dataset)
    dataset.set_augMode(args.aug_mode)
    dataset.set_augStrength(args.aug_strength)
    print(dataset)

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        num_workers=args.num_workers, shuffle=True)

    # set up model
    gnn = GNN(num_layer=args.num_layer, emb_dim=args.emb_dim, JK=args.JK,
              drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)

    model = graphcl(gnn)
    model.to(device)

    # set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    # print(optimizer)

    # pdb.set_trace()
    aug_prob = np.ones(25) / 25
    np.set_printoptions(precision=3, floatmode='fixed')

    for epoch in range(1, args.epochs + 1):
        print('\n\n')
        start_time = time.time()
        dataset.set_augProb(aug_prob)
        pretrain_loss, aug_prob = train(loader, model, optimizer, device, args.gamma)

        print('Epoch: {:3d}\tLoss:{:.3f}\tTime: {:.3f}\tAugmentation Probability:'.format(
            epoch, pretrain_loss, time.time() - start_time))
        print(aug_prob)

    if not args.output_model_dir == '':
        saver_dict = {'model': model.state_dict()}
        torch.save(saver_dict, args.output_model_dir + '_model_complete.pth')
        torch.save(model.gnn.state_dict(), args.output_model_dir + '_model.pth')
