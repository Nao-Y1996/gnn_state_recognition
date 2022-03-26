#!/usr/bin/python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from torch_scatter import scatter_max
import torch.nn.functional as F
from torch_geometric.nn import NNConv
from torch_geometric.loader import DataLoader


class NNConvNet(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, output_dim):
        super(NNConvNet, self).__init__()
        self.edge_fc1 = nn.Linear(edge_feature_dim, node_feature_dim*32)
        self.nnconv1 = NNConv(node_feature_dim, 32, self.edge_fc1, aggr="mean")
        self.edge_fc2 = nn.Linear(edge_feature_dim, 32*48)
        self.nnconv2 = NNConv(32, 48, self.edge_fc2, aggr="mean")
        self.edge_fc3 = nn.Linear(edge_feature_dim, 48*64)
        self.nnconv3 = NNConv(48, 64, self.edge_fc3, aggr="mean")
        self.edge_fc4 = nn.Linear(edge_feature_dim, 64*128)
        self.nnconv4 = NNConv(64, 128, self.edge_fc4, aggr="mean")
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, output_dim)
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.nnconv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.nnconv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.nnconv3(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.nnconv4(x, edge_index, edge_attr)
        x = F.relu(x)
        x, _ = scatter_max(x, data.batch, dim=0)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class classificator():
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    def __init__(self,model,output_dim):
        model_path = model
        self.loading_model = NNConvNet(node_feature_dim=300, edge_feature_dim=3, output_dim=output_dim)
        self.loading_model.load_state_dict(torch.load(model_path))

    def classificate(self, graph):
        input_graph = [graph]
        input_loader = DataLoader(input_graph, batch_size=1, shuffle=True)

        self.loading_model.eval()
        sm = nn.Softmax(dim=1)
        for batch in input_loader:
            pred = self.loading_model(batch)
            probability = sm(pred).tolist()[0]
            # print(probability) 
            return probability