import torch.nn as nn 
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.conv import PNAConv
import torch.nn.functional as F

class SimpleGCN(nn.Module):
    """ 
    https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336124
    """
    def __init__(self,
                 num_node_features: int,
                 num_edge_features: int,
                #  num_classes: int,
                 deg):
        super(SimpleGCN, self).__init__()

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        towers = 4
        pre_layer_num = 2
        post_layer_num = 2
        divide_input=False
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        conv = PNAConv(in_channels=num_node_features, out_channels=32,
                                    aggregators=aggregators, scalers=scalers, deg=deg,
                                    edge_dim=num_edge_features, towers=towers, pre_layers=pre_layer_num, post_layers=post_layer_num,
                                    divide_input=divide_input)
        self.batch_norms.append(nn.BatchNorm(32))
        self.convs.append(conv)

        for j in [32, 64, 128]:
            conv = PNAConv(in_channels=j, out_channels=j*2, aggregators=aggregators, scalers=scalers, deg=deg, edge_dim=num_edge_features, towers=towers, pre_layers=pre_layer_num, post_layers=post_layer_num, divide_input=divide_input)
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm(j*2))

        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(j*2, 1)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))

        x = self.dropout(x)
        x = self.linear(x).flatten()
        return x