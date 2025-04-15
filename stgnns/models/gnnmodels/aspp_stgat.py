import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, LeakyReLU
from torch_geometric.nn import GATConv

from models.gnnmodels.aspp_stgcn import ASPP
from models.gnnmodels.stgcn import OutputLayer



class SpatioGATLayer(nn.Module):

    def __init__(self, c_in, c_out, edge_feat_size=10, edge_hid_size=4, heads=1):
        super(SpatioGATLayer, self).__init__()
        self.edge_feat_size = edge_feat_size
        self.c_in = c_in
        self.c_out = c_out
        self.edge_mlp = torch.nn.Sequential(
            Linear(edge_feat_size, edge_hid_size, bias=False),  # Transform edge features
            LeakyReLU(),
            Linear(edge_hid_size, edge_hid_size, bias=False)
        )
        self.layer = GATConv(c_in, c_out, heads=heads, concat=False, edge_dim=edge_hid_size)


    def forward(self, x, edge_index, edge_attr):
        
        edge_attr = self.edge_mlp(edge_attr)
        
        # Here the input is (t, n, x).
        outputs = []
        for i in range(x.shape[0]):
            # x[:, i, :] has shape (n, x) which is valid for GCNConv
            conv_in = x[i, :, :]            
            conv_out = self.layer(conv_in, edge_index=edge_index, edge_attr=edge_attr)
            conv_out = F.elu(conv_out)
            outputs.append(conv_out)
        # Stack the results along the time dimension
        x = torch.stack(outputs, dim=0)
        x = F.dropout(x, p=0.5, training=self.training)
        
        return x


class ASPP_STGAT(nn.Module):
    def __init__(self, num_nodes, time_window_size, 
                 aspp_dilation_list=[2, 4, 8], aspp_t_out_size = 1, aspp_s_hid_size = 16, 
                 s_feat_size = 32 ,num_spatio_layers = 3, edge_feat_size = 8, edge_hid_size = 6, gat_hid_heads = 1, normalization = True):
        super(ASPP_STGAT, self).__init__()
        self.num_spatio_layers = num_spatio_layers
        self.layers = nn.ModuleList([])
        cnt = 0

        self.ASPP = ASPP(n = num_nodes, c_in = 1, c_out = s_feat_size, c_hid = aspp_s_hid_size, 
                         dia_list = aspp_dilation_list, t_w = time_window_size, t_out_size = aspp_t_out_size, normalization = normalization)

        while (cnt < self.num_spatio_layers):
            heads = 1 if cnt == self.num_spatio_layers - 1 else gat_hid_heads
            self.layers.append(SpatioGATLayer(s_feat_size, s_feat_size, edge_feat_size, edge_hid_size, heads=heads))
            cnt += 1
        self.output = OutputLayer(s_feat_size, aspp_t_out_size, num_nodes, normalization = normalization)

    def forward(self, data):
        
        x = data.x
        edge_index = data.edge_index
        edge_attr  = data.edge_attr
         
        # Here the input is (n, t*x), where n=num_of_graphs*num_of_nodes_per_graph, t=time_window_size, x=c_in (1 in the input layer). 
        # We transpose them into (t, n, x) representing spatio-temporal graphs in the time window.
        x = x.transpose(0, 1)
        x = torch.reshape(x, (x.shape[0], x.shape[1], 1))

        x = self.ASPP(x)
        for i in range(self.num_spatio_layers):
            x = x + self.layers[i](x, edge_index, edge_attr) 
        x = self.output(x)
        return x
