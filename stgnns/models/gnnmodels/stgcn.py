import torch
import torch.nn as nn 
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, ResGatedGraphConv
from torch_geometric.utils import degree



class TemporalConvLayer(nn.Module):
    # for Conv2d in torch, the input shape should be (C_in, H, W) and output is (C_out, H_out, W_out)
    # Here the input is (t, n, x), we need to transpose them into (x, t, n)
    def __init__(self, c_in, c_out, dia = 1):
        super(TemporalConvLayer, self).__init__()
        self.c_out = c_out
        self.c_in = c_in
        self.conv = nn.Conv2d(c_in, c_out, (2, 1), 1, dilation = dia)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(2, 0, 1).float()
        x = self.conv(x) # to (c_out, t-dia, n)
        x = self.relu(x)
        x = x.permute(1, 2, 0) # to (t, n, x)
        return x



class SpatioConvLayer(nn.Module):

    def __init__(self, c_in, c_out, edge_feat_size=0):
        super(SpatioConvLayer, self).__init__()
        self.edge_feat_size = edge_feat_size
        self.c_in = c_in
        self.c_out = c_out
        self.layer = GCNConv(c_in, c_out, normalize=False)
        #self.layer = GCNConv(c_in, c_out)
        self.relu = nn.ReLU()
        if self.edge_feat_size != 0:
            self.e_layer = nn.Linear(edge_feat_size,1)

    def forward(self, x, edge_index, edge_attr):

        num_node = x.shape[1]

        # In PyG, the normalization implemented in GCNConv doesn't allow negative or zero weights of edges (see docs of GCNConv).
        # Otherwise, there will be nan loss when computing symmetric normalization coefficients. Here we calculate the coefficients
        # in a different way, which allows negative weights.
        weight = self.e_layer(edge_attr)
        deg = degree(edge_index[0], num_nodes=num_node)
        sqrt_deg = torch.sqrt(deg)
        src, dst = edge_index[0], edge_index[1]
        sqrt_deg_src = sqrt_deg[src]  # sqrt(deg) for source nodes
        sqrt_deg_dst = sqrt_deg[dst]  # sqrt(deg) for destination nodes
        norm = torch.mul(sqrt_deg_src,sqrt_deg_dst).view(-1,1)
        weight = torch.mul(norm,weight)


        # for GCNConv, the input shape should be (num_nodes, node_features). 
        # Here the input is (t, n, x).
        outputs = []
        for i in range(x.shape[0]):
            # x[i, :, :] has shape (n, x) which is valid for GCNConv
            conv_in = x[i, :, :]
            conv_out = self.layer(conv_in, edge_index=edge_index, edge_weight=weight)
            outputs.append(conv_out)
        # Stack the results along the time dimension
        x = torch.stack(outputs, dim=0)
        x = self.relu(x)
        
        return x


# In this layer, we use nn.LayerNorm for normalization, which implements normalization across features.
# The normalization here is to normalize across nodes and features. After reshape, the shape of x is changed to 
# (batch_size(i.e. number of graphs)*time_window_size, number of nodes per graph, chan_in). The normalization is employed on the last two domains.
class NormalizationLayer(nn.Module):
    def __init__(self, n, c):
        super(NormalizationLayer, self).__init__()
        self.num, self.chan_in = n, c
        self.layer = nn.LayerNorm([n, c])

    def forward(self, x):
        input_shape = x.shape
        shape0 = int( x.numel() / (self.num*self.chan_in) )
        x = torch.reshape(x,(shape0, self.num, self.chan_in)) 
        x = self.layer(x)
        x = torch.reshape(x,input_shape)
        return x




class OutputLayer(nn.Module):
    def __init__(self, c, T, n, normalization = True):
        super(OutputLayer, self).__init__()
        self.c = c
        self.normalization = normalization
        self.tconv1 = nn.Conv2d(c, c, (T, 1), 1)
        if normalization == True:
            self.ln = NormalizationLayer(n, c)
        self.fc = nn.Conv2d(c, 2, 1)


    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = self.tconv1(x)
        x = x.permute(1, 2, 0)

        if self.normalization == True:
            x = self.ln(x)
        
        x = x.permute(2, 0, 1)
        x = self.fc(x)
        x = F.sigmoid(x)
        x = torch.squeeze(x)
        x = x.transpose(0, 1)
        return x



class STGCN(nn.Module):
    def __init__(self, feat_size_list, dilation_list, time_window_size, num_nodes, control_str = 'TSTNTSTN', edge_feat_size = 0):
        super(STGCN, self).__init__()
        self.control_str = control_str # model structure controller
        self.num_layers = len(control_str)
        self.layers = nn.ModuleList([])
        self.in_t_size = time_window_size
        cnt = 0
        diapower = 0
        total_dilation = 0
        for i in range(self.num_layers):
            i_layer = control_str[i]
            if i_layer == 'T': # Temporal Layer
                self.layers.append(TemporalConvLayer(feat_size_list[cnt], feat_size_list[cnt + 1], dia = dilation_list[cnt]))
                total_dilation = total_dilation + dilation_list[cnt]
                diapower += 1
                cnt += 1
            if i_layer == 'S': # Spatio Layer
                self.layers.append(SpatioConvLayer(feat_size_list[cnt], feat_size_list[cnt], edge_feat_size))
            if i_layer == 'N': # Norm Layer
                self.layers.append(NormalizationLayer(num_nodes, feat_size_list[cnt]))
        self.output = OutputLayer(feat_size_list[cnt], time_window_size - total_dilation, num_nodes)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr  = data.edge_attr 

        # Here the input is (n, t*x), where n=num_of_graphs*num_of_nodes_per_graph, t=time_window_size, x=c_in (1 in the input layer). 
        # We transpose them into (t, n, x) representing spatio-temporal graphs in the time window.
        x = x.transpose(0, 1)
        x = torch.reshape(x, (x.shape[0], x.shape[1], 1))

        for i in range(self.num_layers):
            i_layer = self.control_str[i]
            if i_layer == 'S':
                x = self.layers[i](x, edge_index, edge_attr)
            else:
                x = self.layers[i](x)  
        return self.output(x)
