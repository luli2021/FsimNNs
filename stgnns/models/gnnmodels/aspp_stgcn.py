import torch
import torch.nn as nn

from models.gnnmodels.stgcn import TemporalConvLayer, SpatioConvLayer, NormalizationLayer, OutputLayer




class ASPP(nn.Module):
    def __init__(self, n, c_in, c_out, c_hid, dia_list = [2, 4, 8], t_w = 20, t_out_size = 10, normalization = True):
        super(ASPP, self).__init__()
        self.c_out = c_out
        self.c_in = c_in
        self.normalization = normalization

        self.in_layer = nn.Conv2d(c_in, c_hid, (1, 1), 1) 
        self.in_relu = nn.ReLU()
        if self.normalization == True:
            self.in_ln = NormalizationLayer(n, c_hid)

        self.layers = nn.ModuleList([])
        for d in dia_list:
            self.layers.append( TemporalConvLayer(c_hid, c_hid, dia = d) )

        # the output layer is to transform the dimension of time (1st dimension) to t_out_size
        t_w_size = t_w * (len(dia_list) + 1) - sum(dia_list) - t_out_size + 1
        self.out_layer = nn.Conv2d(c_hid, c_out, (t_w_size, 1), 1)
        self.out_relu = nn.ReLU()
        if self.normalization == True:
            self.out_ln = NormalizationLayer(n, c_out)

    def forward(self, x):
        x = x.permute(2, 0, 1).float()
        x = self.in_layer(x)
        x = self.in_relu(x)
        x = x.permute(1, 2, 0)

        if self.normalization == True:
            x = self.in_ln(x)
        x_out = x
        for tconv in self.layers:
            x_t = tconv(x)
            x_out = torch.cat((x_out, x_t),0)

        x_out = x_out.permute(2, 0, 1).float()
        x_out = self.out_layer(x_out)
        x_out = self.out_relu(x_out)
        x_out = x_out.permute(1, 2, 0)
        if self.normalization == True:
            x_out = self.out_ln(x_out)
        
        return x_out



class ASPP_STGCN(nn.Module):
    def __init__(self, time_window_size, num_nodes, 
                 aspp_dilation_list=[2, 4, 8], aspp_t_out_size = 1, aspp_s_hid_size = 16, 
                 s_feat_size_list = [32, 64, 32, 16] ,control_str = 'SNSNSN', edge_feat_size = 0):
        super(ASPP_STGCN, self).__init__()
        self.control_str = control_str # model structure controller
        self.num_layers = len(control_str)
        self.layers = nn.ModuleList([])
        cnt = 0

        self.ASPP = ASPP(n = num_nodes, c_in = 1, c_out = s_feat_size_list[cnt], c_hid = aspp_s_hid_size, 
                         dia_list = aspp_dilation_list, t_w = time_window_size, t_out_size = aspp_t_out_size)

        for i in range(self.num_layers):
            i_layer = control_str[i]
            if i_layer == 'S': # Spatio Layer
                self.layers.append(SpatioConvLayer(s_feat_size_list[cnt], s_feat_size_list[cnt + 1], edge_feat_size))
                cnt += 1
            if i_layer == 'N': # Norm Layer
                self.layers.append(NormalizationLayer(num_nodes, s_feat_size_list[cnt]))
        self.output = OutputLayer(s_feat_size_list[cnt], aspp_t_out_size, num_nodes)

    def forward(self, data):
        
        x = data.x
        edge_index = data.edge_index
        edge_attr  = data.edge_attr       
 
        # Here the input is (n, t*x), where n=num_of_graphs*num_of_nodes_per_graph, t=time_window_size, x=c_in (1 in the input layer). 
        # We transpose them into (t, n, x) representing spatio-temporal graphs in the time window.
        x = x.transpose(0, 1)
        x = torch.reshape(x, (x.shape[0], x.shape[1], 1))       
        
        x = self.ASPP(x)
        for i in range(self.num_layers):
            i_layer = self.control_str[i]
            if i_layer == 'S':
                x = self.layers[i](x, edge_index, edge_attr)
            else:
                x = self.layers[i](x)  
        x = self.output(x)
        return x
