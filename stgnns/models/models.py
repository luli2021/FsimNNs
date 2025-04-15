import importlib

from models.gnnmodels.stgcn import STGCN 
from models.gnnmodels.aspp_stgcn import ASPP_STGCN 
from models.gnnmodels.aspp_stgat import ASPP_STGAT


class STGCNs(STGCN):
    def __init__(self, feat_list, dilation_list, time_window_size, num_nodes, layer_types = 'TSTNTSTN', edge_vector_size = 0, **kwargs):
        super(STGCNs, self).__init__(
                feat_size_list=feat_list,
                dilation_list=dilation_list,
		time_window_size=time_window_size,
                num_nodes=num_nodes,
                control_str=layer_types,
                edge_feat_size=edge_vector_size-2)



class ASPP_STGCNs(ASPP_STGCN):
    def __init__(self, time_window_size, num_nodes, edge_vector_size,
                aspp_dilation_list=[2, 4, 8], 
                aspp_t_out_size = 1, 
                aspp_s_hid_size = 16,
                s_feat_size_list = [32, 64, 32, 16],
                control_str = 'SNSNSN',
                **kwargs):
        super(ASPP_STGCNs, self).__init__(
                time_window_size = time_window_size, 
                num_nodes = num_nodes,
                aspp_dilation_list = aspp_dilation_list, 
                aspp_t_out_size = aspp_t_out_size, 
                aspp_s_hid_size = aspp_s_hid_size,
                s_feat_size_list = s_feat_size_list,
                control_str = control_str, 
                edge_feat_size=edge_vector_size-2)




class ASPP_STGATs(ASPP_STGAT):
    def __init__(self, num_nodes, time_window_size=20,
                aspp_dilation_list=[2, 4, 8], 
                aspp_t_out_size = 1, 
                aspp_s_hid_size = 16,
                s_feat_size = 32,
                num_spatio_layers = 3, 
                edge_vector_size = 10,
                edge_hid_size = 6,
                gat_hid_heads = 1,
                normalization = True,
                **kwargs):
        super(ASPP_STGATs, self).__init__(
                num_nodes = num_nodes,
                time_window_size = time_window_size, 
                aspp_dilation_list = aspp_dilation_list, 
                aspp_t_out_size = aspp_t_out_size, 
                aspp_s_hid_size = aspp_s_hid_size,
                s_feat_size = s_feat_size,
                num_spatio_layers = num_spatio_layers, 
                edge_feat_size = edge_vector_size-2,
                edge_hid_size = edge_hid_size,
                gat_hid_heads = gat_hid_heads,
                normalization = normalization)






def get_model(config):
    def _model_class(class_name):
        m = importlib.import_module('models.models')
        clazz = getattr(m, class_name)
        return clazz

    assert 'model' in config, 'Could not find model configuration'
    model_config = config["model"]
    model_class = _model_class(model_config["name"])
    return model_class(**model_config)


