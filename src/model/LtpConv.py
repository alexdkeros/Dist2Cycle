import numpy as np
import numpy.linalg as npla
import math
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.utils import expand_as_pair
from dgl import function as fn  

device='cuda'

class LtpConv(nn.Module):
    def __init__(self,
                in_feats,
                out_feats,
                aggregator_type='sum',
                weighted_edges=True,
                fc_bias=True, #feature learning bias
                norm=None,
                fc_activation=None,
                initialization='kaiming'):
    
        super(LtpConv, self).__init__()

        self.weighted_edges=weighted_edges
        
        self.initialization=initialization #choices [debug, xavier, kaiming]
        
        if aggregator_type not in ['mean', 'sum']:
            raise KeyError('Aggregator type {} not supported.'.format(aggregator_type))
        self._aggre_type = aggregator_type
        
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats=out_feats
        self.fc_bias=fc_bias
        self.norm=norm
        self.fc_activation=fc_activation
        
        #feature learning
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=self.fc_bias)

        self.reset_parameters()
    
    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The linear weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        """
        fc_alpha=0.0
        
        if isinstance(self.fc_activation, nn.LeakyReLU):
            fc_nonlin='leaky_relu'
            fc_alpha=self.fc_activation.negative_slope
        elif self.fc_activation is not None:
            fc_nonlin=self.fc_activation.__str__()[:-2].lower()
        else:
            fc_nonlin='linear'    
        
        if fc_nonlin not in ['linear', 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'selu']:
            fc_nonlin='sigmoid'
        
        
        fc_gain = nn.init.calculate_gain(fc_nonlin)
        
        if self.initialization=='debug':
            #when debuging the layer, initialize weights with identity
            nn.init.eye_(self.fc_neigh.weight)
                    
        elif self.initialization=='xavier':    
            nn.init.xavier_uniform_(self.fc_neigh.weight, gain=fc_gain)
                    
                    
        elif self.initialization=='kaiming':
            nn.init.kaiming_uniform_(self.fc_neigh.weight,a=fc_alpha, nonlinearity=fc_nonlin)
            
        
    
    def forward(self, g, feat):        
        
        g.srcdata['h']=feat.float()
        
        with g.local_scope():
            #standard weighted message
            if self._aggre_type=='sum':
                if self.weighted_edges:
                    g.update_all(fn.e_mul_u('w', 'h', 'm'), fn.sum('m','h'))
                else:
                    g.update_all(fn.copy_u('h','m'), fn.sum('m','h'))
            elif self._aggre_type=='mean':
                if self.weighted_edges:
                    g.update_all(fn.e_mul_u('w', 'h', 'm'), fn.mean('m','h'))
                else:
                    g.update_all(fn.copy_u('h','m'), fn.mean('m','h'))
                
            
            h_neigh = g.dstdata['h']
            rst=self.fc_neigh(h_neigh)

            # activation
            if self.fc_activation is not None:
                rst = self.fc_activation(rst)

            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            
            return rst
            
