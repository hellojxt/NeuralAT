
import torch
import math
import torch.utils.checkpoint
from typing import List, Optional


from src.gnn.graph_tree import Data
#from torch_geometric.nn import avg_pool
from src.gnn.graph_tree import avg_pool
from src.gnn.graph_tree import GraphTree


bn_momentum, bn_eps = 0.01, 0.001    # the default value of Tensorflow 1.x
# bn_momentum, bn_eps = 0.1, 1e-05   # the default value of pytorch


'''
本文件包含了用于构建 UNet、ResNet 等所需的图神经网络 res-blocks
原名 modules.py
'''



###############################################################
# Util funcs
###############################################################

def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
  r''' Broadcast :attr:`src` according to :attr:`other`, originally from the
  library `pytorch_scatter`.
  '''

  if dim < 0:
    dim = other.dim() + dim

  if src.dim() == 1:
    for _ in range(0, dim):
      src = src.unsqueeze(0)
  for _ in range(src.dim(), other.dim()):
    src = src.unsqueeze(-1)

  src = src.expand_as(other)
  return src


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None,) -> torch.Tensor:
  r''' Reduces all values from the :attr:`src` tensor into :attr:`out` at the
  indices specified in the :attr:`index` tensor along a given axis :attr:`dim`.
  This is just a wrapper of :func:`torch.scatter` in a boardcasting fashion.

  Args:
    src (torch.Tensor): The source tensor.
    index (torch.Tensor): The indices of elements to scatter.
    dim (torch.Tensor): The axis along which to index, (default: :obj:`-1`).
    out (torch.Tensor or None): The destination tensor.
    dim_size (int or None): If :attr:`out` is not given, automatically create
        output with size :attr:`dim_size` at dimension :attr:`dim`. If
        :attr:`dim_size` is not given, a minimal sized output tensor according
        to :obj:`index.max() + 1` is returned.
    '''

  index = broadcast(index, src, dim)

  if out is None:
    size = list(src.size())
    if dim_size is not None:
      size[dim] = dim_size
    elif index.numel() == 0:
      size[dim] = 0
    else:
      size[dim] = int(index.max()) + 1
    out = torch.zeros(size, dtype=src.dtype, device=src.device)

  return out.scatter_add_(dim, index, src)


###############################################################
# Basic Operators on graphs
###############################################################

from torch_geometric.nn import GraphSAGE

class GraphSAGEConv(torch.nn.Module):
    # 最简单的在图上的卷积, SAGE，表达能力很低
    def __init__(self, in_channels: int, out_channels: int,
                 ):
        super().__init__()
        self.conv = GraphSAGE(in_channels, out_channels, num_layers=1)

    def forward(self, input_feature: torch.Tensor, graphtree: GraphTree, depth: int):
        graph = graphtree.treedict[graphtree.depth - depth]
        assert input_feature.shape[0] == graph.x.shape[0]
        out = self.conv(input_feature, graph.edge_index)
        return out


from torch_geometric.nn import GATv2Conv

class GraphAttentionConv(torch.nn.Module):
    # Graph Attention Network ()
    def __init__(self, in_channels: int, out_channels: int,
                 ):
        super().__init__()
        self.conv = GATv2Conv(in_channels, out_channels, num_layers=1)

    def forward(self, input_feature: torch.Tensor, graphtree: GraphTree, depth: int):
        graph = graphtree.treedict[graphtree.depth - depth]
        assert input_feature.shape[0] == graph.x.shape[0]
        out = self.conv(input_feature, graph.edge_index)
        return out
      
      
from torch_geometric.nn import GCNConv

class GraphConvolutionNetworkConv(torch.nn.Module):
    # Graph Attention Network ()
    def __init__(self, in_channels: int, out_channels: int,
                 ):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels, add_self_loops=False)

    def forward(self, input_feature: torch.Tensor, graphtree: GraphTree, depth: int):
        graph = graphtree.treedict[graphtree.depth - depth]
        assert input_feature.shape[0] == graph.x.shape[0]
        out = self.conv(input_feature, graph.edge_index)
        return out


############################################################
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing, HeteroLinear

class EdgeConvOp(MessagePassing):
    # this implementation is from tutorial "message passing" of torch_geometric
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='sum') #  "Max" aggregation.
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)
      

class EdgeConv(torch.nn.Module):
    # Edge Conv
    def __init__(self, in_channels: int, out_channels: int,
                 ):
        super().__init__()
        self.conv = EdgeConvOp(in_channels, out_channels)

    def forward(self, input_feature: torch.Tensor, graphtree: GraphTree, depth: int):
        graph = graphtree.treedict[graphtree.depth - depth]
        assert input_feature.shape[0] == graph.x.shape[0]
        out = self.conv(input_feature, graph.edge_index)
        return out
      
############################################################

class MyConvOp(MessagePassing):
    # this implementation is from tutorial "message passing" of torch_geometric
    def __init__(self, in_channels, out_channels, include_distance):
        super().__init__(aggr='max') #  "Max" aggregation.
        #self.mlp = Seq(Linear(2 * in_channels, out_channels),
        #               ReLU(),
        #               Linear(out_channels, out_channels))
     #   self.self_loop = Linear(in_channels, out_channels)
     #   self.neighbor_matrix_1 = Linear(in_channels, out_channels)
     #   self.neighbor_matrix_2 = Linear(in_channels, out_channels)
        
        self.num_types = 2
        self.lin = HeteroLinear(in_channels, out_channels, num_types=self.num_types, is_sorted=False)
        self.include_distance = include_distance
        

    def forward(self, x, edge_index, edge_attr=None):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # edge_attr has shape [1, E]
        # referred to https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html#implementing-the-gcn-layer
        
        if edge_attr == None:
          # dynamically calculate edge_type
          raise NotImplementedError
        else:
          # use predefined edge_type
          assert edge_attr.min() == 0
          assert edge_attr.max() == self.num_types - 1
          

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        
        
        # last 3 dim is the xyz position of vertices
        # compute the abs value of them
        if self.include_distance == True:
          abs_dist = torch.norm(x_i[..., -3:] - x_j[..., -3:], dim=1, keepdim=True) # shape: [E, 1]
          x_j = torch.cat([x_j[..., :-3], x_i[..., -3:] - x_j[..., -3:], abs_dist], dim=-1)
        
        out = self.lin(x_j, edge_attr)
        return out
      

class MyConv(torch.nn.Module):
    # My Conv, consider neighbor features (relative pos and its abs)
    # and many seperated catagories
    def __init__(self, in_channels: int, out_channels: int,
                 ):
        super().__init__()
        self.include_distance = False#default_settings.get_global_value("include_distance")
        if self.include_distance == False:
          self.conv = MyConvOp(in_channels, out_channels, include_distance=False)
        else:
          self.conv = MyConvOp(in_channels + 3 + 0 + 1, out_channels, include_distance=True)

    def forward(self, input_feature: torch.Tensor, graphtree: GraphTree, depth: int):
        graph = graphtree.treedict[graphtree.depth - depth]
        assert input_feature.shape[0] == graph.x.shape[0]
        # concat input feature
        if self.include_distance == True:
          input_feature = torch.cat([input_feature, graph.x[..., :3]], dim=-1)
        # do the conv
        out = self.conv(input_feature, graph.edge_index, graph.edge_attr)
        return out


########################################################


###############################################################
# Complex Components
###############################################################

# 非常重要：选择你喜欢的图网络卷积方式
# 默认根据config里的设定选择

# GraphSAGEConv: 普通的 SAGE
# GraphAttentionConv: 普通的 GATv2
# DirectionalGraphConv: 分7类的卷积操作
# DistanceGraphConv: 考虑距离的卷积操作

conv_type = "gat"
if conv_type == "sage":
  GraphConv = GraphSAGEConv
elif conv_type == "gat":
  GraphConv = GraphAttentionConv
elif conv_type == "edge":
  GraphConv = EdgeConv
elif conv_type == "dirconv":
  GraphConv = None#DirectionalGraphConv
elif conv_type == "distconv":
  GraphConv = None#DistanceGraphConv
elif conv_type == "my":
  GraphConv = MyConv
elif conv_type == "gcn":
  GraphConv = GraphConvolutionNetworkConv
else:
  raise NotImplementedError


# 非常重要2：BN/LN/GN
from functools import partial

# batch norm
#normalization = partial(torch.nn.BatchNorm1d, eps=bn_eps, momentum=bn_momentum)
# instance norm
#normalization = partial(torch.nn.InstanceNorm1d, eps=bn_eps, momentum=bn_momentum)
# layer norm
#normalization = partial(torch.nn.LayerNorm, eps=bn_eps)
# group norm
normalization = lambda x: torch.nn.GroupNorm(num_groups=4, num_channels=x, eps=bn_eps)



class GraphConvBn(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 ):
        super().__init__()
        self.conv = GraphConv(in_channels, out_channels)
        self.bn = normalization(out_channels)

    def forward(self, data: torch.Tensor, graphtree: GraphTree, depth: int):
        out = self.conv(data, graphtree, depth)
        out = self.bn(out)
        return out


class GraphConvBnRelu(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 ):
        super().__init__()
        self.conv = GraphConv(in_channels, out_channels)
        self.bn = normalization(out_channels)
        self.relu = torch.nn.ReLU() # inplace=True

    def forward(self, data: torch.Tensor, graphtree: GraphTree, depth: int):
        out = self.conv(data, graphtree, depth)
        out = self.bn(out)
        out = self.relu(out)
        return out


class PoolingGraph(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor, graphtree: GraphTree, depth: int):
        cluster = graphtree.cluster[depth]
        out = avg_pool(cluster=cluster, data=Data(x=x, edge_index=torch.zeros([2,1]).long()))  # fake edges XD
        return out.x

class UnpoolingGraph(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor, graphtree: GraphTree, depth: int):
        assert depth != 0
        # TODO: 确认这里有没有错
        feature_dim = x.shape[1]
        cluster = graphtree.cluster[depth][..., None].repeat(1, feature_dim).long()
        out = torch.gather(input=x, dim=0, index=cluster)
        return out


"""
OctreeDeconvBnRelu 和 OctreeDeconvBn 没有对应的函数，因为图神经网络里没有反卷积操作

经 HKU PhD Dr. Qin TZ 确认

"""



class Conv1x1(torch.nn.Module):
  r''' Performs a convolution with kernel :obj:`(1,1,1)`.

  The shape of octree features is :obj:`(N, C)`, where :obj:`N` is the node
  number and :obj:`C` is the feature channel. Therefore, :class:`Conv1x1` can be
  implemented with :class:`torch.nn.Linear`.
  '''

  def __init__(self, in_channels: int, out_channels: int, use_bias: bool = False):
    super().__init__()
    self.linear = torch.nn.Linear(in_channels, out_channels, use_bias)

  def forward(self, data: torch.Tensor):
    r''''''

    return self.linear(data)


class Conv1x1Bn(torch.nn.Module):
  r''' A sequence of :class:`Conv1x1` and :class:`BatchNorm`.
  '''

  def __init__(self, in_channels: int, out_channels: int):
    super().__init__()
    self.conv = Conv1x1(in_channels, out_channels, use_bias=False)
    self.bn = normalization(out_channels)

  def forward(self, data: torch.Tensor):
    r''''''

    out = self.conv(data)
    out = self.bn(out)
    return out


class Conv1x1BnRelu(torch.nn.Module):
  r''' A sequence of :class:`Conv1x1`, :class:`BatchNorm` and :class:`Relu`.
  '''

  def __init__(self, in_channels: int, out_channels: int):
    super().__init__()
    self.conv = Conv1x1(in_channels, out_channels, use_bias=False)
    self.bn = normalization(out_channels)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, data: torch.Tensor):
    r''''''

    out = self.conv(data)
    out = self.bn(out)
    out = self.relu(out)
    return out


class FcBnRelu(torch.nn.Module):
  r''' A sequence of :class:`FC`, :class:`BatchNorm` and :class:`Relu`.
  '''

  def __init__(self, in_channels: int, out_channels: int):
    super().__init__()
    self.flatten = torch.nn.Flatten(start_dim=1)
    self.fc = torch.nn.Linear(in_channels, out_channels, bias=False)
    self.bn = normalization(out_channels)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, data):
    r''''''

    out = self.flatten(data)
    out = self.fc(out)
    out = self.bn(out)
    out = self.relu(out)
    return out