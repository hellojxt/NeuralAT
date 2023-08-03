from src.gnn.graph_unet import GraphUNet
from src.gnn.graph_tree import GraphTree
from src.gnn.graph_tree import Data as GraphData
import torch
import torch.nn as nn

from torch_geometric.nn import knn_graph

class GraphUNetWrapper(nn.Module):
    def __init__(self, in_feature_dim, out_feature_dim):
        super().__init__()
        self.model = GraphUNet(in_channels=in_feature_dim, out_channels=out_feature_dim)
        
        
        
    def forward(self, data, input_concat=False):
        # extract data from data
        feature = data.x        # [total_points, feature_dim]
        position = data.pos     # [total_points, 3]
        batch = data.batch      # [total_points]. It looks like this: [0,0,0,0,1,1,1,1,2,2,2,2,...]
        
        # create GraphData
        graph_tree_list = []
        for i in range(batch.max()+1):
            x_temp = feature[batch==i]
            edge_index_temp = knn_graph(x_temp, k=7, batch=None, loop=False)  # build a graph with knn
            graph_data_temp = GraphData(x=x_temp, edge_index=edge_index_temp)
            graph_tree_temp = GraphTree()
            graph_tree_temp.build_single_graphtree(graph_data_temp)
            graph_tree_list.append(graph_tree_temp)
        
        # merge the graph tree
        graph_tree = GraphTree(batch_size=batch.max()+1)
        graph_tree.merge_graphtree(graph_tree_list)
        
        # forward the graph unet
        output = self.model(data=feature, graphtree=graph_tree, depth=graph_tree.depth)
        
        return output
        
        
        
        