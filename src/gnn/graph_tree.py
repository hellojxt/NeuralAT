from __future__ import annotations
from typing import Callable, Optional, Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric
from torch_geometric.nn import SAGEConv
#from torch_geometric.nn import avg_pool, voxel_grid

#from utils.thsolver import default_settings

from src.gnn.decide_edge_type import *

"""
graph neural network coarse to fine data structure
keep records of a "graph tree"
"""

#from torch_geometric.data import Data

class Data:
    #@profile
    def __init__(self, x=None, edge_index=None, edge_attr=None, 
                 batch=None, edges_size_of_each_subgraph=None):
        """
        a rewrite of torch_geometric.data.Data
        get rid of its self-aleck re-indexing
        :param edge_attr: the type of edges.
        """
        # basic attributes
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        
        # batching information
        self.batched_data = []
        if batch == None or batch.max() == 0:
            self.batch = torch.zeros([x.shape[0]], dtype=torch.int64)
        else:
            self.batch = batch.to(torch.int64)    # batch should be int64 for torch.scatter
            # generate a list of batch indices
            for i in range(batch.max().item()+1):
                # find the index of the first vertex in batch i
                first_idx = torch.where(batch == i)[0][0]
                last_idx = torch.where(batch == i)[0][-1]
                
                # find all edges that connect vertices in batch i
                edges_mask = torch.zeros([edge_index.shape[1]], dtype=torch.bool)
                edges_size_of_each_subgraph = [0] + edges_size_of_each_subgraph
                cumsum_edge_size = torch.cumsum(torch.tensor(edges_size_of_each_subgraph), dim=0)
                edges_mask[cumsum_edge_size[i]:cumsum_edge_size[i+1]] = True
                        
                edges_index_batch = edge_index[:, edges_mask]
                edges_attr_batch = edge_attr[edges_mask]
                x_in_batch = x[first_idx:last_idx+1]
               # batch_debug = batch[first_idx:last_idx+1]
                
                temp_graph = Data(x=x_in_batch, edge_index=edges_index_batch, edge_attr=edges_attr_batch)
                self.batched_data.append(temp_graph)
                

    def to(self, target):
        # trans data from gpu to cpu or vice versa
        self.x = self.x.to(target)
        self.edge_index = self.edge_index.to(target)
        if self.edge_attr != None:
            self.edge_attr = self.edge_attr.to(target)
        if self.batch != None:
            self.batch = self.batch.to(target)
        return self
    
    def __getitem__(self, index: int):
        # get a subgraph according to index and batch
        # 暂时不支持切片操作
        # TODO：使得slice的结果是一个clone
        assert type(index) == int, "index must be an int"
        assert index < self.batch.max().item() + 1, "index out of range"
        if self.batched_data.__len__() == 0 and index == 0:
            # 这里不直接返回self是为了内存上的一致性：如果您返回self，外面的代码可能会修改self的内容，这会导致本类的其他方法出错
            return Data(x=self.x, edge_index=self.edge_index, edge_attr=self.edge_attr, batch=self.batch)
        return self.batched_data[index]
        # TODO: 
        

    def cuda(self):
        return self.to("cuda")

    def cpu(self):
        return self.to("cpu")



def avg_pool(
    cluster: torch.Tensor,
    data: Data,
    transform: Optional[Callable] = None,
) -> Data:
    """a wrapper of torch_geometric.nn.avg_pool"""
    data_torch_geometric = torch_geometric.data.Data(x=data.x, edge_index=data.edge_index)
    new_data = torch_geometric.nn.avg_pool(cluster, data_torch_geometric, transform=transform)
    ret = Data(x=new_data.x, edge_index=new_data.edge_index)
    return ret


def avg_pool_maintain_old(
    cluster: torch.Tensor,
    data: Data,
    transform: Optional[Callable] = None,
):
    """a wrapper of torch_geometric.nn.avg_pool, but maintain the old graph"""
    data_torch_geometric = torch_geometric.data.Data(x=data.x, edge_index=data.edge_index)
    new_layer = torch_geometric.nn.avg_pool(cluster, data_torch_geometric, transform=transform)
    # connect the corresponding node in the two layers
    


def pooling(data: Data, size: float, normal_aware_pooling):
    """
    do pooling according to x. a new graph (x, edges) will be generated after pooling.
    This function is a wrapper of some funcs in pytorch_geometric. It assumes the 
    object's coordinates range from -1 to 1.
    
    normal_aware_pooling: if True, only data.x[..., :3] will be used for grid pooling.
    """
    assert type(size) == float
    if normal_aware_pooling == False:
        x = data.x[..., :3]
    else:
        x = data.x[..., :6]
        # we assume x has 6 feature channels, first 3 xyz, then 3 nxnynz
        # grid size here waits for fine-tuning
        n_size = size * 3.   # TODO: a hyper parameter, controling "how important the normal vecs are, compared to xyz coords"
        size = [size, size, size, n_size, n_size, n_size]

    edges = data.edge_index
    cluster = torch_geometric.nn.voxel_grid(pos=x, size=size, batch=None,
                                            start=[-1, -1, -1.], end=[1, 1, 1.])

    # keep max index smaller than # of unique (e.g., [4,5,4,3] --> [0,1,0,2])
    mapping = cluster.unique()
    mapping += mapping.shape[0]
    cluster += mapping.shape[0]

    for i in range(int(mapping.shape[0])):  # TODO: 这里可以优化一下，循环数量上千
        cluster[cluster == mapping[i]] = i

    # sanity check，左侧是cluster后更coarse层的点数，右边是cluster前更fine层的点数，右边必然大于左边
    assert cluster.unique().shape[0] < cluster.shape[0], "池化过程中计算cluster出错。这可能是因为您没有将输入网格数据归一化到 unit cube 区间内；在某一层没有减小顶点规模也可能触发此警告。"
    
    return cluster #.contiguous()




def add_self_loop(data: Data):
    # avg_pool 会把vertex的自环给清除掉，这里我们补上
    device = data.x.device
    n_vert = data.x.shape[0]
    self_loops = torch.tensor([[i for i in range(int(n_vert))]])
    self_loops = self_loops.repeat(2, 1)
    new_edges = torch.zeros([2, data.edge_index.shape[1] + n_vert], dtype=torch.int64)
    new_edges[:, :data.edge_index.shape[1]] = data.edge_index
    new_edges[:, data.edge_index.shape[1]:] = self_loops

    return Data(x=data.x, edge_index=new_edges).to(device)




class GraphTree:
    """
    build a naive graph tree from a graph
    names of arguments and functions are self-explainable

    notes:
    - coordinates of input vertices should be in [-1, 1]
    - this class ONLY handles xyz coordinates, it does NOT handle features
    (即，这个类只处理坐标，负责构建一个树结构，它不负责处理任何feature)

    """
    def __init__(self, depth: int=4,
                 smallest_grid=2/2**5,
                 batch_size: int=1,
                 adj_layer_connected=False):
        """
        :param adj_layer_connected: 
        if False, each layer only consists of the graph of that layer. 
        if True, each layer consists of the graph of that layer and below layers
        
        处于简单考虑，暂时是这样的：

        假设depth=3，则可以通过self.graphtree访问到如下内容：
        self.graphtree[0] = 原始的graph, type: Data
        self.graphtree[1] = 以 smallest_grid 为栅格进行合并得到的graph, type: Data
        self.graphtree[2] = 以 smallest_grid * (2**1) 为栅格进行合并得到的graph, type: Data
        self.graphtree[3] = 以 smallest_grid * (2**2) 为栅格进行合并得到的graph, type: Data

        __init__ 仅仅是规定了一个graphtree的超参数，真正的构造函数其实应该是 build_single_graphtree 或 merge_graphtree
        """

        assert smallest_grid * (2**(depth-1)) <= 1, "最大的栅格的大小超过了整个空间(-0.5~0.5)，这可能是不恰当的"
        assert depth >= 0

        self.device = "cuda"
        self.depth = depth
        self.batch_size = batch_size
        self.smallest_grid = smallest_grid
        self.normal_aware_pooling = False#default_settings.get_global_value("normal_aware_pooling")

        # 下列两行的数据格式为:
        # 假设本graph tree由Y个图构成，每个图按照我们定义的简化方式可以分为X层，那么
        # self.batch_vertices_sizes[i][j]: 第i个图的第j层有多少 vertices
        # self.batch_edges_sizes[i][j]: 第i个图的第j层有多少 edges
        # 例：{0: [5000, 2000, 400], 1: [3000, 800, 100]}
        self.vertices_sizes = {}
        self.edges_sizes = {}
        #
        self.treedict = {}
        self.cluster = {}

    def build_single_graphtree(self, original_graph: Data):
        """
        build a graph-tree of **one** graph
        """
        #assert type(original_graph) == Data
        #assert original_graph.x.shape[1] == 3, "original_graph.x 的特征维度需为xyz三维"

        graphtree = {}
        cluster = {}
        vertices_size = {}
        edges_size = {}

        for i in range(self.depth+1):
            if i == 0:
                original_graph = add_self_loop(original_graph)
                # if original graph do not have edge types, assign it 
                if original_graph.edge_attr == None:
                    edges = original_graph.x[original_graph.edge_index[0]] \
                            - original_graph.x[original_graph.edge_index[1]]
                    edges_attr = decide_edge_type_distance(edges, return_edge_length=False)
                    original_graph.edge_attr = edges_attr
                graphtree[0] = original_graph
                cluster[0] = None
                edges_size[0] = original_graph.edge_index.shape[1]
                vertices_size[0] = original_graph.x.shape[0]
                continue

            clst = pooling(graphtree[i-1], self.smallest_grid * (2**(i-1)), normal_aware_pooling=self.normal_aware_pooling)
            new_graph = avg_pool(cluster=clst, data=graphtree[i-1], transform=None)
            new_graph = add_self_loop(new_graph)
            # assign edge type
            edges = new_graph.x[new_graph.edge_index[0]] \
                    - new_graph.x[new_graph.edge_index[1]]
            edges_attr = decide_edge_type_distance(edges, return_edge_length=False)
            new_graph.edge_attr = edges_attr

            graphtree[i] = new_graph
            cluster[i] = clst
            edges_size[i] = new_graph.edge_index.shape[1]
            vertices_size[i] = new_graph.x.shape[0]

        self.treedict = graphtree
        self.cluster = cluster
        self.vertices_sizes = vertices_size
        self.edges_sizes = edges_size

        # 下一行代码会生成 mesh pooling layers 的可视化。debug 时可用
        #self.export_obj()


    # @staticmethod
    def merge_graphtree(self, original_graphs: list[GraphTree], debug_report=False):
        """
        多个 graphtree 构成一个大 graphtree。

        举个例子：
        graphtree_1 有两层graph，第一层有500个v，500个f；第二层有100个v，100个f
        graphtree_2 有两层graph，第一层有220个v，220个f；第二层有50个v，50个f
        则合并后得到的大 graphtree 有两层，第一层720个v，720个f；第二层150个v，150个f。注意它们的连接性关系并没有改变

        合并后得到的大 graphtree 各层及其连接关系被存入 self.graphtree, self.cluster 里。
        注意合并时涉及到稍有复杂的index变换，但因为我们的graphtree显得更加复杂。
        这里，您可以重写 torch geometric 的 mini-batching 行为，但这里直接完整重写一遍。

        我有些担心这里for循环有点多是否会导致性能下降，不过考虑到d和i都是相对较小的数（d约为4~6，i约为16），
        这应该不是大问题。而且，这一部分的数据处理是cpu做，应该还好

        """
        assert len(self.cluster) == 0 and len(self.treedict) == 0, "请在全新的实例上调用本方法"
        assert original_graphs.__len__() == self.batch_size, "请确保输入图的数量等于batch_size"

        # re-indexing
        for d in range(self.depth+1):
            # graphtree 每一层都合并一次图
            num_vertices = [0]
            for i, each in enumerate(original_graphs):
                num_vertices.append(each.vertices_sizes[d])
            cum_sum = torch.cumsum(torch.tensor(num_vertices), dim=0)
            for i in range(original_graphs.__len__()):
                original_graphs[i].treedict[d].edge_index += cum_sum[i]
                # 第0层没有cluster，自然不需要处理
                if d != 0:
                    original_graphs[i].cluster[d] += cum_sum[i]

        # merge 生成新的graphtree
        for d in range(self.depth+1):
            graphtrees_x, graphtrees_e, graphtrees_e_type, clusters = [], [], [], []
            batching = []
            for i in range(original_graphs.__len__()):
                graphtrees_x.append(original_graphs[i].treedict[d].x)
                graphtrees_e.append(original_graphs[i].treedict[d].edge_index)
                graphtrees_e_type.append(original_graphs[i].treedict[d].edge_attr)
                clusters.append(original_graphs[i].cluster[d])
                # 生成 batch 信息，即记录每个顶点属于哪个图
                batching.append(torch.ones([original_graphs[i].treedict[d].x.shape[0]], dtype=torch.int32) * i)
                
            # construct new graph
            temp_data = Data(x=torch.cat(graphtrees_x, dim=0).float(),          # convert to float32
                             edge_index=torch.cat(graphtrees_e, dim=1),
                             edge_attr=torch.cat(graphtrees_e_type, dim=0),   # edge_attr shape: [E]
                             batch=torch.cat(batching, dim=0),
                             edges_size_of_each_subgraph=[original_graphs[i].treedict[d].edge_index.shape[1] for i in range(original_graphs.__len__())],
                             )
            # 第0层没有cluster，自然不需要处理
            if d != 0:
                temp_clst = torch.cat(clusters, dim=0)
            else:
                temp_clst = None
            self.treedict[d] = temp_data
            self.cluster[d] = temp_clst
            self.edges_sizes = temp_data.edge_index.shape[1]
            self.vertices_sizes = len(temp_data.x)
        

        # sanity check
        if debug_report == True:
            # a simple unit test
            for d in range(self.depth+1):
                num_edges_before = 0
                for i in range(original_graphs.__len__()):
                    num_edges_before += original_graphs[i].treedict[d].edge_index.shape[1]
                num_edges_after = self.treedict[d].edge_index.shape[1]
                print(f"Before merge, at d={d} there's {num_edges_before} edges; {num_edges_after} afterwards")




    #####################################################
    # Util
    #####################################################

    def cuda(self):
        # move all tensors to cuda
        for each in self.treedict.keys():
            self.treedict[each] = self.treedict[each].cuda()
        for each in self.cluster.keys():
            if self.cluster[each] is None:
                continue
            self.cluster[each] = self.cluster[each].cuda()
        return self


    def export_obj(self, path="logs/visualization/"):
        """
        导出obj格式的多层次graph，做可视化和debug用
        使用时最好保证调用此函数的实例只包含一个graph
        """
        if self.batch_size != 1:
            print("最好保证调用此函数的实例只包含一个graph，否则画出来会很乱")

        def graph_2_obj(graph: Data, file_name: str):
            with open(file_name, "w") as f:
                for line in graph.x:
                    # 创建两个离得很近的几乎重合的 vertex
                    s0 = str(float(line[0]))
                    s1 = str(float(line[1]))
                    s2 = str(float(line[2]))
                    f.write("v " + s0 + " " + s1 + " " + s2 + "\n")
                    s0 = str(float(line[0]) + 0.0000001)
                    s1 = str(float(line[1]) + 0.0000001)
                    s2 = str(float(line[2]) + 0.0000001)
                    f.write("v " + s0 + " " + s1 + " " + s2 + "\n")
                for line in graph.edge_index.t():
                    # obj 文件是 1-index 的，而我们内存里的 edge 是 0-index，所以最前面有一个“1+”
                    # 下面三行，用一个极其细长的三角形来“模拟”一条边
                    s0 = str(1+int(float(line[0])) * 2)
                    s1 = str(1+int(float(line[1])) * 2)
                    s2 = str(1+int(float(line[1])) * 2 + 1)
                    # edge里面有些自环，不画
                    if s0 == s1:
                        continue
                    f.write("f " + s0 + " " + s1 + " " + s2 + "\n")
            return None

        for i, each in enumerate(self.treedict):
            file_name = path + "visualize_graphtree_" + str(i) + ".obj"
            graph_2_obj(self.treedict[each], file_name=file_name)
            print(f"level {i} finished exporting obj!")





