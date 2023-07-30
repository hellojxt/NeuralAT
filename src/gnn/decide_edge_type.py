
import torch

# helper functions, decide the type of a edge

###############################################################
# For directional graph conv
###############################################################

def decide_edge_type_directional(vec: torch.tensor, 
                     catagories):
    """
    classify each vec into many categories. 
    vec: N x 3
    ret: N 
    """
    if catagories == 7:
        return decide_edge_type_7_directional(vec)
    elif catagories == 25:
        assert False, "dont use this"
        return decide_edge_type_25_directional(vec)
    else:
        raise NotImplementedError


def decide_edge_type_7_directional(vec: torch.tensor, epsilon=0.00001, return_edge_length=True):
    """
    classify each vec into 7 categories. category #6 is self-loop
    vec: N x 3
    ret: N
    """
    positive_x = torch.maximum(vec[..., 0], torch.zeros_like(vec[..., 0]))
    positive_y = torch.maximum(vec[..., 1], torch.zeros_like(vec[..., 0]))
    positive_z = torch.maximum(vec[..., 2], torch.zeros_like(vec[..., 0]))
    negative_x = - torch.minimum(vec[..., 0], torch.zeros_like(vec[..., 0]))
    negative_y = - torch.minimum(vec[..., 1], torch.zeros_like(vec[..., 0]))
    negative_z = - torch.minimum(vec[..., 2], torch.zeros_like(vec[..., 0]))

    ary = torch.stack([positive_x, positive_y, positive_z, negative_x, negative_y, negative_z])
    ary = ary.transpose(0, 1)
    edge_type = ary.argmax(axis=1)

    # need to filter out self-edges. calculate the vec_length, those vecs with length<epsilon are self loops
    vec_length = torch.norm(ary, dim=1)
    self_loops_mask = vec_length <= epsilon
    edge_type[self_loops_mask] = 6

    if return_edge_length == True:
        return edge_type, vec_length
    else:
        return edge_type, None

def decide_edge_type_25_directional(vec: torch.tensor, epsilon=0.00001):
    """
    classify each vec into 25 categories, according to the following rules:
    the type of an edge is determined by its two largest direction. for example,
    an edge [0.4, 0.8, -0.45], the largest direction is +y, while second largest
    direction is -z. So its type is (+y-z).
    
    There are 6 * 4 + 1 = 25 catagories of edges.
    
    vec: N x 3
    ret: N
    """
    positive_x = torch.maximum(vec[..., 0], torch.zeros_like(vec[..., 0]))
    positive_y = torch.maximum(vec[..., 1], torch.zeros_like(vec[..., 0]))
    positive_z = torch.maximum(vec[..., 2], torch.zeros_like(vec[..., 0]))
    negative_x = - torch.minimum(vec[..., 0], torch.zeros_like(vec[..., 0]))
    negative_y = - torch.minimum(vec[..., 1], torch.zeros_like(vec[..., 0]))
    negative_z = - torch.minimum(vec[..., 2], torch.zeros_like(vec[..., 0]))

    #breakpoint()
    ary = torch.stack([positive_x, positive_y, positive_z, negative_x, negative_y, negative_z])
    ary = ary.transpose(0, 1)
    largest_edge_idx = ary.argmax(axis=1)
    
    ary[torch.arange(ary.shape[0]), largest_edge_idx] = 0
    second_largest_edge_idx = ary.argmax(axis=1)
    
    # 根据top2方向决定edge type
    edge_type = largest_edge_idx * 10 + second_largest_edge_idx
    mapping = {
                1:0,
                2:1,
                4:2,
                5:3,
                10:4,
                12:5,
                13:6,
                15:7,
                20:8,
                21:9,
                23:10,
                24:11,
                31:12,
                32:13,
                34:14,
                35:15,
                40:16,
                42:17,
                43:18,
                45:19,
                50:20,
                51:21,
                53:22,
                54:23,
                }
    for each in mapping.keys():
        edge_type[edge_type==each] = mapping[each]
    #breakpoint()
    # need to filter out self-edges. calculate the vec_length, those vecs with length<epsilon are self loops
    vec_length = torch.norm(ary, dim=1)
    self_loops_mask = vec_length <= epsilon
    edge_type[self_loops_mask] = 24
    
    assert edge_type.max() < 25

    return edge_type


###############################################################
# For distance graph conv
###############################################################

def decide_edge_type_distance(vec: torch.tensor, 
                        method="predefined",
                        return_edge_length=True):
    """
    classify each vec into many categories. 
    vec: N x 3
    ret: N
    """
    if method == "predefined":
        return decide_edge_type_predefined_distance(vec, return_edge_length=return_edge_length)
    else:
        raise NotImplementedError


def decide_edge_type_predefined_distance(vec: torch.tensor, epsilon=0.00001, return_edge_length=True):
    """
    classify each vec into N categories, according to the length of the vcector. 
    the last category is self-loop
    vec: N x 3
    ret: N
    """
    positive_x = torch.maximum(vec[..., 0], torch.zeros_like(vec[..., 0]))
    positive_y = torch.maximum(vec[..., 1], torch.zeros_like(vec[..., 0]))
    positive_z = torch.maximum(vec[..., 2], torch.zeros_like(vec[..., 0]))
    negative_x = - torch.minimum(vec[..., 0], torch.zeros_like(vec[..., 0]))
    negative_y = - torch.minimum(vec[..., 1], torch.zeros_like(vec[..., 0]))
    negative_z = - torch.minimum(vec[..., 2], torch.zeros_like(vec[..., 0]))

    ary = torch.stack([positive_x, positive_y, positive_z, negative_x, negative_y, negative_z])
    ary = ary.transpose(0, 1)
    
    device = vec.device
 
    edge_type = torch.ones([len(vec), 1]).to(device) * 999
    vec_length = torch.norm(ary, dim=1)
    #breakpoint()
    
    # How this work? for example, say the dist_threshold = [1, 0.1, 0.01, epsilon]
    # edge_length > threshold[0] --> type 0 edge
    # edge_length > threshold[1] --> type 1 edge
    # edge_length > threshold[2] --> type 2 edge
    # edge_length > eps          --> type 3 edge
    # edge_length <= eps         --> type 4 edge (self-loop)
    
    # 由于UNet多尺度的存在，最好使用相对值（如果使用绝对值，会使得UNet的不同层次因尺度天然不同而只学到有限的信息）
    # [0.643+0.143, 0.428+0.143, 0.214+0.143] 是这样来的：对于一般mesh而言，有14.3%的edge是自环。因此，对于剩下的85.7%，我们将其均分成四份
    #thres_1 = torch.quantile(vec_length, torch.tensor([0.643+0.143, 0.428+0.143, 0.214+0.143]).to(device))  # tensor of shape [3]
  #  print(thres_1)
    dist_threshold = [epsilon]
    #dist_threshold = [thres_1[0], thres_1[1], thres_1[2], epsilon]         # TODO: 这里的阈值取了所有edge长度的四分点。这是否可能有问题？
                                                                           # 一些证据暗示对于部分大小较小的mesh，UNet的最高层可能只有自环，这是否可能产生部分问题。
    
    for i in range(len(dist_threshold)-1, -1, -1):
        dist_mask = vec_length > dist_threshold[i]
        edge_type[dist_mask] = i
        
    # self-loop
    self_loops_mask = vec_length <= epsilon
    edge_type[self_loops_mask] = len(dist_threshold)
    
    # squeeze to 1d tensor
    edge_type = edge_type.squeeze(-1)
    edge_type = edge_type.long()
    
    '''
    ratio_0 = torch.sum(edge_type == 0) / len(edge_type)
    ratio_1 = torch.sum(edge_type == 1) / len(edge_type)
    ratio_2 = torch.sum(edge_type == 2) / len(edge_type)
    ratio_3 = torch.sum(edge_type == 3) / len(edge_type)
    ratio_4 = torch.sum(edge_type == 4) / len(edge_type)
    ratio_5 = torch.sum(edge_type == 5) / len(edge_type)
    print(f"Ratio: {ratio_0}, {ratio_1}, {ratio_2}, {ratio_3}, {ratio_4}, {ratio_5}")
    '''
    
    # assertion, suppose there are N thresholds (epsilon included), make sure that 
    # the type of different edges are at most N+1
   # breakpoint()
    assert edge_type.max() == len(dist_threshold)     # there must be edges indexed N, since self-loop exists in all meshes
    assert edge_type.min() >= 0                       # note edge_type indexed N-1 may not exists, since edges of that length may not exists in this mesh

    if return_edge_length == True:
        return edge_type, vec_length
    else:
        return edge_type

