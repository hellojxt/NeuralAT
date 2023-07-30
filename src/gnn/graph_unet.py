# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import torch.nn
from typing import Dict, Optional


from src.gnn.graph_tree import Data
from src.gnn.graph_tree import GraphTree

from src.gnn.resblocks import *

from src.gnn.blocks import *



class GraphUNet(torch.nn.Module):
  r''' UNet but with graph neural network
  no octree data structure

  use graphtree as the substitution of octree
  '''

  def __init__(self, in_channels: int, out_channels: int, interp: str = 'linear',
               nempty: bool = False, **kwargs): 
    super(GraphUNet, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.config_network()
    self.encoder_stages = len(self.encoder_blocks)
    self.decoder_stages = len(self.decoder_blocks)

    # encoder
    self.conv1 = GraphConvBnRelu(in_channels, self.encoder_channel[0])

    self.downsample = torch.nn.ModuleList(
        [PoolingGraph() for i in range(self.encoder_stages)]
    )
    self.encoder = torch.nn.ModuleList(
        [GraphResBlocks(self.encoder_channel[i], self.encoder_channel[i+1],
                        resblk_num=self.encoder_blocks[i], resblk=self.resblk)
         for i in range(self.encoder_stages)]
    )

    # decoder
    channel = [self.decoder_channel[i] + self.encoder_channel[-i-2]
               for i in range(self.decoder_stages)]
    self.upsample = torch.nn.ModuleList(
        [UnpoolingGraph() for i in range(self.decoder_stages)]
    )
    self.decoder = torch.nn.ModuleList(
        [GraphResBlocks(channel[i], self.decoder_channel[i+1],
                        resblk_num=self.decoder_blocks[i], resblk=self.resblk, bottleneck=self.bottleneck)
         for i in range(self.decoder_stages)]
    )

    # header
    # channel = self.decoder_channel[self.decoder_stages]
    #self.octree_interp = ocnn.nn.OctreeInterp(interp, nempty)
    self.header = torch.nn.Sequential(
        Conv1x1BnRelu(self.decoder_channel[-1], self.decoder_channel[-1]),
        Conv1x1(self.decoder_channel[-1], self.out_channels, use_bias=True))
    
    # a embedding decoder function
    self.embedding_decoder_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.out_channels, self.out_channels, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(self.out_channels, self.out_channels, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(self.out_channels, 1, bias=True)            
        )

  def config_network(self):
    r''' Configure the network channels and Resblock numbers.
    '''
    self.encoder_blocks = [2, 3, 3, 3]
    self.decoder_blocks = [2, 3, 3, 3]
    self.encoder_channel = [32, 32, 64, 64, 128]
    self.decoder_channel = [128, 64, 64, 32, 32]

    self.bottleneck = 4
    self.resblk = GraphResBlock2

  def unet_encoder(self, data: torch.Tensor, graphtree: GraphTree, depth: int):
    r''' The encoder of the U-Net.
    '''

    convd = dict()
    convd[depth] = self.conv1(data, graphtree, depth)
    for i in range(self.encoder_stages):
      d = depth - i
      conv = self.downsample[i](convd[d], graphtree, i+1)
      convd[d-1] = self.encoder[i](conv, graphtree, d-1)
    return convd

  def unet_decoder(self, convd: Dict[int, torch.Tensor], graphtree: GraphTree, depth: int):
    r''' The decoder of the U-Net. 
    '''

    deconv = convd[depth]
    for i in range(self.decoder_stages):
      d = depth + i
      deconv = self.upsample[i](deconv, graphtree, self.decoder_stages-i)
      deconv = torch.cat([convd[d+1], deconv], dim=1)  # skip connections
      deconv = self.decoder[i](deconv, graphtree, d+1)
    return deconv

  def forward(self, data: torch.Tensor, graphtree: GraphTree, depth: int):
    """_summary_

    Args:
        data (torch.Tensor): _description_
        graphtree (GraphTree): _description_
        depth (int): _description_
    """

    convd = self.unet_encoder(data, graphtree, depth)
    deconv = self.unet_decoder(convd, graphtree, depth - self.encoder_stages)

   # interp_depth = depth - self.encoder_stages + self.decoder_stages
   # feature = self.octree_interp(deconv, graphtree, interp_depth, query_pts)
    embedding = self.header(deconv)
    
    return embedding


