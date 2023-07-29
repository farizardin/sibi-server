import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

import sys
# sys.path.append("./model/Temporal_shift/")

# from cuda.shift import Shift

num_nodes = 4
weight = 2

## TOOLS PY
def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):  # 除以每列的和
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A

## END TOOLS PY

num_node = 4
self_link = [(i, i) for i in range(num_node)]
#inward = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9), (9, 8),
#         (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0), (17, 15),
#          (16, 14)]
# inward = [(0, 1),(2, 1),(3, 2),(4, 3),(5, 1),(6, 5),(7, 6), (8, 2),(9, 8),(10, 9),(11, 5), (12, 11),(13, 12),  
#                (14, 0), (15, 0), (16, 14), (17, 15)]
inward = [(0, 1),(1, 2),(1, 3)]
outward = [(j, i) for (i, j) in inward]

def get_adjacency_matrix(labeling_mode=None):
    # if labeling_mode is None:
    #     return self.A
    if labeling_mode == 'spatial':
        A = get_spatial_graph(num_node, self_link, inward, outward)
    else:
        raise ValueError()
    return A

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def conv_init(conv):
    nn.init.kaiming_normal(conv.weight, mode='fan_out')
    nn.init.constant(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant(bn.weight, scale)
    nn.init.constant(bn.bias, 0)

class Shift_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(Shift_gcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.Linear_weight = nn.Parameter(torch.zeros(in_channels, out_channels, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0,math.sqrt(1.0/out_channels))

        self.Linear_bias = nn.Parameter(torch.zeros(1,1,out_channels,requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.constant(self.Linear_bias, 0)

        self.Feature_Mask = nn.Parameter(torch.ones(1,num_nodes,in_channels, requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.constant(self.Feature_Mask, 0)

        self.bn = nn.BatchNorm1d(num_nodes*out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        ### Upper section is the same, modify code below
        index_array = np.empty(num_nodes*in_channels).astype(np.int)
        for i in range(num_nodes):
            for j in range(in_channels):
                index_array[i*in_channels + j] = (i*in_channels + j + j*in_channels)%(in_channels*num_nodes)
        self.shift_in = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)
        print("index arr in:",index_array)

        index_array = np.empty(num_nodes*out_channels).astype(np.int)
        for i in range(num_nodes):
            for j in range(out_channels):
                index_array[i*out_channels + j] = (i*out_channels + j - j*out_channels)%(out_channels*num_nodes)
        self.shift_out = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)
        print("index arr out:",index_array)
        #### Code modification ends here

class Shift_gcn_local(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(Shift_gcn_local, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.Linear_weight = nn.Parameter(torch.zeros(in_channels, out_channels, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0,math.sqrt(1.0/out_channels))

        self.Linear_bias = nn.Parameter(torch.zeros(1,1,out_channels,requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.constant(self.Linear_bias, 0)

        self.Feature_Mask = nn.Parameter(torch.ones(1,25,in_channels, requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.constant(self.Feature_Mask, 0)

        self.bn = nn.BatchNorm1d(25*out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        
        # A adalah adjacency matrix yg berbentuk (c*n*n) (c = channel, n = node), dengan c awal = 3
        # contoh:
        # shape = (3, 4, 4)
        # [[[1.  0.  0.  0. ]
        #   [0.  1.  0.  0. ]
        #   [0.  0.  1.  0. ]
        #   [0.  0.  0.  1. ]]

        #  [[0.  0.  0.  0. ]
        #   [1.  0.  0.  0. ]
        #   [0.  0.5 0.  0. ]
        #   [0.  0.5 0.  0. ]]

        #  [[0.  1.  0.  0. ]
        #   [0.  0.  1.  1. ]
        #   [0.  0.  0.  0. ]
        #   [0.  0.  0.  0. ]]]

        # hasil np.sum
        # [[1.  1.  0.  0. ]
        #  [1.  1.  1.  1. ]
        #  [0.  0.5 1.  0. ]
        #  [0.  0.5 0.  1. ]]

        # c1 = hubungan dengan node sendiri
        # c2 = hubungan node tetangga (outward) dengan node awal (inward)
        # c3 = hubungan node awal (inward) dengan node tetangga (outward)
        # np.sum = menambah value tensor A n*c sehingga hasilnya 1 dimensi (c)
        print(A)
        A = np.sum(A,0)
        print(A)

        n = 4
        channels = in_channels
        if channels == 3:
            index_array = np.arange(n*channels).astype(np.int)
            print(index_array)
        else:
            ### Fancy Indexing
            ## NumPy arrays can be indexed with slices, 
            # but also with boolean or integer arrays (masks). 
            # This method is called fancy indexing. It creates copies not views.
            ##
            # Value didalam A yg 0 akan tetap 0
            # Value didalam A yg selain 0 akan menjadi 1
            A[A==0] = 0
            A[A!=0] = 1
            # membuat np array dengan value 0 dengan dimensi n*c (node * channel)
            index_array = np.zeros(n*channels).astype(np.int)
            for i in range(n):
                partition = np.sum(A[i])
                channel_per_partition = channels//partition
                current_A = A[i]
                print(current_A)
                current_A[i] = 0
                neighbors = np.nonzero(current_A)[0]
                for j in range(int(partition)):
                    if j == 0:
                        index_array[int(i*channels):int(i*channels + channel_per_partition)] = np.arange(i*channels,i*channels + channel_per_partition).astype(np.int)
                    else:
                        index_array[int(i*channels + j*channel_per_partition):int(i*channels + (j+1)*channel_per_partition)] = np.arange(neighbors[j-1]*channels + j*channel_per_partition,neighbors[j-1]*channels + (j+1)*channel_per_partition).astype(np.int)
        self.shift_in = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)

        n = 4
        channels = out_channels
        if channels == 3:
            index_array = np.arange(n*channels).astype(np.int)
        else:
            A[A==0] = 0
            A[A!=0] = 1
            index_array = np.zeros(n*channels).astype(np.int)
            for i in range(n):
                partition = np.sum(A[i])
                channel_per_partition = channels//partition
                current_A = A[i]
                print("current A",current_A)
                current_A[i] = 0
                neighbors = np.nonzero(current_A)[0]
                for j in range(int(partition)):
                    if j == 0:
                        index_array[int(i*channels):int(i*channels + channel_per_partition)] = np.arange(i*channels,i*channels + channel_per_partition).astype(np.int)
                    else:
                        index_array[int(i*channels + j*channel_per_partition):int(i*channels + (j+1)*channel_per_partition)] = np.arange(neighbors[j-1]*channels + j*channel_per_partition,neighbors[j-1]*channels + (j+1)*channel_per_partition).astype(np.int)
        self.shift_out = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)
        
class Modified_shift_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(Modified_shift_gcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.Linear_weight = nn.Parameter(torch.zeros(in_channels, out_channels, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0,math.sqrt(1.0/out_channels))

        self.Linear_bias = nn.Parameter(torch.zeros(1,1,out_channels,requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.constant(self.Linear_bias, 0)

        self.Feature_Mask = nn.Parameter(torch.ones(1,num_nodes,in_channels, requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.constant(self.Feature_Mask, 0)

        self.bn = nn.BatchNorm1d(num_nodes*out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

        # print(A)
        A = np.sum(A,0)
        A[A==0] = 0
        A[A!=0] = 1
        node = 4
        weight = 2
        # print(A)

        index_array = self.shift(A,node,in_channels,weight)
        self.shift_in = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)

        index_array = self.shift(A,node,out_channels,weight)
        self.shift_out = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)

    def shift(self, A, node_num, channels, weight):
        print("RUN SHIFT")
        index_array = np.arange(0,node_num*channels)
        index_array = index_array.reshape(node_num,channels)
        index_array_copy = np.copy(index_array)

        for i in range(node_num):
            current_A = A[i]
            # print(current_A)
            current_A[i] = 0
            neighbors = np.nonzero(current_A)[0]
            # print(neighbors)
            j = 0
            counter_i = i + 1
            while j < channels: 
                next_node = counter_i % node_num
                if j < weight:
                    j += weight
                elif next_node == i:
                    j += weight
                    counter_i += 1
                elif next_node  in neighbors:
                    index_array[i][j:j+weight] = index_array_copy[next_node][j:j+weight]
                    j += weight
                    counter_i += 1
                else:
                    index_array[i][j] = index_array_copy[next_node][j]
                    j += 1
                    counter_i += 1
        print(index_array)
        return index_array.flatten()

def main():
    # A = np.random.randint(2, size=(3, 4, 4))
    A = get_adjacency_matrix(labeling_mode='spatial')
    # A = np.random.rand(3,4,4)
    print(A.shape)
    # gcn1 = Shift_gcn_local(3, 64, A)
    # gcn1 = Shift_gcn(3, 64, A)
    gcn1 = Modified_shift_gcn(3, 12, A)
    # gcn1 = Modified_shift_gcn(64, 64, A)

main()