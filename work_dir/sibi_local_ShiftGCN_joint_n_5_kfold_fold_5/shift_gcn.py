import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

import sys
sys.path.append("./model/Temporal_shift/")

from cuda.shift import Shift

num_node = 115

def import_class(name):
    print(name)
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


class tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class Shift_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(Shift_tcn, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bn = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        bn_init(self.bn2, 1)
        self.relu = nn.ReLU(inplace=True)
        self.shift_in = Shift(channel=in_channels, stride=1, init_scale=1)
        self.shift_out = Shift(channel=out_channels, stride=stride, init_scale=1)

        self.temporal_linear = nn.Conv2d(in_channels, out_channels, 1)
        nn.init.kaiming_normal(self.temporal_linear.weight, mode='fan_out')

    def forward(self, x):
        x = self.bn(x)
        # shift1
        x = self.shift_in(x)
        x = self.temporal_linear(x)
        x = self.relu(x)
        # shift2
        x = self.shift_out(x)
        x = self.bn2(x)
        return x


class Shift_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(Shift_gcn, self).__init__()
        print("RUN NON-LOCAL SHIFT")
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

        self.Feature_Mask = nn.Parameter(torch.ones(1,num_node,in_channels, requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.constant(self.Feature_Mask, 0)

        self.bn = nn.BatchNorm1d(num_node*out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        ### Upper section is the same, modify code below
        # print(A)
        # print("SIZE OF A:",A.shape)
        print("Original Non-Local Shift GCN")
        index_array = np.empty(num_node*in_channels).astype(np.int)
        for i in range(num_node):
            for j in range(in_channels):
                index_array[i*in_channels + j] = (i*in_channels + j + j*in_channels)%(in_channels*num_node)
        self.shift_in = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)

        index_array = np.empty(num_node*out_channels).astype(np.int)
        for i in range(num_node):
            for j in range(out_channels):
                index_array[i*out_channels + j] = (i*out_channels + j - j*out_channels)%(out_channels*18)
        self.shift_out = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)
        #### Code modification ends here

    def forward(self, x0):
        n, c, t, v = x0.size()
        x = x0.permute(0,2,3,1).contiguous()

        # shift1
        x = x.view(n*t,v*c)
        x = torch.index_select(x, 1, self.shift_in)
        x = x.view(n*t,v,c)
        ## Code below not exist in local shift
        x = x * (torch.tanh(self.Feature_Mask)+1)
        ## End of the upper comment

        x = torch.einsum('nwc,cd->nwd', (x, self.Linear_weight)).contiguous() # nt,v,c
        x = x + self.Linear_bias

        # shift2
        x = x.view(n*t,-1) 
        x = torch.index_select(x, 1, self.shift_out)
        x = self.bn(x)
        x = x.view(n,t,v,self.out_channels).permute(0,3,1,2) # n,c,t,v

        x = x + self.down(x0)
        x = self.relu(x)
        return x

class Shift_gcn_local(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(Shift_gcn_local, self).__init__()
        print("RUN LOCAL SHIFT")
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

        self.Feature_Mask = nn.Parameter(torch.ones(1,num_node,in_channels, requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.constant(self.Feature_Mask, 0)

        self.bn = nn.BatchNorm1d(num_node*out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        # menambah value tensor A n*c sehingga hasilnya 1 dimensi (c)
        A = np.sum(A,0)

        n = num_node
        channels = in_channels
        if channels == 3:
            index_array = np.arange(n*channels).astype(np.int)
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
                current_A[i] = 0
                neighbors = np.nonzero(current_A)[0]
                for j in range(int(partition)):
                    if j == 0:
                        index_array[int(i*channels):int(i*channels + channel_per_partition)] = np.arange(i*channels,i*channels + channel_per_partition).astype(np.int)
                    else:
                        index_array[int(i*channels + j*channel_per_partition):int(i*channels + (j+1)*channel_per_partition)] = np.arange(neighbors[j-1]*channels + j*channel_per_partition,neighbors[j-1]*channels + (j+1)*channel_per_partition).astype(np.int)
        self.shift_in = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)

        n = num_node
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
                current_A[i] = 0
                neighbors = np.nonzero(current_A)[0]
                for j in range(int(partition)):
                    if j == 0:
                        index_array[int(i*channels):int(i*channels + channel_per_partition)] = np.arange(i*channels,i*channels + channel_per_partition).astype(np.int)
                    else:
                        index_array[int(i*channels + j*channel_per_partition):int(i*channels + (j+1)*channel_per_partition)] = np.arange(neighbors[j-1]*channels + j*channel_per_partition,neighbors[j-1]*channels + (j+1)*channel_per_partition).astype(np.int)
        self.shift_out = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)
        

    def forward(self, x0):
        n, c, t, v = x0.size()
        x = x0.permute(0,2,3,1).contiguous()

        # shift1
        x = x.view(n*t,v*c)
        x = torch.index_select(x, 1, self.shift_in)
        x = x.view(n*t,v,c)

        x = torch.einsum('nwc,cd->nwd', (x, self.Linear_weight)).contiguous() # nt,v,c
        x = x + self.Linear_bias

        # shift2
        x = x.view(n*t,-1) 
        x = torch.index_select(x, 1, self.shift_out)
        x = self.bn(x)
        x = x.view(n,t,v,self.out_channels).permute(0,3,1,2) # n,c,t,v

        x = x + self.down(x0)
        x = self.relu(x)
        return x

class Modified_shift_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, weight=2, coff_embedding=4, num_subset=3):
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

        A = np.sum(A,0)
        A[A==0] = 0
        A[A!=0] = 1
        node = A.shape[0]

        self.Linear_weight = nn.Parameter(torch.zeros(in_channels, out_channels, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0,math.sqrt(1.0/out_channels))

        self.Linear_bias = nn.Parameter(torch.zeros(1,1,out_channels,requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.constant(self.Linear_bias, 0)

        self.Feature_Mask = nn.Parameter(torch.ones(1,node,in_channels, requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.constant(self.Feature_Mask, 0)

        self.bn = nn.BatchNorm1d(node*out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        ### Upper section is the same, modify code below

        
        # print(A)

        index_array = self.shift(A,node,in_channels,weight)
        self.shift_in = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)

        index_array = self.shift(A,node,out_channels,weight)
        self.shift_out = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)

    def shift(self, A, node_num, channels, weight):
        print("RUN MODIFIED SHIFT")
        print("WEIGHT:",weight)
        index_array = np.arange(0,node_num*channels)
        index_array = index_array.reshape(node_num,channels)
        index_array_copy = np.copy(index_array)

        for i in range(node_num):
            current_A = A[i]
            # print(current_A)
            current_A[i] = 0
            neighbors = np.nonzero(current_A)[0] # cari tetangga
            # print(neighbors)
            j = 0 # column
            counter_i = i + 1 # row
            while j < channels: 
                next_node = counter_i % node_num
                if j < weight: # awal pasti skip sebanyak bobot
                    j += weight
                elif next_node == i: # jika next node adalah row operasi maka skip sebanyak bobot
                    j += weight
                    counter_i += 1
                elif next_node in neighbors: # jika neigbor maka apply bobot
                    index_array[i][j:j+weight] = index_array_copy[next_node][j:j+weight]
                    j += weight
                    counter_i += 1
                else: # jika bukan neighbor maka tidak apply bobot
                    index_array[i][j] = index_array_copy[next_node][j]
                    j += 1
                    counter_i += 1
        # print(index_array)
        return index_array.flatten()

    def forward(self, x0):
        n, c, t, v = x0.size()
        x = x0.permute(0,2,3,1).contiguous()

        # shift1
        x = x.view(n*t,v*c)
        x = torch.index_select(x, 1, self.shift_in)
        x = x.view(n*t,v,c)
        ## Code below not exist in local shift
        x = x * (torch.tanh(self.Feature_Mask)+1)
        ## End of the upper comment

        x = torch.einsum('nwc,cd->nwd', (x, self.Linear_weight)).contiguous() # nt,v,c
        x = x + self.Linear_bias

        # shift2
        x = x.view(n*t,-1) 
        x = torch.index_select(x, 1, self.shift_out)
        x = self.bn(x)
        x = x.view(n,t,v,self.out_channels).permute(0,3,1,2) # n,c,t,v

        x = x + self.down(x0)
        x = self.relu(x)
        return x

class Modified_grouped_shift_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, shift_method = 'inner', weight=2, graph_group = None, coff_embedding=4, num_subset=3):
        super(Modified_grouped_shift_gcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        A = np.sum(A,0)
        A[A==0] = 0
        A[A!=0] = 1
        node = A.shape[0]

        self.Linear_weight = nn.Parameter(torch.zeros(in_channels, out_channels, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0,math.sqrt(1.0/out_channels))

        self.Linear_bias = nn.Parameter(torch.zeros(1,1,out_channels,requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.constant(self.Linear_bias, 0)

        self.Feature_Mask = nn.Parameter(torch.ones(1,node,in_channels, requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.constant(self.Feature_Mask, 0)

        self.bn = nn.BatchNorm1d(node*out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        ### Upper section is the same, modify code below

        
        print(shift_method)

        index_array = self.shift(A, shift_method, graph_group, node,in_channels,weight)
        self.shift_in = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)

        index_array = self.shift(A, shift_method, graph_group, node,out_channels,weight)
        self.shift_out = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)

    def shift(self, A, shift_method, graph_group, node_num, channels, weight):
        if shift_method == 'inner':
            index_array = self.inner_shift(A, graph_group, node_num, channels, weight)
        else:
            index_array = self.outer_shift(A, graph_group, node_num, channels, weight)
        
        return index_array

    def inner_shift(self, A, graph_group, node_num, channels, weight):
        print("RUN GROUPED MODIFIED SHIFT")
        print("WEIGHT:",weight)
        index_array = np.arange(0,node_num*channels)
        index_array = index_array.reshape(node_num,channels)
        index_array_copy = np.copy(index_array)

        for i in graph_group:
            lower_bound = np.min(i)
            upper_bound = np.max(i) + 1
            # print(lower_bound, upper_bound)
            for i in range(lower_bound, upper_bound):
                current_A = A[i].copy()
                # print(current_A)
                current_A[i] = 0
                neighbors = np.nonzero(current_A)[0]
                # print(neighbors)
                j = 0
                counter_i = (i - lower_bound) + 1
                while j < channels:
                    # print(index_array)
                    next_node = (counter_i % (upper_bound - lower_bound)) + lower_bound
                    # print(next_node)
                    if j < weight:
                        j += weight
                    elif next_node == i:
                        j += weight
                        counter_i += 1
                    elif next_node in neighbors:
                        # print("current j: ",j)
                        # print("J+weight: ",j+weight)
                        index_array[i][j:j+weight] = index_array_copy[next_node][j:j+weight]
                        j += weight
                        counter_i += 1
                    else:
                        index_array[i][j] = index_array_copy[next_node][j]
                        j += 1
                        counter_i += 1
        return index_array.flatten()

    def outer_shift(self, A, graph_group, node_num, channels, weight):
        print("RUN MODIFIED SHIFT")
        print("WEIGHT:",weight)
        index_array = np.arange(0,node_num*channels)
        index_array = index_array.reshape(node_num,channels)
        index_array_copy = np.copy(index_array)

        for i in range(node_num):
            current_A = A[i]
            # print(current_A)
            current_A[i] = 0
            neighbors = np.nonzero(current_A)[0] # cari tetangga
            # print(neighbors)
            j = 0 # column
            counter_i = i + 1 # row
            current_group = [item for item in graph_group if i in range(item[0], item[1] + 1, 1)][0]
            while j < channels:
                next_node = counter_i % node_num
                # print("shifted:", j)
                # print("next node:", next_node)
                # print(next_node in neighbors)
                if j < weight:
                    j += weight
                elif next_node == i:
                    j += weight
                    counter_i += 1
                elif next_node in neighbors and next_node in range(current_group[0], current_group[1] + 1):
                    # print("current j: ",j)
                    # print("J+weight: ",j+weight)
                    index_array[i][j:j+weight] = index_array_copy[next_node][j:j+weight]
                    # print(index_array)
                    j += weight
                    counter_i += 1
                elif next_node not in neighbors and next_node in range(current_group[0], current_group[1] + 1):
                    index_array[i][j:j+weight-1] = index_array_copy[next_node][j:j+weight-1]
                    # print(index_array)
                    j += (weight - 1)
                    counter_i += 1
                else:
                    index_array[i][j] = index_array_copy[next_node][j]
                    j += 1
                    counter_i += 1
        # print(index_array)
        return index_array.flatten()

    def forward(self, x0):
        n, c, t, v = x0.size()
        x = x0.permute(0,2,3,1).contiguous()

        # shift1
        x = x.view(n*t,v*c)
        x = torch.index_select(x, 1, self.shift_in)
        x = x.view(n*t,v,c)
        ## Code below not exist in local shift
        x = x * (torch.tanh(self.Feature_Mask)+1)
        ## End of the upper comment

        x = torch.einsum('nwc,cd->nwd', (x, self.Linear_weight)).contiguous() # nt,v,c
        x = x + self.Linear_bias

        # shift2
        x = x.view(n*t,-1) 
        x = torch.index_select(x, 1, self.shift_out)
        x = self.bn(x)
        x = x.view(n,t,v,self.out_channels).permute(0,3,1,2) # n,c,t,v

        x = x + self.down(x0)
        x = self.relu(x)
        return x

class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, method="modified", weight=2, graph_group = None):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = None
        # method = 'grouped.inner_shift'
        method = method.split('.')
        print(method)
        if method[0] == "modified":
            self.gcn1 = Modified_shift_gcn(in_channels, out_channels, A, weight=weight)
        elif method[0] == "grouped":
            shift_method = method[1]
            self.gcn1 = Modified_grouped_shift_gcn(in_channels, out_channels, A, shift_method=shift_method, weight=weight, graph_group=graph_group)
        elif method[0] == "nonlocal":
            self.gcn1 = Shift_gcn(in_channels, out_channels, A)
        elif method[0] == "local":
            self.gcn1 = Shift_gcn_local(in_channels, out_channels, A)


        self.tcn1 = Shift_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        # print("gcn")
        # print(self.tcn1(self.gcn1(x)).size())
        # print("residual")
        # print(type(self.residual(x)))
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class Model(nn.Module):
    def __init__(self, 
                num_class=60, 
                num_point=25, 
                num_person=2, 
                graph=None,
                graph_group=None,
                graph_args=dict(), 
                in_channels=3, 
                method="modified", 
                weight=2):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        if weight is None:
            weight = 1

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(3, 64, A, residual=False, method=method, weight=weight, graph_group=graph_group)
        self.l2 = TCN_GCN_unit(64, 64, A,method=method, weight=weight, graph_group=graph_group)
        self.l3 = TCN_GCN_unit(64, 64, A,method=method, weight=weight, graph_group=graph_group)
        self.l4 = TCN_GCN_unit(64, 64, A,method=method, weight=weight, graph_group=graph_group)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2,method=method, weight=weight, graph_group=graph_group)
        self.l6 = TCN_GCN_unit(128, 128, A,method=method, weight=weight, graph_group=graph_group)
        self.l7 = TCN_GCN_unit(128, 128, A,method=method, weight=weight, graph_group=graph_group)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2,method=method, weight=weight, graph_group=graph_group)
        self.l9 = TCN_GCN_unit(256, 256, A,method=method, weight=weight, graph_group=graph_group)
        self.l10 = TCN_GCN_unit(256, 256, A,method=method, weight=weight, graph_group=graph_group)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return self.fc(x)
