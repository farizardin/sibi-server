import numpy as np
import sys

sys.path.extend(['../'])
from graph import tools
import networkx as nx

# Joint index:
# {0,  "WRIST"}
# {1,  "THUMB_CMC"},
# {2,  "THUMB_MCP"},
# {3,  "THUMB_IP"},
# {4,  "THUMB_TIP"},
# {5,  "INDEX_FINGER_MCP"},
# {6,  "INDEX_FINGER_PIP"},
# {7,  "INDEX_FINGER_DIP"},
# {8,  "INDEX_FINGER_TIP"},
# {9,  "MIDDLE_FINGER_MCP"},
# {10,  "MIDDLE_FINGER_PIP"},
# {11,  "MIDDLE_FINGER_DIP"},
# {12,  "MIDDLE_FINGER_TIP"},
# {13,  "RING_FINGER_MCP"},
# {14,  "RING_FINGER_PIP"},
# {15,  "RING_FINGER_DIP"},
# {16,  "RING_FINGER_TIP"},
# {17,  "PINKY_FINGER_MCP"},
# {18,  "PINKY_FINGER_PIP"},
# {19,  "PINKY_FINGER_DIP"},
# {20,  "PINKY_FINGER_TIP"},


# Edge format: (origin, neighbor)

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = 21
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.inward = [(0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20)]
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            # print("num of node:", self.num_node)
            A = tools.get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        else:
            raise ValueError()
        return A

if __name__ == '__main__':
    A = Graph('spatial').get_adjacency_matrix()
    print('')
