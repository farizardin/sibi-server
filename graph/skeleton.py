import networkx as nx
from graph import tools
import numpy as np
import sys
from graph.hsd import Graph

sys.path.extend(['../'])

class Skeleton(Graph):
    def __init__(self, labeling_mode='spatial'):
        self.num_node = 75
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.inward = [(3, 4), (0, 5), (17, 18), (0, 17), (13, 14), (13, 17), (18, 19), (5, 6), (5, 9), (14, 15), (0, 1), (9, 10), (1, 2), (9, 13), (10, 11), (19, 20), (6, 7), (15, 16), (2, 3), (11, 12), (7, 8), (24, 25), (21, 26), (38, 39), (21, 38), (34, 35), (34, 38), (39, 40), (26, 27), (26, 30), (35, 36), (21, 22), (30, 31), (22, 23), (30, 34), (31, 32), (40, 41), (27, 28), (36, 37),
            (23, 24), (32, 33), (28, 29), (57, 63), (58, 62), (60, 62), (45, 49), (56, 58), (65, 67), (70, 72), (53, 65), (69, 73), (48, 50), (57, 59), (66, 68), (58, 64), (46, 47), (47, 48), (71, 73), (54, 66), (65, 66), (42, 43), (51, 52), (43, 44), (42, 46), (53, 55), (72, 74), (70, 74), (57, 61), (58, 60), (67, 69), (68, 70), (54, 56), (59, 61), (44, 45), (53, 54), (69, 71), (55, 57)]
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward
        self.A = self.get_adjacency_matrix(labeling_mode)

if __name__ == '__main__':
    A = Graph('spatial').get_adjacency_matrix()
    print('')
