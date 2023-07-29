import numpy as np
import pandas as pd

A = [[1, 1, 0, 0],
    [1, 1, 1, 1],
    [0, 0.5, 1, 0],
    [0, 0.5, 0, 1]]

A = np.array(A)

A[A==0] = 0
A[A!=0] = 1
n = A.shape[0]
print(n)
in_ch = 3
out_ch = 12
weight = 3

channels = out_ch

index_array = np.arange(0,n*channels)
index_array = index_array.reshape(n,channels)
index_array_copy = np.copy(index_array)

for i in range(n):
    current_A = A[i]
    print(current_A)
    current_A[i] = 0
    neighbors = np.nonzero(current_A)[0]
    print(neighbors)
    j = 0
    counter_i = i + 1
    while j < channels:
        print(index_array)
        next_node = counter_i % n
        if j < weight:
            j += weight
        elif next_node == i:
            j += weight
            counter_i += 1
        elif next_node in neighbors:
            # print("current j: ",j)
            # print("J+weight: ",j+weight)
            index_array[i][j:j+weight] = index_array_copy[next_node][j:j+weight]
            # print(index_array)
            j += weight
            counter_i += 1
        else:
            index_array[i][j] = index_array_copy[next_node][j]
            j += 1
            counter_i += 1

print(index_array)