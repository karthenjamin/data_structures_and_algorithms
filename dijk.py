import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import heapq as h
import random
from time import time
import scipy.optimize as opt

def dij_mat(A, s, t, heap = True):
    A = np.array(A)
    
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        print("array A not square")
        return -1
    
    n_index = A.shape[0]
    
    seen = [False for i in range(n_index)]
    
    
    if heap:
        # format of queue is (cost, node, path from s)
        q = [(0, s, [])]
    
        while q:
            (cost, u, path) = h.heappop(q)
            if not seen[u]:
                seen[u] = True
                path = path + [u]

                if u == t:
                    return cost

                for v,c in enumerate(A[u]):
                    if not seen[v] and c:
                        h.heappush(q, (cost + c, v, path))
                        
    else:
        
        l = [(0, s, [])]
        
        while l:
            (cost, u, path) = l.pop()
            if not seen[u]:
                seen[u] = True
                path = path + [u]

                if u == t:
                    return cost

                for v,c in enumerate(A[u]):
                    if not seen[v] and c:
                        l.append((cost + c, v, path))
                        l = sorted(l, reverse=True)

    print("could not find path")
    return float("inf")

def dij_list(L, s, t, heap = True):
    n_index = len(L)
    
    seen = [False for i in range(n_index)]
    
    if heap:
        
        # format of queue is (cost, node, path from s)
        q = [(0, s, [])]

        while q:
            (cost, u, path) = h.heappop(q)
            u = int(u)
            if not seen[u]:
                seen[u] = True
                path = path + [u]

                if u == t:
                    return cost
                for v,c in L[str(u)].items():
                    v = int(v)
                    if not seen[v] and c:
                        h.heappush(q, (cost + c, str(v), path))
    else:
                           
        l = [(0, s, [])]      
        while l:
            (cost, u, path) = l.pop()
            u = int(u)
            if not seen[u]:
                seen[u] = True
                path = path + [u]

                if u == t:
                    return cost

                for v,c in L[str(u)].items():
                    v = int(v)
                    if not seen[v] and c:
                        l.append((cost + c, str(v), path))
                        l = sorted(l, reverse=True)

        

    print("could not find path")
    return float("inf")

def gen_graph_matrix(size, p = 0.9):
    mat = [[np.random.random() * np.random.choice([0,1], p=[1 - p, p]) for j in range(size)] for i in range(size)]
    for i in range(size):
        mat[i][i] = 0
        
    return mat
    
def gen_graph_list(mat):
    l = {}
    
    for i,row in enumerate(mat):
        l[str(i)] = {}
        for j, c in enumerate(row):
            if c > 0:
                l[str(i)][str(j)] = c

    return l

sizes = [5 * i for i in range(1,40)]
trials = 10

times = {"mat_heap":[[] for x in sizes], "mat_list":[[] for x in sizes], "list_heap":[[] for x in sizes], "list_list":[[] for x in sizes]}

# times = [[] for size in sizes]
# path_lengths = [[] for size in sizes]



for i,size in enumerate(sizes):
    print("SIZE:", size)
    for trial in range(trials):
        X_mat = gen_graph_matrix(size)
        X_list = gen_graph_list(X_mat)
        
        tic = time()
        dij_mat(X_mat, 0, size - 1)
        toc = time()
        
        times["mat_heap"][i].append(toc - tic)
        
        tic = time()
        dij_mat(X_mat, 0, size - 1, False)
        toc = time()
        
        times["mat_list"][i].append(toc - tic)
        
        tic = time()
        dij_list(X_list, 0, size - 1)
        toc = time()
        
        times["list_heap"][i].append(toc - tic)
        
        tic = time()
        dij_list(X_list, 0, size - 1, False)
        toc = time()
        
        times["list_list"][i].append(toc - tic)
       
    

plt.figure(figsize=(10,10))

plt.subplot(211)

means = [np.mean(x) for x in times]
param = opt.curve_fit(lambda t,a: a * t ** 2 * np.log(t),  sizes,  means)


plt.plot(sizes, [np.mean(x) for x in times["mat_list"]], label = "adj. matrix with list")

plt.plot(sizes, [np.mean(x) for x in times["list_list"]], label = "adj. list with list")

plt.legend()
plt.title("runtime of Dijkstra's Algorithm")
plt.xlabel("number of vertices")
plt.ylabel("runtime (s)")

plt.plot(sizes, [np.mean(x) for x in times["mat_heap"]], label = "adj. matrix with heap")
plt.plot(sizes, [np.mean(x) for x in times["list_heap"]], label = "adj. list with heap")


plt.subplot(212)
plt.plot(sizes, [np.mean(x) for x in path_lengths])