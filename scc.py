from grafo import GraphFactory
import numpy as np
from collections import deque

white = -1
grey = 0
black = 1

class SCC:
  def __init__(self, reach):
    self.graph_factory = GraphFactory(reach)
    graph = self.graph_factory.get_graph()
    
    self.cor = {s: 0 for s in graph}
    self.pai = {s: None for s in graph}
    self.discovery_t = {s: 0 for s in graph}
    self.finished_t = {s: 0 for s in graph}
    self.postI = deque()
    self.graph = graph

  def execute(self):
    graph_factory = self.graph_factory
    graph = self.graph
  
    # print("grafo > ")
    # print_grafo(graph)


    self.DFS(graph)

    inverted_graph = graph_factory.invert_graph()

    for v in inverted_graph:
      self.cor[v] = white
      self.pai[v] = None

        

    comps = self.DFS(inverted_graph)

    print(len(comps), 'componentes e', sum([len(i) for i in comps]), 'vertices')
    for i in range(len(comps)):
      if(len(comps[i]) > 1):
        print('Componente', i, '->', len(comps[i]), 'vértices.')

    return comps
  
  def DFS(self, G):
    forest = deque()
    for v in G:
      self.cor[v] = white
      self.pai[v] = None

    time = 0
    for v in G:
      if(self.cor[v] == white):
        tree = deque()
        tree = self.DFS_visit(G, v, time, tree)
        forest.appendleft(tree)
    

    return forest

  def DFS_visit(self, G, v, time, tree):
    self.cor[v] = grey
    time = time + 1
    self.discovery_t[v] = time
  
    for u in G[v]:
      if(self.cor[u] == white):
        self.pai[u] = v
        self.DFS_visit(G, u, time, tree)
    
    self.cor[v] = black
    tree.appendleft(v)

    time = time + 1
    self.finished_t[v] = time
    
    return tree

    



def print_grafo(graph):
  S_index = {s: i for i, s in enumerate(graph)}
  for node in graph:
    print(S_index[node], ': ')
    for succ in graph[node]:
      print(' -->', S_index[succ])


    
