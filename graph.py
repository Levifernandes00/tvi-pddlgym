class GraphFactory:
  def __init__(self, reach):
    self.graph = {s: [] for s in reach}
    self.S_index = {s: i for i, s in enumerate(reach)}
    self.create_graph(reach)

  def create_graph(self, reach):
    for s in reach:
      for a in reach[s]:
        for succ_state in reach[s][a].keys():
          self.graph[s].append(succ_state) if succ_state not in self.graph[s] else self.graph[s]         

  def invert_graph(self):
    S_index = self.S_index
    graph = self.graph
    self.inverted_graph = {s: [] for s in graph}
    
    # invert transitions
    for node in graph:
      for succ_state in graph[node]:
        self.inverted_graph[succ_state].append(node)

    return self.inverted_graph

  def get_graph(self):
    return self.graph
  

