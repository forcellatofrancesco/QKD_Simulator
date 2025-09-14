import networkx as nx
from networkx import DiGraph
import heapq


def effective_degree(graph: DiGraph, u, v, tau_n, tau_e):
    if graph[u][v]['util'] > tau_e:
        graph.remove_edge(u, v)

    if graph.nodes[u]['util_n'] > tau_n:
        for e in graph.neighbors(u):
            graph.remove_edge(u, e)

    return graph.degree[u]

def centflow(graph: DiGraph, source, target, tau_n = 0.9, tau_e = 0.8):


    node_centrality = nx.betweenness_centrality(graph)
    edge_centrality = nx.edge_betweenness_centrality(graph)

    dist = {node: float("inf") for node in graph.nodes}
    dist[source] = 0

    pred = {}
    queue = [(dist[node], node) for node in graph.nodes]
    heapq.heapify(queue)
    visited = set()

    while queue:
        d, u = heapq.heappop(queue)

        if u in visited:
            continue
        visited.add(u)

        if u == target:
            break

        for v in graph.neighbors(u):

            cnb = node_centrality[v]
            ceb = edge_centrality[(u,v)]
            ed = effective_degree(graph, u, v, tau_n, tau_e)

            new_dist = dist[u]\
                        + graph.nodes[v]['util_n'] * cnb * ed\
                        + graph[u][v]['util'] * ceb

            if dist[v] > new_dist:
                dist[v] = new_dist
                pred[v] = u

    # computing next hop
    t = target
    while pred[t] != source:
        t = pred[t]

    return t








    pass
