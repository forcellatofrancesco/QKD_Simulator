import networkx as nx
from networkx import DiGraph
import heapq

def effective_degree(G, node, thresh):
    neighbors = list(G.neighbors(node))
    return sum(
        1 for nbr in neighbors if G[node][nbr]["weight"] < thresh
    )

def centflow(graph: DiGraph, source, target, tau_e = 0.95) -> list | dict:
    queue = []
    node_centrality = nx.betweenness_centrality(graph)
    edge_centrality = nx.edge_betweenness_centrality(graph)

    heapq.heappush(queue, (0, source, []))
    visited = set()

    while queue:
        cost, node, path = heapq.heappop(queue)
        if node in visited:
            continue
        visited.add(node)
        path = path + [node]

        if node == target:
            return path

        for neighbor in graph.neighbors(node):
            if neighbor in visited:
                continue

            # After the graph recomputation, weight holds the utilization of the link
            e_util = graph[node][neighbor]["weight"]

            cnb = node_centrality[neighbor]
            ceb = edge_centrality.get(
                (node, neighbor), edge_centrality.get((neighbor, node), 0)
            )
            deg_eff = effective_degree(graph, neighbor, tau_e)
            node_weight = cnb * deg_eff
            edge_weight = e_util * ceb
            weight = node_weight + edge_weight + 1e-6

            heapq.heappush(queue, (cost + weight, neighbor, path))
    return []  # No path found


