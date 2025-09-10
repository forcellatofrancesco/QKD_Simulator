import heapq
import networkx as nx
from networkx import Graph

# useless, we are dealing only with edge utilization
# THRESHOLD_NODE = 0.7  # node utilization optional threshold


THRESHOLD_LINK = 0.7  # link utilization optional threshold


def edge_utilization(G, u, v):
    """
    Calculate the utilization of an edge in a graph.

    Parameters:
        G (networkx.Graph): The graph containing the edge.
        u (hashable): The source node of the edge.
        v (hashable): The target node of the edge.

    Returns:
        float: The ratio of the edge's current usage to its capacity.

    Raises:
        KeyError: If the edge (u, v) does not exist or required attributes are missing.
        ZeroDivisionError: If the edge's capacity is zero.
    """
    return G[u][v]["usage"] / G[u][v]["capacity"]


""" # SHOULD BE USELESS
def node_utilization(G, n):
    Calculate the utilization of a node in a graph.

    Parameters:
        G (networkx.Graph): The graph containing the node.
        n (hashable): The node identifier.

    Returns:
        float: The ratio of the node's current usage to its capacity.

    Raises:
        KeyError: If the node does not have 'usage' or 'capacity' attributes.
        ZeroDivisionError: If the node's capacity is zero.

    return G.nodes[n]["usage"] / G.nodes[n]["capacity"] """


def effective_degree(G, node):
    """
    Calculates the effective degree of a node in the graph based on utilization thresholds.
    The effective degree is defined as the number of neighboring nodes connected by edges whose utilization is below a specified threshold, provided the node's own utilization is also below a node threshold. If the node's utilization exceeds or equals the node threshold, the effective degree is zero.
    Args:
        G (networkx.Graph): The graph containing the node and its edges.
        node (hashable): The node for which to calculate the effective degree.
    Returns:
        int: The effective degree of the node, or 0 if the node's utilization exceeds the threshold.
    Dependencies:
        - node_utilization(G, node): Function that returns the utilization of the node.
        - edge_utilization(G, node1, node2): Function that returns the utilization of the edge between node1 and node2.
        - THRESHOLD_NODE (float): Utilization threshold for nodes.
        - THRESHOLD_LINK (float): Utilization threshold for edges.
    """

    # we don't use node utilization
    # if node_utilization(G, node) >= THRESHOLD_NODE:
    #     return 0

    neighbors = list(G.neighbors(node))
    return sum(
        1 for nbr in neighbors if edge_utilization(G, node, nbr) < THRESHOLD_LINK
    )


""" SHOULD BE USELESS
def ensure_graph_attributes(G, node_capacity: int, link_capacity: int):
    """"""
    Ensures that all nodes and edges in the given graph have 'capacity' and 'usage' attributes.

    For each node in the graph `G`, if the 'capacity' or 'usage' attribute is missing, it sets:
        - 'capacity' to the provided `node_capacity`
        - 'usage' to 0

    For each edge in the graph `G`, if the 'capacity' or 'usage' attribute is missing, it sets:
        - 'capacity' to the provided `link_capacity`
        - 'usage' to 0

    Parameters:
        G (networkx.Graph): The graph whose nodes and edges will be checked and updated.
        node_capacity (int): The default capacity value to assign to nodes missing the 'capacity' attribute.
        link_capacity (int): The default capacity value to assign to edges missing the 'capacity' attribute.
    """"""
    # Only set node attributes if any node is missing them
    if any("capacity" not in G.nodes[n] or "usage" not in G.nodes[n] for n in G.nodes):
        for n in G.nodes:
            if "capacity" not in G.nodes[n]:
                G.nodes[n]["capacity"] = node_capacity
            if "usage" not in G.nodes[n]:
                G.nodes[n]["usage"] = 0
    # Only set edge attributes if any edge is missing them
    if any("capacity" not in G[u][v] or "usage" not in G[u][v] for u, v in G.edges):
        for u, v in G.edges:
            if "capacity" not in G[u][v]:
                G[u][v]["capacity"] = link_capacity
            if "usage" not in G[u][v]:
                G[u][v]["usage"] = 0 """


def centflow_shortest_path(G: Graph, source, target) -> list | dict:
    """
    Finds the shortest path between a source and target node in a graph using a custom cost function
    that incorporates node and edge betweenness centrality, utilization, and effective degree.

    Args:
        G (Graph): A NetworkX graph object.
        source: The starting node for the path.
        target: The destination node for the path.

    Returns:
        list: A list of nodes representing the shortest path from source to target.
              Returns an empty list if no path is found.

    Notes:
        - The path cost is computed as a combination of node utilization, edge utilization,
          node betweenness centrality, edge betweenness centrality, and effective degree.
        - Utilization and effective degree are calculated using external functions:
          `edge_utilization`, `node_utilization`, and `effective_degree`.
        - The algorithm uses a priority queue (min-heap) to explore paths with the lowest cumulative cost.
    """
    
    queue = []
    node_centrality = nx.betweenness_centrality(G)
    edge_centrality = nx.edge_betweenness_centrality(G)

    # It shouldn't be necessary since the Graph used is a GlobalGraph which surely has these attributes.
    # ensure_graph_attributes(G, NODE_CAPACITY, LINK_CAPACITY)
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

        for neighbor in G.neighbors(node):
            if neighbor in visited:
                continue

            # weight computation
            e_util = edge_utilization(G, node, neighbor)
            # useless since we use only edge utilization
            # n_util = node_utilization(G, neighbor)

            cnb = node_centrality[neighbor]
            ceb = edge_centrality.get(
                (node, neighbor), edge_centrality.get((neighbor, node), 0)
            )
            deg_eff = effective_degree(G, neighbor)
            # node_weight = n_util * cnb * deg_eff
            node_weight = cnb * deg_eff
            edge_weight = e_util * ceb
            weight = node_weight + edge_weight + 1e-6

            heapq.heappush(queue, (cost + weight, neighbor, path))
    return []  # No path found



def weighted_shortest_path(G: Graph, source, target) -> list | dict:
    """
    Finds the shortest path between a source and target node in a graph using edge centrality betweenness as a weight.

    Args:
        G (DiGraph): A NetworkX graph object.
        source: The starting node for the path.
        target: The destination node for the path.

    Returns:
        list: A list of nodes representing the shortest path from source to target.
              Returns an empty list if no path is found.

    Notes:
        - The path cost is computed as edge centrality betweenness which should encourage to avoid possibly heavily used nodes.
        - The cost of an edge (u, v) is the centrality value of that edge.
    """

    edge_betweenness = nx.edge_betweenness_centrality(G)

    for u, v in G.edges():
            weight = edge_betweenness[(u, v)] + 1    # +1 to avoid 0 as centrality value
            G[u][v]['weight'] = weight

    path = nx.dijkstra_path(G, source, target, 'weight')
    return path

