import json
import networkx as nx
from path_algorithms import centflow_shortest_path
import matplotlib.pyplot as plt
import os
import pandas as pd


class GlobalGraph:

    def __init__(self, filename: str, location: str):
        with open(filename, "r") as f:
            data = json.load(f)

        # Creating the graph in NetworkX
        # self.graph = nx.Graph()
        self.graph = nx.DiGraph()

        # Adding nodes with IDs and attributes
        for node in data["nodes"]:
            node_id = node["id"]        
            attrs = {k: v for k, v in node.items() if k != "id"} 
            self.graph.add_node(node_id, **attrs)

        # Adding attribute with value 'usage' set to 0
        # Adding attribute with value 'capacity' set to 20
        # Or 'capacity' attribute could be set using qber.py and taking the BER
        for edge in data["links"]:
            src = edge["source"]
            dst = edge["target"]
            attrs = {k: v for k, v in edge.items() if k not in ["source", "target"]}

            attrs.setdefault("usage", 0)
            # To set with buffer_capacity, the default value is 10, so we take 20 since the same link is used by 2 transceivers.
            attrs.setdefault("capacity", 20)

            self.graph.add_edge(src, dst, **attrs)
            # if the graph is directed
            self.graph.add_edge(dst, src, **attrs)

        # location where to save the graph plots
        self.location = location

  
    # To edit as we want
    def increase_usage(self, tr_name: str):
        _, u, _, v = tr_name.split("_")
        u = int(u.replace("node", ""))
        v = int(v.replace("node", ""))

        if self.graph[u][v]['usage'] < 20:
            self.graph[u][v]['usage'] += 1


    # To edit as we want
    def decrease_usage(self, tr_name: str):
        _, u, _, v = tr_name.split("_")
        u = int(u.replace("node", ""))
        v = int(v.replace("node", ""))

        if self.graph[u][v]['usage'] > 0:
            self.graph[u][v]['usage'] -= 1


    def best_available_path(self, src, dest):
        return centflow_shortest_path(self.graph, src, dest)


    """ def save_graph(self, tl):
        os.makedirs(f"{self.location}graph", exist_ok=True)

        edge_data = [
            {"timeline": tl.now(), "source": u, "target": v, **attrs}
            for u, v, attrs in self.graph.edges(data=True)
        ]

        df_edges = pd.DataFrame(edge_data)
        df_edges.to_csv(f"{self.location}graph/graph_status{self.num_saves}.csv", index=False)
        self.num_saves += 1 """

    def save_graph(self, tl):
        os.makedirs(f"{self.location}graph", exist_ok=True)

        for u, v, attrs in self.graph.edges(data=True):
            row = {
                "timeline": tl.now(),
                "source": u,
                "target": v,
                "usage": attrs.get("usage"),
                "capacity": attrs.get("capacity"),
            }

            filename = f"{self.location}graph/status_link_{u}_to_{v}.csv"
            file_exists = os.path.isfile(filename)

            with open(filename, "a", newline="") as f:
                df = pd.DataFrame([row])
                df.to_csv(f, index=False, header=not file_exists)