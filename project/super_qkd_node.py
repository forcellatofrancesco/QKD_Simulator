import json
from global_graph import GlobalGraph

class SuperQKDNode:

    # To add the global graph reference to update the global graph
    # here or inside the Transceiver or MessagingProtocol class
    def __init__(self, name, GG: GlobalGraph, routing: bool = False):
        self.name = name
        print(f"Sono il supernodo {self.name}  tipo variable self.name: " + str(type(self.name)))
        self.transceivers = {}
        self.routing_table = {}
        self.routing = routing
        self.gg = GG

    # To change to use the centflow algorithm
    # dest_node should be of the form "nodeX", where is the node number
    def send_message(self, tl, dest_node, plaintext_msg, forwarding):
        src = int(self.name.replace("node", ""))
        dst = int(dest_node.replace("node", ""))
        next_hop_int = self.gg.best_available_path(src, dst)[1]
        next_hop_name = f"node{next_hop_int}"

        # next_hop_name = self.routing_table[dest_node][1]

        # print('PLAINTEXT: ', plaintext_msg)

        # packet = json.loads(plaintext_msg)
        # hop = packet['hop']
        # route = packet['route']

        # if self.routing:
        #     next_hop_name = route[hop]

        for tr in self.transceivers.values():
            if tr.qkd_node.name.endswith(next_hop_name):
                tr.send_message(tl, plaintext_msg, forwarding)
