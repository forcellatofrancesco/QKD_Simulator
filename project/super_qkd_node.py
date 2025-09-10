import json

class SuperQKDNode:

    # To add the global graph reference to update the global graph, I don't actually know for sure if it has to be done
    # here or inside the Transceiver or MessagingProtocol class
    def __init__(self, name, routing: bool = False):
        self.name = name
        self.transceivers = {}
        self.routing_table = {}
        self.routing = routing

    def send_message(self, tl, dest_node, plaintext_msg, forwarding):
        next_hop_name = self.routing_table[dest_node][1]

        # print('PLAINTEXT: ', plaintext_msg)

        packet = json.loads(plaintext_msg)
        hop = packet['hop']
        route = packet['route']

        if self.routing:
            next_hop_name = route[hop]

        for tr in self.transceivers.values():
            if tr.qkd_node.name.endswith(next_hop_name):
                tr.send_message(tl, plaintext_msg, forwarding)
