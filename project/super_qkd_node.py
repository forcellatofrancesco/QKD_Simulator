class SuperQKDNode:

    def __init__(self, name):
        self.name = name
        self.transceivers = {}
        self.routing_table = {}

    def send_message(self, tl, dest_node, plaintext_msg, forwarding):
        next_hop_name = self.routing_table[dest_node][1]
        for tr in self.transceivers.values():
            if tr.qkd_node.name.endswith(next_hop_name):
                tr.send_message(tl, plaintext_msg, forwarding)
    
        
        
        