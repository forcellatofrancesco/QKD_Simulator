from sequence.qkd.BB84 import pair_bb84_protocols
from sequence.qkd.cascade import pair_cascade_protocols
from sequence.kernel.process import Process
from sequence.kernel.event import Event

import numpy
import json

class Transceiver:

    def __init__(self, qkd_node, qkd_node_p):
        self.qkd_node = qkd_node
        self.qkd_node_p = qkd_node_p
        self.qkd_node_km = None

    def add_key_manager(self, qkd_node_km):
        self.qkd_node_km = qkd_node_km
        self.qkd_node_p.add_key_manager(qkd_node_km)
        self.qkd_node_km.mp = self.qkd_node_p

    def send_message(self, tl, plaintext, forwarding):
        if forwarding:
            process = Process(self.qkd_node_p, "start", [plaintext, tl, True])
            event = Event(tl.now(), process)
            tl.schedule(event)
        else:
            time = numpy.random.exponential(self.qkd_node_p.mess_rate, 1)[0]
            packet = json.loads(plaintext)
            packet["time"] = tl.now() + (time * 1.0e12)
            plaintext = json.dumps(packet)
            process = Process(self.qkd_node_p, "start", [plaintext, tl, False])
            event = Event(tl.now() + (time * 1.0e12), process)
            tl.schedule(event)
 
    def start_qkd(self):
        self.qkd_node_km.send_request()
        

    