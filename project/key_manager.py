from sequence.kernel.process import Process
from sequence.kernel.event import Event

import numpy

from memory_profiler import profile

class KeyManager():

    def __init__(self, own, timeline, keysize, num_keys):
        self.own = own                      # QKDNode object
        self.timeline = timeline            # global timeline
        self.lower_protocols = []
        self.keysize = keysize              # 128 bits
        self.num_keys = num_keys            # Infinite
        self.keys = []                      # List of available keys
        self.mp = None                      # MessagingProtocol
        self.first = True
        self.count_keys = 0                 # Total number of keys generated so far
        self.keys_available = 0             # Number of keys available from previous inspection (left over)
        self.keys_gen = 0                   # Number of keys generated from previous inspection
        self.keys_used = 0                  # Number of keys used from previous inspection
        
    def send_request(self):
        for p in self.lower_protocols:
            p.push(self.keysize, self.num_keys)
            
    def pop(self, key):
        self.keys.append(key)
        self.keys_gen += 1
        self.count_keys += 1

        if self.first:
            time = self.mp.packet_period
            process = Process(self.mp, "send", [self.timeline])
            event = Event(self.timeline.now(), process)
            self.timeline.schedule(event)

            self.first = False
    
    def consume(self):
        self.keys_used += 1
        return self.keys.pop(0)
    
    def utilization(self):
        if (self.keys_available + self.keys_gen) == 0:
            usage = 0
        else:
            usage = self.keys_used / (self.keys_available + self.keys_gen)

        self.keys_available = len(self.keys)
        self.keys_gen = 0
        self.keys_used = 0

        return usage
    

