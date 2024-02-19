from sequence.kernel.process import Process
from sequence.kernel.event import Event

import numpy

from memory_profiler import profile

class KeyManager():

    def __init__(self, own, timeline, keysize, num_keys):
        self.own = own
        self.timeline = timeline
        self.lower_protocols = []
        self.keysize = keysize
        self.num_keys = num_keys
        self.keys = []
        self.mp = None
        self.first = True
        self.count_keys = 0
        
    def send_request(self):
        for p in self.lower_protocols:
            p.push(self.keysize, self.num_keys)
            
    def pop(self, key):
        self.keys.append(key)
        self.count_keys += 1

        if self.first:
            time = self.mp.packet_period
            process = Process(self.mp, "send", [self.timeline])
            event = Event(self.timeline.now(), process)
            self.timeline.schedule(event)

            self.first = False
    
    def consume(self):
        return self.keys.pop(0)
    

