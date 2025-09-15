from enum import Enum, auto
import json
import numpy
import collections
import csv

from sequence.topology.node import Node
from sequence.protocol import Protocol
from sequence.message import Message
from sequence.qkd.BB84 import pair_bb84_protocols
from sequence.qkd.cascade import pair_cascade_protocols
from sequence.kernel.process import Process
from sequence.kernel.event import Event
from global_graph import GlobalGraph
from global_graph import GlobalGraph

from onetimepad import OneTimePad
from pympler import asizeof


class MsgType(Enum):

    TEXT_MESS = auto()


class MessagingProtocol(Protocol):

    sim_path = None

    def __init__(
        self,
        own: Node,
        name: str,
        other_name: str,
        other_node: str,
        superQKD,
        print_gap: int = 10_000,
    ):
        super().__init__(own, name)
        self.own = own                  # QKDNode type
        self.own = own                  # QKDNode type
        own.protocols.append(self)
        self.other_name = other_name
        self.other_node = other_node    # Name to use for the link identification
        self.other_node = other_node    # Name to use for the link identification
        self.other = None
        self.super_qkd = superQKD       # SuperQKDNode of reference
        self.super_qkd = superQKD       # SuperQKDNode of reference
        self.key_manager = None
        self.token = True

        self.mess_rate = None
        self.packet_period = None

        # For metrics
        self.del_mess = 0
        self.drop_mess = 0
        self.mess_ric = 0

        # This should be the buffer the professor was talking about. It should be used to compute the utilization of a specific link
        # Buffer
        self.buffer = collections.deque()
        self.buffer_capacity = None
        self.print_gap: int = print_gap
        self.print_idx: int = 0

    def init(self):
        pass

    # This appears to be the function that actually sends the message (among the several ones)
    # So maybe the utilization update should be done here
    # This should mean that the easiest place to put the global graph reference, as a class field
    def send(self, tl):
        if (
            len(self.buffer) > 0 and len(self.key_manager.keys) > 0 and self.token
        ):  # and len(self.other.qkd_node_p.key_manager.keys):
            key = self.key_manager.consume()
            key = key.to_bytes((key.bit_length() + 7) // 8, "big")

            packet = json.loads(self.buffer.pop())

            message = packet["payload"]
            message = bytes(message)  # b'str'

            key = key[0 : len(message)]

            ciphertext = OneTimePad.encrypt(message, key)
            ciphertext = list(ciphertext)

            packet["payload"] = ciphertext

            new_msg = Message(MsgType.TEXT_MESS, self.other_name)
            new_msg.payload = json.dumps(packet)
            new_msg.protocol_type = type(self)

            # send_message QKDNode
            self.own.send_message(self.other_node, new_msg)  # send_message di QKDNode

            self.token = False
            self.other.qkd_node_p.token = False

        # Simula service time tra invii
        time = self.packet_period
        process = Process(self, "send", [tl])
        event = Event(tl.now() + (time * 1.0e12), process)
        tl.schedule(event)

    def start(self, text: str, tl, forwarding):
        if not forwarding:
            # print(f"[{self.own.name}]\nMessage sent. At simulation time: {self.own.timeline.now() * 1.0e-12} s\n")

            time = numpy.random.exponential(self.mess_rate, 1)[0]
            packet = json.loads(text)
            packet["time"] = tl.now() + (time * 1.0e12)
            plaintext = json.dumps(packet)

            process = Process(self, "start", [plaintext, tl, False])
            event = Event(tl.now() + (time * 1.0e12), process)
            tl.schedule(event)

        if len(self.buffer) < self.buffer_capacity:
            self.buffer.appendleft(text)
        else:
            self.drop_mess += 1
            packet = json.loads(text)

            if forwarding:
                self.append_metrics(
                    packet["src"],
                    packet["dest"],
                    True,
                    False,
                    self.own.name,
                    packet["hop"],
                    packet["time"] * 1.0e-12,
                    self.own.timeline.now() * 1.0e-12,
                    (self.own.timeline.now() * 1.0e-12) - (packet["time"] * 1.0e-12),
                )
            else:
                self.append_metrics(
                    packet["src"],
                    packet["dest"],
                    False,
                    False,
                    self.own.name,
                    packet["hop"],
                    packet["time"] * 1.0e-12,
                    self.own.timeline.now() * 1.0e-12,
                    (self.own.timeline.now() * 1.0e-12) - (packet["time"] * 1.0e-12),
                )

    def received_message(self, src: str, msg: Message):
        assert msg.msg_type == MsgType.TEXT_MESS

        if len(self.key_manager.keys) <= 0:
            return

        self.mess_ric += 1

        if len(self.buffer) > 0:
            self.token = True
        else:
            self.other.qkd_node_p.token = True

        key = self.key_manager.consume()
        key = key.to_bytes((key.bit_length() + 7) // 8, "big")

        packet = json.loads(msg.payload)

        message = packet["payload"]
        message = bytes(message)  # b'str'

        key = key[0 : len(message)]

        plaintext = OneTimePad.decrypt(message, key)

        packet["hop"] += 1
        is_print = self.print_idx % self.print_gap == 0
        self.print_idx += 1
        if packet["dest"] == self.super_qkd.name:
            self.del_mess += 1
            self.append_metrics(
                packet["src"],
                packet["dest"],
                True,
                True,
                None,
                packet["hop"],
                packet["time"] * 1.0e-12,
                self.own.timeline.now() * 1.0e-12,
                (self.own.timeline.now() * 1.0e-12) - (packet["time"] * 1.0e-12),
            )
            if is_print:
                print(
                    f"[{self.own.name}]\nMessage received. At simulation time: {self.own.timeline.now() * 1.0e-12} s\nEncrypted Message: {message}\nDecrypted Message: {plaintext}\n"
                )
        else:
            packet["payload"] = list(plaintext)
            if is_print:
                print(
                    f"[{self.own.name}]\nMessage received. At simulation time: {self.own.timeline.now() * 1.0e-12} s. Forwarding ...\nEncrypted Message: {message}\nDecrypted Message: {plaintext}\n"
                )
            self.super_qkd.send_message(
                self.own.timeline, packet["dest"], json.dumps(packet), True
            )

    def add_key_manager(self, key_manager):
        self.key_manager = key_manager

    # For metrics
    def append_metrics(
        self, src, dest, sent, deliv, drop, num_hop, send_time, sim_time, time
    ):
        row = [
            MessagingProtocol.sim_command,
            str(src),
            str(dest),
            str(sent),
            str(deliv),
            str(drop),
            str(num_hop),
            str(send_time),
            str(sim_time),
            str(time),
        ]

        with open(MessagingProtocol.sim_path + "packet_result.csv", "a") as file:
            writer = csv.writer(file)
            writer.writerow(row)
