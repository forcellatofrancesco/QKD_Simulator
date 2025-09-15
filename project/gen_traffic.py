import json
import random


def gen_outer_traffic():
    outer_nodes: list[int] = [7, 8, 9, 10, 11, 12, 13, 14]
    # inner_nodes: list[int] = [0, 1, 2, 3, 4, 5, 6]
    packets: list[dict] = []
    for node in outer_nodes:
        dst_nodes: list[str] = list(
            map(
                lambda x: f"node{x}",
                filter(
                    lambda x: x != node,
                    outer_nodes,
                ),
            )
        )
        packet = {"src_node": f"node{node}", "dst_node": dst_nodes}
        packets.append(packet)


def gen_random_traffic():
    length = 15
    nodes = list(range(0, length))
    packets: list[dict] = []
    for node in nodes:
        dst_nodes: list[int] = []
        while len(dst_nodes) < length // 2:
            r = random.randint(0, length - 1)
            if r != node and r not in dst_nodes:
                dst_nodes.append(r)

        packet = {
            "src_node": f"node{node}",
            "dst_node": list(map(lambda x: f"node{x}", sorted(dst_nodes))),
        }

        packets.append(packet)
    return packets


def main():
    output_file: str = "./project/file/traffic_15_nodes_random.json"
    packets = gen_random_traffic()
    traffic = {"packets": packets}
    with open(output_file, "w") as f:
        json.dump(traffic, f, indent=4)


if __name__ == "__main__":
    main()
