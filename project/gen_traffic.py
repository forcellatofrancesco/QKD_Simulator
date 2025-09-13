import json


def main():
    output_file: str = "./project/file/traffic_15_nodes_outer_chatting.json"
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
    traffic = {"packets": packets}
    with open(output_file, "w") as f:
        json.dump(traffic, f, indent=4)


if __name__ == "__main__":
    main()
