TRAFFIC_FILE="project/file/traffic_15_nodes_outer_chatting.json"
python3 project/sim_ext.py --sim-time 7.5 --seq-graph project/file/graph_15_nodes.json --traffic "$TRAFFIC_FILE" --inspection-rate 0.001 --mess-rate 0.01 --algorithm betweenness &
python3 project/sim_ext.py --sim-time 1.5 --seq-graph project/file/graph_15_nodes.json --traffic "$TRAFFIC_FILE" --inspection-rate 0.001 --mess-rate 0.002 --algorithm betweenness &
python3 project/sim_ext.py --sim-time 7.5 --seq-graph project/file/graph_15_nodes.json --traffic "$TRAFFIC_FILE" --inspection-rate 0.001 --mess-rate 0.01 --algorithm shortest &
python3 project/sim_ext.py --sim-time 1.5 --seq-graph project/file/graph_15_nodes.json --traffic "$TRAFFIC_FILE" --inspection-rate 0.001 --mess-rate 0.002 --algorithm shortest &
python3 project/sim_ext.py --sim-time 7.5 --seq-graph project/file/graph_15_nodes.json --traffic "$TRAFFIC_FILE" --inspection-rate 0.001 --mess-rate 0.01 --algorithm greedy &
python3 project/sim_ext.py --sim-time 1.5 --seq-graph project/file/graph_15_nodes.json --traffic "$TRAFFIC_FILE" --inspection-rate 0.001 --mess-rate 0.002 --algorithm greedy &
wait
