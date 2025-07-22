echo "Low load simulation on 15 nodes graph"
python3 project/sim_ext.py --sim-time 7.5 --seq-graph project/file/graph_15_nodes.json --traffic project/file/traffic_15_nodes_net.json --inspection-rate 0.001 --mess-rate 0.01

echo "Heavy load simulation on 15 nodes graph"
python3 project/sim_ext.py --sim-time 1.5 --seq-graph project/file/graph_15_nodes.json --traffic project/file/traffic_15_nodes_net.json --inspection-rate 0.001 --mess-rate 0.002
