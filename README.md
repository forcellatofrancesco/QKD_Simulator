# Quantum Key Distribution Performance: Extending the SeQUeNCe simulator to Estimate the Speed of free-space Optic Links

This repository contains the **SeQUeNCe** simulator extension code. The extension uses the various components of SeQUeNCe and is intended to implement a **Quantum Key Distribution (QKD)** network where a message exchange from node A and node B is possible only every hop on the path, made on **Free Space Optic (FSO)** links, connecting the two nodes has a quantum key generated with the **Bennett-Brassard (BB84)** protocol and subsequently analyze its performance.


# Installation

## Virtual Environment

Let's create a virtual environment using the Python module venv. We download the repository from GitHub (or via git clone), then inside the simulator folder and  let's go create a virtual environment as follows:

```bash
python3 -m venv path/to/env
```

Next we activate the virtual environment:

```bash
source path-to-venv/bin/activate
```

## Installation

Once the virtual environment has been created and activated, we are going to install the various requirements that the extension requires. Inside the extension folder we start the installation with the following commands:
```bash
pip install -r requirements.txt
cd SeQUeNCe & pip install .
```
# Simulations

Simulation parameters are customizable using the following CLI arguments when launch-
ing a simulation:

- `--sim-time SIM_TIME`: Establishes the simulation time in seconds.
- `--netx-graph NETX_GRAPH`: Specifies the path to a network json file in Net-
workX.
- `--seq-graph SEQ_GRAPH`: Specifies the path to the json file of a network in Se-
quence.
- `--key-size KEY_SIZE`: Indicates the length of keys that QKD will generate for node
pairs.
- `--mess-rate MESS_RATE`: Specifies the load rate, i.e. how many packets each
node will generate per second on average, times generated by an exponential random variable with mess-rate as parameter.
- `--num-nodes NUM_NODES`: If we want to generate a random network with Net-
workX, this parameter indicates how many nodes we want in our network.
- `--buff-capacity BUFF_CAPACITY`: Indicates the capacity of the buffers in the
nodes.
- `--inspection-rate INSPECTION_RATE`: This rate is used to inspect the buffers
and keys in each node, that is, how often we visit the nodes to obtain information.
- `--traffic TRAFFIC`: A path must be specified to a json file that contains custom
traffic.

Example of a command:

```bash
python3 project/sim_ext.py --seq-graph example_graph.json --traffic traffic.json --inspection-rate 0.001 --sim-time 1.0 --mess-rate 0.1
```

