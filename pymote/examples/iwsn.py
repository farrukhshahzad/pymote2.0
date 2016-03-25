
# built-in packages
import time
import datetime

# external packages
import scipy as sp
import numpy as np
from em import expectation_maximization
from numpy import sign, sqrt, array, pi, sin, cos
from numpy.random import rand,seed
import networkx as nx
from networkx.algorithms import approximation as approx

# internal packages
from pymote import *
from pymote.conf import global_settings
from beacon import Beacon, MAX_TRIES
from toplogies import Toplogy
from pymote import propagation
from pymote.utils import plotter
from pymote.utils.filing import get_path, date2str,\
     DATA_DIR, TOPOLOGY_DIR, CHART_DIR, DATETIME_DIR


class Manual(Toplogy):

    def generate_zigbee_network(self, name=None, randomness=0):

        from pymote.conf import global_settings
        from pymote.environment import Environment2D

        self.n_count = 0
        self.name = name or "Manual"
        #net = Network(environment=Environment2D(shape=(200,200)), commRange=75)
        net = Network(commRange=self.comm_range, **self.kwargs)
        h, w = net.environment.im.shape

        node = Node(node_type='B')
        net.add_node(node, pos=(w/2, h/5))
        self.n_count +=1

        node = Node(node_type='N')
        net.add_node(node, pos=(3*w/10, h/16))
        self.n_count +=1

        node = Node()
        net.add_node(node, pos=(3*w/10, 2*h/5))
        self.n_count +=1

        node = Node()
        net.add_node(node, pos=(w/2, h/2))
        self.n_count +=1

        node = Node()
        net.add_node(node, pos=(7*w/10, 2*h/5))
        self.n_count +=1

        node = Node()
        net.add_node(node, pos=(7*w/10, h/16))
        self.n_count +=1

        node = Node()
        net.add_node(node, pos=(6*w/10, 15*h/20))
        self.n_count +=1

        return net

    def generate_mesh_network(self, name=None, type=0):

        from pymote.conf import global_settings
        from pymote.environment import Environment2D

        self.n_count = 0
        self.name = name or "Manual"
        #net = Network(environment=Environment2D(shape=(200,200)), commRange=75)
        net = Network(commRange=self.comm_range, **self.kwargs)
        h, w = net.environment.im.shape

        node = Node(node_type='N')
        net.add_node(node, pos=(w/2, h/5))
        net.nodes()[0].type = 'C'
        self.n_count +=1

        node = Node()
        net.add_node(node, pos=(3*w/10, 2*h/5))
        self.n_count +=1

        if type<2:
            node = Node()
            net.add_node(node, pos=(w/2, h/2))
            self.n_count +=1


        node = Node()
        net.add_node(node, pos=(7*w/10, 2*h/5))
        self.n_count +=1

        if type==0:
            node = Node()
            net.add_node(node, pos=(3*w/10, 5*h/8))
            self.n_count +=1

            node = Node()
            net.add_node(node, pos=(7*w/10, 5*h/8))
            self.n_count +=1

        if type==0 or type==2:
            node = Node()
            net.add_node(node, pos=(w/2, 4*h/5 - type/2*h/5))
            self.n_count +=1

        return net

# Network/Environment setup
seed(123)  # to get same random sequence for each run so that simulation can be reproduce

# Network/Environment setup
global_settings.ENVIRONMENT2D_SHAPE = (500, 500) # desired network size for simulation
global_settings.COMM_RANGE = 160

net = Network()
h, w = net.environment.im.shape
propagation.PropagationModel.P_TX = 0.0144  # Node Tx Power

max_n = 100  # max no. of nodes
clusters = 10
x_radius = 0.9*global_settings.COMM_RANGE
y_radius = 0.9*global_settings.COMM_RANGE

n_power = []
n_range = np.arange(20, max_n+1, 10)
start_time = time.time()
project_name = "IWSN"
folder = DATETIME_DIR+ "-" + project_name

# network topology setup
Node.cid = 1
net_gen = Manual()
net = net_gen.generate_zigbee_network(name="Zigbee")
n = len(net)
net.name = "Zigbee Experiment" + "\nN=%s, R=%s m" \
                                 %(n, global_settings.COMM_RANGE)
filename = net.name.replace("\n", "-")
# saving topology as PNG image
net.savefig(fname=get_path(folder, filename),   title=net.name,
            xlabel="X-coordinate (m)", ylabel="Y-coordinate (m)",
            show_labels=True, format="png", label_color='yellow')
print net.__len__(), len(net)
net.reset()

Node.cid = 1
net_gen = Manual()
net = net_gen.generate_mesh_network(name="WirelessHart Mesh")
n = len(net)
net.name = "WirelessHart Experiment" + "\nN=%s, R=%s m" \
                                 %(n, global_settings.COMM_RANGE)
filename = net.name.replace("\n", "-")
# saving topology as PNG image
net.savefig(fname=get_path(folder, filename),   title=net.name,
            xlabel="X-coordinate (m)", ylabel="Y-coordinate (m)",
            show_labels=True, format="png", label_color='yellow')
print net.__len__(), len(net)
net.reset()

Node.cid = 1
net_gen = Manual()
net = net_gen.generate_mesh_network(name="ISA100 Mesh", type=1)
n = len(net)
net.name = "ISA100 Plane Experiment" + "\nN=%s, R=%s m" \
                                 %(n, global_settings.COMM_RANGE)
filename = net.name.replace("\n", "-")
# saving topology as PNG image
net.savefig(fname=get_path(folder, filename),   title=net.name,
            xlabel="X-coordinate (m)", ylabel="Y-coordinate (m)",
            show_labels=True, format="png", label_color='yellow')
print net.__len__(), len(net)
net.reset()

Node.cid = 1
net_gen = Manual()
net = net_gen.generate_mesh_network(name="ISA100 Mesh", type=2)
n = len(net)
net.name = "ISA100 Experiment (obstacle)" + "\nN=%s, R=%s m" \
                                 %(n, global_settings.COMM_RANGE)
filename = net.name.replace("\n", "-")
# saving topology as PNG image
net.savefig(fname=get_path(folder, filename),   title=net.name,
            xlabel="X-coordinate (m)", ylabel="Y-coordinate (m)",
            show_labels=True, format="png", label_color='yellow')
print net.__len__(), len(net)
net.reset()

# save data to text file for further analysis
end_time = time.time() - start_time
print("Execution time:  %s seconds ---" % round(end_time,2) )

# sp.savetxt(get_path(folder, "Energy Consumption.csv"),
#            n_power,
#            delimiter="\t", fmt="%s",
#            header="Nodes\tEnergy(m^2)\tefficeincy",
#            comments='')
#
# plotter.plots(n_range, [e[1]/10e6 for e in n_power],
#               get_path(folder, "Energy"),
#               title="Energy Consumption, Edat",
#               more_plots=[[e[3]/10e6 for e in n_power]],
#               xlabel="Number of nodes", ylabel="Energy, Edat ($10^6, m^2$)")
#
# plotter.plots(n_range, [e[2]/10e-4 for e in n_power],
#               get_path(folder, "Efficiency"),
#               title="Efficiency",
#               xlabel="Number of Nodes", ylabel="Efficiency ($10^{-4}$)")
