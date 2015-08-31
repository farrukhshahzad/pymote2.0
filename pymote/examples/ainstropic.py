import datetime

import scipy as sp

from pymote import *
from pymote.conf import global_settings

from pymote import propagation
from toplogies import Toplogy
from pymote.utils import plotter
from pymote.utils.filing import get_path, date2str,\
     DATA_DIR, TOPOLOGY_DIR, CHART_DIR, DATETIME_DIR
import numpy as np
from numpy.random import seed
from numpy import array, sqrt, power

from pymote.algorithms.niculescu2003.trilaterate import Trilaterate
from pymote.simulation import Simulation
from pymote.sensor import TruePosSensor
from pymote.networkgenerator import NetworkGenerator
from pymote.algorithms.niculescu2003.dvhop import DVHop


seed(123)  # to get same random sequence for each run

# Network/Environment setup
global_settings.ENVIRONMENT2D_SHAPE = (600, 600)
net = Network()
h, w = net.environment.im.shape

max_n = 101  # no. of nodes
n = 200
p_anchors = 10 # in %
c_range = 100  # communication radii

clusters = 1
x_radius = w/3/clusters
y_radius = h/3/clusters

degree = 10
Node.cid = 1
net_gen = Toplogy(n_count=n, degree=degree, n_max=n, n_min=n, connected=False)
net = net_gen.generate_grid_network(randomness=0.2)
n_anchors = (int)(100 / p_anchors)
for node in net.nodes():
    if (node.id % n_anchors==0):  # anchor nodes
        node.compositeSensor = (TruePosSensor,)
        node.type = 'C'  # Anchors

# saving topology as PNG image
avg_deg = round(net.avg_degree())
net.name = "Grid - Nodes=%s, Avg degree=%s, Range=%s" % (net.__len__(), int(avg_deg), int(node.commRange))
net.savefig(fname=get_path(TOPOLOGY_DIR, net.name),   title=net.name,
            x_label="X", y_label="Y", show_labels=False)

print net.__len__(), avg_deg, node.commRange
net.reset()


# network topology setup
Node.cid = 1
xpositions = []
ypositions = []
net_gen = NetworkGenerator(n_count=n, degree=degree)
net = net_gen.generate_homogeneous_network()
n_anchors = (int)(100 / p_anchors)
for node in net.nodes():
    xpositions.append(net.pos[node][0])
    ypositions.append(net.pos[node][1])
    if (node.id % n_anchors==0):  # anchor nodes
        node.compositeSensor = (TruePosSensor,)
        node.type = 'C'  # Anchors

avg_deg = round(net.avg_degree())
net.name = "Homogenous - Nodes=%s, Avg degree=%s, Range=%s" % (net.__len__(), int(avg_deg), int(node.commRange))
net.savefig(fname=get_path(TOPOLOGY_DIR, net.name),
            title=net.name, format='svg',
            x_label="X", y_label="Y", show_labels=False)
print net.__len__(),avg_deg,  node.commRange, n_anchors
print xpositions
print range(1,len(xpositions))
# plotter.plot_bars(np.arange(1, n+1), positions,
#                   get_path(TOPOLOGY_DIR, "energy consumption-", prefix=n),
#                   title="Energy Consumption (%s nodes)" %n,
#                   xlabel="Nodes", ylabel="mJ")
plotter.plots(np.arange(len(xpositions)), xpositions,
                  get_path(TOPOLOGY_DIR, "energy consumption-", prefix=n),
                  title="Energy Consumption (%s nodes)" %n,
                  xlabel="Nodes", ylabel="mJ", format='png')

net.reset()
# network topology setup
settings.COMM_RANGE = 50
settings.ENVIRONMENT2D_SHAPE = (200, 200)
Node.cid = 1
net_gen = Toplogy()
net = net_gen.generate_manual_network()
node = net.nodes()[1]
avg_deg = round(net.avg_degree())
net.name = "Manual - Nodes=%s, Avg degree=%s, Range=%s" % (net.__len__(), int(avg_deg), int(node.commRange))
net.savefig(fname=get_path(TOPOLOGY_DIR, net.name), format='png',
            title="Manual", author="FS", x_label="X", y_label="Y", show_labels=True)
print net.__len__(),avg_deg,  node.commRange

# plotter.gethtml(range(1,len(xpositions)), [xpositions, ypositions],
#                 fname=get_path(TOPOLOGY_DIR, "Node Coordinates"),
#                 xlabel="Node", ylabel="Coordinate",
#                 title="Node Coordinates", open=1)
