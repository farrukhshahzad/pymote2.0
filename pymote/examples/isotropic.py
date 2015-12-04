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
from numpy.random import seed,rand

from pymote.algorithms.niculescu2003.trilaterate import Trilaterate
from pymote.simulation import Simulation
from pymote.sensor import TruePosSensor
from pymote.networkgenerator import NetworkGenerator
from pymote.algorithms.niculescu2003.dvhop import DVHop


seed(100)  # to get same random sequence for each run

# Network/Environment setup
#net = Network(environment=Environment2D(shape=(200,200)), commRange=75)
#h, w = net.environment.im.shape
global_settings.ENVIRONMENT2D_SHAPE = (100, 100)
global_settings.COMM_RANGE = 10
c_range = 10  # communication radii
net = Network(commRange=c_range)
h, w = net.environment.im.shape

n = 17*17  # no. of nodes
p_anchors = 10 # in %

clusters = 1
x_radius = w/3/clusters
y_radius = h/3/clusters
degree = 10


# Topology setup
Node.cid = 1
net_gen = NetworkGenerator(n_count=n, degree=degree)
net = net_gen.generate_homogeneous_network()

f_anchors = (int)(100 / p_anchors)
for node in net.nodes():
    if node.id % f_anchors == 0:  # anchor nodes
        node.compositeSensor = (TruePosSensor,)
        node.type = 'C'  # Anchors

# saving topology as PNG image
avg_deg = net.avg_degree()
na = net.__len__()
n_anchors = (int)(na *  p_anchors/100.0)
net.name = "%s - $N=$%s(%s), $D=$%s, $R=$%s m\n$A=$%s$\\times 10^3m^2$, $ND=$%s$/10^3.m^2$" \
           % (net_gen.name, na, n_anchors, round(avg_deg,1), round(node.commRange,1),
              round(net_gen.area/1000.0, 2), round(net_gen.net_density*1000, 1))
filename = (net.name.split("\n")[0]).replace("$","")
net.savefig(fname=get_path(TOPOLOGY_DIR, filename),   title=net.name,
            xlabel="X-coordinate (m)", ylabel="Y-coordinate (m)", show_labels=True, format="pdf")

print net.__len__(), avg_deg, node.commRange
net.reset()


Node.cid = 1
net_gen = Toplogy(n_count=n, degree=degree, n_max=n, n_min=n, connected=False)
net = net_gen.generate_grid_network(name="Randomized Grid", randomness=1.0)

f_anchors = (int)(100 / p_anchors)
for node in net.nodes():
    if node.id % f_anchors == 0:  # anchor nodes
        node.compositeSensor = (TruePosSensor,)
        node.type = 'C'  # Anchors

# saving topology as PNG image
avg_deg = net.avg_degree()
na = net.__len__()
n_anchors = (int)(na *  p_anchors/100.0)
net.name = "%s - $N=$%s(%s), $D=$%s, $R=$%s m\n$A=$%s$\\times 10^3m^2$, $ND=$%s$/10^3.m^2$" \
           % (net_gen.name, na, n_anchors, round(avg_deg,1), round(node.commRange,1),
              round(net_gen.area/1000.0, 2), round(net_gen.net_density*1000, 1))
filename = (net.name.split("\n")[0]).replace("$","")
net.savefig(fname=get_path(TOPOLOGY_DIR, filename),   title=net.name,
            xlabel="X-coordinate (m)", ylabel="Y-coordinate (m)", show_labels=True, format="pdf")

print na, avg_deg, node.commRange, net_gen.area, net_gen.net_density
net.reset()


# Manual network topology setup
# Node.cid = 1
# net_gen = Toplogy()
# net = net_gen.generate_manual_network()
# node = net.nodes()[1]
# avg_deg = round(net.avg_degree())
# net.name = "%s - Nodes=%s, Avg degree=%s, Range=%s" \
#            % (net_gen.name, net.__len__(), int(avg_deg), int(node.commRange))
#
# net.savefig(fname=get_path(TOPOLOGY_DIR, net.name), format='png',
#             title=net.name, author="FS", xlabel="X", ylabel="Y", show_labels=True)
# print net.__len__(),avg_deg,  node.commRange
#
