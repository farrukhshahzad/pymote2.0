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
net_gen = Toplogy(n_count=n, n_max=2*n, n_min=n/2, connected=False, comm_range=c_range)
# cut_shape is a rectangle with top-left and bottom-right coordinates
net = net_gen.generate_grid_network(name="O-shaped Grid", randomness=0.2,
                                    cut_shape=[[(w/4,3*h/4), (3*w/4,h/4)]])

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
#net_gen = Toplogy(n_count=n, degree=degree, n_max=n, n_min=n, connected=False)
net = net_gen.generate_grid_network(name="C-shaped Grid", randomness=0.2,
                                    cut_shape=[[(w/4,3*h/4), (w, h/4)]])
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

print na, avg_deg, node.commRange
net.reset()

Node.cid = 1
#net_gen = Toplogy(n_count=n, degree=degree, n_max=n, n_min=n, connected=False)
net = net_gen.generate_grid_network(name="S-shaped Grid", randomness=0.1,
                                    cut_shape=[[(w/4,3*h/4), (w,7*h/12)], [(0,5*h/12), (3*w/4, h/4)]])
anchors = [10, 30, 42, 84, 119, 110]
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

print na, avg_deg, node.commRange
net.reset()



Node.cid = 1
#net_gen = Toplogy(n_count=n, degree=degree, n_max=n, n_min=n, connected=False)
net = net_gen.generate_grid_network(name="W-shaped Grid", randomness=0.1,
                                    cut_shape=[[(w/4,h), (5*w/12,h/3)], [(7*w/12,h), (3*w/4, h/3)]])
f_anchors = (int)(100 / p_anchors)
for node in net.nodes():
    if node.id % f_anchors == 0:  # anchor nodes
        node.compositeSensor = (TruePosSensor,)
        node.type = 'C'  # Anchors

# saving topology as PNG image
na = net.__len__()
n_anchors = (int)(na *  p_anchors/100.0)
avg_deg = net.avg_degree()
net.name = "%s - $N=$%s(%s), $D=$%s, $R=$%s m\n$A=$%s$\\times 10^3m^2$, $ND=$%s$/10^3.m^2$" \
           % (net_gen.name, na, n_anchors, round(avg_deg,1), round(node.commRange,1),
              round(net_gen.area/1000.0, 2), round(net_gen.net_density*1000, 1))
filename = (net.name.split("\n")[0]).replace("$","")
net.savefig(fname=get_path(TOPOLOGY_DIR, filename),   title=net.name,
            xlabel="X-coordinate (m)", ylabel="Y-coordinate (m)", show_labels=True, format="pdf")

print net.__len__(), avg_deg, node.commRange
net.reset()

Node.cid = 1
#net_gen = Toplogy(n_count=n, degree=degree, n_max=n, n_min=n, connected=False)
net = net_gen.generate_grid_network(name="8-shaped Grid", randomness=0.1,
                                    cut_shape=[[(w/4,3*h/4), (3*w/4,7*h/12)], [(w/4,5*h/12), (3*w/4, h/4)]])
na = net.__len__()
n_anchors = (int)(na *  p_anchors/100.0)
f_anchors = (int)(100 / p_anchors)
# Random Anchor selection
for i in range(1, n_anchors):
        node = net.nodes()[int(rand() * na)]
        node.compositeSensor = (TruePosSensor,)
        node.type = 'C'  # Anchors

# saving topology as PNG image
avg_deg = net.avg_degree()
net.name = "%s - $N=$%s(%s), $D=$%s, $R=$%s m\n$A=$%s$\\times 10^3m^2$, $ND=$%s$/10^3.m^2$" \
           % (net_gen.name, na, n_anchors, round(avg_deg,1), round(node.commRange,1),
              round(net_gen.area/1000.0, 2), round(net_gen.net_density*1000, 1))
filename = (net.name.split("\n")[0]).replace("$","")
net.savefig(fname=get_path(TOPOLOGY_DIR, filename),   title=net.name,
            xlabel="X-coordinate (m)", ylabel="Y-coordinate (m)", show_labels=True, format="pdf")

print net.__len__(), avg_deg, node.commRange
net.reset()

Node.cid = 1
#net_gen = Toplogy(n_count=n, degree=degree, n_max=n, n_min=n, connected=False)
net = net_gen.generate_grid_network(name="H-shaped Grid", randomness=0.1,
                                    cut_shape=[[(w/4,h), (3*w/4,2*h/3)], [(w/4,h/3), (3*w/4, 0)]])
na = net.__len__()
n_anchors = (int)(na *  p_anchors/100.0)
f_anchors = (int)(100 / p_anchors)
# Random Anchor selection
for i in range(1, n_anchors):
        node = net.nodes()[int(rand() * na)]
        node.compositeSensor = (TruePosSensor,)
        node.type = 'C'  # Anchors

# saving topology as PNG image
avg_deg = net.avg_degree()
net.name = "%s - $N=$%s(%s), $D=$%s, $R=$%s m\n$A=$%s$\\times 10^3m^2$, $ND=$%s$/10^3.m^2$" \
           % (net_gen.name, na, n_anchors, round(avg_deg,1), round(node.commRange,1),
              round(net_gen.area/1000.0, 2), round(net_gen.net_density*1000, 1))
filename = (net.name.split("\n")[0]).replace("$","")
net.savefig(fname=get_path(TOPOLOGY_DIR, filename),   title=net.name,
            xlabel="X-coordinate (m)", ylabel="Y-coordinate (m)", show_labels=True, format="pdf")

print net.__len__(), avg_deg, node.commRange
net.reset()


Node.cid = 1
#net_gen = Toplogy(n_count=n, degree=degree, n_max=n, n_min=n, connected=False)
net = net_gen.generate_grid_network(name="T-shaped Grid", randomness=0.2,
                                    cut_shape=[[(0,2*h/3), (w/3,0)], [(2*w/3,2*h/3), (w, 0)]])
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

print net.__len__(), avg_deg, node.commRange, net_gen.area, net_gen.net_density
net.reset()

Node.cid = 1
#net_gen = Toplogy(n_count=n, degree=degree, n_max=n, n_min=n, connected=False)
net = net_gen.generate_grid_network(name="I-shaped Grid", randomness=0.2,
                                    cut_shape=[[(0,5*h/6), (w/3,h/6)], [(2*w/3,5*h/6), (w, h/6)]])
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

print net.__len__(), avg_deg, node.commRange, net_gen.area, net_gen.net_density
net.reset()

Node.cid = 1
#net_gen = Toplogy(n_count=n, degree=degree, n_max=n, n_min=n, connected=False)
net = net_gen.generate_grid_network(name="Plus-shaped Grid", randomness=0.2,
                                    cut_shape=[[(0,h), (w/3, h/6)],[(3*w/3,h), (w, h/6)],
                                    [(0,5*h/6), (w/3, 0)], [(2*w/3,5*h/6), (w, 0)]])
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

print net.__len__(), avg_deg, node.commRange, net_gen.area, net_gen.net_density
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
