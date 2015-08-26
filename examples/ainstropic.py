import datetime

import scipy as sp

from pymote import *
from pymote.conf import global_settings
from beacon import Beacon, MAX_TRIES
from pymote import propagation
from toplogies import Star
from pymote.utils import plotter
from pymote.utils.filing import get_path, date2str,\
     DATA_DIR, TOPOLOGY_DIR, CHART_DIR, DATETIME_DIR
import numpy as np

from numpy import array, sqrt, power

from pymote.algorithms.niculescu2003.trilaterate import Trilaterate
from pymote.simulation import Simulation
from pymote.sensor import TruePosSensor
from pymote.networkgenerator import NetworkGenerator
from pymote.algorithms.niculescu2003.dvhop import DVHop


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
net_gen = NetworkGenerator(n_count=n, degree=degree, n_max=n, n_min=n, connected=False)
net = net_gen.generate_random_network()
n_anchors = (int)(p_anchors * net.__len__()/100.0)
for node in net.nodes()[:20]:
        node.compositeSensor = (TruePosSensor,)
        node.type = 'C'  # Anchors

net.name = "random Deg: %s" %degree
# saving topology as PNG image
avg_deg = round(net.avg_degree())
net.savefig(fname=get_path(TOPOLOGY_DIR, "Random-%s-%i" %(n, avg_deg)),
            title="%s - %s nodes, %s Avg. degree" % (net.name, n, avg_deg),
            x_label="X", y_label="Y", show_labels=False)

print net.__len__(), avg_deg, node.commRange
net.reset()


# network topology setup
Node.cid = 1
net_gen = NetworkGenerator(n_count=n, degree=degree)
net = net_gen.generate_homogeneous_network()
n_anchors = (int)(p_anchors * net.__len__()/100.0)
for node in net.nodes():
    if (node.id % p_anchors==0):  # anchor nodes
        node.compositeSensor = (TruePosSensor,)
        node.type = 'C'  # Anchors
avg_deg = round(net.avg_degree())
net.savefig(fname=get_path(TOPOLOGY_DIR, "Homogenous-%s-%i" %(n, avg_deg)),
            title="Homogeneous - %s nodes, %s Avg degree" % (n, avg_deg),
            x_label="X", y_label="Y", show_labels=False)
print net.__len__(),avg_deg,  node.commRange, n_anchors

