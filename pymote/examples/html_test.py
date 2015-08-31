import time

import numpy as np
from numpy.random import seed, rand
from numpy import array, sqrt, power

from pymote import *
from pymote.conf import global_settings
from pymote.utils import plotter
from pymote.utils.filing import get_path, date2str,\
     DATA_DIR, TOPOLOGY_DIR, CHART_DIR, DATETIME_DIR
from pymote.sensor import TruePosSensor
from pymote.networkgenerator import NetworkGenerator

from toplogies import Toplogy

#seed(100)  # to get same random sequence for each run

# Network/Environment setup
global_settings.ENVIRONMENT2D_SHAPE = (600, 600)
net = Network()
h, w = net.environment.im.shape

n = 200
p_anchors = 20 # in %
c_range = 100  # communication radii

degree = 10
Node.cid = 1

xpositions = []
xestpositions = []
deltapos = []
positions = []
newpos = []
anchpositions = []

start_time = time.time()

#net_gen = NetworkGenerator(n_count=n, degree=degree)
#net = net_gen.generate_homogeneous_network()
net_gen = Toplogy(n_count=n, degree=degree, n_max=n, n_min=n, connected=False)
net = net_gen.generate_grid_network(randomness=0.2)

print net_gen.name

f_anchors = (int)(100 / p_anchors)
n_anchors = (int)(n *  p_anchors/100.0)
for node in net.nodes():
    newx = net.pos[node][0] + (rand() - 0.5) * 100
    newy = net.pos[node][1] + (rand() - 0.5) * 100
    xpositions.append(net.pos[node][0])
    if (node.id % f_anchors==0):  # anchor nodes
        newx = net.pos[node][0]
        newy = net.pos[node][1]
        node.compositeSensor = (TruePosSensor,)
        node.type = 'C'  # Anchors
        anchpositions.append({'x': net.pos[node][0], 'y': net.pos[node][1],
                              'name': str(node.id), 'color': 'red',
                              'marker': {'symbol': 'circle', 'radius': '8'}})
    else:
        positions.append({'x': net.pos[node][0], 'y': net.pos[node][1],
                          'name': 'Node: ' + str(node.id)})
        newpos.append({'x': newx, 'y': newy,
                       'name': 'Node: ' + str(node.id)})

    xestpositions.append(newx)
    deltapos.append(net.pos[node][0] - newx)

avg_deg = round(net.avg_degree())
net.name = "%s - Nodes=%s, Avg degree=%s, Range=%s" \
           % (net_gen.name, net.__len__(), int(avg_deg), int(node.commRange))

net.savefig(fname=get_path(TOPOLOGY_DIR, net.name),
            title=net.name, format='pdf',
            x_label="X", y_label="Y", show_labels=False)
print net.__len__(),avg_deg,  node.commRange, n_anchors
end_time = time.time() - start_time
ntx = 345
comments = "Anchors: " + str(n_anchors) +"="+str(p_anchors) +"%"+ \
           ",   Runtime(sec): "+ str(round(end_time,2)) + \
           ",   No. of Tx: " + str(ntx)


plotter.gethtmlScatter(xpositions, [anchpositions, positions, newpos],
                fname="Topology-"+net.name, folder=TOPOLOGY_DIR,
                xlabel="X", ylabel="Y", labels=['Anchor','Regular','Localized'],
                title="Topology-"+net.name, open=1, range={'xmin':0, 'ymin':0, 'xmax': w, 'ymax': h},
                comment=comments,
                plot_options=["color: 'red', visible: false,", "color: 'blue',", "color: 'pink', visible: false,"])

plotter.gethtmlLine(range(1,len(xpositions)), [xpositions, xestpositions, deltapos],
                fname="X-"+net.name, folder=TOPOLOGY_DIR,
                xlabel="Node", ylabel="X-Coordinate", labels=['Actual', 'Estimated', 'Error'],
                title="X-"+net.name, open=1,
                comment=comments,
                plot_options=["color: 'red',", "color: 'blue',", "type: 'areaspline', color: 'grey', visible: false,"])

dt = [range(1,len(xpositions)+1), xpositions, xestpositions, deltapos]
np.savetxt(get_path(TOPOLOGY_DIR, net.name+".csv"),
           np.column_stack((dt)),
           delimiter=",", fmt="%s", comments='',
           header="Node,actual,estimated,error")

print("Execution time:  %s seconds ---" % round(end_time,2) )
