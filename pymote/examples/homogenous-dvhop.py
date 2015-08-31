import time
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

n = 200
p_anchors = 20 # in %
c_range = 100  # communication radii

#x_radius = w/3
#y_radius = h/3

degree = 10
xpositions = []
xestpositions = []
deltapos = []
positions = []
newpos = []
anchpositions = []
message_stats = []
position_stats = []

# network topology setup
Node.cid = 1

net_gen = NetworkGenerator(n_count=n, degree=degree)
net = net_gen.generate_homogeneous_network()
f_anchors = (int)(100 / p_anchors)
n_anchors = (int)(n *  p_anchors/100.0)
for node in net.nodes():
    xpositions.append(net.pos[node][0])
    if (node.id % f_anchors==0):  # anchor nodes
        node.compositeSensor = (TruePosSensor,)
        node.type = 'C'  # Anchors
        anchpositions.append({'x': net.pos[node][0], 'y': net.pos[node][1],
                              'name': str(node.id), 'color': 'red',
                              'marker': {'symbol': 'circle', 'radius': '8'}})
    else:
        positions.append({'x': net.pos[node][0], 'y': net.pos[node][1],
                          'name': 'Node: ' + str(node.id)})

avg_deg = round(net.avg_degree())
net.name = "%s - Nodes=%s, Avg degree=%s, Range=%s" \
           % (net_gen.name, net.__len__(), int(avg_deg), int(node.commRange))

net.savefig(fname=get_path(DATETIME_DIR, net.name),
            title=net.name, format='png',
            x_label="X", y_label="Y", show_labels=False)

print net.__len__(),avg_deg,  node.commRange, n_anchors

#Select algorithm
net.algorithms = ((DVHop, {'truePositionKey': 'tp',
                                  'hopsizeKey': 'hs',
                                  'dataKey': 'I'
                                  }),
                   (Trilaterate, {'truePositionKey': 'tp',
                                        'hopsizeKey': 'hs',
                                        'positionKey': 'pos',
                                        'dataKey': 'I'}),
                    )

start_time = time.time()
# simulation start
sim = Simulation(net)
sim.run()
# simulation ends
end_time = time.time() - start_time
print("Execution time:  %s seconds ---" % round(end_time,2) )

# Now data capturing/analysis and visualization
k=0
err_sum=0.0
total_tx = 0
for node in net.nodes():
    total_tx += node.n_transmitted
    message_stats.append((node.id, node.n_transmitted,
                              node.n_transmitted_failed_power,
                              node.n_received,
                              node.n_received_failed_power,
                              node.n_received_failed_loss
                              ))
    if node.type == 'N' and 'pos' in node.memory:
        act = node.get_dic()['Info']['position']
        est = node.memory['pos']
        newx = est[0]
        newy = est[1]
        err = sqrt(sum(pow(act - est, 2)))
        print node.id, node.type, act, est, err
        position_stats.append((node.id, newx, newy, err))
        err_sum += err
        k += 1
    elif node.type ==' C':
        newx = net.pos[node][0]
        newy = net.pos[node][1]
    else:
        newx = 0
        newy = 0
    xestpositions.append(newx)

    newpos.append({'x': newx, 'y': newy,
                  'name': 'Node: ' + str(node.id)})
    deltapos.append(net.pos[node][0] - newx)


comments = "Anchors: " + str(n_anchors) +"="+str(p_anchors) +"%"+ \
           ",   Runtime(sec): "+ str(round(end_time,2)) + \
           ",   No. of Tx: " + str(total_tx) + \
           ",  Avg. error: " + str(err_sum/k)


print ("%s nodes localized, Avg error: %s m" % (k, err_sum/k))
sim.reset()

sp.savetxt(get_path(DATETIME_DIR, "stats-%s.csv" % n),
           message_stats,
           delimiter=",", fmt="%8s", comments='',
           header="Node,transmit,transmit_failed_power,"
                  "received,received_failed_power,"
                  "received_failed_loss")

sp.savetxt(get_path(DATETIME_DIR, "localization_error-%s.csv" % n),
           position_stats,
           delimiter=",", fmt="%8s", comments='',
           header="Node,actual,estimated,error")

plotter.gethtmlScatter(xpositions, [anchpositions, positions, newpos],
                fname="Topology-"+net.name, folder=DATETIME_DIR,
                xlabel="X", ylabel="Y", labels=['Anchor','Regular','Localized'],
                title="Topology-"+net.name, open=1, range={'xmin':0, 'ymin':0, 'xmax': w, 'ymax': h},
                comment=comments,
                plot_options=["color: 'red', visible: false,", "color: 'blue',", "color: 'pink', visible: false,"])

plotter.gethtmlLine(range(1,len(xpositions)), [xpositions, xestpositions, deltapos],
                fname="X-"+net.name, folder=DATETIME_DIR,
                xlabel="Node", ylabel="X-Coordinate", labels=['Actual', 'Estimated', 'Error'],
                title="X-"+net.name, open=1,
                comment=comments,
                plot_options=["color: 'red',", "color: 'blue',", "type: 'areaspline', color: 'grey', visible: false,"])

# plotter.plots(n_range, c_power,  #[e[0] for e in n_power], [e[1] for e in n_power],
#               get_path(DATETIME_DIR, "energy-Coordinator"),
#               more_plots=[[e[2] for e in n_power],
#                           [e[3] for e in n_power],
#                           [e[4] for e in n_power]],
#               labels=["Coordinator (mJ)","EHWSN nodes (mJ)",
#                       "Received Packets", "Lost Packets"],
#               title="Energy Consumption & Received Packet Stat",
#               xlabel="Nodes", ylabel="")
#
# # save data to text file for further analysis
# sp.savetxt(get_path(DATETIME_DIR, "energy-Coordinator.csv"),
#            n_power,
#            delimiter="\t", fmt="%7.0f",
#            header="Nodes\tEnergy(mJ)\tTotal(mJ)\tReceived\tLost",
#            comments='')