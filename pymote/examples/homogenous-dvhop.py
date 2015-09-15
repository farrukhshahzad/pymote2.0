"""
    This is an example of python script to show a complete simulation example.
    In this example, the famous range-free localization algrothm called 'DV-hop'
    is simulated. Follow the comments in the script,
"""

'''First we need to include required packages'''

# built-in packages
import time

# external packages
import scipy as sp
import numpy as np
from numpy.random import seed
from numpy import sqrt, amax, amin, std, var, mean

# Internal packages
from pymote import *
from pymote.conf import global_settings
from pymote import propagation
from toplogies import Toplogy
from pymote.utils import plotter
from pymote.utils.filing import get_path, load_metadata,\
     DATA_DIR, TOPOLOGY_DIR, CHART_DIR, DATETIME_DIR

from pymote.simulation import Simulation
from pymote.sensor import TruePosSensor
from pymote.networkgenerator import NetworkGenerator
from pymote.algorithms.niculescu2003.dvhop import DVHop
from pymote.algorithms.niculescu2003.trilaterate import Trilaterate

'''Start of Script'''

# get the pymote version and print it
meta = load_metadata()
print meta['name'], meta['version']

seed(123)  # to get same random sequence for each run so that simulation can be reproduce

# Network/Environment setup
n = 100  # total no of nodes
p_anchors = 20  # No. of anchors in %age
c_range = 100  # communication radii of each node
degree = 10   # Desired degree or connectivity (how many nodes are in range)
global_settings.ENVIRONMENT2D_SHAPE = (600, 600)  # desired network size for simulation
net = Network(commRange=c_range)  # Initiate the network object
h, w = net.environment.im.shape  # get the network width and height(should be as above)

# Start out with empty arrays(list) to be filled in after simulation from result
xpositions = []
xestpositions = []
deltapos = []
esterror = []
positions = []
newpos = []
anchpositions = []
message_stats = []
position_stats = []
consume = []
energy = []

# Network Topology setup
Node.cid = 1  # start node id

net_gen = NetworkGenerator(n_count=n, degree=degree)
net = net_gen.generate_homogeneous_network()  # A random homogeneous topology

#net_gen = Toplogy(n_count=n, degree=degree, n_max=n, n_min=n, connected=False)
#net = net_gen.generate_grid_network(randomness=0.2)

# Computes no. of anchors
f_anchors = (int)(100 / p_anchors)
n_anchors = (int)(n *  p_anchors/100.0)

# Set some nodes as anchor based on number of desired anchors
# Two arrays are populated with location of nodes to be plotted later
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


nn = net.__len__()
n_anchors = (int)(nn *  p_anchors/100.0)
avg_deg = round(net.avg_degree())
comm_range = node.commRange
# set the network name based on no. of Nodes, degree and Comm. Range
net.name = "%s - $N=$%s(%s), $D=$%s, $R=$%s m\n$A=$%s$\\times 10^3m^2$, $ND=$%s$/10^3.m^2$" \
           % (net_gen.name, nn, n_anchors, round(avg_deg,1), round(comm_range,1),
              round(net_gen.area/1000.0, 2), round(net_gen.net_density*1000, 1))
filename = (net.name.split("\n")[0]).replace("$","")
net.savefig(fname=get_path(DATETIME_DIR, filename),   title=net.name,
            x_label="X-coordinate (m)", y_label="Y-coordinate (m)", show_labels=False, format="pdf")


# Now select the algorithm for simulation
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
total_tx = 0  # Total number of Tx by all nodes
total_rx = 0  # Total number of Rx by all nodes
total_energy = 0  # Total energy consumption by all nodes

for node in net.nodes():
    total_tx += node.n_transmitted
    total_rx += node.n_received
    err=0
    message_stats.append([node.id, node.n_transmitted,
                              node.n_transmitted_failed_power,
                              node.n_received,
                              node.n_received_failed_power,
                              node.n_received_failed_loss
                              ])
    consume.append(round(node.power.energy_consumption * 1000.0, 2))
    energy.append(node.power.energy * 1000.0)
    total_energy += node.power.energy_consumption

    if node.type == 'N' and 'pos' in node.memory:
        act = node.get_dic()['Info']['position']
        est = node.memory['pos']
        newx = est[0]
        newy = est[1]
        err = sqrt(sum(pow(act - est, 2)))
        print node.id, node.type, act, est, err
        position_stats.append([node.id, newx, newy, err])
        err_sum += err
        k += 1
        newpos.append({'x': newx, 'y': newy,
                  'name': 'Node: ' + str(node.id)})
    elif node.type == 'C':
        newx = net.pos[node][0]
        newy = net.pos[node][1]
    else:
        newx = -1
        newy = -1
    xestpositions.append(newx)
    esterror.append(err)
    deltapos.append(net.pos[node][0] - newx)

# Summary of simulation result/Metrics
comments = "Anchors: " + str(n_anchors) + " = " +str(p_anchors) +"%"+ \
           ",  Runtime(sec): "+ str(round(end_time,2)) + \
           ",  Tx/Rx per node: " + str(round(total_tx/nn)) + "/" + \
                                       str(round(total_rx/nn)) + \
           ",  Energy/node (mJ): " + str(round(1000*total_energy/nn, 3)) + \
           ",  Avg. error: " + str(round(err_sum/k, 2))


print ("%s nodes localized, %s" % (k, comments))
print "max=" + str(np.amax(message_stats, axis=0))
print "std=" + str(np.std(message_stats, axis=0))
print "mean=" + str(np.mean(message_stats, axis=0))

print "Position max=" + str(np.amax(position_stats, axis=0))

try:
    X2 = np.sort([col[3]/comm_range for col in position_stats])
    F2 = np.array(range(k))/float(k)
    H, X1 = np.histogram([col[3] for col in position_stats], bins=20, normed=True)
    dx = X1[1] - X1[0]
    F1 = np.cumsum(H)*dx
    print X2.__len__(), F2.__len__(), F1.__len__()
    plotter.plots(X1[1:]/comm_range,F1,get_path(DATETIME_DIR, "Error CDF-%d" % nn),
                  title="CDF-"+net.name, xlabel="Normalized Localization error",
                  ylabel="CDF")

    plotter.plots(X2,F2,get_path(DATETIME_DIR, "Error CDF2-%d" % nn),
                  title="CDF-"+net.name, xlabel="Normalized Localization error",
                  ylabel="CDF")

except Exception as exc:
    print exc
    pass

sim.reset()

# Create CSV, plots and interactive html/JS charts based on collected data
# np.savetxt(get_path(DATETIME_DIR, "message_stats-%s.csv" % nn),
#            message_stats,
#            delimiter=",", fmt="%8s", comments='',
#            header="Node,transmit,transmit_failed_power,"
#                   "received,received_failed_power,"
#                   "received_failed_loss")
#
# sp.savetxt(get_path(DATETIME_DIR, "localization_error-%s.csv" % nn),
#            position_stats,
#            delimiter=",", fmt="%8s", comments='',
#            header="Node,actual,estimated,error")
#
# sp.savetxt(get_path(DATETIME_DIR, "energy-%s.csv" %nn),
#                list(zip(range(1, n+1), energy, consume)),
#                delimiter="\t", fmt="%s",
#                header="Nodes\tEnergy Left (mJ)\tConsumed", comments='')
#
# # Create html/JS file for network Topology
# plotter.gethtmlScatter(xpositions, [anchpositions, positions, newpos],
#                 fname=filename, folder=DATETIME_DIR,
#                 xlabel="X", ylabel="Y", labels=['Anchor','Regular','Localized'],
#                 title="Topology-"+filename, open=1, axis_range={'xmin':0, 'ymin':0, 'xmax': w, 'ymax': h},
#                 comment=comments, show_range=str(int(node.commRange)),
#                 plot_options=["color: 'red', visible: false,", "color: 'blue',", "color: 'green', visible: false,"])
#
# plotter.gethtmlLine(range(1,len(xpositions)), [xpositions, xestpositions, deltapos, esterror],
#                 fname="X-"+filename, folder=DATETIME_DIR,
#                 xlabel="Node", ylabel="meters", labels=['Actual', 'Estimated', 'X-Error', 'Est. Error'],
#                 title="X-"+filename, open=1,
#                 comment=comments,
#                 plot_options=["color: 'red',", "color: 'blue',", "type: 'areaspline', color: 'grey', visible: false,"])
#
# plotter.gethtmlLine(range(1,len(xpositions)), [consume, [row[1] for row in message_stats]],
#                 fname="Power-"+filename, folder=DATETIME_DIR,
#                 xlabel="Node", ylabel="mJ", labels=['Energy','No. of Tx'],
#                 open=1, comment=comments, plot_type='line',
#                 plot_options=["color: 'red',", "color: 'blue',", "type: 'areaspline', color: 'grey', visible: false,"])