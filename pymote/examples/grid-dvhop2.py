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
from pymote import propagation,energy
from toplogies import Toplogy
from pymote.utils import plotter
from pymote.utils.filing import getDateStr, get_path, load_metadata,\
     DATA_DIR, TOPOLOGY_DIR, CHART_DIR, DATETIME_DIR

from pymote.simulation import Simulation
from pymote.sensor import TruePosSensor
from pymote.networkgenerator import NetworkGenerator
from pymote.algorithms.niculescu2003.dvhop import DVHop
from pymote.algorithms.niculescu2003.trilaterate import Trilaterate
from pymote.algorithms.shazad2015.dvhop import DVHop as DVmaxhop
from pymote.algorithms.shazad2015.trilaterate import Trilaterate as Trilateratemax

from pymote.algorithms.ral2009.dvhop import DVHop as RalHop
from pymote.algorithms.ral2009.trilaterate import Trilaterate as TrilaterateRal

'''Start of Script'''

# get the pymote version and print it
meta = load_metadata()
print meta['name'], meta['version']

seed(100)  # to get same random sequence for each run so that simulation can be reproduce

# Network/Environment setup
global_settings.ENVIRONMENT2D_SHAPE = (100, 100) # desired network size for simulation
global_settings.COMM_RANGE = 10
global_settings.CHANNEL_TYPE = 'Doi'
global_settings.DOI = 0.4

maxHop = 10
method="DV-maxHop"
algorithm = {"RAL": [RalHop, TrilaterateRal], "DV-hop": [DVHop, Trilaterate], "DV-maxHop": [DVmaxhop, Trilateratemax]}

clusters=30
n = 300  # total no of nodes
p_anchors = 11   # No. of anchors in %age
c_range = 10  # communication radii of each node
degree = 10   # Desired degree or connectivity (how many nodes are in range)
x_radius = 0.9*global_settings.COMM_RANGE
y_radius = 0.9*global_settings.COMM_RANGE

net = Network(commRange=c_range)  # Initiate the network object
h, w = net.environment.im.shape  # get the network width and height(should be as above)

propagation.PropagationModel.P_TX = energy.EnergyModel.P_TX  # 0.0144  # Node Tx Power
# The distance below which signal is received without any interference
propagation.PropagationModel.MAX_DISTANCE_NO_LOSS = 2 # in m
# The received packet will be assumed lost/corrupted by the receiver if SNR is below this number
propagation.PropagationModel.P_RX_THRESHOLD = -70 # in dbm

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
energys = []
unlocalized = []

# Network Topology setup
Node.cid = 1  # start node id
doi=global_settings.DOI
#net_gen = NetworkGenerator(n_count=n, n_min=n, n_max=n, degree=5)
#net = net_gen.generate_homogeneous_network(name='Sparse Random')

net_gen = Toplogy(n_count=n, n_max=n, n_min=n, connected=False, doi=global_settings.DOI)

net, p = net_gen.generate_cluster_network(name="8",
         x_radius=x_radius, y_radius=y_radius,  sector=1,
         clusters=clusters,  randomness=2.8, method="EM",
         cut_shape=[[(w/4,3*h/4), (3*w/4,7*h/12)],
                   [(w/4,5*h/12), (3*w/4, h/4)]])
#net = net_gen.generate_gaussian_network(clusters=clusters, randomness=0.5, method='EM')
#net = net_gen.generate_cluster_network(name=None,
#        x_radius=x_radius, y_radius=y_radius,  sector=1.0,
#        clusters=clusters,  randomness=3.8, method=None)
# cut_shape is a rectangle with top-left and bottom-right coordinates
#net = net_gen.generate_grid_network(name="O-shaped Grid", randomness=0.1,
#                                    p_anchors=p_anchors,
#                                    cut_shape=[[(w/4,3*h/4), (3*w/4,h/4)]])
#net = net_gen.generate_grid_network(name="Random", randomness=0.5)
#net = net_gen.generate_grid_network(name="C-shaped Grid", randomness=0.2, p_anchors=p_anchors,
#                                    cut_shape=[[(w/4,3*h/4), (w, h/4)]])
#net = net_gen.generate_grid_network(name="S-shaped Grid", randomness=0.1, p_anchors=p_anchors,
#                                    cut_shape=[[(w/4,3*h/4), (w,7*h/12)], [(0,5*h/12), (3*w/4, h/4)]])
#net = net_gen.generate_grid_network(name="W-shaped Grid", randomness=0.1,
#                                    cut_shape=[[(w/4,h), (5*w/12,h/3)], [(7*w/12,h), (3*w/4, h/3)]])
#net = net_gen.generate_grid_network(name="H-shaped Grid", randomness=0.1,
#                                     cut_shape=[[(w/4,h), (3*w/4,2*h/3)], [(w/4,h/3), (3*w/4, 0)]])
#net = net_gen.generate_grid_network(name="I-shaped Grid", randomness=0.1,
#                                    cut_shape=[[(0,5*h/6), (w/3,h/6)], [(2*w/3,5*h/6), (w, h/6)]])
#net = net_gen.generate_grid_network(name="8-shaped Grid", randomness=0.1,
#                                    cut_shape=[[(w/4,3*h/4), (3*w/4,7*h/12)], [(w/4,5*h/12), (3*w/4, h/4)]])
# Computes no. of anchors
n_anchors = net_gen.anchors
# Set some nodes as anchor based on number of desired anchors
# Two arrays are populated with location of nodes to be plotted later
for node in net.nodes():
    xpositions.append(net.pos[node][0])
    if (node.type=='C'):  # anchor nodes
        anchpositions.append({'x': net.pos[node][0], 'y': net.pos[node][1],
                              'name': str(node.id), 'color': 'red',
                              'marker': {'symbol': 'circle', 'radius': '8'}})
    else:
        positions.append({'x': net.pos[node][0], 'y': net.pos[node][1],
                          'name': 'Node: ' + str(node.id), 'color': 'blue',
                          'marker': {'radius': '5'}})


nn = net.__len__()
avg_deg = round(net.avg_degree())
comm_range = node.commRange

folder = DATETIME_DIR+ "-" + net_gen.name+"-hop=" + str(maxHop)
if 'DV-maxHop' not in method:
        maxHop=None
        folder = DATETIME_DIR+ "-" + net_gen.name+"-" + method

# set the network name based on no. of Nodes, degree and Comm. Range
net.name = "%s $N=$%s(%s), $D=$%s, $R=$%s m\n" \
               "$A=$%s$\\times 10^3m^2$, $N_D=$%s$/10^3.m^2$, $DOI$=%s" \
               % (net_gen.name, nn, n_anchors, round(avg_deg,1),
                  round(comm_range,1), round(net_gen.area/1000.0, 2),
                  round(net_gen.net_density*1000, 1), round(doi, 1))

filename = (net.name.split("\n")[0]).replace("$","")

area = "A: %s x 1000 m^2, ND: %s /1000 m^2, DOI: %s" \
           %(round(net_gen.area/1000.0, 2), round(net_gen.net_density*1000, 1),
             round(doi, 1))
net.savefig(fname=get_path(folder, filename),   title=net.name,
            xlabel="X-coordinate (m)", ylabel="Y-coordinate (m)",
            show_labels=True, format="pdf")


# Now select the algorithm for simulation
net.algorithms = ((algorithm[method][0], {'truePositionKey': 'tp',
                                  'hopsizeKey': 'hs',
                                  'dataKey': 'I',
                                   'maxHop': maxHop
                                  }),
                   (algorithm[method][1], {'truePositionKey': 'tp',
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
hops=0
for node in net.nodes():
    total_tx += node.n_transmitted
    total_rx += node.n_received
    err=0
    message_stats.append([node.id, node.n_transmitted,
                              node.n_transmitted_failed_power,
                              node.n_received,
                              node.n_received_failed_power,
                              node.n_received_failed_loss, np.mean(node.snr)
                              ])
    consume.append(round(node.power.energy_consumption * 1000.0, 2))
    energys.append(node.power.energy * 1000.0)
    total_energy += node.power.energy_consumption

    if node.type == 'N' and 'pos' in node.memory:
        act = node.get_dic()['Info']['position']
        est = node.memory['pos']
        newx = est[0]
        newy = est[1]
        err = sqrt(sum(pow(act - est, 2)))
        if 'reliable' in node.memory:
            hops = len(node.memory['reliable'])
            #print node.id, hops , node.memory['reliable']

        position_stats.append([node.id, act[0], act[1], newx, newy, err, hops])
        esterror.append(err)
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
        unlocalized.append([node.id, node.memory.get('I')])

    xestpositions.append(newx)
    deltapos.append(net.pos[node][0] - newx)


X2 = np.sort(esterror)
F2 = 1. * np.arange(len(esterror))/(len(esterror)+1)

# Summary of simulation result/Metrics
comments = "Runtime(s): "+ str(round(end_time,2)) + "(" + \
               str(round(end_time/nn,3)) + ")"+  \
               ", Nodes Localized: " + str(k) + "/" + str(len(unlocalized)) + \
               ", Avg. error: " + str(round(err_sum/k, 2)) + ", DOI: " + str(round(doi, 1)) + \
               "<br>Tx/Rx per node: " + str(int(total_tx/nn)) + "/" + \
                                           str(int(total_rx/nn)) + \
               ",  Energy/node (mJ): " + str(round(1000*total_energy/nn, 2)) + \
               ", |Err>R|: " + str(len(X2[X2>comm_range]))
if maxHop:
   comments += ", MaxHop: " + str(maxHop)
else:
    comments += ", " + method

print (comments)
stats = "Simulation Start at: " + getDateStr(sim.sim_start) + " UTC <br><br>"

position_stats.append(np.round(np.mean(position_stats, axis=0),2))
position_stats.append(np.round(np.min(position_stats, axis=0),2))
position_stats.append(np.round(np.max(position_stats, axis=0),2))
position_stats.append(np.round(np.std(position_stats, axis=0),2))

stats += "Localization: [xr, yr, xl, yl, error, hops]" \
         "<br>Avg: " + str(position_stats[-4][1:]) + \
        "<br>Min: " + str(position_stats[-3][1:]) + \
        "<br>Max: " + str(position_stats[-2][1:]) + \
        "<br>Std: " + str(position_stats[-1][1:]) + "<br>"

message_stats.append(np.round(np.mean(message_stats, axis=0),2))
message_stats.append(np.round(np.min(message_stats, axis=0),2))
message_stats.append(np.round(np.max(message_stats, axis=0),2))
message_stats.append(np.round(np.std(message_stats, axis=0),2))

stats += "<br>Messages: [Tx, Tx-, Rx, Rx-, Rx-, SNR]" \
         "<br>Avg: " + str(np.array(message_stats[-4][1:]).tolist()) + \
        "<br>Min: " + str(np.array(message_stats[-3][1:]).tolist()) + \
        "<br>Max: " + str(np.array(message_stats[-2][1:]).tolist()) + \
        "<br>Std: " + str(np.array(message_stats[-1][1:].tolist())) + "<br>"

stats += "<br>Unlocalized Nodes: " + str(unlocalized) + "<br>"

consume.append(np.mean(consume, axis=0))
consume.append(np.min(consume, axis=0))
consume.append(np.max(consume, axis=0))
consume.append(np.std(consume, axis=0))
consume.append(np.var(consume, axis=0))

energys.append(np.mean(energys, axis=0))
energys.append(np.min(energys, axis=0))
energys.append(np.max(energys, axis=0))
energys.append(np.std(energys, axis=0))
energys.append(np.var(energys, axis=0))

nnd = range(1, nn+1)
nnd.append('mean')
nnd.append('min')
nnd.append('max')
nnd.append('std')
nnd.append('var')

stats += "<br>Simulation Finish at: " + getDateStr(sim.sim_end) + " UTC<br>"

#print nnd

sim.reset()

#Create CSV, plots and interactive html/JS charts based on collected data
np.savetxt(get_path(folder, "Msg_stats-%s.csv" % filename),
           message_stats,
           delimiter=",", fmt="%8.2f", comments='',
           header="Node,transmit,transmit_failed_power,"
                  "received,received_failed_power,"
                  "received_failed_loss,SNR",
           footer=net.name + "\n" + comments)

sp.savetxt(get_path(folder, "Localization-%s.csv" % filename),
           position_stats,
           delimiter=",", fmt="%8.2f", comments='',
           header="Node,Xactual,Yactual,Xestimated,Yestimated,RMSerror, Anchors",
           footer=net.name + "\n" + comments)


# Create html/JS file for network Topology
plotter.gethtmlScatter(xpositions, [anchpositions, positions, newpos],
                fname=filename, folder=folder,
                xlabel="X-coordinates", ylabel="Y-coordinates", labels=['Anchor','Regular','Localized'],
                title="Topology-"+filename, open=1, axis_range={'xmin':0, 'ymin':0, 'xmax': w, 'ymax': h},
                comment=comments, show_range=int(node.commRange),
                report=area+"<br>"+comments+"<br><br>" + str(stats),
                plot_options=["color: 'red', visible: true,", "color: 'blue',",
                              "color: 'green', visible: true,"])

plotter.gethtmlLine(range(1,len(xpositions)), [esterror, [row[6] for row in position_stats]],
                fname="Error-"+filename, folder=folder,
                xlabel="Node", ylabel="meters", labels=['Estimated Errors', 'Good Anchors'],
                title="Error-"+filename, comment=comments, open=1,
                plot_options=["color: 'red',", "type: 'bar', color: 'grey', visible: false,"])

# plotter.gethtmlLine(range(1,len(xpositions)), [consume, [row[1] for row in message_stats]],
#                 fname="Power-"+filename, folder=folder,
#                 xlabel="Node", ylabel="mJ", labels=['Energy','No. of Tx'],
#                 open=1, comment=comments, plot_type='line',
#                 plot_options=["color: 'red',", "color: 'blue',", "type: 'areaspline', color: 'grey', visible: false,"])
#
#
try:
#
#     sp.savetxt(get_path(folder, "Energy-%s.csv" %filename),
#                list(zip(nnd, energys, consume)),
#                delimiter="\t", fmt="%s",
#                header="Nodes\tEnergy Left (mJ)\tConsumed", comments='')
#
      #X2 = np.sort([col[3]/comm_range for col in position_stats])

#     H, X1 = np.histogram(esterror,  normed=True)
#     dx = X1[1] - X1[0]
#     F1 = np.cumsum(H)*dx
#     #print X2, F2 #np.column_stack
      lis  = np.array([X2/comm_range, F2]).tolist()
#     #print map(list, zip(*lis))
#
#
#     plotter.plots(X1[1:]/comm_range,F1,get_path(folder, "Error CDF-%d" % nn),
#                   title="CDF-"+net.name, xlabel="Normalized Localization error",
#                   ylabel="CDF")
#
#     plotter.plots(X2/comm_range,F2,get_path(folder, "Error CDF2-%d" % nn),
#                   title="CDF-"+net.name, xlabel="Normalized Localization error",
#                   ylabel="CDF")
#
      plotter.gethtmlmultiplots(X2,  map(list, zip(*lis)),
                 fname="CDF-"+filename, folder=folder, plot_type='spline',
                 xlabel="Normalized Localization Error (R)", ylabel="CDF", labels=['CDF','Sort'],
                 open=1, comment=comments, axis_range={'xmin':0, 'ymin':0, 'ymax': 1},
                 plot_options=["color: 'red', lineWidth:1, marker: {enabled: false},", "color: 'blue',"])

except Exception as exc:
     print exc
     pass
