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
from pymote.utils.filing import get_path, load_metadata,\
     DATA_DIR, TOPOLOGY_DIR, CHART_DIR, DATETIME_DIR

from pymote.simulation import Simulation
from pymote.sensor import TruePosSensor
from pymote.networkgenerator import NetworkGenerator
from pymote.algorithms.shazad2015.dvhop import DVHop
from pymote.algorithms.shazad2015.trilaterate import Trilaterate

'''Start of Script'''

# get the pymote version and print it
meta = load_metadata()
print meta['name'], meta['version']

#seed(123)  # to get same random sequence for each run so that simulation can be reproduce

# Network/Environment setup
global_settings.ENVIRONMENT2D_SHAPE = (100, 100) # desired network size for simulation
global_settings.COMM_RANGE = 10

n = 289  # total no of nodes
p_anchors = 11  # No. of anchors in %age
c_range = 10  # communication radii of each node
degree = 10   # Desired degree or connectivity (how many nodes are in range)
net = Network(commRange=c_range)  # Initiate the network object
h, w = net.environment.im.shape  # get the network width and height(should be as above)

propagation.PropagationModel.P_TX = energy.EnergyModel.P_TX  # 0.0144  # Node Tx Power
# The distance below which signal is receibved without any interference
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
position_stats2 = []
consume = []
energys = []

# Network Topology setup
Node.cid = 1  # start node id

net_gen = Toplogy(n_count=n, n_max=n, n_min=n, connected=False)
# cut_shape is a rectangle with top-left and bottom-right coordinates
net = net_gen.generate_grid_network(name="S-shaped Grid", randomness=0.1,
                                    cut_shape=[[(w/4,3*h/4), (w,7*h/12)], [(0,5*h/12), (3*w/4, h/4)]])

# select some anchors
anchors = [10, 29, 42, 84, 117, 69, 104]
# Computes no. of anchors
n_anchors = len(anchors)

# Set some nodes as anchor based on number of desired anchors
f_anchors = (int)(100 / p_anchors)
# Two arrays are populated with location of nodes to be plotted later
for region in range(4):
    for node in net.nodes():
        x = net.pos[node][0]
        y = net.pos[node][1]
        xpositions.append(x)
        if node.id % f_anchors == 0 and x<w/2 and y>h/2:  # anchor nodes
            node.compositeSensor = (TruePosSensor,)
            node.type = 'C'  # Anchors
            anchpositions.append({'x': x, 'y': y,
                                  'name': str(node.id), 'color': 'red',
                                  'marker': {'symbol': 'circle', 'radius': '8'}})
        else:
            positions.append({'x': x, 'y': y,
                              'name': str(node.id), 'color': 'blue',
                              'marker': {'radius': '5'}})


    nn = net.__len__()
    n_anchors = (int)(nn *  p_anchors/100.0)
    avg_deg = round(net.avg_degree())
    comm_range = node.commRange
    # set the network name based on no. of Nodes, degree and Comm. Range
    net.name = "%s - $N=$%s(%s), $D=$%s, $R=$%s m\n$A=$%s$\\times 10^3m^2$, $ND=$%s$/10^3.m^2$" \
               % (net_gen.name, nn, n_anchors, round(avg_deg,1), round(comm_range,1),
                  round(net_gen.area/1000.0, 2), round(net_gen.net_density*1000, 1))
    filename = (net.name.split("\n")[0]).replace("$","")
    folder = DATETIME_DIR+ "-" + net_gen.name
    net.savefig(fname=get_path(folder, filename),   title=net.name,
                x_label="X-coordinate (m)", y_label="Y-coordinate (m)", show_labels=True, format="pdf")


    # Now select the algorithm for simulation
    net.algorithms = ((DVHop, {'truePositionKey': 'tp',
                                      'hopsizeKey': 'hs',
                                      'dataKey': 'I'
                                      }),
                       (Trilaterate, {'truePositionKey': 'tp',
                                            'hopsizeKey': 'hs',
                                            'positionKey': 'pos'+region,
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
    k2=0
    err_sum2=0
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
            est = node.memory['pos'+region]
            hs = node.memory['hs']
            newx = est[0]
            newy = est[1]
            err = sqrt(sum(pow(act - est, 2)))
            position_stats.append([node.id, act[0], act[1], newx, newy, err, hs])
            if act[0]<50 and act[1]>50:
                position_stats2.append([node.id, act[0], act[1], newx, newy, err, hs])
                k2 += 1
                err_sum2 += err
            err_sum += err
            k += 1
            newpos.append({'x': newx, 'y': newy,
                      'name': str(node.id)})
        elif node.type == 'C':
            newx = net.pos[node][0]
            newy = net.pos[node][1]
        else:
            newx = -1
            newy = -1
        xestpositions.append(newx)
        esterror.append(err)
        deltapos.append(net.pos[node][0] - newx)

    print node.memory

    # Summary of simulation result/Metrics
    comments = "Nodes Localized: " + str(k) + \
               ",  Runtime(sec): "+ str(round(end_time,2)) + \
               ",  Tx/Rx per node: " + str(round(total_tx/nn)) + "/" + \
                                           str(round(total_rx/nn)) + \
               "<br>Energy/node (mJ): " + str(round(1000*total_energy/nn, 3)) + \
               ",  Avg. error: " + str(round(err_sum/k, 2))

    comments2 = "Nodes Localized: " + str(k2) + \
        ",  Avg. error: " + str(round(err_sum2/k2, 2))
    print (comments)

    print (comments2)
    stats2 = "Localization:<br>Avg: " + str(np.mean(position_stats2, axis=0)[5]) + \
            ", Min: " + str(np.min(position_stats2, axis=0)[5]) + \
            ", Max: " + str(np.max(position_stats2, axis=0)[5]) + \
            ", Std: " + str(np.std(position_stats2, axis=0)[5]) + "<br>"
    print stats2

    stats = "Simulation Start at: " + str(sim.sim_start) + "UTC <br><br>"

    position_stats.append(np.round(np.mean(position_stats, axis=0),2))
    position_stats.append(np.round(np.min(position_stats, axis=0),2))
    position_stats.append(np.round(np.max(position_stats, axis=0),2))
    position_stats.append(np.round(np.std(position_stats, axis=0),2))
    stats += "Localization:<br>Avg: " + str(position_stats[-4][1:]) + \
            "<br>Min: " + str(position_stats[-3][1:]) + \
            "<br>Max: " + str(position_stats[-2][1:]) + \
            "<br>Std: " + str(position_stats[-1][1:]) + "<br>"

    message_stats.append(np.round(np.mean(message_stats, axis=0),2))
    message_stats.append(np.round(np.min(message_stats, axis=0),2))
    message_stats.append(np.round(np.max(message_stats, axis=0),2))
    message_stats.append(np.round(np.std(message_stats, axis=0),2))

    stats += "<br>Messages:<br>Avg: " + str(message_stats[-4][1:]) + \
            "<br>Min: " + str(message_stats[-3][1:]) + \
            "<br>Max: " + str(message_stats[-2][1:]) + \
            "<br>Std: " + str(message_stats[-1][1:]) + "<br>"

    stats += "<br>Simulation Finish at: " + str(sim.sim_end) + "UTC<br>"

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


    #print nnd

    sim.reset()

    #Create CSV, plots and interactive html/JS charts based on collected data
    np.savetxt(get_path(folder, "Msg_stats-%s.csv" % filename),
               message_stats,
               delimiter=",", fmt="%8.2f", comments='',
               header="Node,transmit,transmit_failed_power,"
                      "received,received_failed_power,"
                      "received_failed_loss,SNR", footer=net.name + "\n" + comments)

    sp.savetxt(get_path(folder, "Localization-%s.csv" % filename),
               position_stats,
               delimiter=",", fmt="%8.2f", comments='',
               header="Node,Xactual,Yactual,Xestimated,Yestimated,RMSerror")

    sp.savetxt(get_path(folder, "Localization2-%s.csv" % filename),
               position_stats2,
               delimiter=",", fmt="%8.2f", comments='',
               header="Node,Xactual,Yactual,Xestimated,Yestimated,RMSerror")


    # Create html/JS file for network Topology
    plotter.gethtmlScatter(xpositions, [anchpositions, positions, newpos],
                    fname=filename, folder=folder,
                    xlabel="X-coordinates", ylabel="Y-ccordinates", labels=['Anchor','Regular','Localized'],
                    title="Topology-"+filename, open=1, axis_range={'xmin':0, 'ymin':0, 'xmax': w, 'ymax': h},
                    comment=comments, show_range=int(node.commRange),report=comments+"<br><br>" + str(stats),
                    plot_options=["color: 'red', visible: false,", "color: 'blue',",
                                  "color: 'green', visible: true,"])

net.reset()

# plotter.gethtmlLine(range(1,len(xpositions)), [xpositions, xestpositions, deltapos, esterror],
#                 fname="X-"+filename, folder=folder,
#                 xlabel="Node", ylabel="meters", labels=['Actual', 'Estimated', 'X-Error', 'Est. Error'],
#                 title="X-"+filename, open=1,
#                 comment=comments,
#                 plot_options=["color: 'red',", "color: 'blue',", "type: 'areaspline', color: 'grey', visible: false,"])
#
# plotter.gethtmlLine(range(1,len(xpositions)), [consume, [row[1] for row in message_stats]],
#                 fname="Power-"+filename, folder=folder,
#                 xlabel="Node", ylabel="mJ", labels=['Energy','No. of Tx'],
#                 open=1, comment=comments, plot_type='line',
#                 plot_options=["color: 'red',", "color: 'blue',", "type: 'areaspline', color: 'grey', visible: false,"])
#
#
# try:
#
#     sp.savetxt(get_path(folder, "Energy-%s.csv" %filename),
#                list(zip(nnd, energys, consume)),
#                delimiter="\t", fmt="%s",
#                header="Nodes\tEnergy Left (mJ)\tConsumed", comments='')
#
#     #X2 = np.sort([col[3]/comm_range for col in position_stats])
#     X2 = np.sort(esterror)
#     F2 = 1. * np.arange(len(esterror))/(len(esterror)+1)
#     H, X1 = np.histogram(esterror,  normed=True)
#     dx = X1[1] - X1[0]
#     F1 = np.cumsum(H)*dx
#     #print X2, F2 #np.column_stack
#     lis  = np.array([X2/comm_range, F2]).tolist()
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
#     plotter.gethtmlScatter(X2,  map(list, zip(*lis)),
#                 fname="CDF-"+filename, folder=folder,
#                 xlabel="Normalized Localization Error (R)", ylabel="CDF", labels=['CDF','Sort'],
#                 open=1, comment=comments,
#                 plot_options=["color: 'red', lineWidth:1, marker: {enabled: false},", "color: 'blue',"])
#
# except Exception as exc:
#     print exc
#     pass
