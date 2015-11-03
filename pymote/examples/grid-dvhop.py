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
#from pymote.algorithms.shazad2015.dvhop import DVHop
#from pymote.algorithms.shazad2015.trilaterate import Trilaterate

'''Start of Script'''

# get the pymote version and print it
meta = load_metadata()
print meta['name'], meta['version']

seed(100)  # to get same random sequence for each run so that simulation can be reproduce

# Network/Environment setup
global_settings.ENVIRONMENT2D_SHAPE = (100, 100) # desired network size for simulation
global_settings.COMM_RANGE = 10

method="DV-hop"
n = 289  # total no of nodes
p_anchors = 10  # No. of anchors in %age
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


# Network Topology setup
Node.cid = 1  # start node id


# cut_shape is a rectangle with top-left and bottom-right coordinates
#net = net_gen.generate_grid_network(name="O-shaped Grid", randomness=0.2,
#                                    cut_shape=[[(w/4,3*h/4), (3*w/4,h/4)]])
#net = net_gen.generate_grid_network(name="Randomized Grid", randomness=0.5)
#net = net_gen.generate_grid_network(name="C-shaped Grid", randomness=0.2,
#                                    cut_shape=[[(w/4,3*h/4), (w, h/4)]])
#net = net_gen.generate_grid_network(name="S-shaped Grid", randomness=0.1,
#                                    cut_shape=[[(w/4,3*h/4), (w,7*h/12)], [(0,5*h/12), (3*w/4, h/4)]])
#net = net_gen.generate_grid_network(name="W-shaped Grid", randomness=0.1,
#                                    cut_shape=[[(w/4,h), (5*w/12,h/3)], [(7*w/12,h), (3*w/4, h/3)]])

#net = net_gen.generate_grid_network(name="I-shaped Grid", randomness=0.2,
#                                    cut_shape=[[(0,5*h/6), (w/3,h/6)], [(2*w/3,5*h/6), (w, h/6)]])
#net = net_gen.generate_grid_network(name="8-shaped Grid", randomness=0.1,
#                                    cut_shape=[[(w/4,3*h/4), (3*w/4,7*h/12)], [(w/4,5*h/12), (3*w/4, h/4)]])
#net = net_gen.generate_grid_network(name="T-shaped Grid", randomness=0.2,
#                                    cut_shape=[[(0,2*h/3), (w/3,0)], [(2*w/3,2*h/3), (w, 0)]])
#net = net_gen.generate_grid_network(name="H-shaped Grid", randomness=0.1,
#                                    cut_shape=[[(w/4,h), (3*w/4,2*h/3)], [(w/4,h/3), (3*w/4, 0)]])
#net_gen = NetworkGenerator(n_count=n)
#net = net_gen.generate_homogeneous_network(name='Sparse Random')
sr = 0
stats=''
loc_err = []
vary_name = "Degree"
experiment = "Effect of Connectivity"


for vary in range(5,25,5):
    net_gen = Toplogy(n_count=n, degree=vary, maxn=n, n_min=n, connected=False)
    net = net_gen.generate_grid_network(name="Randomized Grid", randomness=0.5)
    #net = net_gen.generate_grid_network(name="O-shaped Grid", randomness=0.2,
    #                             cut_shape=[[(w/4,3*h/4), (3*w/4,h/4)]])
    nn = net.__len__()
    folder = DATETIME_DIR+ "-" + net_gen.name+"-N=" + str(nn) + "-" + vary_name
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

    # Computes no. of anchors
    f_anchors = (int)(100 / p_anchors)
    n_anchors = 0

    # Set some nodes as anchor based on number of desired anchors
    # Two arrays are populated with location of nodes to be plotted later
    for node in net.nodes():
        xpositions.append(net.pos[node][0])
        if (node.id % f_anchors==0):  # anchor nodes
            n_anchors += 1
            node.compositeSensor = (TruePosSensor,)
            node.type = 'C'  # Anchors
            anchpositions.append({'x': net.pos[node][0], 'y': net.pos[node][1],
                                  'name': str(node.id), 'color': 'red',
                                  'marker': {'symbol': 'circle', 'radius': '8'}})
        else:
            node.type = 'N'  # Normal
            node.compositeSensor = ('NeighborsSensor',)
            positions.append({'x': net.pos[node][0], 'y': net.pos[node][1],
                              'name': 'Node: ' + str(node.id), 'color': 'blue',
                              'marker': {'radius': '5'}})


    avg_deg = round(net.avg_degree())
    comm_range = node.commRange
    # set the network name based on no. of Nodes, degree and Comm. Range
    net.name = "%s(%s) - $N=$%s(%s), $D=$%s, $R=$%s m\n$A=$%s$\\times 10^3m^2$, $ND=$%s$/10^3.m^2$" \
               % (net_gen.name, vary, nn, n_anchors, round(avg_deg,1), round(comm_range,1),
                  round(net_gen.area/1000.0, 2), round(net_gen.net_density*1000, 1))

    filename = (net.name.split("\n")[0]).replace("$","")

    area =  "A: %s x 1000 m^2, ND: %s /1000 m^2" %(round(net_gen.area/1000.0, 2), round(net_gen.net_density*1000, 1))

    net.savefig(fname=get_path(folder, filename),   title=net.name,
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
            position_stats.append([node.id, act[0], act[1], newx, newy, err])
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
        esterror.append(err)
        deltapos.append(net.pos[node][0] - newx)

    # Summary of simulation result/Metrics
    comments = "Runtime(s): "+ str(round(end_time,2)) + \
               ", Nodes Localized: " + str(k) + "/" + str(len(unlocalized)) + \
               ", Avg. error: " + str(round(err_sum/k, 2)) + \
               "<br>Tx/Rx per node: " + str(int(total_tx/nn)) + "/" + \
                                           str(int(total_rx/nn)) + \
               ",  Energy/node (mJ): " + str(round(1000*total_energy/nn, 2))

    print (comments)
    stats = "Simulation Start at: " + getDateStr(sim.sim_start) + " UTC <br><br>"

    position_stats.append(np.round(np.mean(position_stats, axis=0),2))
    position_stats.append(np.round(np.min(position_stats, axis=0),2))
    position_stats.append(np.round(np.max(position_stats, axis=0),2))
    position_stats.append(np.round(np.std(position_stats, axis=0),2))

    stats += "Localization: [xr, yr, xl, yl, error]" \
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

    loc_err.append([vary, degree, k, len(unlocalized), round(end_time/nn,3),
                    position_stats[-4][5], position_stats[-2][5], position_stats[-1][5],
                    message_stats[-4][1],message_stats[-4][3],message_stats[-4][6], consume[-5]])

    stats += "<br>Simulation Finish at: " + getDateStr(sim.sim_end) + " UTC<br>"

    #print nnd

    sim.reset()
    net.reset()
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


    # Create html/JS file for network Topology
    plotter.gethtmlScatter(xpositions, [anchpositions, positions, newpos],
                    fname=filename, folder=folder,
                    xlabel="X-coordinates", ylabel="Y-coordinates", labels=['Anchor','Regular','Localized'],
                    title="Topology-"+filename, open=1, axis_range={'xmin':0, 'ymin':0, 'xmax': w, 'ymax': h},
                    comment=comments, show_range=int(node.commRange),
                    report=area+"<br>"+comments+"<br><br>" + str(stats),
                    plot_options=["color: 'red', visible: false,", "color: 'blue',",
                                  "color: 'green', visible: true,"])

    plotter.gethtmlLine(range(1,len(xpositions)), [xpositions, xestpositions, deltapos, esterror],
                    fname="X-"+filename, folder=folder,
                    xlabel="Node", ylabel="meters", labels=['Actual', 'Estimated', 'X-Error', 'Est. Error'],
                    title="X-"+filename, open=1,
                    comment=comments,
                    plot_options=["color: 'red',", "color: 'blue',", "type: 'areaspline', color: 'grey', visible: false,"])

    sr +=1

print loc_err
sp.savetxt(get_path(folder, "%s-%s.csv" % (experiment, filename)),
               loc_err,
               delimiter=",", fmt="%8.2f", comments='',
               header='     Anc%    Anchors Localized   Not    runtime  avg_err  max_err  std_err  TX      Rx       SNR     Energy')
