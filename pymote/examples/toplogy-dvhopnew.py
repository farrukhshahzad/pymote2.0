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
#from pymote.algorithms.niculescu2003.dvhop import DVHop
#from pymote.algorithms.niculescu2003.trilaterate import Trilaterate
from pymote.algorithms.shazad2015.dvhop import DVHop
from pymote.algorithms.shazad2015.trilaterate import Trilaterate

'''Start of Script'''

# get the pymote version and print it
meta = load_metadata()
print meta['name'], meta['version']

seed(123)  # to get same random sequence for each run so that simulation can be reproduce

# Network/Environment setup
global_settings.ENVIRONMENT2D_SHAPE = (100, 100) # desired network size for simulation
global_settings.COMM_RANGE = 10

c_range = 10  # communication radii of each node
#net = Network(commRange=c_range)  # Initiate the network object
#h, w = net.environment.im.shape  # get the network width and height(should be as above)

propagation.PropagationModel.P_TX = energy.EnergyModel.P_TX  # 0.0144  # Node Tx Power
# The distance below which signal is received without any interference
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
degree = 10   # Desired degree or connectivity (how many nodes are in range)
vary_name = "Degree"
experiment = "Effect of Degree"
maxHop = 0
method="DV-hop"
n = 429  # total no of nodes
p_anchors = 11.0  # No. of anchors in %age
doi = 0

for vary in range(5,25,5):
    net = Network(commRange=c_range)  # Initiate the network object
    h, w = net.environment.im.shape  # get the network width and height(should be as above)
    Node.cid = 1  # start node id
    #degree = 9-vary
    #if vary == 6:
    #    degree = 4
    n = vary * 47
    #p_anchors = vary
    #if vary in [5, 15]:
    #     p_anchors = 9
    #else:
    #     p_anchors = 10
    #net_gen = NetworkGenerator(n_count=n, degree=vary)
    #net = net_gen.generate_homogeneous_network(name='Sparse Random')
    net_gen = Toplogy(n_count=n, maxn=n, n_min=n, connected=True, doi=doi)
    #net = net_gen.generate_grid_network(name="Randomized Grid", randomness=0.5)
    net = net_gen.generate_grid_network(name="O-shaped Grid", randomness=0.1,
                                 cut_shape=[[(w/4,3*h/4), (3*w/4,h/4)]])
    #net = net_gen.generate_grid_network(name="C-shaped Grid", randomness=0.2,
    #                                    cut_shape=[[(w/4,3*h/4), (w, h/4)]])
    #net = net_gen.generate_grid_network(name="S-shaped Grid", randomness=0.2,
    #                                    cut_shape=[[(w/4,3*h/4), (w,7*h/12)],
    #                                               [(0,5*h/12), (3*w/4, h/4)]])
    #net = net_gen.generate_grid_network(name="Plus-shaped Grid", randomness=0.3,
    #                                cut_shape=[[(0,h), (w/3, 2*h/3)],[(2*w/3, h), (w, 2*h/3)],
    #                                [(0, h/3), (w/3, 0)], [(2*w/3, h/3), (w, 0)]])
    nn = net.__len__()
    folder = DATETIME_DIR+ "-" + net_gen.name+"-" + vary_name + "-hop=" + str(maxHop)
    if 'DV-hop' in method:
        maxHop=0
        folder = DATETIME_DIR+ "-" + net_gen.name+"-" + vary_name + "-" + method

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
    f_anchors = (int)(100.0 / p_anchors)
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
    area = "A: %s x 1000 m^2, ND: %s /1000 m^2, DOI: %s" \
           %(round(net_gen.area/1000.0, 2), round(net_gen.net_density*1000, 1),
             round(doi, 1))
    net.name = "%s(%s) - $N=$%s(%s), $D=$%s, $R=$%s m\n" \
               "$A=$%s$\\times 10^3m^2$, $ND=$%s$/10^3.m^2$, $DOI$=%s" \
               % (net_gen.name, vary, nn, n_anchors, round(avg_deg,1),
                  round(comm_range,1), round(net_gen.area/1000.0, 2),
                  round(net_gen.net_density*1000, 1), round(doi, 1))

    filename = (net.name.split("\n")[0]).replace("$","")

    net.savefig(fname=get_path(folder, filename),   title=net.name,
                x_label="X-coordinate (m)", y_label="Y-coordinate (m)",
                show_labels=True, format="pdf")
    net.reset()
