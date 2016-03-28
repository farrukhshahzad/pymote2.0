
# built-in packages
import time
import datetime

# external packages
import scipy as sp
import numpy as np
from em import expectation_maximization
from numpy import sign, sqrt, array, pi, sin, cos
from numpy.random import rand,seed
import networkx as nx
from networkx.algorithms import approximation as approx

# internal packages
from pymote import *
from pymote.conf import global_settings
from pymote.sensor import TruePosSensor
from pymote.sensor import NeighborsSensor
from beacon import Beacon, MAX_TRIES
from toplogies import Topology
from pymote import propagation, energy
from pymote.utils import plotter
from pymote.utils.filing import get_path, date2str,\
     DATA_DIR, TOPOLOGY_DIR, CHART_DIR, DATETIME_DIR


# Network/Environment setup
seed(123)  # to get same random sequence for each run so that simulation can be reproduce

# Network/Environment setup
global_settings.ENVIRONMENT2D_SHAPE = (500, 500) # desired network size for simulation
global_settings.COMM_RANGE = 75
net = Network()
h, w = net.environment.im.shape
energy.EnergyModel.P_TX = 0.0144
energy.EnergyModel.P_RX = 0.014
propagation.PropagationModel.P_TX = energy.EnergyModel.P_TX  # 0.084  # Node Tx Power
propagation.PropagationModel.P_RX_THRESHOLD = -80 # in dbm

max_n = 121  # max no. of nodes
n = 49
clusters = 1

n_power = []
c_power = []
overall_energy = 0
total_tx = total_rx = total_loss = 0
n_range = np.arange(3, int(sqrt(max_n))+1, 2)
start_time = time.time()
project_name = "IWSN - ISA100.11a"
folder = DATETIME_DIR+ "-" + project_name
c_range = [150, 100, 75, 60, 47]
k=0
for nn in n_range:
    # network topology setup
    n = nn * nn
    global_settings.COMM_RANGE = c_range[k]

    k += 1
    Node.cid = 1
    net_gen = Topology(n_count=n, connected=False,
                       commRange=global_settings.COMM_RANGE, comm_range=global_settings.COMM_RANGE)
    net = net_gen.generate_grid_network(name="Grid", randomness=0, p_anchors=None)
    net, p = net_gen.get_clusters(net, clusters=clusters, method="EM")
    node = net.nodes()[int(n/2)]
    net.remove_node(node)

    n = len(net)
    net.name = "Grid Topology" + "\nN=%s, R=%s m" \
                                     %(n, global_settings.COMM_RANGE)
    filename = net.name.replace("\n", "-")
    # saving topology as PNG image
    net.savefig(fname=get_path(folder, filename),   title=net.name,
                xlabel="X-coordinate (m)", ylabel="Y-coordinate (m)",
                show_labels=True, format="png", label_color='yellow')
    net.name = filename
    net.save_json(get_path(folder, filename+".json"), scale=(1, 1))

    print net.__len__(), len(net)

    coordinator_node = net.nodes()[n-1]
    coordinator_node.type = 'B'
    coordinator_node.compositeSensor = (NeighborsSensor,)   # (TruePosSensor,)

    # simulation start
    net.algorithms = ((Beacon, {'informationKey': 'Data', 'detination': coordinator_node}),)
    coordinator_node.memory['Data'] = 'Beacon 1'

    sim = Simulation(net)
    sim.run()
    # simulation ends

    # capturing results
    consume = []
    energy = []
    message_stats = []
    total_energy = 0
    t_transmit = 0
    t_received_failed = 0
    t_received = 0
    expected = MAX_TRIES + 2
    snr_base = []
    xpositions = []
    positions = []
    anchpositions = []
    loss = []

    for nd in net.nodes():
            print nd.id, "Registrations: %s" % nd.memory.get("Registration", []), \
                "\n\rData: %s" % nd.memory, \
                "\n\rEnergy: %s" % nd.power.energy_consumption
            xpositions.append(net.pos[nd][0])
            if (nd.type=='B'):  # base nodes
                anchpositions.append({'x': net.pos[nd][0], 'y': net.pos[nd][1],
                                  'name': str(nd.id), 'color': 'red',
                                  'marker': {'symbol': 'circle', 'radius': '8'}})
            else:
                positions.append({'x': net.pos[nd][0], 'y': net.pos[nd][1],
                              'name': 'Node: ' + str(nd.id), 'color': 'blue',
                              'marker': {'radius': '5'}})
            if nd.id == n-1:
                snr_base = nd.snr
            t_transmit += nd.n_transmitted
            t_received += nd.n_received

            t_received_failed += nd.n_received_failed_loss

            message_stats.append((nd.id, nd.n_transmitted,
                                  nd.n_transmitted_failed_power,
                                  nd.n_received,
                                  nd.n_received_failed_power,
                                  nd.n_received_failed_loss
                                  ))
            consume.append(nd.power.energy_consumption * 1000.0)
            energy.append(nd.power.energy * 1000.0)
            loss.append(nd.n_received_failed_loss)
            total_energy += nd.power.energy_consumption * 1000.0

    c_power.append(consume[n-1])  # coordinator

    n_power.append((n, consume[n-1], total_energy,  # sum of other node consumption
                    coordinator_node.n_received,
                    coordinator_node.n_received_failed_loss))

    sp.savetxt(get_path(folder, "energy-%s.csv" %n),
               list(zip(range(1, n+1), energy)),
               delimiter="\t", fmt="%s",
               header="Nodes\tEnergy (mJ)", comments='')


    sp.savetxt(get_path(folder, "stats-%s.csv" %n),
               message_stats,
               delimiter=",", fmt="%8s", comments='',
               header="Node,transmit,transmit_failed_power,"
                      "received,received_failed_power,"
                      "received_failed_loss")


    # plotting results
    #print len(snr_base)
    lqi_base = 6 * np.array(snr_base) + 538
    #print lqi_base
    plotter.plots(np.arange(len(snr_base[1:])), snr_base[1:],
                  get_path(folder, "SNR_Base_Station-%s" % n),
                  title=net.name + "\nSNR levels (Base Station)",format='png',
                  xlabel="Packet Number", ylabel="SNR")

#    plotter.plots(np.arange(len(snr_base[1:])), lqi_base[1:],
#                  get_path(folder, "LQI_Base_Station-%s" % n),
#                  title="LQI's", format='png',
#                  xlabel="Packet Number", ylabel="LQI")

    plotter.plot_bars(np.arange(1, n+1), consume,
                      get_path(folder, "Energy Consumption-", prefix=n),
                      ymax=1.1*max(consume),
                      ymin=0.9*min(consume),
                      title=net.name + "\nEnergy Consumption (%s nodes)" % n,
                      xlabel="Nodes", ylabel="mJ", format='png')

    plotter.plot_bars(np.arange(1, n+1), loss,
                      get_path(folder, "Loss Packets-", prefix=n),
                      ymax=1.1*max(loss),
                      ymin=0.9*min(loss),
                      title= net.name + "\nLoss Packets (%s nodes)" % n,
                      xlabel="Nodes", ylabel="No.", format='png')

    net.reset()
    sim.reset()
    overall_energy += total_energy
    total_tx += t_transmit
    total_rx += t_received
    total_loss += t_received_failed
    #energy.append({'x': n, 'y': consume[n-1] ,'name': 'Energy Consumption', 'color': 'red'})

end_time = time.time() - start_time

comments = "Runtime (sec): " + str(round(end_time, 2)) + \
           ", Total Energy (mJ): " + str(round(total_energy, 1)) + \
           "<br>Tx & Rx: " + str(int(t_transmit)) + " & " + str(int(t_received)) + \
           ", Total Packet Loss: " + str(int(t_received_failed))

print comments

# Create html/JS file for network Topology
plotter.gethtmlScatter(xpositions, [anchpositions, positions],
                fname=filename, folder=folder,
                xlabel="X-coordinates", ylabel="Y-coordinates", labels=['Base Station','Sensor'],
                title="Topology-"+filename, axis_range={'xmin':0, 'ymin':0, 'xmax': w, 'ymax': h},
                comment=comments,
                show_range=int(global_settings.COMM_RANGE),
                plot_options=["color: 'red', visible: true,", "color: 'blue',"])

plotter.gethtmlLine([9, 25, 49, 81, 121],
                [c_power, [e[3] for e in n_power],[e[4] for e in n_power]],
                fname="Stat-"+filename, folder=folder,
                xlabel="Nodes", ylabel="",
                comment=comments,
                labels=["Base Station (mJ)","Received Packets", "Lost Packets"],
                title="Energy Consumption & Received Packet Stat",
                plot_options=["color: 'red',"])

#simulation level

plotter.plots([9, 25, 49, 81, 121], c_power,  #[e[0] for e in n_power], [e[1] for e in n_power],
              get_path(folder, "energy-Base Station"),
              more_plots=[
                          [e[3] for e in n_power],
                          [e[4] for e in n_power]],
              labels=["Base Station (mJ)",
                      "Received Packets", "Lost Packets"],
              title=project_name+"\nEnergy Consumption & Received Packet Stat",
              xlabel="Nodes", ylabel="", format='png')

# save data to text file for further analysis
sp.savetxt(get_path(folder, "energy-Base Station.csv"),
           n_power,
           delimiter="\t", fmt="%7.0f",
           header="Nodes\tEnergy(mJ)\tTotal(mJ)\tReceived\tLost",
           comments='')

