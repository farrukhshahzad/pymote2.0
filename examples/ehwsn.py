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
from numpy import sqrt, array

# Network/Environment setup
global_settings.ENVIRONMENT2D_SHAPE = (600, 600)
net = Network()
h, w = net.environment.im.shape

max_n = 10  # no. of nodes
clusters = 1
x_radius = w/3/clusters
y_radius = h/3/clusters

c_power = []
n_power = []
n_range = np.arange(11, max_n, 5)

for n in n_range:

    # network topology setup
    Node.cid = 1
    net_gen = Star(degree=2, n_count=n)
    net = net_gen.generate_star_ehwsnnetwork(
        x_radius=x_radius,
        y_radius=y_radius,
        sector=1,
        clusters=clusters,
        is_random=1)
    net.name = "New scheme with EHWSN"

    # saving topology as PNG image
    net.savefig(fname=get_path(DATETIME_DIR, "star-%s" %n),
                title="EHWSN nodes around coordinator - %s nodes" %n,
                x_label="X", y_label="Y")

    # simulation start
    net.algorithms = ((Beacon, {'informationKey': 'Data'}), )
    coordinator_node = net.nodes()[0]
    coordinator_node.memory['Data'] = 'Beacon 1'
    sim = Simulation(net)
    sim.run()
    # simulation ends

    # capturing results
    consume = []
    energy = []
    distances = []
    displacement = []
    message_stats = []
    total_energy = 0
    t_transmit = 0
    t_transmit_failed = 0
    isMoved = False
    recovered_nodes = []
    expected = MAX_TRIES + 2

    for nd in net.nodes():
        print nd.id, "Registrations: %s" % nd.memory.get("Registration", []), \
            "\n\rData: %s" % nd.memory, \
            "\n\rEnergy: %s" % nd.power.energy_consumption

        t_transmit += nd.n_transmitted
        t_transmit_failed += nd.n_transmitted_failed_power

        message_stats.append((nd.id, nd.n_transmitted,
                              nd.n_transmitted_failed_power,
                              nd.n_received,
                              nd.n_received_failed_power,
                              nd.n_received_failed_loss
                              ))
        consume.append(nd.power.energy_consumption * 1000.0)
        energy.append(nd.power.energy * 1000.0)
        total_energy += nd.power.energy_consumption * 1000.0
        moved = array((nd.mobility.moved_x, nd.mobility.moved_y))
        d2 = sqrt(sum(pow(moved, 2)))
        displacement.append(d2)
        if len(nd.distance) > 0:
            d = nd.distance.pop()
            #d2 = nd.distance.pop()
            pr = net.propagation.get_power_ratio(d=d)
            pr = propagation.PropagationModel.pw_to_dbm(pr)
            distances.append((nd.id, d, pr, d2))

        if nd.id > 1 and  nd.n_transmitted > 0 and \
                        nd.n_transmitted < expected:
            recovered_nodes.append((nd.id, nd.energy))
        if nd.mobility.have_moved():
            isMoved = isMoved or True

    if isMoved:
           net.savefig(fname=get_path(DATETIME_DIR, "star_final-%s") % n,
                title="EHWSN nodes around coordinator - %s nodes (final)" %n,
                x_label="X", y_label="Y")

    c_power.append(consume[0])  # coordinator

    n_power.append((n, consume[0], total_energy,  # sum of other node consumption
                    coordinator_node.n_received,
                    coordinator_node.n_received_failed_loss))

    sp.savetxt(get_path(DATETIME_DIR, "energy-%s.csv" %n),
               list(zip(range(1, n+1)[1:], energy[1:])),
               delimiter="\t", fmt="%s",
               header="Nodes\tEnergy (mJ)", comments='')


    sp.savetxt(get_path(DATETIME_DIR, "stats-%s.csv" %n),
               message_stats,
               delimiter=",", fmt="%8s", comments='',
               header="Node,transmit,transmit_failed_power,"
                      "received,received_failed_power,"
                      "received_failed_loss")

    sp.savetxt(get_path(DATETIME_DIR, "distances-%s.csv" %n),
               distances,
               delimiter="\t", fmt="%7.1f", comments='',
               header="Node\tdistance(m)\tdbm\tdisplacement")

    if len(recovered_nodes) > 1:
        #print recovered_nodes
        sp.savetxt(get_path(DATETIME_DIR, "energy_level-%s.csv" % n),
               zip(recovered_nodes[0][1], recovered_nodes[1][1]),
               header="%s\t%s" %(recovered_nodes[0][0],
                                 recovered_nodes[1][0]),
               delimiter="\t", fmt="%s", comments='')
        plotter.plots(np.arange(len(recovered_nodes[0][1])),
                      recovered_nodes[0][1],
              get_path(DATETIME_DIR, "energy_level-%s" % n),
              more_plots=[recovered_nodes[1][1]],
              title="Energy Level change",
              xlabel="Time", ylabel="Energy (Joules)",
              labels=["Node %s" % recovered_nodes[0][0],
                      "Node %s" % recovered_nodes[1][0]])

    # plotting results
    plotter.plot_bars(np.arange(1, n+1)[1:], consume[1:],
                  get_path(DATETIME_DIR, "energy consumption-", prefix=n),
                  ymax=1.1*max(consume[1:]),
                  ymin=0.9*min(consume[1:]),
                  title="Energy Consumption (%s nodes)" %n,
                  xlabel="Nodes", ylabel="mJ")

    plotter.plot_bars(np.arange(1, n+1), displacement,
                  get_path(DATETIME_DIR, "Node movement-", prefix=n),
                  ymax=1.05*max(displacement),
                  title="Node Displacement during simulation", color='g',
                  xlabel="Nodes", ylabel="Displacement (m)")

    sim.reset()
#simulation level

plotter.plots(n_range, c_power,  #[e[0] for e in n_power], [e[1] for e in n_power],
              get_path(DATETIME_DIR, "energy-Coordinator"),
              more_plots=[[e[2] for e in n_power],
                          [e[3] for e in n_power],
                          [e[4] for e in n_power]],
              labels=["Coordinator (mJ)","EHWSN nodes (mJ)",
                      "Received Packets", "Lost Packets"],
              title="Energy Consumption & Received Packet Stat",
              xlabel="Nodes", ylabel="")

# save data to text file for further analysis
sp.savetxt(get_path(DATETIME_DIR, "energy-Coordinator.csv"),
           n_power,
           delimiter="\t", fmt="%7.0f",
           header="Nodes\tEnergy(mJ)\tTotal(mJ)\tReceived\tLost",
           comments='')

