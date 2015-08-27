import datetime

import scipy as sp

from pymote import *
from pymote.conf import global_settings
from beacon import Beacon, MAX_TRIES
from pymote import propagation
from pymote.utils import plotter
from pymote.utils.filing import get_path, date2str,\
     DATA_DIR, TOPOLOGY_DIR, CHART_DIR, DATETIME_DIR
import numpy as np

from numpy import sign, sqrt, array, pi, sin, cos
from numpy.random import rand


class LOS(NetworkGenerator):

    def generate_network(self, center=None, x_radius=100, y_radius=100, sector=1.0, clusters=1, is_random=0):
        """
        Generates network where nodes are located around a center/coordinator node.
        Parameter is_random controls random perturbation of the nodes
        """
        net = Network(propagation_type=2, **self.kwargs)
        h, w = net.environment.im.shape
        if center is None:
            center = (h/2, w/2)  # middle
        if clusters < 1:
            clusters = 1
        n_nets = int(self.n_count/clusters)

        for k in range(clusters):
            # Base Station
            node = Node(commRange=(x_radius + y_radius)/clusters, node_type='B',
                        power_type=0, **self.kwargs)
            mid = (center[0] - 2 * k * x_radius,
                   center[1] - 2 * k * y_radius)
            net.add_node(node, pos=(mid[0], mid[1]))
            for n in range(n_nets)[1:]:
                    # Regular sensor node
                    node = Node(commRange=5, power_type=1, mobile_type=0, **self.kwargs)
                    node.power.energy = 0.5 + rand()*2.0
                    rn = (rand(2) - 0.5)*(x_radius + y_radius)/2/clusters
                    ang = n *2*pi/n_nets * sector + pi*(1.0 - sector)
                    net.add_node(node, pos=(mid[0] + cos(ang)*(x_radius + rn[0]*is_random),
                                            mid[1] + sin(ang)*(y_radius + rn[0]*is_random)
                    ))

        return net

# Network/Environment setup
global_settings.ENVIRONMENT2D_SHAPE = (600, 600)
net = Network()
h, w = net.environment.im.shape
propagation.PropagationModel.P_TX = 0.0144  # Node Tx Power

max_n = 51  # no. of nodes
clusters = 1
x_radius = w/3/clusters
y_radius = h/3/clusters

c_power = []
n_power = []
n_range = np.arange(5, max_n, 5)

for n in n_range:

    # network topology setup
    Node.cid = 1
    net_gen = LOS(degree=2, n_count=n)
    net = net_gen.generate_network(
        x_radius=x_radius,
        y_radius=y_radius,
        sector=0.5,
        clusters=clusters,
        is_random=0)
    net.name = "Yokogawa Industrial Sensor Experiment"

    # saving topology as PNG image
    net.savefig(fname=get_path(DATETIME_DIR, "topology-%s" %n),
                title="Industrial sensor nodes around Base station - %s nodes" %n,
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
    snr_base = []

    for nd in net.nodes():
        print nd.id, "Registrations: %s" % nd.memory.get("Registration", []), \
            "\n\rData: %s" % nd.memory, \
            "\n\rEnergy: %s" % nd.power.energy_consumption
        #print len(nd.distance), nd.distance
        #print len(nd.snr), nd.snr
        if nd.id == 1:
            snr_base = nd.snr
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
                title="Industrial sensor nodes around coordinator - %s nodes (final)" %n,
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

    if isMoved:
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
    #print len(snr_base)
    lqi_base = 6 * np.array(snr_base) + 538
    #print lqi_base
    plotter.plots(np.arange(len(snr_base[1:])), snr_base[1:],
                  get_path(DATETIME_DIR, "SNR_Base_Station-%s" % n),
                  title="SNR levels",
                  xlabel="Time", ylabel="SNR")

    plotter.plots(np.arange(len(snr_base[1:])), lqi_base[1:],
                  get_path(DATETIME_DIR, "LQI_Base_Station-%s" % n),
                  title="LQI's",
                  xlabel="Time", ylabel="LQI")
    plotter.plot_bars(np.arange(1, n+1)[1:], consume[1:],
                  get_path(DATETIME_DIR, "energy consumption-", prefix=n),
                  ymax=1.1*max(consume[1:]),
                  ymin=0.9*min(consume[1:]),
                  title="Energy Consumption (%s nodes)" % n,
                  xlabel="Nodes", ylabel="mJ")

    if isMoved:
        plotter.plot_bars(np.arange(1, n+1), displacement,
                  get_path(DATETIME_DIR, "Node movement-", prefix=n),
                  ymax=1.05*max(displacement),
                  title="Node Displacement during simulation", color='g',
                  xlabel="Nodes", ylabel="Displacement (m)")


    sim.reset()
#simulation level

plotter.plots(n_range, c_power,  #[e[0] for e in n_power], [e[1] for e in n_power],
              get_path(DATETIME_DIR, "energy-Base Station"),
              more_plots=[[e[2] for e in n_power],
                          [e[3] for e in n_power],
                          [e[4] for e in n_power]],
              labels=["Base Station (mJ)","Sensor nodes (mJ)",
                      "Received Packets", "Lost Packets"],
              title="Energy Consumption & Received Packet Stat",
              xlabel="Nodes", ylabel="")

# save data to text file for further analysis
sp.savetxt(get_path(DATETIME_DIR, "energy-Base Station.csv"),
           n_power,
           delimiter="\t", fmt="%7.0f",
           header="Nodes\tEnergy(mJ)\tTotal(mJ)\tReceived\tLost",
           comments='')

