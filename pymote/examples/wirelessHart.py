
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
from beacon import Beacon, MAX_TRIES
from toplogies import Topology
from pymote import propagation, energy
from pymote.utils import plotter
from pymote.utils.filing import get_path, date2str,\
     DATA_DIR, TOPOLOGY_DIR, CHART_DIR, DATETIME_DIR


class Manual(Topology):

    def generate_zigbee_network(self, name=None, randomness=0):

        from pymote.conf import global_settings
        from pymote.environment import Environment2D

        self.n_count = 0
        self.name = name or "Manual"
        #net = Network(environment=Environment2D(shape=(200,200)), commRange=75)
        net = Network(commRange=self.comm_range, **self.kwargs)
        h, w = net.environment.im.shape

        node = Node(node_type='B')
        net.add_node(node, pos=(w/2, h/5))
        self.n_count +=1

        node = Node(node_type='N')
        net.add_node(node, pos=(3*w/10, h/16))
        self.n_count +=1

        node = Node()
        net.add_node(node, pos=(3*w/10, 2*h/5))
        self.n_count +=1

        node = Node()
        net.add_node(node, pos=(w/2, h/2))
        self.n_count +=1

        node = Node()
        net.add_node(node, pos=(7*w/10, 2*h/5))
        self.n_count +=1

        node = Node()
        net.add_node(node, pos=(7*w/10, h/16))
        self.n_count +=1

        node = Node()
        net.add_node(node, pos=(6*w/10, 15*h/20))
        self.n_count +=1

        return net

    def generate_mesh_network(self, name=None, type=0):

        from pymote.conf import global_settings
        from pymote.environment import Environment2D

        self.n_count = 0
        self.name = name or "Manual"
        #net = Network(environment=Environment2D(shape=(200,200)), commRange=75)
        net = Network(commRange=self.comm_range, **self.kwargs)
        h, w = net.environment.im.shape

        node = Node(node_type='N')
        net.add_node(node, pos=(w/2, h/5))
        net.nodes()[0].type = 'B'
        self.n_count +=1

        node = Node()
        net.add_node(node, pos=(3*w/10, 2*h/5))
        self.n_count +=1

        if type<2:
            node = Node()
            net.add_node(node, pos=(w/2, h/2))
            self.n_count +=1


        node = Node()
        net.add_node(node, pos=(7*w/10, 2*h/5))
        self.n_count +=1

        if type==0:
            node = Node()
            net.add_node(node, pos=(3*w/10, 5*h/8))
            self.n_count +=1

            node = Node()
            net.add_node(node, pos=(7*w/10, 5*h/8))
            self.n_count +=1

        if type==0 or type==2:
            node = Node()
            net.add_node(node, pos=(w/2, 4*h/5 - type/2*h/5))
            self.n_count +=1

        return net

# Network/Environment setup
seed(123)  # to get same random sequence for each run so that simulation can be reproduce

print propagation.PropagationModel.dbm_to_pw(11.6)

# Network/Environment setup
global_settings.ENVIRONMENT2D_SHAPE = (500, 500) # desired network size for simulation
global_settings.COMM_RANGE = 160

net = Network()
h, w = net.environment.im.shape
energy.EnergyModel.P_TX = 0.0378
energy.EnergyModel.P_RX = 0.027
propagation.PropagationModel.P_TX = energy.EnergyModel.P_TX
propagation.PropagationModel.P_RX_THRESHOLD = -80 # in dbm

max_n = 100  # max no. of nodes
clusters = 10
x_radius = 0.9*global_settings.COMM_RANGE
y_radius = 0.9*global_settings.COMM_RANGE

n_power = []
n_range = np.arange(20, max_n+1, 10)
start_time = time.time()
project_name = "IWSN - WirelessHART"
folder = DATETIME_DIR+ "-" + project_name

# network topology setup
Node.cid = 1
net_gen = Manual()
net = net_gen.generate_mesh_network(name="WirelessHart Mesh")
n = len(net)
net.name = "WirelessHart Simulation" + "\nN=%s, R=%s m" \
                                 %(n, global_settings.COMM_RANGE)
filename = net.name.replace("\n", "-")
# saving topology as PNG image
net.savefig(fname=get_path(folder, filename),   title=net.name,
            xlabel="X-coordinate (m)", ylabel="Y-coordinate (m)",
            show_labels=True, format="png", label_color='yellow')
print net.__len__(), len(net)

# simulation start
net.algorithms = ((Beacon, {'informationKey': 'Data'}),)
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
        loss.append(nd.n_received_failed_loss)
        total_energy += nd.power.energy_consumption * 1000.0

n_power.append((n, consume[0], total_energy,  # sum of other node consumption
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
              title="SNR levels", format='png',
              xlabel="Packet Number", ylabel="SNR")

plotter.plots(np.arange(len(snr_base[1:])), lqi_base[1:],
              get_path(folder, "LQI_Base_Station-%s" % n),
              title="LQI's",format='png',
              xlabel="Packet Number", ylabel="LQI")
plotter.plot_bars(np.arange(1, n+1), consume,
                  get_path(folder, "energy consumption-", prefix=n),
                  ymax=1.1*max(consume),
                  ymin=0.9*min(consume),format='png',
                  title="Energy Consumption (%s nodes)" % n,
                  xlabel="Nodes", ylabel="mJ")

plotter.plot_bars(np.arange(1, n+1), loss,
                  get_path(folder, "Loss Packets-", prefix=n),
                  ymax=1.1*max(loss),
                  ymin=0.9*min(loss),format='png',
                  title="Loss Packets (%s nodes)" % n,
                  xlabel="Nodes", ylabel="No.")
sim.reset()

# Create html/JS file for network Topology
plotter.gethtmlScatter(xpositions, [anchpositions, positions],
                fname=filename, folder=folder,
                xlabel="X-coordinates", ylabel="Y-coordinates", labels=['Base Station','Sensor'],
                title="Topology-"+filename, open=1, axis_range={'xmin':0, 'ymin':0, 'xmax': w, 'ymax': h},
                show_range=int(global_settings.COMM_RANGE),
                plot_options=["color: 'red', visible: true,", "color: 'blue',"])

plotter.gethtmlLine(np.arange(len(snr_base[1:])), [snr_base[1:]],
                fname="SNR-"+filename, folder=folder,
                xlabel="Packet", ylabel="SNR", labels=['SNR'],
                title="SNR-"+filename, open=1,
                plot_options=["color: 'red',"])