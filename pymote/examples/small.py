from pymote import *

net_gen = NetworkGenerator(degree=2, n_count=20, connected=False)
#net = net_gen.generate_random_network()
net = net_gen.generate_homogeneous_network()
#net = net_gen.generate_neigborhood_network()
print net_gen.n_count
from pymote.algorithms.broadcast import Flood
net.algorithms = ( (Flood, {'informationKey':'I'}), )
some_node = net.nodes()[0]
some_node.memory['I'] = 'Farrukh'
sim = Simulation(net)
sim.run()

print net.get_size()
print net.get_dic()
net.savefig(fname="homogeneous_network")

#net.reset()
sim.reset()

net_gen = NetworkGenerator(degree=2, n_count=10, commRange=20)
net = net_gen.generate_neigborhood_network()

net.algorithms = ( (Flood, {'informationKey':'I'}), )
some_node = net.nodes()[0]
some_node.memory['I'] = 'Beacon'
sim = Simulation(net)
sim.run()


net.savefig(fname="neigborhood_network", title="Neigborhood Network - 12 nodes", x_label="X", y_label="Y")
net.get_tree_net(None)

net.savefig(fname="tree_network", title="Tree Network - 12 nodes", x_label="X", y_label="Y")

print net.info()

