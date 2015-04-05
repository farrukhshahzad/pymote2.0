from pymote import *
net_gen = NetworkGenerator(100)
net = net_gen.generate_random_network()
from pymote.algorithms.broadcast import Flood
net.algorithms = ( (Flood, {'informationKey':'I'}), )
some_node = net.nodes()[0]
some_node.memory['I'] = 'Hello distributed world'
sim = Simulation(net)
sim.run()
net.show()

raw_input()
