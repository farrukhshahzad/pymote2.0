__author__ = 'farrukh'

from pymote import *

from numpy import sign, sqrt, array, pi, sin, cos
from numpy.random import rand


class Star(NetworkGenerator):

    def generate_star_network(self, center=None, x_radius=100, y_radius=100, sector=1.0, clusters=1, is_random=0):
        """
        Generates network where nodes are located around a center/coordinator node.
        Parameter is_random controls random perturbation of the nodes
        """

        net = Network(**self.kwargs)
        h, w = net.environment.im.shape
        if center is None:
            center = (h/2, w/2)  # middle
        if clusters < 1:
            clusters = 1
        n_nets = int(self.n_count/clusters)

        for k in range(clusters):
            # Coordinator
            node = Node(commRange=(x_radius + y_radius)/clusters, node_type='C',
                        power_type=0, **self.kwargs)
            mid = (center[0] - 2 * k * x_radius,
                   center[1] - 2 * k * y_radius)
            net.add_node(node, pos=(mid[0], mid[1]))
            for n in range(n_nets)[1:]:
                    # Regular sensor node
                    node = Node(commRange=5, power_type=2, mobile_type=2, **self.kwargs)
                    if node.id == 5:
                        node.power.energy = node.power.E_MIN - 0.003
                    elif node.id == 10:  # just enough energy to send few messages
                        node.power.energy = node.power.E_MIN + 0.002
                        node.power.P_CHARGING = 0  # no charging
                    else:  # random energy level from 0 to 2 Joules
                        node.power.energy = rand()*2.0
                    rn = (rand(2) - 0.5)*(x_radius + y_radius)/2/clusters
                    ang = n *2*pi/n_nets * sector + pi*(1.0 - sector)
                    net.add_node(node, pos=(mid[0] + cos(ang)*(x_radius + rn[0]*is_random),
                                            mid[1] + sin(ang)*(y_radius + rn[0]*is_random)
                    ))

        return net
