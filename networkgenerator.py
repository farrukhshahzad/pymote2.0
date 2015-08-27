from networkx import is_connected, Graph
from numpy.core.numeric import Inf
from pymote.network import Network
from pymote.logger import logger
from pymote.conf import settings
from numpy import sign, sqrt, array, pi, sin, cos
from pymote.node import Node
from numpy.random import rand
from itertools import product


class NetworkGenerator(object):

    def __init__(self, n_count=None, n_min=0, n_max=Inf, connected=True,
                 degree=None, comm_range=None, method=None,
                 **kwargs):
        """
        Arguments:
            n_count (int):
                number of nodes, if None settings.N_COUNT is used
            n_min (int):
                minimum number of nodes
            n_max (int):
                maximum number of nodes
            connected (bool):
                if True network must be fully connected
            degree (int):
                average number of neighbors per node
            comm_range (int):
                nodes communication range, if None settings.COMM_RANGE is used
                and it is a signal that this value can be changed if needed to
                satisfy other wanted properties (connected and degree)
            method (str):
                sufix of the name of the method used to generate network
        kwargs can be network and node __init__ kwargs i.e.:
            environment (:class:`Environment`):
                environment in which the network should be created, if None
                settings.ENVIRONMENT is used
            channelType (:class:`ChannelType`)
            algorithms (tuple)
            commRange (int):
                overrides `comm_range`
            sensors (tuple)

        Basic usage:

        >>> net_gen = NetworkGenerator()
        >>> net = net_gen.generate()

        """
        self.n_count=n_count if n_count else settings.N_COUNT
        if self.n_count<n_min or self.n_count>n_max:
            raise NetworkGeneratorException('Number of nodes must be between '
                                            'n_min and n_max.')
        if degree and degree>=n_max:
            raise NetworkGeneratorException('Degree % d must be smaller than '
                                            'maximum number of nodes %d.'
                                            % (degree, n_max))
        #TODO: optimize recalculation of edges on bigger commRanges
        if degree and degree>16 and n_count!=Inf:
            logger.warning("Generation could be slow for large degree"
                           "parameter with bounded n_max.")
        self.n_min = n_min
        self.n_max = n_max
        self.connected = connected
        self.degree = degree
        self.comm_range = kwargs.pop('commRange', comm_range)
        #TODO: use subclass based generators instead of method based
        if (method):
            self.generate = self.__getattribute__("generate_" + method)
        self.kwargs = kwargs

    def _create_modify_network(self, net=None, step=1):
        """Helper method for creating new or modifying given network.

        Arguments:
            net (int):
                network to modify, if None create from scratch
            step:
                if >0 new network should be more dense for <0 less dense

        """
        if net is None:
            net = Network(**self.kwargs)
            for _n in range(self.n_count):
                node = Node(commRange=self.comm_range, **self.kwargs)
                net.add_node(node)
        else:
            if step>0:
                if len(net)<self.n_max:
                    node = Node(**self.kwargs)
                    net.add_node(node)
                    logger.debug("Added node, number of nodes: %d"
                                 % len(net))
                elif not self.comm_range:
                    for node in net.nodes():
                        node.commRange += step
                    logger.debug("Increased commRange to %d"
                                 % node.commRange)
                else:
                    return None
            else:
                if len(net)>self.n_min and len(net)>1:
                    net.remove_node(net.nodes()[0])
                    logger.debug("Removed node, nodes left: %d"
                                 % len(net))
                elif not self.comm_range:
                    for node in net:
                        node.commRange += step
                    logger.debug("Decreased commRange to %d"
                                 % net.nodes()[0].commRange)
                else:
                    return None
        return net

    def _are_conditions_satisfied(self, net):
        cr = net.nodes()[0].commRange
        if self.connected and not is_connected(net):
            logger.debug("Not connected")
            return round(0.2*cr)
        elif self.degree:
            diff = self.degree-net.avg_degree()
            if (abs(diff)<0.5):
                return 0
            logger.debug("Degree not satisfied %f" % net.avg_degree())
            diff = sign(diff)*min(abs(diff), 7)
            return round((sign(diff)*(round(diff))**2)*cr/100)
        return 0

    def generate_random_network(self, net=None):
        """Basic method: generates network with randomly positioned nodes."""
        #TODO: try some more advanced algorithm for situation when
        # both connected network and too small degree are needed
        # that is agnostic to actual dimensions of the environment
        steps = [0]
        while True:
            net = self._create_modify_network(net, steps[-1])
            if not net:
                break
            steps.append(self._are_conditions_satisfied(net))
            if len(steps)>1000:
                break
            if steps[-1]==0:
                return net

        logger.error("Could not generate connected network with given "
                     "parameters. Try removing and/or modifying some of "
                     "them.")

    def generate_neigborhood_network(self):
        """
        Generates network where all nodes are in one hop neighborhood of
        at least one node.

        Finds out node in the middle, that is the node with minimum maximum
        distance to all other nodes and sets that distance as new commRange.

        """
        net = self._create_modify_network()

        max_distances = []
        for node in net:
            distances = [sqrt(sum((net.pos[node]-net.pos[neighbor])**2))\
                                   for neighbor in net]
            max_distances.append(max(distances))
        min_distance = min(max_distances)
        for node in net:
            node.commRange = min_distance+1
        return net

    def generate_homogeneous_network(self, randomness=0.11):
        """
        Generates network where nodes are located approximately homogeneous.

        Parameter randomness controls random perturbation of the nodes, it is
        given as a part of the environment size.

        """
        net = self._create_modify_network()
        n = len(net)
        h, w = net.environment.im.shape
        assert net.environment.dim==2  # works only for 2d environments
        size = w

        positions = generate_mesh_positions(net.environment, n)
        for i in range(n):
            pos = array([-1, -1])  # some non space point
            while(not net.environment.is_space(pos)):
                pos = positions[i, :n] + (rand(2) - 0.5)*(size*randomness)
            net.pos[net.nodes()[i]] = pos
        net.recalculate_edges()
        #TODO: this is not intuitive but generate_random network with net
        # given as argument will check if conditions are satisfied and act
        # accordingly, to change only commRange set limits to number of nodes
        self.n_count = self.n_max = self.n_min = n
        return self.generate_random_network(net)

def generate_mesh_positions(env, n):
    """
    Strategy: put rectangle mesh with intersections distance d above
    environment image and try to minimize difference between number of
    intersections in environment's free space and n by varying d.
    """
    h, w = env.im.shape
    # initial d
    d = sqrt(h*w/n)

    def get_mesh_pos(d, dx, dy, w, h):
        return map(lambda (xi, yi): (xi*d+dx, yi*d+dy),
                   product(range(int(round(w/d))), range(int(round(h/d)))))
    n_mesh = 0
    direction = []
    while True:
        n_mesh = len([1 for pos in get_mesh_pos(d, 0.5*d, 0.5*d, w, h)
                      if env.is_space(pos)])
        direction.append(sign(n-n_mesh))
        if n_mesh==n or \
            (len(direction)>=10 and abs(sum(direction[-3:]))<3 and n_mesh>n):
            break
        d *= sqrt(n_mesh/float(n))
    return array(tuple(pos for pos in get_mesh_pos(d, 0.5*d, 0.5*d, w, h)
                       if env.is_space(pos)))
    #TODO: n_mesh could be brought closer to n with modification of dx and dy
    #dx = 0.5*d
    #dy = 0.5*d


class NetworkGeneratorException(Exception):
    pass
