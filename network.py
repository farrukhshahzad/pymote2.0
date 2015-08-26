import inspect
from pymote.logger import logger
from pymote.conf import settings
from networkx import Graph, is_connected
import networkx as nx
from environment import Environment
from channeltype import ChannelType
from node import Node
from pymote import propagation

from numpy.random import rand
from numpy.core.numeric import Inf, allclose
from numpy import array, pi, sign, max, min
from numpy.lib.function_base import average
from algorithm import Algorithm
from pymote.sensor import CompositeSensor
from pymote.utils.helpers import pymote_equal_objects
from copy import deepcopy


class Network(Graph):

    def __init__(self, environment=None, channelType=None, algorithms=(),
                 networkRouting=True, propagation_type=2, **kwargs):

        Graph.__init__(self)
        self._environment = environment or Environment()
        # assert(isinstance(self.environment, Environment))
        self.channelType = channelType or ChannelType(self._environment)
        self.channelType.environment = self._environment
        self.propagation = propagation.PropagationModel(propagation_type=propagation_type)
        self.pos = {}
        self.ori = {}
        self.labels = {}
        #self.star = star_graph
        self.name = "WSN"
        self._algorithms = ()
        self.algorithms = algorithms or settings.ALGORITHMS
        self.algorithmState = {'index': 0, 'step': 1, 'finished': False}
        self.outbox = []
        self.networkRouting = networkRouting
        logger.info("Instance of Network has been initialized with %s" % self.propagation)

    def subgraph(self, nbunch):
        """ Returns Graph instance with nbunch nodes, see subnetwork. """
        return Graph(self).subgraph(nbunch)

    # TODO: incomplete add other properties
    def subnetwork(self, nbunch, pos=None):
        """ Returns Network instance with nbunch nodes, see subgraph. """
        if not pos:
            pos = self.pos
        H = Graph.subgraph(self, nbunch)
        for node in H:
            H.pos.update({node: pos[node][:2]})
            if len(pos[node]) > 2:
                H.ori.update({node: pos[node][2]})
            else:
                H.ori.update({node: self.ori[node]})
            H.labels.update({node: self.labels[node]})
        H._environment = deepcopy(self._environment)
        assert(isinstance(H, Network))
        return H

    def nodes(self, data=False):
        """ Override, sort nodes by id, important for message ordering."""
        return list(sorted(self.nodes_iter(data=data), key=lambda k: k.id))

    @property
    def algorithms(self):
        """
        Set algorithms by passing tuple of Algorithm subclasses.

        >>> net.algorithms = (Algorithm1, Algorithm2,)

        For params pass tuples in form (Algorithm, params) like this

        >>> net.algorithms = ((Algorithm1, {'param1': value,}), Algorithm2)

        """
        return self._algorithms

    @algorithms.setter
    def algorithms(self, algorithms):
        self.reset()
        self._algorithms = ()
        if not isinstance(algorithms, tuple):
            raise PymoteNetworkError('algorithm')
        for algorithm in algorithms:
            if inspect.isclass(algorithm) and issubclass(algorithm, Algorithm):
                self._algorithms += algorithm(self),
            elif (isinstance(algorithm, tuple) and
                  len(algorithm) == 2 and
                  issubclass(algorithm[0], Algorithm) and
                  isinstance(algorithm[1], dict)):
                self._algorithms += algorithm[0](self, **algorithm[1]),
            else:
                raise PymoteNetworkError('algorithm')

    @property
    def environment(self):
        return self._environment

    @environment.setter
    def environment(self, environment):
        """ If net environment is changed all nodes are moved into and
            corresponding channelType environment must be changed also. """
        self._environment = environment
        self.channelType.environment = environment
        for node in self.nodes():
            self.remove_node(node)
            self.add_node(node)
        logger.warning('All nodes are moved into new environment.')

    def remove_node(self, node):
        """ Remove node from network. """
        if node not in self.nodes():
            logger.error("Node not in network")
            return
        Graph.remove_node(self, node)
        del self.pos[node]
        del self.labels[node]
        node.network = None
        logger.debug('Node with id %d is removed.' % node.id)

    def add_node(self, node=None, pos=None, ori=None, commRange=None):
        """
        Add node to network.

        Attributes:
          `node` -- node to add, default: new node is created
          `pos` -- position (x,y), default: random free position in environment
          `ori` -- orientation from 0 to 2*pi, default: random orientation

        """
        if (not node):
            node = Node(commRange=commRange)
        assert(isinstance(node, Node))
        if not node.network:
            node.network = self
        else:
            logger.warning('Node is already in another network, can\'t add.')
            return None

        pos = pos if pos is not None else self.find_random_pos(n=100)
        ori = ori if ori is not None else rand() * 2 * pi
        ori = ori % (2 * pi)

        if (self._environment.is_space(pos)):
            Graph.add_node(self, node)
            self.pos[node] = array(pos)
            self.ori[node] = ori
            self.labels[node] = ('C' if node.type == 'C' else "") + str(node.id)
            logger.debug('Node %d is placed on position %s [energy=%5.3f].'
                         % (node.id, pos, node.power.energy))
            self.recalculate_edges([node])
        else:
            logger.error('Given position is not free space.')
        return node

    def node_by_id(self, id_):
        """ Returns first node with given id. """
        for n in self.nodes():
            if (n.id == id_):
                return n
        logger.error('Network has no node with id %d.' % id_)
        return None

    def avg_degree(self):
        return average(self.degree().values())

    def modify_avg_degree(self, value):
        """
        Modifies (increases) average degree based on given value by
        modifying nodes commRange."""
        # assert all nodes have same commRange
        assert allclose([n.commRange for n in self], self.nodes()[0].commRange)
        #TODO: implement decreasing of degree, preserve connected network
        assert value + settings.DEG_ATOL > self.avg_degree()  # only increment
        step_factor = 7.
        steps = [0]
        #TODO: while condition should call validate
        while not allclose(self.avg_degree(), value, atol=settings.DEG_ATOL):
            steps.append((value - self.avg_degree())*step_factor)
            for node in self:
                node.commRange += steps[-1]
            # variable step_factor for step size for over/undershoot cases
            if len(steps)>2 and sign(steps[-2])!=sign(steps[-1]):
                step_factor /= 2
        logger.debug("Modified degree to %f" % self.avg_degree())

    def get_current_algorithm(self):
        """ Try to return current algorithm based on algorithmState. """
        if len(self.algorithms) == 0:
            logger.warning('There is no algorithm defined in a network.')
            return None
        if self.algorithmState['finished']:
            if len(self.algorithms) > self.algorithmState['index'] + 1:
                self.algorithmState['index'] += 1
                self.algorithmState['step'] = 1
                self.algorithmState['finished'] = False
            else:
                return None
        return self.algorithms[self.algorithmState['index']]

    def reset(self):
        logger.info('Resetting network.')
        self.algorithmState = {'index': 0, 'step': 1, 'finished': False}
        self.reset_all_nodes()

    def show(self, *args, **kwargs):
        fig = self.get_fig(*args, **kwargs)
        fig.show()

    def savefig(self, fname='network.png', figkwargs={}, *args, **kwargs):
        self.get_fig(*args, **kwargs).savefig(fname, **figkwargs)

    def get_fig(self, title="Topology", x_label="X-axis", y_label="Y-axis",
                positions=None, edgelist=None, nodeColor='b',
                show_labels=True):
        try:
            from matplotlib import pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib required for show()")

        # TODO: environment when positions defined
        node_size = 30  # in points^2
        # calculate label delta based on network size
        label_delta = self.get_size()/max([30, len(self)])
        label_delta = self.get_size()/30
        dpi = 90
        size =  self._environment.im.shape[0] + 50, self._environment.im.shape[1] + 50
        fig_size = (tuple(array(size)/ dpi))

        # figsize in inches
        # default matplotlibrc is dpi=80 for plt and dpi=100 for savefig
        fig = plt.figure(figsize=fig_size, dpi=dpi, frameon=False)
        plt.imshow(self._environment.im, cmap='binary_r', vmin=0,
                   origin='lower')
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid()
        if positions:
            # truncate positions to [x, y], i.e. lose theta
            for k, v in positions.items():
                positions[k] = v[:2]
            pos = positions
            net = self.subnetwork(pos.keys())
            label_delta = (max(pos.values(), axis=0) -
                           min(pos.values(), axis=0))/max([60, len(positions)])
        else:
            pos = self.pos
            net = self
        label_pos = {}
        node_sizes = []
        node_colors = []
        colors = {'C': 'red', 'B': 'green', 'N': 'cyan'}
        sizes = {'C': 50, 'B': 70, 'N': 30}
        for n in net.nodes():
            label_pos[n] = pos[n].copy() - label_delta
            node_sizes.append(sizes.get(n.type, 20))
            if n.power.have_energy():
                node_colors.append(colors.get(n.type, 'pink'))
            else:
                node_colors.append('light'+colors.get(n.type, 'pink'))

        nx.draw_networkx_edges(net, pos, alpha=0.5, edgelist=edgelist, style='dashed', edge_color='b')
        nx.draw_networkx_nodes(net, pos, node_size=node_sizes,
                               node_color=node_colors, cmap='YlOrRd')
        if show_labels:
            nx.draw_networkx_labels(net, label_pos, labels=net.labels, font_size=10)

        return fig

    def recalculate_edges(self, nodes=[]):
        """ Recalculate edges for given nodes or for all self.nodes().
        Edge between nodes n1 and n2 are added if both are
        ChannelType.in_comm_range of each other"""
        if(not nodes):
            nodes = self.nodes()
        for n1 in nodes:
            for n2 in self.nodes():
                if (n1 != n2):
                    if (self.channelType.in_comm_range(self, n1, n2)):
                        Graph.add_edge(self, n1, n2)
                    elif (Graph.has_edge(self, n1, n2)):
                        Graph.remove_edge(self, n1, n2)

    def add_edge(self):
        logger.warn('Edges are auto-calculated from channelType and commRange')

    def find_random_pos(self, n=100):
        """ Returns random position, position is free space in environment if
         it can find free space in n iterations """
        while (n > 0):
            pos = rand(self._environment.dim) * \
                tuple(reversed(self._environment.im.shape))
            if self._environment.is_space(pos):
                break
            n -= 1
        return pos

    def reset_all_nodes(self):
        for node in self.nodes():
            node.reset()
        logger.info('Resetting all nodes.')

    def communicate(self):
        """Pass all messages from node's outboxes to its neighbors inboxes."""
        # Collect messages
        for node in self.nodes():
            self.outbox.extend(node.outbox)
            node.outbox = []
        while self.outbox:
            message = self.outbox.pop(0)
            #print message.data
            if (message.destination == None and message.nexthop == None):
                # broadcast
                self.broadcast(message)
            elif (message.nexthop != None):
                # Node routing
                try:
                    self.send(message.nexthop, message)
                except PymoteMessageUndeliverable, e:
                    print e.message
            elif (message.destination != None):
                # Destination is neighbor
                if (message.source in self.nodes() and
                    message.destination in self.neighbors(message.source)):
                    self.send(message.destination, message)
                elif (self.networkRouting):
                # Network routing
                # TODO: program network routing so it goes hop by hop only
                #       in connected part of the network
                    self.send(message.destination, message)
                else:
                    raise PymoteMessageUndeliverable('Can\'t deliver message.',
                                                      message)

    def broadcast(self, message):
        if message.source in self.nodes():
            for neighbor in self.neighbors(message.source):
                neighbors_message = message.copy()
                neighbors_message.destination = neighbor
                self.send(neighbor, neighbors_message)
        else:
            raise PymoteMessageUndeliverable('Source not in network. \
                                             Can\'t broadcast', message)

    def send(self, destination, message):
        logger.debug('Sending message from %s to %s (%s).' %
                      (repr(message.source), destination, message.data))

        if destination in self.nodes():
            destination.push_to_inbox(message)
            #if message.source:
                #message.source.power.decrease_tx_energy(message.message_length())  # TODO
        else:
            raise PymoteMessageUndeliverable('Destination not in network.',
                                             message)

    def info(self):
        return self.__str__()

    def get_size(self):
        """ Returns network width and height based on nodes positions. """
        return max(self.pos.values(), axis=0) - min(self.pos.values(), axis=0)
    
    def get_dic(self):
        """ Return all network data in form of dictionary. """
        algorithms = {'%d %s' % (ind, alg.name): 'active'
                      if alg == self.algorithms[self.algorithmState['index']]
                      else '' for ind, alg in enumerate(self.algorithms)}
        pos = {n: 'x: %.2f y: %.2f theta: %.2f deg' %
               (self.pos[n][0], self.pos[n][1], self.ori[n] * 180. / pi)
               for n in self.nodes()}
        return {'nodes': pos,
                'algorithms': algorithms,
                'algorithmState': {'name':
                                   self.get_current_algorithm().name
                                   if self.get_current_algorithm() else '',
                                  'step': self.algorithmState['step'],
                                  'finished': self.algorithmState['finished']}}

    def get_tree_net(self, treeKey):
        """
        Returns new network with edges that are not in a tree removed.

        Tree is defined in nodes memory under treeKey key as a list of tree
        neighbors or a dict with 'parent' (node) and 'children' (list) keys.

        """
        edgelist = []
        for node in self.nodes():
            nodelist = []
            if not treeKey in node.memory:
                continue
            if isinstance(node.memory[treeKey], list):
                nodelist = node.memory[treeKey]
            elif (isinstance(node.memory[treeKey], dict) and
                  'children' in node.memory[treeKey]):
                nodelist = node.memory[treeKey]['children']
            edgelist.extend([(node, neighbor) for neighbor in nodelist
                              if neighbor in self.nodes()])
        treeNet = self.copy()
        for e in treeNet.edges():
            if e not in edgelist and (e[1], e[0]) not in edgelist:
                treeNet.remove_edge(*e)
        return treeNet

    def validate_params(self, params):
        """ Validate if given network params match its real params. """
        logger.info('Validating params')
        count = params.get('count', None)  #  for unit tests
        if count:
            if isinstance(count, list):
                assert(len(self) in count)
            else:
                assert(len(self)==count)
        n_min = params.get('n_min', 0)
        n_max = params.get('n_max', Inf)
        assert(len(self)>=n_min and len(self)<=n_max)
        for param, value in params.items():
            if param=='connected':
                assert(not value or is_connected(self))
            elif param=='degree':
                assert(allclose(self.avg_degree(), value,
                                atol=settings.DEG_ATOL))
            elif param=='environment':
                assert(self.environment.__class__==value.__class__)
            elif param=='channelType':
                assert(self.channelType.__class__==value.__class__)
            elif param=='comm_range':
                for node in self:
                    assert(node.commRange==value)
            elif param=='sensors':
                compositeSensor = CompositeSensor(Node(), value)
                for node in self:
                    assert(all(map(lambda s1, s2: pymote_equal_objects(s1, s2),
                                   node.sensors, compositeSensor.sensors)))
            elif param=='aoa_pf_scale':
                for node in self:
                    for sensor in node.sensors:
                        if sensor.name()=='AoASensor':
                            assert(sensor.probabilityFunction.scale==value)
            elif param=='dist_pf_scale':
                for node in self:
                    for sensor in node.sensors:
                        if sensor.name()=='DistSensor':
                            assert(sensor.probabilityFunction.scale==value)
            #TODO: refactor this part as setting algorithms resets nodes
            """
            elif param=='algorithms':
                alg = self._algorithms
                self.algorithms = value
                assert(all(map(lambda a1, a2: pymote_equal_objects(a1, a2),
                               alg, self.algorithms)))
                #restore alg
                self._algorithms = alg
            """


class PymoteMessageUndeliverable(Exception):
    def __init__(self, e, message):
        self.e = e
        self.message = message

    def __str__(self):
        return self.e + repr(self.message)


class PymoteNetworkError(Exception):
    def __init__(self, type_):
        if type_ == 'algorithm':
            self.message = ('\nAlgorithms must be in tuple (AlgorithmClass,)'
                            ' or in form: ((AlgorithmClass, params_dict),).'
                            'AlgorithmClass should be subclass of Algorithm')
        else:
            self.message = ''

    def __str__(self):
        return self.message
