__author__ = 'farrukh'

from pymote.algorithm import NodeAlgorithm
from pymote.message import Message
from numpy.random import rand
import time
from pymote.logger import logger

MAX_TRIES = 14
DATA_RATE = 1


class Beacon(NodeAlgorithm):
    required_params = ('informationKey',)
    default_params = {'neighborsKey': 'Neighbors'}
    tries = 0

    def step(self, node):
        """ Executes one step of the algorithm for given node."""
        while True:
            message = node.receive()
            if not message:
                break
            # logger.info("GOTTT %s-%s" % (node.status, node.id))
            if message and "RECEIVE" in node.status and node.type != 'B':
                # node.outbox.insert(0, Message(header="Registration",source=node,
                #                              data="Registration-" + str(node.id),
                #                              destination=message.source))

                node.send(Message(header="Registration", source=node,
                                  data="Registration-" + str(node.id),
                                  destination=message.source))
            if message:
                if (message.destination == None or message.destination == node):
                    # when destination is None it is broadcast message
                    self._process_message(node, message)
                elif (message.nexthop == node.id):
                    self._forward_message(node, message)

    def initializer(self):
        ini_nodes = []
        for node in self.network.nodes():
            print node.compositeSensor.read()
            node.memory[self.neighborsKey] = \
                node.compositeSensor.read()['Neighbors']
            node.status = 'RECEIVE'
            if self.informationKey in node.memory:
                node.status = 'INITIATOR'
                ini_nodes.append(node)
        # buffer Beacon
        #        node.send(Message(header=NodeAlgorithm.INI))
        for ini_node in ini_nodes:
            self.network.outbox.insert(0, Message(header=NodeAlgorithm.INI,
                                                  destination=ini_node))

    def initiator(self, node, message):
        # logger.info("INI")
        if message.header == NodeAlgorithm.INI:
            # default destination: send to every neighbor
            node.send(Message(header='Information',
                              data=node.memory[self.informationKey]))
        node.status = 'RECEIVE'

    def receive(self, node, message):
        # logger.info("RECV")

        node.status = 'SEND'

    def send_data(self, node, message):
        # logger.info("SEND")
        # node.status = 'DONE'
        global tries
        for child_node in self.network.nodes():

            # child_node.outbox.insert(0, Message(header="DataPacket", source=child_node,
            #                                    data="Data-" + str(child_node.id)))
            #
            detination = None
            if 'destinaion' in child_node.memory:
                detination = [child_node.memory['destination']]
            child_node.send(Message(header="DataPacket", source=child_node, destination=detination,
                                    data="Data-" + str(self.tries) + "*" + str(child_node.id) + '+' +
                                         str(child_node.power.energy)))

            # str(range(int(rand(1)[0]*10))) ))
        # manage node
        for child_node in self.network.nodes():
            if child_node.mobility.mobile_type != 0:
                x, y, z = child_node.mobility.drift()
                child_node.network.pos[child_node] += (x, y)

            child_node.power.increase_energy(charging_time=DATA_RATE)  # charging
            child_node.power.decrease_energy(discharging_time=DATA_RATE)  # idle discharge
            child_node.energy.append(child_node.power.energy)

        self.tries += 1
        print self.tries
        if self.tries > MAX_TRIES:
            node.status = 'DONE'
        else:
            node.status = 'RECEIVE'
            # print "Sleeping.."
            # time.sleep(DATA_RATE)
            # print "Done"

    def done(self, node, message):
        print "DONE"
        pass

    STATUS = {
        'INITIATOR': initiator,
        'RECEIVE': receive,
        'SEND': send_data,
        'DONE': done,
    }
