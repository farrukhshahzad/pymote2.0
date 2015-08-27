from pymote.algorithm import NodeAlgorithm
from pymote.message import Message


class Flood(NodeAlgorithm):
    required_params = ('informationKey',)
    default_params = {'neighborsKey': 'Neighbors'}

    def initializer(self):
        ini_nodes = []
        for node in self.network.nodes():
            node.memory[self.neighborsKey] = \
                node.compositeSensor.read()['Neighbors']
            node.status = 'IDLE'
            if self.informationKey in node.memory:
                node.status = 'INITIATOR'
                ini_nodes.append(node)
        for ini_node in ini_nodes:
            self.network.outbox.insert(0, Message(header=NodeAlgorithm.INI,
                                                 destination=ini_node))

    def initiator(self, node, message):
        if message.header == NodeAlgorithm.INI:
            # default destination: send to every neighbor
            node.send(Message(header='Information',
                              data=node.memory[self.informationKey]))
            node.status = 'DONE'

    def idle(self, node, message):
        if message.header == 'Information':
            node.memory[self.informationKey] = message.data
            destination_nodes = list(node.memory[self.neighborsKey])
            # send to every neighbor-sender
            destination_nodes.remove(message.source)
            if destination_nodes:
                node.send(Message(destination=destination_nodes,
                                  header='Information',
                                  data=message.data))
        node.status = 'DONE'

    def done(self, node, message):
        pass

    STATUS = {
              'INITIATOR': initiator,
              'IDLE': idle,
              'DONE': done,
             }
