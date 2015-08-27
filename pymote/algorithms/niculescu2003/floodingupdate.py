from pymote.message import Message
from pymote.algorithm import NodeAlgorithm


class FloodingUpdate(NodeAlgorithm):
    """
    This is modified Flooding algorithm (Santoro2007 p.13) so that every node
    continues to forward flood messages as long as information gathered is
    updating its knowledge.
    Note: does not have global termination detection
    Costs: ?
    """

    required_params = ('dataKey',)  # memory key for data being updated
    default_params = {}

    def initializer(self):
        """ Starts in every node satisfying initiator condition. """

        for node in self.network.nodes():
            if self.initiator_condition(node):
                self.network.outbox.insert(0, Message(destination=node,
                                                     header=NodeAlgorithm.INI))
            node.status = 'FLOODING'

    def flooding(self, node, message):
        if message.header == NodeAlgorithm.INI:
            node.send(Message(header='Flood',
                              data=self.initiator_data(node)))

        if message.header == 'Flood':
            updated_data = self.handle_flood_message(node, message)
            if updated_data:
                node.send(Message(header='Flood',
                                  data=updated_data))

    def initiator_condition(self, node):
        raise NotImplementedError

    def initiator_data(self, node):
        raise NotImplementedError

    def handle_flood_message(self, node, message):
        raise NotImplementedError

    STATUS = {'FLOODING': flooding,  # init,term
              }
