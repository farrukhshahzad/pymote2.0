from pymote.logger import logger
from pymote.sensor import CompositeSensor
from pymote.conf import settings
from pymote.mobility import MobilityModel
from pymote.energy import EnergyModel
from pymote.propagation import PropagationModel

import logging

from numpy import sqrt

class Node(object):

    cid = 1

    def __init__(self, network=None, commRange=None, sensors=None,
                 node_type=None, power_type=1, mobile_type=0, **kwargs):
        """
        Initialize the node object.

        node_type: 'N' regular, 'B' base station/Sink, 'C' coordinator/cluster head/relay/Anchor
        power_type = {0: "External", 1: "Battery", 2: "Energy Harvesting"}
        Mobility_Type = {0: "Fixed", 1: "Mobile-Vel", 2: "Mobile-Acc", 3: "Random"}

        """
        self._compositeSensor = CompositeSensor(self, sensors or
                                                settings.SENSORS)
        self.network = network
        self._commRange = commRange  # or settings.COMM_RANGE
        self.id = self.__class__.cid
        self.__class__.cid += 1
        self._inboxDelay = True
        self.type = node_type or 'N'
        self.power = EnergyModel(power_type=power_type)
        self.mobility = MobilityModel(mobile_type=mobile_type)

        self.reset()

    def __repr__(self):
        return "<Node id=%s>" % self.id
        # return "<Node id=%s at 0x%x>" % (self.id, id(self))

    def __deepcopy__(self, memo):
        return self

    def reset(self):
        self.outbox = []
        self._inbox = []
        self.status = ''
        self.memory = {}
        self.energy = []
        self.distance = []
        self.snr = []
        self.n_received = 0
        self.n_received_failed_power = 0
        self.n_received_failed_loss = 0
        self.n_transmitted = 0
        self.n_transmitted_failed_power = 0

    def send(self, message):
        """
        Send a message to nodes listed in message's destination field.

        Note: Destination should be a list of nodes or one node.

        Update message's source field and  inserts in node's outbox one copy
        of it for each destination.

        """
        if not self.power.have_energy():
            self.n_transmitted_failed_power += 1
            self.energy.append(self.power.energy)
            logger.debug("Node %d doesn't have  enough energy to send message [energy=%5.3f]" %
                        (self.id, self.power.energy))
            return

        message.source = self
        message.destination = isinstance(message.destination, list) and\
                              message.destination or [message.destination]
        msg_len = message.message_length()

        for destination in message.destination:
            self.power.decrease_tx_energy(msg_len)
            self.energy.append(self.power.energy)
            self.n_transmitted += 1
            logger.debug('Node %d sent message %s [%d].' %
                         (self.id, message.data, msg_len))
            m = message.copy()
            m.destination = destination
            self.outbox.insert(0, m)

    def receive(self):
        """
        Pop message from inbox but only if it has been there at least one step.

        Messages should be delayed for one step for visualization purposes.
        Messages are processed without delay only if they are pushed into empty
        inbox. So if inbox is empty when push_to_inbox is called _inboxDelay is
        set to True.

        This method is used only internally and is not supposed to be used
        inside algorithms.

        """
        if self._inbox and not self._inboxDelay:

            message = self._inbox.pop()
            if not message:
                return message
            msg_len = message.message_length()
            if not self.power.have_energy():
                self.n_received_failed_power += 1
                logger.debug("Node %d doesn't have enough energy to receive message [energy=%5.3f]" %
                         (self.id, self.power.energy))
            else:
                self.power.decrease_rx_energy(msg_len)

                if not message.source:
                    message.source = self
                p1 = self.network.pos[message.source]
                p2 = self.network.pos[message.destination]
                d = sqrt(sum(pow(p1 - p2, 2)))
                self.distance.append(d)
                prt = self.network.propagation.get_power_ratio(d=d)
                self.snr.append(PropagationModel.pw_to_dbm(prt))
                rx_ok = self.network.propagation.is_rx_ok(d=d, prt=prt)

                logger.debug('Node %d received message %s [%d] - %s m (%s)' %
                             (self.id, message.data, msg_len, d, PropagationModel.pw_to_dbm(prt)))

                if rx_ok:
                    self.n_received += 1
                    if message.header not in self.memory:
                        self.memory[message.header] = []
                    self.memory[message.header].append(message.data)
                else:
                    self.n_received_failed_loss += 1
                    logger.debug("Receive Failed due to signal loss: %s" %
                                 self.network.propagation.get_power_ratio(d=d))
            self.energy.append(self.power.energy)
        else:
            message = None
        self._inboxDelay = False

        return message

    @property
    def inbox(self):
        return self._inbox

    def push_to_inbox(self, message):
        # TODO: for optimization remove _inboxDelay when not visualizing

        self._inboxDelay = self._inboxDelay or not self._inbox
        self._inbox.insert(0, message)
        print "Got" + str(len(self._inbox)) + str(self._inboxDelay)

    @property
    def compositeSensor(self):
        return self._compositeSensor

    @compositeSensor.setter
    def compositeSensor(self, compositeSensor):
        self._compositeSensor = CompositeSensor(self, compositeSensor)

    @property
    def sensors(self):
        return self._compositeSensor.sensors

    @sensors.setter
    def sensors(self, sensors):
        self._compositeSensor = CompositeSensor(self, sensors)

    @property
    def commRange(self):
        return self._commRange

    @commRange.setter
    def commRange(self, commRange):
        self._commRange = commRange
        if self.network:
            self.network.recalculate_edges([self])

    def get_log(self):
        """ Special field in memory used to log messages from algorithms. """
        if not 'log' in self.memory:
            self.memory['log'] = []
        return self.memory['log']

    def log(self, message, level=logging.WARNING):
        """ Insert a log message in node memory. """
        assert isinstance(message, str)
        context = {
                   'algorithm': str(self.network.get_current_algorithm()),
                   'algorithmState': self.network.algorithmState,
                   }
        if not 'log' in self.memory:
            self.memory['log'] = [(level, message, context)]
        else:
            self.memory['log'].append((level, message, context))

    def get_dic(self):
        return {'Info': {'id': self.id,
                    'status': self.status,
                    'type': self.type,
                    'position': self.network.pos[self],
                    'orientation': self.network.ori[self]},
                'Communication': {'range': self.commRange,
                                  'inbox': self.box_as_dic('inbox'),
                                  'outbox': self.box_as_dic('outbox')},
                'Memory': self.memory,
                'Sensors': {sensor.name(): '%s(%.3f)' %
                                (sensor.probabilityFunction.name,
                                 sensor.probabilityFunction.scale)
                                 if hasattr(sensor, 'probabilityFunction') and
                                    sensor.probabilityFunction is not None
                                 else ('', 0)
                              for sensor in self.compositeSensor.sensors}}

    def box_as_dic(self, box):
        messagebox = self.__getattribute__(box)
        dic = {}
        for i, message in enumerate(messagebox):
            dic.update({'%d. Message' % (i + 1,): {'1 header': message.header,
                                          '2 source': message.source,
                                          '3 destination': message.destination,
                                          '4 data': message.data}})
        return dic
