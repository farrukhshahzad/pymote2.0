from copy import deepcopy
from copy import copy


class Message(object):

    def __init__(self, source=None, destination=None, nexthop=None, header='',
                 data={}):
        self.source = source
        self.destination = destination
        self.nexthop = nexthop
        self.header = header
        self.data = data
        self.overhead_bytes = 60

    # in bytes
    def message_length(self):
        return len(self.header + str(self.data)) + self.overhead_bytes

    def __repr__(self):
        destination = self.destination
        if self.destination is None:
            destination = 'Broadcast'
        elif isinstance(self.destination, list) and\
             len(self.destination) == 1 and\
             self.destination[0] is None:
            destination = 'Broadcasting'
        return ("\n------ Message ------ \n     source = %s \ndestination = %s"
                " \n     header = '%s' \nid(message) = 0x%x>")\
                % (self.source, destination, self.header, id(self))

    def copy(self):
        # nodes are protected from copying by __deepcopy__()
        self.data = deepcopy(self.data)
        return copy(self)
