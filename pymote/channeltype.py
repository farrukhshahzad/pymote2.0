from pymote.conf import settings
from numpy import sqrt
from numpy.random import random


class ChannelType(object):
    """ChannelType abstract base class."""

    def __new__(self, environment=None, **kwargs):
        """Return instance of default ChannelType."""
        for cls in self.__subclasses__():
            print "***************", settings.CHANNEL_TYPE, cls.__name__, settings.DOI
            if (cls.__name__ == settings.CHANNEL_TYPE):
                return object.__new__(cls, environment)
        # if self is not ChannelType class (as in pickle.load_newobj) return
        # instance of self
        return object.__new__(self, environment, **kwargs)

    def in_comm_range(self, network, node1, node2):
        raise NotImplementedError

    def set_params(self, doi):
        raise NotImplementedError

class Udg(ChannelType):
    """Unit disc graph channel type."""

    def __init__(self, environment):
        self.environment = environment

    def in_comm_range(self, network, node1, node2):
        """Two nodes are in communication range if they can see each other and
        are positioned so that their distance is smaller than commRange."""
        p1 = network.pos[node1]
        p2 = network.pos[node2]
        d = sqrt(sum(pow(p1 - p2, 2)))
        if (d < node1.commRange or d < node2.commRange):
            #if (self.environment.are_visible(p1, p2)):
                return True
        return False


class SquareDisc(ChannelType):
    """ Probability of connection is 1-d^2/r^2 """

    def __init__(self, environment):
        self.environment = environment

    def in_comm_range(self, network, node1, node2):
        p1 = network.pos[node1]
        p2 = network.pos[node2]
        d = sqrt(sum(pow(p1 - p2, 2)))
        if random() > d ** 2 / node1.commRange ** 2:
            if (self.environment.are_visible(p1, p2)):
                #assert node1.commRange == node2.commRange
                return True
        return False


class Doi(ChannelType):
    """ Probability of connection is
    """

    def __init__(self, environment):
        self.environment = environment
        self.doi = settings.DOI

    def set_params(self, doi):
        self.doi = doi

    def in_comm_range(self, network, node1, node2):
        p1 = network.pos[node1]
        p2 = network.pos[node2]
        d = sqrt(sum(pow(p1 - p2, 2)))
        dr = d/node1.commRange

        comm = None
        if dr < (1.0 - self.doi):
                comm = True
        elif dr > (1.0 + self.doi):
            comm =  False
        else:
            p = int(0.5 * self.doi * (dr-1) + 0.5)
            comm = False
            if p > 1:
                comm = True
        #print "****", dr, self.doi, comm
        if comm:
            if (self.environment.are_visible(p1, p2)):
                #assert node1.commRange == node2.commRange
                return True
        return False
