"""
Sensors provide a way for node to interact with its environment.

Sensors can also be used to satisfy algorithm prerequisites. For example
if algorithm depends on the assumption that all nodes know who their neighbors
are then nodes should be equipped with :class:`NeighborsSensor`.

Generally sensors should incorporate some model of measurement insecurity that
is inherent in real world sensors. This is implemented as a
:class:`ProbabilityFunction`.

Basic usage:

>>> node.compositeSensor = ('NeighborsSensor','AoASensor')
>>> node.compositeSensor.sensors
(<pymote.sensor.NeighborsSensor at 0x6d3fbb0>,
 <pymote.sensor.AoASensor at 0x6d3f950>)

To manually set sensor parameters first make an sensor instance:

>>> import scipy.stats
>>> aoa_sensor = AoASensor({'pf': scipy.stats.norm, 'scale': 10*pi/180 })
>>> node.compositeSensor = (aoa_sensor,)

"""

from pymote.conf import settings
from numpy import arctan2, pi, sqrt
import inspect


class Sensor(object):

    """
    Abstract base class for all Sensors.

    Sensor provides a certain capability for a node, information about the
    outside world. It could could be a capability to detect neighbors, distance
    to them or to get the environment temperature.

    """

    pf_settings_key = ''

    def __init__(self, pf_params={}):
        pf_params_final = getattr(settings, self.pf_settings_key, {})
        pf_params_final.update(pf_params)
        if pf_params_final:
            self.probabilityFunction = ProbabilityFunction(**pf_params_final)
        else:
            self.probabilityFunction = None

    def name(self):
        return self.__class__.__name__

    def read(self):
        """This method should be overriden in subclass."""
        pass


def node_in_network(fun):
    """Decorator function that checks if node is in network."""
    def f(sensor, node):
        if not node.network:
            raise Exception('Cannot take a sensor reading if node is'
                            ' outside of a network.')
        return fun(sensor, node)
    return f


class NeighborsSensor(Sensor):

    """Provides list of node's neighbors."""

    @node_in_network
    def read(self, node):
        return {'Neighbors': node.network.neighbors(node)}


class AoASensor(Sensor):

    """Provides azimuth between node and its neighbors."""

    pf_settings_key = 'AOA_PF_PARAMS'

    @node_in_network
    def read(self, node):
        network = node.network
        measurements = {}
        p = network.pos[node]
        o = network.ori[node]
        for neighbor in network.neighbors(node):
            v = network.pos[neighbor] - p
            measurement = (arctan2(v[1], v[0]) - o) % (2 * pi)
            measurement = self.probabilityFunction.getNoisyReading(measurement)
            measurements.update({neighbor: measurement})
        return {'AoA': measurements}


class DistSensor(Sensor):

    """Provides distance between node and its neighbors."""

    pf_settings_key = 'DIST_PF_PARAMS'

    @node_in_network
    def read(self, node):
        network = node.network
        measurements = {}
        p = network.pos[node]
        for neighbor in network.neighbors(node):
            pn = network.pos[neighbor]
            measurement = sqrt(sum(pow(p - pn, 2)))
            measurement = self.probabilityFunction.getNoisyReading(measurement)
            measurements.update({neighbor: measurement})
        return {'Dist': measurements}


class TruePosSensor(Sensor):

    """Provides node's true position."""

    @node_in_network
    def read(self, node):
        return {'TruePos': node.network.pos[node]}


class CompositeSensor(object):

    """
    Wrap multiple sensors, coalesce results and return composite readout.

    This class is not a sensor itself, i.e. subclass of :class:`Sensor`,
    instead it serves as a placeholder for multiple sensors that can be
    attached to a :class:`Node`.

    """

    def __init__(self, node, componentSensors=None):
        """
        Arguments:
            node (:class:`Node`):
                Node that has this composite sensor is attached to.
            componentSensors (tuple):
                Tuple of :class:`Sensor` subclasses or their class names.

        """
        self.node = node
        self._sensors = ()
        self.sensors = componentSensors or ()

    @property
    def sensors(self):
        return self._sensors

    @sensors.setter
    def sensors(self, sensors):
        self._sensors = ()
        # instantiate sensors passed by class name
        for cls in Sensor.__subclasses__():
            if (cls.__name__ in sensors):
                self._sensors += cls(),
        # instantiate sensors passed by class
        for cls in sensors:
            if inspect.isclass(cls) and issubclass(cls, Sensor):
                self._sensors += cls(),
        # add sensors that are already instantiated
        for sensor in sensors:
            if isinstance(sensor, Sensor):
                self._sensors += sensor,

    def get_sensor(self, name):
        sensor = [s for s in self._sensors if s.name()==name]
        if len(sensor)!=1:
            raise Exception("Multiple or no sensors found with name %s"
                            % name)
        return sensor[0]

    def read(self):
        measurements = {}
        for sensor in self._sensors:
            measurements.update(sensor.read(self.node))
        return measurements


class ProbabilityFunction(object):

    """Provides a way to get noisy reading."""

    def __init__(self, scale, pf):
        """
        pf: probability function (i.e. :py:data:`scipy.stats.norm`)
        scale: pf parameter
        """
        self.pf = pf  # class or gen object
        self.name = self.pf.__class__.__name__
        self.scale = scale

    def getNoisyReading(self, value):
        return self.pf.rvs(scale=self.scale, loc=value)
