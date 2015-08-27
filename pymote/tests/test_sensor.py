import unittest
from pymote.node import Node
from pymote.network import Network
from pymote.sensor import NeighborsSensor, DistSensor
import scipy.stats


class TestSensor(unittest.TestCase):

    def test_node_without_network(self):
        """Test if node without network raises exception on read sensor."""
        node = Node()
        self.assertRaises(Exception, node.compositeSensor.read)

    def test_read(self):
        """Test read compositeSensor"""
        net = Network()
        node = net.add_node()
        node.compositeSensor.read()

    def test_set_compositeSensor(self):
        """Test setting compositeSensors on a node"""
        net = Network()
        node = net.add_node()
        dist_sensor = DistSensor({'pf': scipy.stats.norm, 'scale': 10})
        node.compositeSensor = (NeighborsSensor, 'AoASensor', dist_sensor)
        self.assertTrue(len(node.compositeSensor.sensors) == 3)
        readings = node.compositeSensor.read()
        self.assertTrue('Neighbors' in readings.keys() and
                        'AoA' in readings.keys() and
                        'Dist' in readings.keys())

        #TODO: check normal distribution
