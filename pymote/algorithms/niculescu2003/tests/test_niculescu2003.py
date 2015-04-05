import unittest
from pymote.algorithms.niculescu2003.trilaterate import Trilaterate
from pymote.simulation import Simulation
from pymote.sensor import TruePosSensor
from pymote.networkgenerator import NetworkGenerator
from pymote.algorithms.niculescu2003.dvhop import DVHop
from pymote import *

class TestNiculescu2003(unittest.TestCase):

    def setUp(self):
        net_gen = NetworkGenerator(100)
        self.net = net_gen.generate_random_network()
        self.net.algorithms = ((DVHop, {'truePositionKey': 'tp',
                                  'hopsizeKey': 'hs',
                                  'dataKey': 'I'
                                  }),
                          (Trilaterate, {'truePositionKey': 'tp',
                                        'hopsizeKey': 'hs',
                                        'positionKey': 'pos',
                                        'dataKey': 'I'}),
                          )
        for node in self.net.nodes()[:10]:
            node.compositeSensor = (TruePosSensor,)

    def test_niculescu2003_sim(self):
        """Test niculescu2003 default simulation."""
        sim = Simulation(self.net)
        sim.run()
        for node in self.net.nodes():
            self.assertTrue(len(node.memory.get('pos', [None, None])) == 2\
                            or 'tp' in node.memory)

    def test_sim(self):
        netgen = NetworkGenerator(degree=9, n_min=100, n_max=300)
        for lm_pct in [5, 10, 20, 33]:
            for net_count in range(100):
                net = netgen.generate()
                for node in net.nodes()[:int(lm_pct * len(net.nodes())/100)]:
                    node.compositeSensor = CompositeSensor(('TruePosSensor'))
            net.algorithms = ALGORITHMS
            sim = Simulation(net)
            sim.run()
            write_npickle(net, '%d-%d.gz' % (net_count,lm_pct))
