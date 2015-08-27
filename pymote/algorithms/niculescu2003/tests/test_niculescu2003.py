import unittest
from numpy import array, sqrt, dot

from pymote.algorithms.niculescu2003.trilaterate import Trilaterate
from pymote.simulation import Simulation
from pymote.sensor import TruePosSensor
from pymote.networkgenerator import NetworkGenerator
from pymote.algorithms.niculescu2003.dvhop import DVHop
from pymote import *
from pymote.utils.filing import get_path, DATETIME_DIR

class TestNiculescu2003(unittest.TestCase):

    def setUp(self):
        self.n = 50;
        net_gen = NetworkGenerator(self.n)
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
            node.type = 'C'  # Anchors

    def test_niculescu2003_sim(self):
        """Test niculescu2003 default simulation."""
        sim = Simulation(self.net)
        sim.run()
        i=0;
        for node in self.net.nodes():
            i += 1
            self.assertTrue(len(node.memory.get('pos', [None, None])) == 2\
                            or 'tp' in node.memory)

        self.net.savefig(fname=get_path(DATETIME_DIR, "dv-%s" %self.n),
                title="DV hop Default Simulation - %s nodes" % self.n,
                x_label="X", y_label="Y")

        print i, self.net.__len__()
        print self.net.nodes()[1].get_dic()
        print self.net.nodes()[20].get_dic()
        #print self.net.get_dic()

        print self.net.nodes()[0].type, \
            self.net.nodes()[0].get_dic()['Info']['position'],\
            self.net.nodes()[0].memory['tp']

        for node in self.net.nodes():
            act = node.get_dic()['Info']['position']
            est = node.memory['pos']
            if node.type=='C':
                err = sqrt(sum(pow(act - est, 2)))
                print node.type, act, est, err


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
