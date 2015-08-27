import unittest
from pymote.networkgenerator import NetworkGenerator
from pymote.algorithm import NodeAlgorithm, NetworkAlgorithm, \
        PymoteAlgorithmException
from pymote.network import PymoteNetworkError
from pymote import Node


def set_algorithms(net, algorithms):
    net.algorithms = algorithms


class SomeNodeAlgorithm(NodeAlgorithm):
    required_params = ('rp1',
                       'rp2',
                       'rp3')
    default_params = {'dp1': 'dp1_value',
                      'dp2': 'dp2_value',
                      'dp3': 'dp3_value'}


class SomeNetworkAlgorithm(NetworkAlgorithm):
    default_params = {'dp1': 'dp1_value',
                      'dp2': 'dp2_value',
                      'dp3': 'dp3_value'}


class SomeAlgorithmWithInheritance(SomeNodeAlgorithm):
    required_params = ('rp4',)
    default_params = {'dp4': 'dp4_value', }


class SomeAlgorithmWithInheritanceChild(SomeAlgorithmWithInheritance):
    required_params = ('rp5', 'rp6')
    default_params = {'dp2': 'overriden_dp2_value', }


def rp_multiple():
    class SomeAlgorithmWhereRpIsRedefined(SomeAlgorithmWithInheritance):
        required_params = ('rp2',)


def dp_is_rp():
    class SomeAlgorithmWhereDpIsInheritedRp(SomeAlgorithmWithInheritance):
        default_params = {'rp2': 'dp2_value', }


def rp_is_dp():
    class SomeAlgorithmWhereRpIsInheritedDp(SomeAlgorithmWithInheritance):
        required_params = ('dp2',)


class TestAlgorithmsSetter(unittest.TestCase):

    def setUp(self):
        net_gen = NetworkGenerator(100)
        self.net = net_gen.generate_random_network()
        self.algorithms_ok = ((SomeNodeAlgorithm,
                              {'rp1': 1, 'rp2': 2, 'rp3': 3, }),

                              (SomeNetworkAlgorithm,
                              {}),

                              SomeNetworkAlgorithm,

                              (SomeAlgorithmWithInheritance,
                               {'rp1': 1, 'rp2': 2, 'rp3': 3, 'rp4': 4}),

                              (SomeAlgorithmWithInheritanceChild,
                               {'rp1': 1, 'rp2': 2, 'rp3': 3, 'rp4': 4,
                                'rp5': 5, 'rp6': 6}),
                              )
        self.check = [
                # wrong_format
                (PymoteNetworkError, [(SomeNodeAlgorithm,
                                       {'rp1': 1, 'rp2': 2, 'rp3': 3}), ]),
                # wrong_base_class
                (PymoteNetworkError, ((Node, {}),)),
                # missing_req_params
                (PymoteAlgorithmException, ((SomeNodeAlgorithm,
                                             {'rp1': 1, }),)),
                (PymoteAlgorithmException, ((SomeAlgorithmWithInheritance,
                                             {'rp1': 1, }),)),
                ]

    def test_setter(self):
        """Test different algorithm initialization formats and params."""
        set_algorithms(self.net, self.algorithms_ok)
        for exc, alg in self.check:
            self.assertRaises(exc, set_algorithms, self.net, alg)

    def test_params_inheritance(self):
        """Test default params inheritance algorithm classes."""
        self.net.algorithms = ((SomeAlgorithmWithInheritanceChild,
                                {'rp1': 1, 'rp2': 2, 'rp3': 3, 'rp4': 4,
                                'rp5': 5, 'rp6': 6}),)
        self.assertTrue(self.net.algorithms[0].dp1 == 'dp1_value')
        self.assertTrue(self.net.algorithms[0].dp2 == 'overriden_dp2_value')
        self.assertTrue(self.net.algorithms[0].dp3 == 'dp3_value')
        self.assertRaises(AssertionError, rp_multiple)
        self.assertRaises(AssertionError, dp_is_rp)
        self.assertRaises(AssertionError, rp_is_dp)

    def test_default_params(self):
        """Test default params."""
        self.net.algorithms = ((SomeNetworkAlgorithm,
                                {'dp2': 'overriden_dp2_value', }),
                               )
        self.assertTrue(self.net.algorithms[0].dp1 == 'dp1_value')
        self.assertTrue(self.net.algorithms[0].dp2 == 'overriden_dp2_value')
        self.assertTrue(self.net.algorithms[0].dp3 == 'dp3_value')

