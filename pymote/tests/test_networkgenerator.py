#@PydevCodeAnalysisIgnore
import unittest
from numpy.core.numeric import Inf
from pymote.networkgenerator import NetworkGenerator, NetworkGeneratorException
from pymote.conf import settings
from pymote.environment import Environment2D
from pymote.channeltype import Udg
from pymote.sensor import NeighborsSensor
from pymote.algorithms.readsensors import ReadSensors
from inspect import isclass


class TestNetworkGeneration(unittest.TestCase):

    def setUp(self):
        # Raises NetworkGeneratorException
        # returns None
        # else expected network/node properties dictionary
        env = Environment2D(shape=(600,600))
        channelType = Udg(env)
        algorithms = (ReadSensors,)
        sensors = (NeighborsSensor,)
        self.in_out = [
                  # default N_COUNT and COMM_RANGE and ENVIRONMENT should be compatible
                  ({"n_count": None, "n_min": 0, "n_max": Inf, "connected": True, "environment": None, "degree": None, "comm_range": None}, {'count': range(100,1001)}),  
                  # regular default params
                  ({"n_count": 100, "n_min": 0, "n_max": Inf,  "connected": True, "environment": env,  "degree": None, "comm_range": 100},  {'count': range(100,1001)}),
                  
                  ############## connected True degree False
                  # increase node number
                  ({"n_count": 10, "n_min": 0, "n_max": Inf, "connected": True, "environment": env, "degree": None, "comm_range": 100},  {'count': range(11,301)}),
                  # increase commRange
                  ({"n_count": 10, "n_min": 0, "n_max": 10,  "connected": True, "environment": env, "degree": None, "comm_range": None}, {'count': 10}),
                  # decrease commRange
                  ({"n_count": 10, "n_min": 10, "n_max": 10,  "connected": True, "environment": env, "degree": 0, "comm_range": None}, {'count': 10}),
                  
                  ############## connected True degree True
                  # increase node number
                  ({"n_count": 10, "n_min": 0, "n_max": 200, "connected": True, "environment": env, "degree": 11, "comm_range": 100},  {'count': range(10,201)}),
                  # increase commRange
                  ({"n_count": 10, "n_min": 0, "n_max": 10,  "connected": True, "environment": env, "degree": 9,  "comm_range": None}, {'count': 10}),
                  # low degree with connected, alternating directions problem
                  ({"n_count": 10, "n_min": 0, "n_max": 10,  "connected": True, "environment": env, "degree": 3,  "comm_range": 30},   None),
                  
                  ############## connected False degree True
                  # increase node number
                  ({"n_count": 10, "n_min": 0, "n_max": 200, "connected": False, "environment": env, "degree": 8,  "comm_range": 100},   {'count': range(10,201)}),
                  # increase commRange
                  ({"n_count": 10, "n_min": 0, "n_max": 200, "connected": False, "environment": env, "degree": 11, "comm_range": None},  {'count': range(10,201)}),

                  # low degree 
                  ({"n_count": 10, "n_min": 0, "n_max": 100, "connected": False, "environment": env, "degree": 3,  "comm_range": 100},    {'count': range(10,101)}),
                  # degree too high for node number
                  ({"n_count": 10, "n_min": 0, "n_max": 10,  "connected": False, "environment": env, "degree": 10,   "comm_range": None}, NetworkGeneratorException),
                  ({"n_count": 11, "n_min": 0, "n_max": 10,  "connected": False, "environment": env, "degree": None, "comm_range": None}, NetworkGeneratorException),
                  ({"n_count": 9, "n_min": 10, "n_max": 10,  "connected": False, "environment": env, "degree": None, "comm_range": None}, NetworkGeneratorException),
                  
                  ############## connected False degree False - no need for modifying initially created network
                  # also remove environment from kwargs to test default and change comm_range to commRange 
                  ({"n_count": 10, "n_min": 0, "n_max": 100, "connected": False, "degree": None,  "commRange": 100},  {'count': 10}),
                  ({"n_count": 20, "n_min": 0, "n_max": 100, "connected": False, "degree": None,  "commRange": None}, {'count': 20}),
                  ({"n_count": 30, "n_min": 0, "n_max": 100, "connected": False, "degree": None,  "commRange": 30},   {'count': 30}),

                  ############## Check sensors and algorithms
                  ({"n_count": 10, "n_min": 0, "n_max": 100, "connected": False, "channelType": channelType, "algorithms": algorithms, "commRange": 100, "sensors": sensors}, 
                   {'count': 10, "channelType": channelType, "algorithms": algorithms, "commRange": 100, "sensors": sensors}),
              ]
        

    def test_random_generation(self):
        """Test different random generation parameters"""
        for input, output in self.in_out:
            if isclass(output) and issubclass(output, Exception):
                self.assertRaises(output, NetworkGenerator, input)
                continue
            net_gen = NetworkGenerator(**input)
            if output==None:
                self.assertEqual(None, net_gen.generate_random_network())
            elif isinstance(output, dict):
                net = net_gen.generate_random_network()
                try:
                    net.validate_params(output)
                except AssertionError:
                    self.fail("Network params did not validate.")
