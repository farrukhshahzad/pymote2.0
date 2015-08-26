import unittest
from pymote.network import Network
from pymote.node import Node
from pymote.environment import Environment2D
from pymote.channeltype import ChannelType
from pymote.conf import settings


class TestNetworkCreation(unittest.TestCase):

    def setUp(self):
        self.net = Network()
        self.net.environment.im[22, 22] = 0
        self.node1 = self.net.add_node(pos=[22.8, 21.8])
        self.node2 = self.net.add_node(pos=[21.9, 22.9])
        self.node3 = self.net.add_node(pos=[21.7, 21.7])

    def test_nodes(self):
        """Make sure the nodes are added."""
        self.assertTrue(isinstance(self.node1, Node))
        self.assertEqual(len(self.net.node), 3)
        if (isinstance(self.net.environment, Environment2D)):
            self.assertEqual(self.net.environment.im.shape,
                             settings.ENVIRONMENT2D_SHAPE,
                             'incorrect default size')
        self.assertTrue(isinstance(self.net.channelType, ChannelType))

    def test_visibility(self):
        """
        Pixel 22,22 is not space so node1 and node2 should not be visible
        but node3 is visible.
        """
        self.assertFalse(self.net.environment\
                             .are_visible(self.net.pos[self.node1],
                                          self.net.pos[self.node2]))
        self.assertTrue(self.net.environment\
                            .are_visible(self.net.pos[self.node2],
                                         self.net.pos[self.node3]))
