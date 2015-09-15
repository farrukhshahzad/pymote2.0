from PySide.QtCore import QThread, SIGNAL
import logging
import datetime
from pymote.network import Network
from pymote.algorithm import NetworkAlgorithm
from pymote.algorithm import NodeAlgorithm


class Simulation(QThread):
    """ Controls single network algorithm and node algorithms simulation.
        It is responsible for visualization and logging, also. """

    def __init__(self, network, logLevel=None, **kwargs):
        assert(isinstance(network, Network))
        self._network = network
        self.sim_start = self.sim_end = None
        self.stepsLeft = 0
        self.logger = logging.getLogger('pymote.simulation')
        self.logger.level = logLevel or logging.DEBUG
        self.logger.debug('Simulation %s created successfully!.' %
                          (hex(id(self))))
        QThread.__init__(self)

    def __del__(self):
        self.exiting = True
        self.wait()

    def run_all(self, stepping=False):
        """ Run simulation form beginning. """
        self.reset()
        self.sim_start = datetime.datetime.utcnow()
        self.logger.info('Simulation %s starts running.' % hex(id(self)))
        if stepping:
            self.run(1)
            self.logger.info('Simulation pause. Use sim.run(n) to continue n '
                             'steps or sim.run() to continue without '
                             'stepping.')
        else:
            self.run()
            self.sim_end = datetime.datetime.utcnow()
            self.logger.info('Simulation end.')

    def run(self, steps=0):
        """
        Run simulation from current state.

        If steps = 0 it runs until all algorithms are finished.
        If steps > 0 simulation is in stepping mode.
        If steps > number of steps to finish current algorithm it finishes it.
        """
        self.sim_start = datetime.datetime.utcnow()
        self.stepsLeft = steps
        while True:
            algorithm = self.network.get_current_algorithm()
            if not algorithm:
                self.sim_end = datetime.datetime.utcnow()
                self.logger.info('Simulation has finished. There are no '
                                 'algorithms left to run. '
                                 'To run it from the start use sim.reset().')
                self.emit(SIGNAL('redraw()'))
                break
            self.run_algorithm(algorithm)
            self.emit(SIGNAL('redraw()'))
            if self.stepsLeft >= 0:
                break

    def run_algorithm(self, algorithm):
        """
        Run given algorithm on given network.

        Update stepsLeft and network.algorithmState['step'].
        If stepsLeft hit 0 it may return unfinished.
        """
        if isinstance(algorithm, NetworkAlgorithm):
            self.stepsLeft -= 1
            algorithm.run()
        elif isinstance(algorithm, NodeAlgorithm):
            if self.network.algorithmState['step'] == 1:
                algorithm.initializer()
                if not self.network.outbox:
                    self.logger.warning('Initializer didn\'t send INI message')
            while not self.is_halted():
                self.stepsLeft -= 1
                self.network.communicate()
                for node in self.network.nodes():
                    nodeTerminated = algorithm.step(node)
                self.emit(SIGNAL('updateLog(QString)'),
                          '[%s] Step %d finished' %
                          (algorithm.name,
                           self.network.algorithmState['step']))
                self.logger.debug('[%s] Step %d finished' %
                                  (algorithm.name,
                                   self.network.algorithmState['step']))
                self.network.algorithmState['step'] += 1
                if nodeTerminated:
                    break
                if self.stepsLeft == 0:
                    return  # not finished
        self.emit(SIGNAL('updateLog(QString)'), '[%s] Algorithm finished' %
                                                      (algorithm.name))
        self.logger.debug('[%s] Algorithm finished' % (algorithm.name))
        self.network.algorithmState['finished'] = True
        return

    def run_step(self):
        self.run(1)

    def reset(self):
        self.logger.info('Resetting simulation.')
        self._network.reset()

    def is_halted(self):
        """ Check if distributed algorithm have come to end or deadlock
            i.e. no messages to pass. """
        if len(self._network.outbox) > 0 or \
           any([len(node.outbox) for node in self.network.nodes()]) or \
           any([len(node.inbox) for node in self.network.nodes()]):
            return False
        else:
            return True

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, network):
        self._network.simulation = None
        self._network = network
        self._network.simulation = self
        self.emit(SIGNAL('updateLog(QString)'), 'Network loaded')
        self.emit(SIGNAL('redraw()'))
