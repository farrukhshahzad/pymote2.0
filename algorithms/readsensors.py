from pymote.algorithm import NetworkAlgorithm


class ReadSensors(NetworkAlgorithm):
    """Read all sensors and save data in memory for latter usage."""
    required_params = ()
    default_params = {'sensorReadingsKey': 'sensorReadings'}

    def run(self):
        for node in self.network.nodes():
            node.memory.update({self.sensorReadingsKey:
                                node.compositeSensor.read()})
