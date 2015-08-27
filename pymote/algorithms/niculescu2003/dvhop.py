from pymote.algorithms.niculescu2003.floodingupdate import FloodingUpdate
from numpy import concatenate, array, sqrt, dot


class DVHop(FloodingUpdate):
    """
    Data is {landmark: [x,y,hop_count], ...}
    """

    required_params = ('truePositionKey', 'hopsizeKey')

    def initiator_condition(self, node):
        node.memory[self.truePositionKey] = node.compositeSensor.read().\
                                            get('TruePos', None)
        # true if node is one of the landmarks
        return node.memory[self.truePositionKey] is not None

    def initiator_data(self, node):
        return {node:
                concatenate((node.memory[self.truePositionKey][:2], [1]))}

    def handle_flood_message(self, node, message):
        if not self.dataKey in node.memory:
            node.memory[self.dataKey] = {}
        updated_data = {}
        for landmark, landmark_data in message.data.items():
            # skip if landmark in message data is current node
            if landmark == node:
                continue
            # update only if this is first received data from landmark or new
            # hopcount is smaller than previous minimum
            if not landmark in node.memory[self.dataKey] or \
                   landmark_data[2] < node.memory[self.dataKey][landmark][2]:
                node.memory[self.dataKey][landmark] = array(landmark_data)
                # increase hopcount
                landmark_data[2] += 1
                updated_data[landmark] = landmark_data

        # if node is one of the landmarks then it should recalculate hopsize
        if node.memory[self.truePositionKey] is not None:
            self.recalculate_hopsize(node)

        return updated_data

    def recalculate_hopsize(self, node):
        pos = node.memory[self.truePositionKey]
        try:
            landmarks_count = len(node.memory[self.dataKey])
        except KeyError:
            pass
        else:
            dist = lambda x, y: sqrt(dot(x - y, x - y))
            if landmarks_count > 0:
                node.memory[self.hopsizeKey] = \
                    sum([dist(lp[:2], pos)
                         for lp in node.memory[self.dataKey].values()]) / \
                    sum([lp[2] for lp in node.memory[self.dataKey].values()])
