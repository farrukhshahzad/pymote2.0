from pymote.algorithms.ral2009.floodingupdate import FloodingUpdate
from pymote.logger import logger
from numpy import array, sqrt, average, dot, diag, ones
from numpy import linalg


class Trilaterate(FloodingUpdate):

    required_params = (
                       # key in memory for true position data (only landmarks)
                       'truePositionKey',
                       # key in memory for storing estimated position
                       'positionKey',
                       # key in memory for storing hopsize data
                       'hopsizeKey',
                       )

    def initiator_condition(self, node):
        return node.memory[self.truePositionKey] is not None  # if landmark

    def initiator_data(self, node):
        return node.memory.get(self.hopsizeKey)

    def handle_flood_message(self, node, message):
        if self.hopsizeKey in node.memory:
            return None
        node.memory[self.hopsizeKey] = message.data
        self.estimate_position(node)
        return node.memory[self.hopsizeKey]

    def estimate_position(self, node):
        TRESHOLD = .1
        MAX_ITER = 10
        landmarks = []

        # get landmarks with hopsize data
        if self.dataKey in node.memory:
            landmarks = node.memory[self.dataKey].keys()
        # calculate estimated distances
        if len(landmarks) >= 3:
            dist = lambda x, y: sqrt(dot(x - y, x - y))
            landmark_max_positions = [array(node.memory[self.dataKey][lm][:2])
                                  for lm in landmarks]
            # take centroid as initial estimation
            pos = average(landmark_max_positions, axis=0)
            landmark_distances = []
            landmark_positions = []
            # only reliable anchors
            for lp in node.memory[self.dataKey].values():
                 threshold = FloodingUpdate.lookup.get(lp[2], 0.75) * node.commRange
                 hl = dist(lp[:2], pos)/lp[2]
                 logger.debug("Node=%s, Hop=%s, threshold=%s, hoplen=%s" %(node.id, lp[2], threshold, hl))
                 if hl > threshold and self.hopsizeKey in node.memory:  # reliable
                    landmark_distances.append(lp[2] * (node.memory[self.hopsizeKey] or 1))
                    landmark_positions.append(array(lp[:2]))

            # take centroid as initial estimation
            W = diag(ones(len(landmark_positions)))
            counter = 0
            while True:
                J = array([(lp - pos) / dist(lp, pos)
                           for lp in landmark_positions])
                range_correction = array([dist(landmark_positions[li], pos) -
                                          landmark_distances[li]
                                          for li, lm in enumerate(landmark_positions)])
                pos_correction = dot(linalg.inv(dot(dot(J.T, W), J)),
                                     dot(dot(J.T, W), range_correction))
                logger.debug("Est. %s, %s, %s" %(node.id, pos, pos_correction))
                pos = pos + pos_correction
                counter += 1
                if sqrt(sum(pos_correction ** 2)) < \
                   TRESHOLD or counter >= MAX_ITER:
                    logger.info("Trilaterate break %s" % counter)
                    break
            if counter <= MAX_ITER:
                node.memory[self.positionKey] = pos
                node.memory['reliable'] = landmark_positions
