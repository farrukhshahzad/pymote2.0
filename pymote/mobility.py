from pymote.logger import logger
from numpy import sqrt, pi, sin, cos
from numpy.random import rand

Mobility_Type = {0: "Fixed", 1: "Mobile-Uniform Velocity", 2: "Mobile-Uniform Velocity-Random Heading",
                 3: "Mobile-Random"}
MAX_VELOCITY = 50  # m/s


class MobilityModel(object):

    # Velocity
    VELOCITY = 20.0   # m/s
    HEADING = pi/4  # in rad (pi = 180 deg)

    MAX_RANDOM_MOVEMENT = 30  # m

    def __init__(self, mobile_type=0, node_type=None, **kwargs):
        """
        Initialize the node object.

        node_type: 'N' regular, 'B' base station/Sink, 'C' coordinator/cluster head/relay

        """
        self.type = node_type or 'N'
        self.mobile_type = mobile_type
        self.moved_x = 0
        self.moved_y = 0
        self.moved_z = 0

    def __repr__(self):
        return "<Mobility Type=%s>" % (Mobility_Type[self.mobile_type])

    def drift(self, time=1.0):
        x_drift = 0
        y_drift = 0
        z_drift = 0
        velocity = min(self.VELOCITY, MAX_VELOCITY)

        if self.mobile_type == 1:  # S = v * t
            x_drift = velocity * cos(self.HEADING) * time
            y_drift = velocity * sin(self.HEADING) * time
        elif self.mobile_type == 2:  # uniform velocity but random direction
            rnd = rand(2)
            x_drift = velocity * cos(rnd[0]*2*pi) * time * time
            y_drift = velocity * sin(rnd[0]*2*pi) * time * time
        elif self.mobile_type == 3:  # random
            rnd = rand(2)
            x_drift = (rnd[0] - 0.5) * self.MAX_RANDOM_MOVEMENT
            y_drift = (rnd[1] - 0.5) * self.MAX_RANDOM_MOVEMENT

        self.moved_x += x_drift
        self.moved_y += y_drift
        self.moved_z += z_drift

        return x_drift, y_drift, z_drift

    def have_moved(self):
        if abs(self.moved_x) + abs(self.moved_y) + abs(self.moved_z) > 1e-5:
            return True
        return False

    @property
    def mobilityType(self):
        return self.mobile_type

    @mobilityType.setter
    def mobilityType(self, mobile_type):
        self.mobile_type = mobile_type


