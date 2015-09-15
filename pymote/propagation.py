from pymote.logger import logger
from numpy import sqrt, pi, sin, cos, log10
from numpy.random import normal


"""
 These models are used to predict the received signal power of each packet. At the physical layer of each wireless node,
 there is a receiving threshold. When a packet is received, if its signal power is below the receiving threshold,
 it is marked as error and dropped by the MAC layer.

"""
Propagation_Type = {0: "Free space", 1: "two-ray ground", 2: "Shadowing"}


class PropagationModel(object):

    G_TX = 1.0  # Tx antenna gain
    G_RX = 1.0  # Rx antenna gain
    L = 1.0  # System loss >= 1.0
    C = 3E8  # Speed of light/wave propagation (m/s)
    FREQ = 2.4E9  # 2.4Ghz
    BETA = 2.5  # path loss exponent
    SIGMA_DB = 4  # Gaussian noise standard deviation
    REF_DIST = 1  # Reference Distance

    P_RX_THRESHOLD = -70  # dbm
    MAX_DISTANCE_NO_LOSS = 10  # m

    P_TX = 1.0  # Watts

    def __init__(self, propagation_type=0, node_type=None, **kwargs):
        """
        Initialize the node object.

        node_type: 'N' regular,
        'B' base station/Sink,
        'C' coordinator/cluster head/relay

        """
        self.type = node_type or 'N'
        self.propagation_type = propagation_type

    def __repr__(self):
        return "<Propagation Type=%s>" % (Propagation_Type[self.propagation_type])

    @staticmethod
    def pmw_to_dbm(p_mw):
        return 10 * log10(p_mw)

    @staticmethod
    def pw_to_dbm(p_w):
        return 10 * log10(p_w/1e-3)

    @staticmethod
    def dbm_to_pw(dbm):
        return 1e-3 * pow(10, dbm/10.0)

    def free_space(self, d=1.0, p_tx=None):
        """
        /*
         * Friis free space equation:
         *
         *       Pt * Gt * Gr * (lambda^2)
         *   P = --------------------------
         *       (4 * pi * d)^2 * L
         */
        """
        p_tx = p_tx or PropagationModel.P_TX
        wavelength = self.C/self.FREQ
        m = wavelength/(4 * pi * d)

        p_r_t = p_tx * self.G_RX * self.G_TX * \
                m * m / max(self.L, 1.0)

        return p_r_t

    def free_space_distance(self, p_tx=None, p_rx=1.0):
        """
        Distance based on transmitted and received power
        """
        p_tx = p_tx or PropagationModel.P_TX
        wavelength = self.C/self.FREQ
        d = sqrt((p_tx * self.G_RX * self.G_RX *
                  wavelength * wavelength)/
                 (self.L * p_rx)) / (4 * pi)

        return d

    def cross_over_distance(self,  h_tx=1.0, h_rx=1.0):
        """
        /*
                4 * PI * hr * ht
        d = -----------------------
                lambda
        * At the crossover distance, the received power predicted by the two-ray
        * ground model equals to that predicted by the Friis equation.
        */
        """
        wavelength = self.C/self.FREQ  # lambda

        dc = 4 * pi * h_tx * h_rx / wavelength

        return dc

    def two_ray_ground(self, d=1.0, p_tx=None,
                       h_tx=1.0, h_rx=1.0):
        """
        /*
         *  Two-ray ground reflection model.
         *
         *	     Pt * Gt * Gr * (ht^2 * hr^2)
         *  Pr = ----------------------------
         *           d^4 * L
         */
        """
        p_tx = p_tx or PropagationModel.P_TX
        d = max(d, 1.0)
        m = h_rx * h_tx/(d * d)

        p_r_t = p_tx * self.G_RX * self.G_TX * \
                m * m / max(self.L, 1.0)

        return p_r_t

    def two_ray_ground_distance(self, p_tx=None, p_rx=1.0,
                                h_tx=1.0, h_rx=1.0):
        """
        Distance based on transmitted and received power
        :param p_tx:
        :param p_rx:
        :param h_tx:
        :param h_rx:
        :return: distance in meters
        """
        p_tx = p_tx or PropagationModel.P_TX
        d = sqrt(sqrt(p_tx * self.G_RX * self.G_RX *
                      (h_rx * h_rx * h_tx * h_tx)/p_rx))

        return d

    def shadowing(self, d=1.0, p_tx=None):
        """
        The received power at certain distance is a random variable
        due to multipath propagation effects, which is also known as fading effects.
        The shadowing model consists of two parts: path loss component and
        a Gaussian random variable with zero mean and standard deviation in DB,
        which represent the variation of the received power at certain distance.
        :param d: distance from transmitter (m)
        :param p_tx: transmit power (Watts)
        :return: received power (Watts)
        """
        d0 = PropagationModel.REF_DIST  # ref. distance
        p_tx = p_tx or PropagationModel.P_TX
        pr0 = self.free_space(d=d0, p_tx=p_tx)
        avg_db = -10.0 * self.BETA * log10(d/d0)

        p_loss_db = avg_db + normal(0, self.SIGMA_DB)
        p_r = pr0 * pow(10, p_loss_db/10.0)
        return p_r

    def shadowing_rssi(self, d=1.0, p_tx=None):
        d0 = PropagationModel.REF_DIST
        p_tx = p_tx or PropagationModel.P_TX
        pr0 = self.free_space(d=d0, p_tx=p_tx)
        avg_db = -10.0 * self.BETA * log10(d/d0)

        p_loss_db = avg_db + normal(0, self.SIGMA_DB)
        #print p_loss_db, PropagationModel.pw_to_dbm(pr0)
        rssi = PropagationModel.pw_to_dbm(pr0) + p_loss_db
        return rssi

    def get_power_ratio(self, h_tx=1.0, h_rx=1.0,
                        d=1.0, p_tx=None):
        """
        :param h_tx:
        :param h_rx:
        :param d:
        :return: power ratio, and True if greater than threshold
        """
        p_tx = p_tx or PropagationModel.P_TX
        if d <= self.MAX_DISTANCE_NO_LOSS:
            return p_tx

        if self.propagation_type == 1 and \
           d > self.cross_over_distance(h_tx, h_rx):
            prt = self.two_ray_ground(d, h_tx, h_rx, p_tx)
        elif self.propagation_type == 2:
            prt = self.shadowing(d, p_tx)
        else:
            prt = self.free_space(d, p_tx)

        return prt

    def is_rx_ok(self, h_tx=1.0, h_rx=1.0,
                 d=1.0, p_tx=None, prt=None):
        p_tx = p_tx or PropagationModel.P_TX
        if prt is None:
            prt = self.get_power_ratio(h_tx, h_rx, d, p_tx)
        prt_dbm = PropagationModel.pw_to_dbm(prt)
        return prt_dbm >= self.P_RX_THRESHOLD
