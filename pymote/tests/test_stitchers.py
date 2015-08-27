import unittest
from numpy import array
from pymote.utils.localization.aoastitcher import AoAStitcher
from pymote.utils.localization.diststitcher import DistStitcher
from pymote.utils.localization.helpers import get_rms
from numpy.core.numeric import dot, concatenate, nan
from numpy.ma.core import cos, sin
from scipy.constants.constants import pi
from numpy.testing.utils import rand
from pymote.utils.memory.positions import Positions


class TestStitchers(unittest.TestCase):

    def setUp(self):
        # stitch_test.cdr
        self.test_cluster = {0: array([0., 0., nan]),
                             1: array([3., -2., nan]),
                             2: array([6., 2., nan]),
                             3: array([2., 5., nan]),
                             4: array([-2., 4., nan]),
                             5: array([2., -5., nan]),
                             6: array([7., -3., nan])
                             }

    def test_aoa_stitcher(self):
        src = self.setup_positions([[0, 1, 2, 3, 4]])
        dst = self.setup_positions([[0, 1, 2, 5, 6]])

        #theta = 60/180.*pi
        #R = array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])  #ccw
        #self.transform(src, R, [-1.62, -0.8], 0.5)  #transform src positions
        self.transform(src)

        stitcher = AoAStitcher()
        stitcher.stitch(dst, src)

        rms = get_rms(self.test_cluster, dst)
        self.assertTrue(rms<0.001, 'Stitching was not good enough.')
        self.assertTrue(len(dst)==1, 'Wrong number of subclusters.')

    def test_dist_stitcher(self):
        src = self.setup_positions([[0, 1, 2, 3, 4]])
        dst = self.setup_positions([[0, 1, 2, 5, 6]])

        self.transform(src, s_i=1)

        stitcher = DistStitcher()
        stitcher.stitch(dst, src)

        self.assertTrue(len(dst)==1, 'Wrong number of subclusters.')
        rms = get_rms(self.test_cluster, dst)
        self.assertTrue(rms<0.001, 'Stitching was not good enough.')

    def test_dist_stitcher2(self):
        """ Test dist stitch with only 2 common nodes - not enough. """
        src = self.setup_positions([[0, 1, 3, 4]])
        dst = self.setup_positions([[0, 1, 2, 5, 6]])
        self.transform(src, s_i=1)  #transform src positions

        stitcher = DistStitcher()
        stitcher.stitch(dst, src)

        self.assertTrue(len(dst)==2, 'Wrong number of subclusters.')
        rms = get_rms(self.test_cluster, dst)
        self.assertFalse(rms<0.001, 'Stitching was too good.')
        # with align rms should be ok
        rms_align = get_rms(self.test_cluster, dst, align=True)
        self.assertTrue(rms_align<0.001, 'Stitching was not good enough.')

    def test_dist_stitcher3(self):
        """ Test dist stitch with only 1 common node - not enough. """
        src = self.setup_positions([[0, 3, 4]])
        dst = self.setup_positions([[0, 1, 2, 5, 6]])
        self.transform(src, s_i=1)  #transform src positions

        stitcher = DistStitcher()
        stitcher.stitch(dst, src)

        self.assertTrue(len(dst)==2, 'Wrong number of subclusters.')
        # without align new subcluster should be wrongly (not) rotated
        rms = get_rms(self.test_cluster, dst)
        self.assertFalse(rms<0.001, 'Stitching was too good.')
        # with align rms should be ok
        rms_align = get_rms(self.test_cluster, dst, align=True)
        self.assertTrue(rms_align<0.001, 'Stitching was not good enough.')

    def test_dist_intrastitch(self):
        """
        Test dist intrastitch

        Clusters are setup so that first stitch is src[0] with dst[1] and
        then dst[0] and dst[1] can be intrastitched.

        """
        src = self.setup_positions([[1, 2, 4, 5, 6]])
        dst = self.setup_positions([[0, 2, 3, 4], [0, 1, 2, 6]])
        self.transform(src, s_i=1)  #transform src positions
        self.transform(dst, s_i=1)  #transform src positions

        stitcher = DistStitcher()
        stitcher.stitch(dst, src)

        self.assertTrue(len(dst)==1, 'Wrong number of subclusters.')
        # without align new subcluster should be wrongly (not) rotated
        rms = get_rms(self.test_cluster, dst)
        self.assertFalse(rms<0.001, 'Stitching was too good.')
        # with align rms should be ok
        rms_align = get_rms(self.test_cluster, dst, align=True)
        self.assertTrue(rms_align<0.001, 'Stitching was not good enough.')

    def setup_positions(self, subclusters):
        """ Return new cluster with nodes form main_cluster and with
            subclusters defined as in subcluster argument. """

        positions = [{k: v for k, v in self.test_cluster.items()
                      if k in sub}
                     for sub in subclusters]
        return Positions(positions)

    def transform(self, positions, R_i=None, t_i=None, s_i=None, flip=False):
        """
        Return subclusters with (randomly) rotated translated and scaled
        positions. If R_i, s_i or t_i is given then that part of transformation
        is not random.

        """

        for sub in positions:
            t = t_i or rand(2)*10
            s = s_i or rand()*2
            if R_i is None:
                th = 2*pi*rand()
                # ccw
                R = array([[cos(th), -sin(th)], [sin(th), cos(th)]])
            else:
                R = R_i
            if flip:
                #TODO: make R with flip
                pass

            for node, pos in sub.items():
                sub[node] = concatenate((dot(dot(s, R), pos[:2])+t, [nan]))


suite = unittest.TestLoader().loadTestsFromTestCase(TestStitchers)
unittest.TextTestRunner(verbosity=2).run(suite)
