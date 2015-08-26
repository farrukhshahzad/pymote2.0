from numpy import dot, arctan2, sqrt, array, sin, cos
from numpy.linalg import det
from pymote.utils.localization.stitchsubclusterselectors import \
            MaxCommonNodeSelector, StitchSubclusterSelectorBase
from pymote.utils.localization.basestitcher import BaseStitcher


class AoAStitcher(BaseStitcher):
    """
    Base class implementing subcluster stitching methods based on degrees of
    freedom given by localization using angle of arrival measurements i.e.
    translation, rotation/reflection and scaling.

    Reliable stitch with R, s and t that are not None is based on two or more
    common nodes between them.

    If rotation matrix includes reflection it is not reliable so return Nones.

    """

    def __new__(cls, *args, **kwargs):
        """ Legacy: by default returns AoAStitcherHorn instance. """
        if cls is not AoAStitcher:
            return super(AoAStitcher, cls).__new__(cls)
        return AoAStitcherHorn()

    def __init__(self, selector=None, **kwargs):
        self.selector = selector or MaxCommonNodeSelector(cn_count_treshold=2)
        assert(isinstance(self.selector, StitchSubclusterSelectorBase))

    def _get_scaling_factor(self, commonNodes, dstSubPos, srcSubPos, p_d, p_s,
                            w_d):
        s = sqrt(sum([sum((dstSubPos[cn][:2] - p_d) ** 2) * w_d[cn]
              for cn in commonNodes]) /
         sum([sum((srcSubPos[cn][:2] - p_s) ** 2) * w_d[cn]
              for cn in commonNodes]))
        return s

    def _get_rotation_matrix_2_common_nodes(self, commonNodes, dstSubPos,
                                            srcSubPos):
        """
        Calculates rotation matrix based only on two common nodes

        Warning: should not be used when local systems could be
        reflected i.e. when they are formed by distance measurements.

        """
        vector1 = dstSubPos[commonNodes[0]] - dstSubPos[commonNodes[1]]
        vector2 = srcSubPos[commonNodes[0]] - srcSubPos[commonNodes[1]]
        theta1 = arctan2(vector1[1], vector1[0])
        theta2 = arctan2(vector2[1], vector2[0])
        theta = theta1 - theta2
        # in ccw direction
        R = array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
        return R


class AoAStitcherHorn(AoAStitcher):

    def __init__(self, *args, **kwargs):
        """
        ``reflectable`` controls whether reflection is accepted when
        calculating rotation matrix.
        """
        self.reflectable = kwargs.get('reflectable', False)
        self.get_rotation_matrix = self._get_rotation_matrix_horn
        super(AoAStitcherHorn, self).__init__(*args, **kwargs)

    def stitch_subclusters(self, dstSubPos, srcSubPos):
        commonNodes = self._get_common_nodes(dstSubPos, srcSubPos)
        assert len(commonNodes) > 1
        (p_s, p_d, w_d) = self._get_centroids(commonNodes, dstSubPos,
                                              srcSubPos)
        s = self._get_scaling_factor(commonNodes, dstSubPos, srcSubPos,
                                     p_d, p_s, w_d)
        if len(commonNodes) == 2:
            R = self._get_rotation_matrix_2_common_nodes(commonNodes,
                                                         dstSubPos, srcSubPos)
        else:
            R = self.get_rotation_matrix(commonNodes, dstSubPos, srcSubPos,
                                         p_d, p_s)
        t = p_d - dot(dot(s, R), p_s)

        # if rotation matrix reflects nodes then it is not reliable
        if det(R)<0 and not self.reflectable:
            return (None, None, None)

        return (R, s, t)
