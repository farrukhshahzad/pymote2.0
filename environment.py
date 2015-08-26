from pymote.conf import settings
import png
from itertools import imap
from numpy import vstack, uint8, ones
from numpy.core.numeric import sign, sqrt, Inf


class Environment(object):
    """ Environment abstract base class """

    def __new__(self, **kwargs):
        """ return instance of default Environment """
        for cls in self.__subclasses__():
            if (cls.__name__ == settings.ENVIRONMENT):
                return object.__new__(cls, **kwargs)
        # if self is not Environment class (as in pickle.load_newobj) return
        # instance of self
        return object.__new__(self, **kwargs)

    def is_space(self, xy):
        raise NotImplementedError

    def are_visible(self, xy1, xy2):
        raise NotImplementedError


class Environment2D(Environment):
    """
    Base class for 2D environment.
    The Environment2D allows to define map and scale of 2D environment.
    """

    def __init__(self, path='', scale=None, shape=None):
        shape = shape if shape else settings.ENVIRONMENT2D_SHAPE
        if (path):
            try:
                r = png.Reader(path)
                planes = r.read()[3]['planes']
                self.im = vstack(imap(uint8, r.asDirect()[2]))[:, ::planes]
                self.im = self.im[::-1, :]  # flip-up-down
                assert((r.height, r.width) == self.im.shape)
            except IOError:
                print 'Can\'t open %s creating new default environment.' % path

                self.im = uint8(ones((shape)) * 255)
        else:
            self.im = uint8(ones((shape)) * 255)

        self.dim = 2
        scale = not scale and 1 or int(scale)
        if scale > 1:
            raise NotImplementedError

    def __deepcopy__(self, memo):
        dup = Environment2D()
        dup.im = self.im.copy()
        return dup

    def is_space(self, xy):
        """ Returns true if selected space (x,y) is space. If point xy
        is exactly on the edge or crossing check surrounding pixels. """
        x, y = xy
        h, w = self.im.shape
        if x<0 or x>w or y<0 or y>h:
            return False
        check = True
        points = [xy]
        if (xy[0] % 1 == 0):
            points.append([xy[0] - 1, xy[1]])
        if (xy[1] % 1 == 0):
            points.append([xy[0], xy[1] - 1])
        if (xy[0] % 1 == 0 and xy[1] % 1 == 0):
            points.append([xy[0] - 1, xy[1] - 1])
        try:
            for p in points:
                check = check and self.im[int(p[1]), int(p[0])] != 0
        except IndexError:
            check = False
        return check

    def are_visible(self, xy0, xy1):
        """
        Returns true if there is line of sight between source (x0,y0) and
        destination (x1,y1).

        This is floating point version of Bresenham algorithm that does
        not spread on diagonal pixels.

        """
        x = x0 = xy0[0]
        y = y0 = xy0[1]
        x1 = xy1[0]
        y1 = xy1[1]
        d = sqrt(pow(x1 - x0, 2) + pow(y1 - y0, 2))
        incrE = (x1 - x0) / d  # incrE is cos in direction of x axis
        incrN = (y1 - y0) / d  # incrN is sin in direction of y axis

        # check if pixel (x,y) is target pixel (x1,y1) or
        # if float (x,y) is on N or E edge then check also W or S neighbor
        while (not((int(x) == int(x1) or\
                    (x % 1 == 0 and int(x) - 1 == int(x1))) and
                   (int(y) == int(y1) or\
                    (y % 1 == 0 and int(y) - 1 == int(y1))))):
            if incrE > 0:
                dx = 1 - x % 1
            else:
                dx = x % 1 == 0 and 1.0 or x % 1
            if incrN > 0:
                dy = 1 - y % 1
            else:
                dy = y % 1 == 0 and 1.0 or y % 1

            # check whether the path will hit first E/W or N/S pixel edge
            # by calculating length of paths
            if incrE != 0:
                cx = abs(dx / incrE)
            else:
                cx = Inf
            if incrN != 0:
                cy = abs(dy / incrN)
            else:
                cy = Inf

            # if path needed to hit N/S edge is longer than E/W
            if cx < cy:
                x = round(x + sign(incrE) * dx)  # spread on E
                y = y + cx * incrN
            else:
                x = x + cy * incrE
                y = round(y + sign(incrN) * dy)  # spread on N

            # logger.debug('x = %s, y = %s' % (str(x),str(y)))
            if (not self.is_space([x, y])):
                return False
        return True
