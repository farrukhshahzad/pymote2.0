from copy import copy
from pymote.utils.memory import MemoryStructure


class Positions(MemoryStructure):
    """Class to represent subclusters positions data in node memory.

    Position instances initializer has only one attribute ``subclusters`` a
    list of dictionaries in form {node: position_array,}. Node can be part of
    multiple subclusters.

    """

    def __init__(self, subclusters=None):
        self.subclusters = []
        if subclusters is not None:
            # list of dictionaries of positioned subclusters
            assert(isinstance(subclusters, list))
            # create subclusters by copying from given list and its dicts
            ini_subclusters = list(subclusters)
            self.subclusters = []
            for subcluster in ini_subclusters:
                new_subcluster = {}
                for n, p in subcluster.items():
                    new_subcluster.update({n: copy(p)})
                self.subclusters.append(new_subcluster)
        self.old_style_positions = {}
        self.old_style_subclusters = []

    @classmethod
    def create(cls, obj):
        """ Create Positions instance from obj which can be dict or list.

        If it's dict this method wraps it in list. Returns Position object
        dereferenced from the initial list or dict.

        For example ``obj`` can be ``net.pos`` dict.

        """
        if not isinstance(obj, cls):
            if not isinstance(obj, list):
                assert isinstance(obj, dict)
                obj = [obj]
            obj = cls(obj)
        return obj

    @property
    def subclusters_nodes(self):
        """ Returns list of subclusters without positions. """
        cl = []
        for c in self.subclusters:
            cl.append(c.keys())
        return cl

    def set_pos_copy(self, positions):
        """ Empty all subclusters and copy in data from ``positions``. """
        assert isinstance(positions, Positions)
        self.subclusters = []
        for subcluster in positions.subclusters:
            new_subcluster = {}
            for n, p in subcluster.items():
                new_subcluster.update({n: copy(p)})
            self.subclusters.append(new_subcluster)

    def __len__(self):
        return len(self.subclusters)

    def get_nodes(self):
        """ Returns nodes in all subclusters. """
        nodes = []
        for subcluster in self.subclusters:
            nodes += [node for node in subcluster if node not in nodes]
        return nodes

    def get_largest_subcluster(self):
        """ Returns new Positions instance with only one largest subcluster.
            If multiple subclusters have maximum number of nodes, first one is
            returned. """
        return Positions([reduce(lambda x, y: len(x)>len(y) and
                                 x or y, self.subclusters)])

    def get_dic(self):
        dic = {}
        for i, sc in enumerate(self.subclusters):
            dic['subclusters[%d] (%d nodes)' % (i, len(sc))] = sc
        return dic

    def __str__(self):
        return str(self.subclusters)

    # Backward compatibility - usage raises exception
    # -----------------------------------------------
    def __getitem__(self, key):
        if key is 'positions':
            key = 'old_style_positions'
            raise(Exception('Old style positions usage, please refactor code'))
        if key is 'subclusters':
            key = 'old_style_subclusters'
            raise(Exception('Old style positions usage, please refactor code'))
        if isinstance(key, int):
            return self.subclusters[key]
        return self.__getattribute__(key)

    def has_key(self, key):
        raise(Exception('Old style positions usage, please refactor code'))
        try:
            self.__getitem__(key)
        except AttributeError:
            return False
        return True

    def keys(self):
        raise (Exception('Old style positions usage, please refactor code'))
        return ['positions', 'subclusters']

    def items(self):
        raise (Exception('Old style positions usage, please refactor code'))
        return [('positions', self.old_style_positions),
                ('subclusters', self.old_style_subclusters)]
