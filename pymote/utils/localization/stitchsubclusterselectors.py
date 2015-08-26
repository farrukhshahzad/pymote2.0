class StitchSubclusterSelectorBase(object):
    """ Subclasses of this class are used to make choice which subclusters
        will be stitched next based on criteria. """

    def __init__(self, *args, **kwargs):
        pass

    def select(self, dst, src, stitched, is_intra=False):
        raise NotImplementedError


class MaxCommonNodeSelector(StitchSubclusterSelectorBase):
    """
    Return subclusters that have maximum common node count.
    """
    def __init__(self, cn_count_treshold=0):
        self.cn_count_treshold = cn_count_treshold

    def select(self, dst, src, stitched, is_intra=False):
        cn_count_max = 0
        dstSubIndex = None
        srcSubIndex = None
        for src_index, src_sc in enumerate(src):
            for dst_index, dst_sc in enumerate(dst):
                # same subcluster
                if is_intra and src_index==dst_index:
                    continue
                # already stitched
                if (dst_index, src_index) in stitched:
                    continue
                # src_sc is subset of dst_sc mark src_sc stitched with dst
                if set(src_sc.keys())<=set(dst_sc.keys()):
                    stitched[(dst_index, src_index)] = ()
                    continue

                commonNodes = [node for node in dst_sc.keys()
                               if node in src_sc.keys()]
                cn_count = len(commonNodes)
                if cn_count>cn_count_max and cn_count>=self.cn_count_treshold:
                    cn_count_max = cn_count
                    dstSubIndex = dst_index
                    srcSubIndex = src_index

        # in intraStitch make sure that dstSub is larger one
        if is_intra and cn_count_max!=0 and \
        len(src[srcSubIndex])>len(dst[dstSubIndex]):
            return (srcSubIndex, dstSubIndex)  # swapped src and dst indexes
        return (dstSubIndex, srcSubIndex)
