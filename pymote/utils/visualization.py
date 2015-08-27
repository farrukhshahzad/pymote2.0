from pymote.utils.tree import get_root_node


def show_mst(net, treeKey='mst'):
    """
    Show tree representation of network.

    treeKey -- key in nodes memory (dictionary) where parent and
               children data is stored in format:
                {'parent': parent_node,
                 'children': [child_node1, child_node2 ...]}
    """
    nodesToCheck = [(get_root_node(net, treeKey), 0)]
    edgelist = []
    levels = [0]*len(net.nodes())  # level of node in tree, root is 0
    while nodesToCheck:
        (node, level) = nodesToCheck.pop()
        edgelist += [(node, child)
                      for child in node.memory[treeKey]['children']]
        levels[net.nodes().index(node)] = level
        nodesToCheck += [(child, level+1)
                         for child in node.memory[treeKey]['children']]
    net.show(edgelist=edgelist, nodeColor=levels)
    from matplotlib.pyplot import gca
    gca().set_title('Minimum spanning tree in memory[\'%s\']' % treeKey)
