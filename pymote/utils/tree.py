"""
Set of utils that operate on network nodes when they have defined tree
in their memory under treeKey key.
treeKey -- key in node memory (dictionary) where parent and
           children data is stored in format:
           {'parent': parent_node,
            'children': [child_node1, child_node2 ...]}
"""


def get_root_node(net, treeKey='mst'):
    """
    Return root node in network tree.
    """
    check_tree_key(net, treeKey)

    node = net.nodes()[0]
    while node.memory[treeKey]['parent'] and \
          node.memory[treeKey]['parent'] in net.nodes():
        node = node.memory[treeKey]['parent']
    return node


def change_root_node(net, root, treeKey='mst'):
    """
    Change root node by altering nodes' memory.
    """
    check_tree_key(net, treeKey)
    curr_root = get_root_node(net, treeKey)
    # flip all edges on a path from current root to new root
    p = get_path(curr_root, root, treeKey)
    for i in range(len(p)-1):
        _flip_root_edge(p[i], p[i+1], treeKey)


def _flip_root_edge(root, child, treeKey):
    """
    Flips the edge orientation between root and child.
    """
    if root.memory[treeKey]['parent']:
        raise NodeNotRoot(treeKey, root)
    root.memory[treeKey]['parent'] = child
    root.memory[treeKey]['children'].remove(child)
    child.memory[treeKey]['parent'] = None
    child.memory[treeKey]['children'].append(root)


def get_path(n1, n2, treeKey):
    """
    Return path between nodes n1 and n2.
    """
    nodesToCheck = [n1]
    checkedNodes = []
    tree = {}
    path = []
    while nodesToCheck:
        node = nodesToCheck.pop()
        checkedNodes.append(node)
        if node == n2:
            path.insert(0, n2)
            break
        parent = node.memory[treeKey]['parent']
        children = node.memory[treeKey]['children']
        if parent and parent not in checkedNodes:
            nodesToCheck.append(parent)
            tree[parent] = node
        for child in children:
            if child not in checkedNodes:
                nodesToCheck.append(child)
                tree[child] = node
    if path:
        next = n2  # @ReservedAssignment
        while next!=n1:
            next = tree[next]  # @ReservedAssignment
            path.insert(0, next)
    return path


def check_tree_key(net, treeKey):
    for node in net.nodes():
        if not treeKey in node.memory:
            raise MissingTreeKey(treeKey)


class MissingTreeKey(Exception):

    def __init__(self, treeKey):
        self.treeKey = treeKey

    def __str__(self):
        return 'At least one node is missing \'%s\' key in memory.' \
                % self.treeKey


class NodeNotRoot(Exception):
    def __init__(self, treeKey, node):
        self.treeKey = treeKey
        self.node = node

    def __str__(self):
        return ("Node with id=%d is is not root in tree defined by "
                "\'%s\' key in memory." % (self.node.id, self.treeKey))
