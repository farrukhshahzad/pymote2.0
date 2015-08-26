from pymote.logger import logger
from numpy import sum as nsum
from numpy import asarray, sqrt, dot, concatenate, diag, mean, zeros, min, \
max, arctan2, sin, cos, square, tile
from pymote.utils.localization.aoastitcher import AoAStitcher
from numpy.linalg import inv, pinv
from copy import deepcopy
from pymote.utils.localization.diststitcher import DistStitcher
from pymote.utils.memory.positions import Positions
from xml.dom import NotSupportedErr


def align_clusters(dst, src, scale):
    """ Scale, rotate, translate srcLoc w.r.t. dst. """
    assert(isinstance(src, Positions))
    assert(isinstance(dst, Positions))
    if scale:
        stitcher = AoAStitcher(reflectable=True)
    else:
        stitcher = DistStitcher()
    stitcher.align(dst, src)


def get_rms(truePos, estimated, align=False, scale=False, norm=False):
    """
    Returns root mean square error of estimated positions.

    Set align if estimated positions need to be transformed (rotated and
    translated) before calculating rms error of localization. Set scale=True
    if they needs to be scaled also.

    If norm is True then rms is divided with norm of the true formation
    translated to centroid.

    """
    truePos = Positions.create(truePos)
    assert(len(truePos) == 1)
    estimated = Positions.create(estimated)

    if align:
        estimated = deepcopy(estimated)
        align_clusters(truePos, estimated, scale)

    suma = 0
    node_count = 0
    for estimated_subcluster in estimated:
        for n in estimated_subcluster:
            suma += (estimated_subcluster[n][0] - truePos[0][n][0]) ** 2 + \
                    (estimated_subcluster[n][1] - truePos[0][n][1]) ** 2
            node_count += 1

    rms = sqrt(suma / node_count)
    if norm:
        sc = truePos.subclusters[0]
        truePos.subclusters[0] = {n: p for n, p in sc.items()
                                  if n in estimated.subclusters[0]}
        rms = rms/get_pos_norm(truePos)
    return rms


def construct_G(pos, edges, u, sensor):
    """
    Construct Jacobian of vector mu(pos) that defines expectation of
    measurement given  positions.

    u - number of unknowns i.e. 3 - (x, y, theta)
    edges - list of edges as tuples, if there is (i, j) then there should not
            be (j, i)
    sensor - 'DistSensor' or 'AoASensor'

    For AoA and measurement between nodes i and j:
    mu(pos) = arctan((pos_y_j - pos_y_i)/(pos_x_j - pos_x_i) - alpha_i)

    """
    nodes = pos.keys()
    neighbors = {n: [] for n in nodes}
    for n1, n2 in edges:
        neighbors[n1].append(n2)
        neighbors[n2].append(n1)
    # number of measurements x number of unknowns
    G = zeros((2 * len(edges), u * len(nodes)))
    m = 0
    for r, node in enumerate(nodes):
        (x_r, y_r) = pos[node][:2]
        for neighbor in neighbors[node]:
            t = nodes.index(neighbor)
            (x_t, y_t) = pos[neighbor][:2]
            d = sqrt((x_r - x_t) ** 2 + (y_r - y_t) ** 2)
            if sensor == 'DistSensor':
                G[m, r*u] = (x_r-x_t)/d
                G[m, r*u+1] = (y_r-y_t)/d
                G[m, t*u] = (x_t-x_r)/d
                G[m, t*u+1] = (y_t-y_r)/d
            elif sensor == 'AoASensor':
                G[m, r*u] = (y_t-y_r)/d**2
                G[m, r*u+1] = (x_r-x_t)/d**2
                G[m, t*u] = (y_r-y_t)/d**2
                G[m, t*u+1] = (x_t-x_r)/d**2
                if u == 3:
                    G[m, t * u + 2] = -1.
            m += 1  # next measurement
    return G


#TODO: only one per row -1 element in G?
def get_crb(net, sensor, compass='off', loc_type='anchor-free', anchors=[]):
    """
    Calculates Cramer-Rao lower bound on covariance matrix of unbiased multihop
    location estimator based on sensor (distance/angle) measurement between
    in-commrange neighbors.

    POS is n by 2 matrix of X and Y coordinates of n nodes
    SIGMA is standard deviation on measurement vector.

    Arguments:
     `compass` -- 'off' (default) or 'on' which means all nodes have compass
     `loc_type` -- 'anchor-free' (default) or 'anchored'
     `anchors` -- list of anchor nodes, used only when loc_type is 'anchored'

    References:
        Savvides2003 - Savvides et al,
                        On the Error Characteristics of Multihop Node
                        Localization in Ad-Hoc Sensor Networks, 2003
        Patwari2005 - Patwari et al,
                        Locating the nodes: cooperative localization in
                        wireless sensor networks
        Chang2006 - Chen Chang et al,
                        Cramer-Rao-Type Bounds for Localization, 2006
    """

    if loc_type == 'anchored' and not anchors:
        raise Exception('Anchors not defined.')
    if loc_type == 'anchor-free' and anchors:
        logger.warning('Anchors are ignored')
        anchors = []

    nodes = [node for node in net.nodes() if node not in anchors]

    if sensor.name()=='AoASensor' and compass=='off':
        u = 3  # x, y, theta
    else:  # DistSensor or AoASensor with compass on
        u = 2  # x, y

    G = construct_G(net.pos, net.edges(), u, sensor.name())

    for s in nodes[0].compositeSensor.sensors:
        if s.name() == sensor.name():
            sigma = s.probabilityFunction.scale
            break
    else:
        raise Exception('Sensor not found in nodes')

    J = (dot(G.T, G)) / sigma ** 2

    #print matrix_rank(J)

    if loc_type == 'anchor-free':
        cov = pinv(J)
    elif loc_type=='anchored':
        cov = inv(J)

    # Chang2006 (28)
    di = diag(cov)
    # extract only position from cov
    di = concatenate((di[::3], di[1::3]))

    #return is lower bound on rms -> sqrt(1/n*sum_i^n((x_i'-x_i)^2+(y_i'-y_i)^2))
    return sqrt(2*mean(di))


def get_crb_norm(*args, **kwargs):
    """
    Calculates crb that is not dependent on scale i.e. to check correlation
    with gdop.
    """
    net = args[0]
    norm = get_pos_norm(net.pos)
    return get_crb(*args, **kwargs)/norm


def get_pos_norm(pos):
    """ Translate positions so that centroid is in the origin and return mean
    norm of the translated positions. """
    pos = Positions.create(pos)
    assert(len(pos) == 1)
    n = len(pos[0])
    p = zeros((n, 2))
    for i, node in enumerate(pos[0]):
        p[i, :] = pos[0][node][:2]
    centroid = p.sum(axis=0)/n
    p -= tile(centroid, (n, 1))
    p_norm = nsum(sqrt(nsum(square(p), axis=1)))/n
    return p_norm


def get_aoa_gdop_rel(estimated):
    """
    Calculation of relative GDOP is based on CRB.

    GDOP_rel = sigma_CRB/sigma_d = sqrt(tr((G^TG)^-1)/(sum(Di^2))/M)
    As regular GDOP, it is not dependent on scale nor on sigma_AoA.
    """
    estimated = Positions.create(estimated)
    assert len(estimated.subclusters)==1
    pos = estimated.subclusters[0]
    nodes = pos.keys()
    edges = [e for e in pos.keys()[0].network.edges()
             if e[0] in nodes and e[1] in nodes]
    G = construct_G(pos, edges, 3, 'AoASensor')
    J = dot(G.T, G)
    cov = pinv(J)
    di = diag(cov)
    di = concatenate((di[::3], di[1::3]))
    var_p = sum(di)/len(nodes)
    distances = [sqrt(dot(pos[n1][:2] - pos[n2][:2],
                          pos[n1][:2] - pos[n2][:2]))
                 for n1, n2 in edges]
    var_d = sum(square(distances))/len(edges)
    return sqrt(var_p/var_d)


def get_aoa_gdop_node(estimated, node):
    """
    Calculate geometric dilution of precision for node in estimated formation.
    Node is in the formation and all other nodes should be neighbors of
    node. Node should have Neighbors sensorReadings in it memory.
    """
    if 'AoASensor' not in [sensor.name() for sensor in node.sensors]:
        raise NotSupportedErr('Only angle of arrival based gdop is supported')
    sensor = node.compositeSensor.get_sensor('AoASensor')
    sigma = sensor.probabilityFunction.scale
    for n in estimated:
        sensor = n.compositeSensor.get_sensor('AoASensor')
        if sensor.probabilityFunction.scale!=sigma:
            raise NotSupportedErr('All nodes\' AoA sensors should have '
                                  'same scale')
    if len(estimated)<=2:
        return 0.
    # Note from Torrieri, Statistical Theory of Passive Location Systems
    # if measurement sigmas are all equal gdop doesn't depend on sigma.
    sigma = 1

    x, y = estimated[node][:2]
    fi = []
    d = []

    for n in estimated:
        xi, yi = estimated[n][:2]
        if n != node and n in node.memory['sensorReadings']['Neighbors']:
            fi.append(arctan2(y - yi, x - xi))
            d.append(sqrt((x - xi) ** 2 + (y - yi) ** 2))

    mi = ni = l = 0
    for fii, di in zip(fi, d):
        mi += (cos(fii) / di / sigma) ** 2  # (129)
        l += (sin(fii) / di / sigma) ** 2  # (130)
        ni += sin(fii) * cos(fii) / (di * sigma) ** 2  # (131)

    sigma1 = sqrt(mi / (mi * l - ni ** 2))  # (126)
    sigma2 = sqrt(l / (mi * l - ni ** 2))  # (127)
    # sigma12 = sqrt(ni/(mi*l-ni**2))

    sigmad = sqrt(sum((di * sigma) ** 2 for di in d) / len(d))  # (138)
    return sqrt((sigma1 ** 2 + sigma2 ** 2)) / sigmad  # (139)


def get_aoa_gdops(estimated):
    estimated = Positions.create(estimated)
    assert len(estimated.subclusters)==1
    estimated = estimated.subclusters[0]
    return [get_aoa_gdop_node(estimated, node) for node in estimated]


def get_aoa_gdop(estimated):
    return sum(get_aoa_gdops(estimated))


def show_subclusters(net, subclusters):
    """ Show colored subclusters. """
    colors = [node in subclusters[0] and 'g' or 'r' for node in net.nodes()]
    net.show(nodeColor=colors)


def show_localized(net, estimated, scale=False, align=True,\
                   display_loc_err=True, show_labels=True):
    """
    Display estimated positions.

    estimated should be a list of dictionaries.

    """
    from matplotlib.pylab import gca
    from matplotlib.collections import LineCollection

    truePos = Positions.create(net.pos)
    estimated = Positions.create(estimated)
    # copy estimated so that passed estimated remains unchanged
    estimated = deepcopy(estimated)
    if align:
        # rotate, translate and if needed scale estimated w.r.t. true positions
        align_clusters(truePos, estimated, scale)

    #TODO: implement display of all subclusters
    if len(estimated)>1:
        raise(NotImplementedError)
    else:
        estimated_sc = estimated[0]

        #net.show(positions=estimated_sc, show_labels=show_labels)
        fig = net.get_fig(positions=estimated_sc, show_labels=show_labels)
        ax = fig.gca()
        minpos = min(estimated_sc.values(), axis=0)
        maxpos = max(estimated_sc.values(), axis=0)
        minpos -= (maxpos-minpos)*0.1
        maxpos += (maxpos-minpos)*0.1

        ax.set_xlim(minpos[0], maxpos[0])
        ax.set_ylim(minpos[1], maxpos[1])
        #fig.show()
        if display_loc_err:
            #TODO: not working in ipython notepad
            ax = gca()
            ax.set_title('Localized positions')
            ax.set_title('Localization error display')
            edge_pos = asarray([(net.pos[n], estimated_sc[n])
                                for n in estimated_sc.keys()])
            errorCollection = LineCollection(edge_pos, colors='r',
                                             transOffset=ax.transData)
            errorCollection.set_zorder(1)  # errors go behind nodes
            ax.add_collection(errorCollection)
            ax.figure.canvas.draw()
