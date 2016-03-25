__author__ = 'farrukh'

from pymote import *
from pymote.sensor import TruePosSensor
from pymote import gmm

from numpy import sign, sqrt, array, pi, sin, cos
from numpy.random import rand
from matplotlib.pylab import *
from pypr.clustering.kmeans import *
#import pypr.clustering.gmm as gmm

tests = 1
def divide_equally(n):
    if (n<3):
        return [n, 1]
    result = list()
    for i in range(1, int(n ** 0.5) + 1):
        div, mod = divmod(n, i)
        #ignore 1 and n itself as factors
        if mod == 0 and i != 1 and div != n:
            result.append(div)
            result.append(i)
    if len(result)==0: # if no factors than add 1
        return divide_equally(n+1)
    return result[len(result)-2:]


class Topology(NetworkGenerator):

    def generate_star_ehwsn_network(self, center=None, x_radius=100, y_radius=100, sector=1.0, clusters=1, is_random=0):
        """
        Generates network where nodes are located around a center/coordinator node.
        Parameter is_random controls random perturbation of the nodes
        """
        self.name = "Star EHWSN"
        net = Network(commRange=self.comm_range, **self.kwargs)
        h, w = net.environment.im.shape
        if center is None:
            center = (h/2, w/2)  # middle
        if clusters < 1:
            clusters = 1
        n_nets = int(self.n_count/clusters)

        for k in range(clusters):
            # Coordinator
            node = Node(commRange=(x_radius + y_radius)/clusters, node_type='C',
                        power_type=0, **self.kwargs)
            mid = (center[0] - 2 * k * x_radius,
                   center[1] - 2 * k * y_radius)
            net.add_node(node, pos=(mid[0], mid[1]))
            for n in range(n_nets)[1:]:
                    # Regular sensor node
                    node = Node(commRange=5, power_type=2, mobile_type=2, **self.kwargs)
                    if node.id == 5:
                        node.power.energy = node.power.E_MIN - 0.003
                    elif node.id == 10:  # just enough energy to send few messages
                        node.power.energy = node.power.E_MIN + 0.002
                        node.power.P_CHARGING = 0  # no charging
                    else:  # random energy level from 0 to 2 Joules
                        node.power.energy = rand()*2.0
                    rn = (rand(2) - 0.5)*(x_radius + y_radius)/2/clusters
                    ang = n *2*pi/n_nets * sector + pi*(1.0 - sector)
                    net.add_node(node, pos=(mid[0] + cos(ang)*(x_radius + rn[0]*is_random),
                                            mid[1] + sin(ang)*(y_radius + rn[0]*is_random)
                    ))

        return net

    def generate_star_network(self, center=None, x_radius=100, y_radius=100, sector=1.0, clusters=1, is_random=0):
        """
        Generates network where nodes are located around a center/coordinator node.
        Parameter is_random controls random perturbation of the nodes
        """
        self.name = "Star"
        net = Network(commRange=self.comm_range, **self.kwargs)
        h, w = net.environment.im.shape
        if center is None:
            center = (h/2, w/2)  # middle
        if clusters < 1:
            clusters = 1
        n_nets = int(self.n_count/clusters)

        for k in range(clusters):
            # Coordinator
            mid = (center[0] - 2 * k * x_radius,
                   center[1] - 2 * k * y_radius)
            #net.add_node(node, pos=(mid[0], mid[1]))
            for n in range(n_nets):
                    # Regular sensor node
                    node = Node(**self.kwargs)
                    rn = (rand(2) - 0.5)*(x_radius + y_radius)/4/clusters
                    ang = n *2*pi/n_nets * sector + pi*(1.0 - sector)
                    x = mid[0] + cos(ang)*(x_radius + rn[0]*is_random)
                    y = mid[1] + sin(ang)*(y_radius + rn[0]*is_random)
                    net.add_node(node, pos=(x, y))

        return net

    def generate_ring_network(self, center=None, x_radius=100, y_radius=100, sector=1.0, n_ring=5):
        """
        Generates network where nodes are located approximately homogeneous.

        Parameter randomness controls random perturbation of the nodes, it is
        given as a part of the environment size.

        """
        self.name = "Ring"
        net = Network(commRange=self.comm_range, **self.kwargs)
        h, w = net.environment.im.shape
        if center is None:
            center = (h/2, w/2)  # middle
        for _n in range(self.n_count/n_ring):

                    ang = n_ring * (_n + 1)*2*pi/self.n_count * sector  + pi/4  # start from 45 deg
                    for _r in range(n_ring):
                        rn = (rand(2) - 0.5)*(x_radius + y_radius)/3
                        node = Node(**self.kwargs)
                        net.add_node(node, pos=(center[0] + cos(ang)*x_radius + rn[0],
                                                center[1] + sin(ang)*y_radius + rn[1]),
                                                ori=ang)


        return net

    def generate_two_ring_network(self, center=None, x_radius=100, y_radius=100, sector=1.0, n_ring=5):
        """
        Generates network where nodes are located approximately homogeneous.

        Parameter randomness controls random perturbation of the nodes, it is
        given as a part of the environment size.

        """
        self.name = "Two Ring"
        net = Network(commRange=self.comm_range, **self.kwargs)
        h, w = net.environment.im.shape
        dir = 1.0  # ccw
        if center is None:
            center = (w/2, 3*h/4)  # one third up
        for _n in range(self.n_count/n_ring/2):

                    ang = dir * 2* n_ring * (_n + 1)*2*pi/self.n_count * sector  + pi/4  # start from 45 deg
                    for _r in range(n_ring):
                        rn = (rand(2) - 0.5)*(x_radius + y_radius)/4
                        node = Node(**self.kwargs)
                        net.add_node(node, pos=(center[0] + cos(ang)*x_radius + rn[0],
                                                center[1] + sin(ang)*y_radius + rn[1]),
                                                ori=ang)

        center = (w/2, h/4)  # one third
        dir = -1.0  # cw
        sector = 0.5
        for _n in range(self.n_count/n_ring/2):

                    ang = dir * (2 * n_ring * (_n + 1)*2*pi/self.n_count * sector)  # start from 0 deg
                    for _r in range(n_ring):
                        rn = (rand(2) - 0.5)*(x_radius + y_radius)/4
                        node = Node(**self.kwargs)
                        net.add_node(node, pos=(center[0] + cos(ang)*x_radius + rn[0],
                                                center[1] + sin(ang)*y_radius + rn[1]),
                                                ori=ang)

        return net


    def generate_web_network(self, randomness=0):
        """
        Generates network where nodes are located approximately homogeneous in a grid

        Parameter randomness controls random perturbation of the nodes, it is
        given as a part of the environment size.

        """
        self.name = "Grid"
        net = Network(commRange=self.comm_range, **self.kwargs)
        h, w = net.environment.im.shape
        area = h * w
        sq = int(round(sqrt(self.n_count)))
        nr = sqrt(self.n_count) * area/self.n_count/h
        k = 0
        print nr, sq
        for x in range(self.n_count/sq + 1):
            for y in range(self.n_count/sq):
                k += 1
                rn = rand(2)*randomness
                node = Node(**self.kwargs)
                net.add_node(node, pos=(int((x + rn[0])*nr), int((y + rn[1])*nr)))
                if (k >= self.n_count):
                    return net

        return net

    def generate_grid_network(self, name=None, randomness=0, p_anchors=10, cut_shape=None):
        """
        Generates network where nodes are located approximately homogeneous in a grid

        Parameter randomness controls random perturbation of the nodes, it is
        given as a part of the environment size.
        Default: 10% anchors

        """
        self.name = name or "Grid"
        self.anchors = 0
        f_anchors = None
        if p_anchors:
            f_anchors = (int)(100 / p_anchors)
        net = Network(commRange=self.comm_range, **self.kwargs)
        h, w = net.environment.im.shape
        area = h * w
        sq = int(round(sqrt(self.n_count)))
        nh = h/(sq+1) #((sqrt(self.n_count)+1) * area / self.n_count / h)
        nw = w/(sq+1)
        k = 0
        cut_area = 0
        if cut_shape:
            for box in cut_shape:
                cut_area += (box[1][0] - box[0][0])*(box[0][1] - box[1][1])
        self.area = area - cut_area
        done = False
        for x in range(0, self.n_count / sq):
            if done:
                break
            for y in range(0, self.n_count / sq):
                rn = rand(2) * randomness
                xpos = nw + round((x + rn[0]) * nw, 2)
                ypos = nh + round((y + rn[1]) * nh, 2)
                inside = True
                if cut_shape:
                    for box in cut_shape:
                        #print box[0], box[1],  xpos, ypos
                        if xpos-nw >= box[0][0] and xpos+nh <= box[1][0] and \
                           ypos+nh <= box[0][1] and ypos-nh >= box[1][1]:
                            #print "not in"
                            inside = False
                if inside:
                    k += 1

                    node = Node(**self.kwargs)
                    #print node.id, (xpos, ypos)
                    if (f_anchors and node.id % f_anchors==0):  # anchor nodes
                        node.compositeSensor = (TruePosSensor,)
                        node.type = 'C'  # Anchors
                    net.add_node(node, pos=(xpos, ypos))


                if (k >= self.n_count):
                    done = True
                    break

        if p_anchors:
            self.anchors = (int)(len(net) *  p_anchors/100.0)

        if self.degree:
            self.n_max = len(net)
            net = self.generate_random_network(net)  # adjust for degree by changing comm_range

        self.net_density = 1.0 * len(net)/self.area
        return net


    def generate_gaussian_network(self, name=None, center=None, randomness=0.5, cut_shape=None):
        """
        Generates network where nodes are located approximately homogeneous.

        Parameter randomness controls random perturbation of the nodes, it is
        given as a part of the environment size.

        """
        net = Network(commRange=self.comm_range, **self.kwargs)
        h, w = net.environment.im.shape
        self.area = h * w
        if center is None:
            center = (h/2, w/2)  # middle
        centroids = [ array([w/4,h/4]), array([w/4,3*h/4]),
                      array([3*w/4,h/4]), array([3*w/4, 3*h/4]) ]
        ccov=[array([[4*w*randomness, 0],[0, 4*h*randomness]]),
              array([[4*w*randomness, 0],[0, 4*h*randomness]]),
              array([[4*w*randomness, 0],[0, 4*h*randomness]]),
              array([[4*w*randomness, 0],[0, 4*h*randomness]])]
        cut_area = 0
        if cut_shape:
            for box in cut_shape:
                cut_area += (box[1][0] - box[0][0])*(box[0][1] - box[1][1])
        self.area = self.area - cut_area
        p_nk = None
        X = gmm.sample_gaussian_mixture(centroids, ccov, mc=None, samples=self.n_count)
        k=0
        for _n in range(len(X)):
            xpos, ypos = (X[_n,0], X[_n,1])
            inside = True
            if cut_shape:
                    for box in cut_shape:
                        #print box[0], box[1],  xpos, ypos
                        if xpos >= box[0][0] and xpos <= box[1][0] and \
                                        ypos <= box[0][1] and ypos >= box[1][1]:
                            #print "not in"
                            inside = False
            if inside:
                node = Node(**self.kwargs)
                if net.add_node(node, pos=(xpos, ypos), find_random=False):
                    k += 1

        self.name = name or 'Gaussian'
        self.net_density = 1.0 * len(net)/self.area
        return net

    def generate_cluster_network(self, name=None, center=None, x_radius=100.0, y_radius=100.0,
                         sector=1.0, cluster_divide=None, randomness=0, cut_shape=None):
        """
        Generates network where nodes are located around a center/coordinator node.
        Parameter is_random controls random perturbation of the nodes
        """
        self.name = name or "Cluster-%s" %cluster_divide
        net = Network(propagation_type=2, **self.kwargs)
        h, w = net.environment.im.shape
        self.area = h * w
        cut_area = 0
        if cut_shape:
            for box in cut_shape:
                cut_area += (box[1][0] - box[0][0])*(box[0][1] - box[1][1])
        self.area = self.area - cut_area

        if center is None:
            center = (h/2, w/2)  # middle
        if not cluster_divide:
            cluster_divide = 1

        n_nets = int(self.n_count/cluster_divide) # nodes per cluster
        p_nk = None
        fact = divide_equally(cluster_divide)
        # get two last factors
        x = fact[0]
        y = fact[1]
        hh = divmod(h, y)[0]
        ww = divmod(w, x)[0]
        done = False
        k=0
        for i in range(x):
            if done:
                break
            for j in range(y):
                if done:
                    break
                rn = (rand(2) - 0.5)*(hh+ww)/cluster_divide
                mid = (ww*(i+1) - ww/2 + rn[0]*randomness,
                       hh*(j+1) - hh/2 + rn[1]*randomness)
                for n in range(n_nets):
                    rn = (rand(2) - 0.5)*(hh+ww)/2/cluster_divide
                    ang = n *2*pi/n_nets * sector + pi*(1.0 - sector)
                    xpos = mid[0] + cos(ang)*(x_radius - rn[0]*randomness)
                    ypos = mid[1] + sin(ang)*(y_radius - rn[1]*randomness)
                    inside = True
                    if cut_shape:
                        for box in cut_shape:
                                #print box[0], box[1],  xpos, ypos
                           if xpos >= box[0][0] and xpos <= box[1][0] and \
                                 ypos <= box[0][1] and ypos >= box[1][1]:
                               inside = False
                    if inside:
                        node = Node(**self.kwargs)
                        if net.add_node(node, pos=(xpos, ypos), find_random=False):
                            k += 1
                    if (net.__len__() >= self.n_count):
                            done = True
                            break
        self.net_density = 1.0 * len(net)/self.area
        return net

    def get_clusters(self, net, clusters=1, method=None):

        k = len(net)
        p_nk = None

        centeriods = []
        h, w = net.environment.im.shape
        locations = np.zeros((len(net), 2))
        k = 0
        for node in net.nodes():
            if node.type == 'C':
                 if method == 'Random':
                    node.type = 'N'
                    node.compositeSensor = ('NeighborsSensor',)
                 else:
                    net.remove_node(node)
                 continue
            xpos = net.pos[node][0]
            ypos = net.pos[node][1]
            locations[k, :] = [xpos, ypos]
            k += 1

        if method == 'Random':
            for i in range(clusters):
                rn = rand(2) * k
                rn = int(rn[0])
                node = net.nodes()[rn]
                node.type = 'C'
                node.compositeSensor = (TruePosSensor,)

        elif method == 'EM':
            centeriods, m, p_k, logL, p_nk = gmm.em_gm(locations[:k, :], K=clusters,
                                                       max_iter=100, verbose=False)
        elif method:
            method = "K-Means"
            m, centeriods = kmeans(locations[:k, :], clusters)

        if method != 'Random':
            if p_nk is None:
                p_nk= np.ones((k, clusters))

            #print centeriods
            for _n in range(len(centeriods)):
                node = Node(**self.kwargs)
                node.compositeSensor = (TruePosSensor,)
                node.type = 'C'  # Anchors
                if centeriods[_n][0] < 0:
                    centeriods[_n][0] = 1
                if centeriods[_n][1] < 0:
                    centeriods[_n][1] = 1
                if centeriods[_n][0] > w:
                    centeriods[_n][0] = w-1
                if centeriods[_n][1] > h:
                    centeriods[_n][1] = h-1

                net.add_node(node, pos=(centeriods[_n][0], centeriods[_n][1]))

        self.net_density = 1.0 * len(net)/self.area
        self.anchors = clusters
        self.method = method
        self.name = self.name + " - " + method

        return net, p_nk

    def generate_manual_network(self, randomness=0):

        from pymote.conf import global_settings
        from pymote.environment import Environment2D

        self.n_count = 0
        self.name = "Manual"
        net = Network(environment=Environment2D(shape=(200,200)), commRange=75)
        h, w = net.environment.im.shape

        node = Node(node_type='C')
        net.add_node(node, pos=(h/2, w/2))
        self.n_count +=1

        node = Node(node_type='N')
        net.add_node(node, pos=(h/3, w/3))
        self.n_count +=1

        node = Node(node_type='N', mobile_type=2, power_type=2)
        net.add_node(node, pos=(2*h/3, 2*w/3))
        self.n_count +=1

        node = Node(node_type='B')
        net.add_node(node, pos=(50, 150))
        self.n_count +=1

        #auto add a node 'N'
        net.add_node(pos=(150, 50))
        self.n_count +=1

        #auto add a node 'N' at random location
        net.add_node()
        self.n_count +=1

        return net
