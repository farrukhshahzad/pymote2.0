
"""Gaussian Mixture Models
"""
import copy
import types

import numpy as np
import numpy.linalg as linalg

from pypr.clustering import kmeans
from pypr.clustering import gauss_diff
import pypr.preprocessing as preproc

def mulnormpdf(X, MU, SIGMA):
    """Evaluates the PDF for the multivariate Guassian distribution.

    Parameters
    ----------
    X : np array
        Inputs/entries row-wise. Can also be a 1-d array if only a 
        single point is evaluated.
    MU : nparray
        Center/mean, 1d array. 
    SIGMA : 2d np array
        Covariance matrix.

    Returns
    -------
    prob : 1d np array
        Probabilities for entries in `X`.
    
    Examples
    --------
    ::

        from pypr.clustering import *
        from numpy import *
        X = array([[0,0],[1,1]])
        MU = array([0,0])
        SIGMA = diag((1,1))
        gmm.mulnormpdf(X, MU, SIGMA)

    """
    # Check if inputs are ok:
    if MU.ndim != 1:
        raise ValueError, "MU must be a 1 dimensional array"
    
    # Evaluate pdf at points or point:
    mu = MU
    x = X.T
    if x.ndim == 1:
        x = np.atleast_2d(x).T
    sigma = np.atleast_2d(SIGMA) # So we also can use it for 1-d distributions

    N = len(MU)
    ex1 = np.dot(linalg.inv(sigma), (x.T-mu).T)
    ex = -0.5 * (x.T-mu).T * ex1
    if ex.ndim == 2: ex = np.sum(ex, axis = 0)
    K = 1 / np.sqrt ( np.power(2*np.pi, N) * linalg.det(sigma) )
    return K*np.exp(ex)

def logmulnormpdf(X, MU, SIGMA):
    """
    Evaluates natural log of the PDF for the multivariate Guassian distribution.

    Parameters
    ----------
    X : np array
        Inputs/entries row-wise. Can also be a 1-d array if only a 
        single point is evaluated.
    MU : nparray
        Center/mean, 1d array. 
    SIGMA : 2d np array
        Covariance matrix.

    Returns
    -------
    prob : 1d np array
        Log (natural) probabilities for entries in `X`.
    
    """
    # Check if inputs are ok:
    if MU.ndim != 1:
        raise ValueError, "MU must be a 1 dimensional array"
    
    # Evaluate pdf at points or point:
    #ex = _cal_ex(X, MU, SIGMA)
    mu = MU
    x = X.T
    if x.ndim == 1:
        x = np.atleast_2d(x).T
    sigma = np.atleast_2d(SIGMA) # So we also can use it for 1-d distributions

    N = len(MU)
    ex1 = np.dot(linalg.inv(sigma), (x.T-mu).T)
    ex = -0.5 * (x.T-mu).T * ex1
    if ex.ndim == 2: ex = np.sum(ex, axis = 0)
    K = -(N/2)*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(SIGMA))
    return ex + K

def gmm_pdf(X, centroids, ccov, mc, individual=False):
    """Evaluates the PDF for the multivariate Guassian mixture.

    Draw samples from a Mixture of Gaussians (MoG)

    Parameters
    ----------
    centroids : list
        List of cluster centers - [ [x1,y1,..],..,[xN, yN,..] ]
    ccov : list
        List of cluster co-variances DxD matrices
    mc : list
        Mixing cofficients for each cluster (must sum to one)
                  by default equal for each cluster.
    individual : bool
        If True the probability density is returned for each cluster component.

    Returns
    -------
    prob : 1d np array
        Probability density values for entries in `X`.
    """
    if individual:
        pdf = np.zeros((len(X), len(centroids)))
        for i in range(len(centroids)):
            pdf[:,i] = mulnormpdf(X, centroids[i], ccov[i]) * mc[i]
        return pdf
    else:
        pdf = None
        for i in range(len(centroids)):
            pdfadd = mulnormpdf(X, centroids[i], ccov[i]) * mc[i]
            if pdf==None:
                pdf = pdfadd
            else:
                pdf = pdf + pdfadd
        return pdf

def sample_gaussian_mixture(centroids, ccov, mc = None, samples = 1):
    """
    Draw samples from a Mixture of Gaussians (MoG)

    Parameters
    ----------
    centroids : list
        List of cluster centers - [ [x1,y1,..],..,[xN, yN,..] ]
    ccov : list
        List of cluster co-variances DxD matrices
    mc : list
        Mixing cofficients for each cluster (must sum to one)
                  by default equal for each cluster.

    Returns
    -------
    X : 2d np array
         A matrix with samples rows, and input dimension columns.

    Examples
    --------
    ::

        from pypr.clustering import *
        from numpy import *
        centroids=[array([10,10])]
        ccov=[array([[1,0],[0,1]])]
        samples = 10
        gmm.sample_gaussian_mixture(centroids, ccov, samples=samples)

    """
    cc = centroids
    D = len(cc[0]) # Determin dimensionality
    
    # Check if inputs are ok:
    K = len(cc)
    if mc is None: # Default equally likely clusters
        mc = np.ones(K) / K
    if len(ccov) != K:
        raise ValueError, "centroids and ccov must contain the same number" +\
            "of elements."
    if len(mc) != K:
        raise ValueError, "centroids and mc must contain the same number" +\
            "of elements."

    # Check if the mixing coefficients sum to one:
    EPS = 1E-15
    if np.abs(1-np.sum(mc)) > EPS:
        raise ValueError, "The sum of mc must be 1.0"

    # Cluster selection
    cs_mc = np.cumsum(mc)
    cs_mc = np.concatenate(([0], cs_mc))
    sel_idx = np.random.rand(samples)

    # Draw samples
    res = np.zeros((samples, D))
    for k in range(K):
        idx = (sel_idx >= cs_mc[k]) * (sel_idx < cs_mc[k+1])
        ksamples = np.sum(idx)
        drawn_samples = np.random.multivariate_normal(\
            cc[k], ccov[k], ksamples)
        res[idx,:] = drawn_samples
    return res

def gauss_ellipse_2d(centroid, ccov, sdwidth=1, points=100):
    """Returns x,y vectors corresponding to ellipsoid at standard deviation sdwidth.
    """
    # from: http://www.mathworks.com/matlabcentral/fileexchange/16543
    mean = np.c_[centroid]
    tt = np.c_[np.linspace(0, 2*np.pi, points)]
    x = np.cos(tt); y=np.sin(tt);
    ap = np.concatenate((x,y), axis=1).T
    d, v = np.linalg.eig(ccov);
    d = np.diag(d)
    d = sdwidth * np.sqrt(d); # convert variance to sdwidth*sd
    bp = np.dot(v, np.dot(d, ap)) + np.tile(mean, (1, ap.shape[1])) 
    return bp[0,:], bp[1,:]



def gmm_init(X, K, verbose = False,
                    cluster_init = 'sample', \
                    cluster_init_prop = {}, \
                    max_init_iter = 5, \
                    cov_init = 'var'):
    """Initialize a Gaussian Mixture Model (GMM).
    Generates a set of inital parameters for the GMM.

    Paramseters
    -----------
    cluster_init : string
        How to initalize centroids: 'sample', 'box', or 'kmeans'
    cluster_init_prop : dict
        Passed to the k-means (if used) as keyword arguments
    max_init_iter : int
        How often to try k-means (best final result is used)
    cov_init : string
        Either 'iso' or 'var'

    Returns
    -------
    center_list : list
        A K-length list of cluster centers
    cov_list : list
        A K-length list of co-variance matrices
    p_k : list
        An K length array with mixing cofficients
    """
    samples, dim = np.shape(X)
    if cluster_init == 'sample':
        if verbose: print "Using sample GMM initalization."
        center_list = []
        for i in range(K):
            center_list.append(X[np.random.randint(samples), :])
    elif cluster_init == 'box':
        if verbose: print "Using box GMM initalization."
        center_list = []
        X_max = np.max(X, axis=0)
        X_min = np.min(X, axis=0)
        for i in range(K):
            init_point = ((X_max-X_min)*np.random.rand(1,dim)) + X_min
            center_list.append(init_point.flatten())            
    elif cluster_init == 'kmeans':
        if verbose: print "Using K-means GMM initalization."
        # Normalize data (K-means is isotropic)
        normalizerX = preproc.Normalizer(X)
        nX = normalizerX.transform(X)
        center_list = []
        best_icv = np.inf
        for i in range(max_init_iter):
            m, kcc = kmeans.kmeans(nX, K, iter=100, **cluster_init_prop)
            icv = kmeans.find_intra_cluster_variance(X, m, kcc)
            if best_icv > icv:
                membership = m
                cc = kcc
                best_icv = icv
        cc = normalizerX.invtransform(cc)
        for i in range(cc.shape[0]):
            center_list.append(cc[i,:])
        print cc
    else:
        raise "Unknown initialization of EM of MoG centers."

    # Initialize co-variance matrices
    cov_list = []
    if cov_init=='iso':
        for i in range(K):
            cov_list.append(np.diag(np.ones(dim)/1e10))
            #cov_list.append(np.diag(np.ones(dim)))
    elif cov_init=='var':
        for i in range(K):
            cov_list.append(np.diag(np.var(X, axis=0)/1e10))
    else:
        raise ValueError('Unknown option used for cov_init')

    p_k = np.ones(K) / K # Uniform prior on P(k)
    return (center_list, cov_list, p_k)


def em_gm(X, K, max_iter = 50, verbose = False, \
                iter_call = None,\
                delta_stop = 1e-6,\
                init_kw = {}, \
                max_tries = 10,\
                diag_add = 1e-3):
    """Find K cluster centers in X using Expectation Maximization of Gaussian Mixtures.
   
    Parameters
    -----------
    X : NxD array
        Input data. Should contain N samples row wise, and D variablescolumn wise.
    max_iter : int
        Maximum allowed number of iterations/try.
    iter_call : callable
        Called for each iteration: iter_call(center_list, cov_list, p_k, i)
    delta_stop : float
        Stop when the change in the mean negative log likelihood goes below this
        value.
    max_tries : int
        The co-variance matrix for some cluster migth end up with NaN values, then
        the algorithm will restart; max_tries is the number of allowed retries.
    diag_add : float
        A scalar multiplied by the variance of each feature of the input data, 
        and added to the diagonal of the covariance matrix at each iteration.

    Centroid initialization is given by *cluster_init*, the only available options
    are 'sample' and 'kmeans'. 'sample' selects random samples as centroids. 'kmeans'
    calls kmeans to find the cluster centers.

    Returns
    -------
    center_list : list
        A K-length list of cluster centers
    cov_list : list
        A K-length list of co-variance matrices
    p_k : list
        An K length array with mixing cofficients (p_k)
    logLL : list
         Log likelihood (how well the data fits the model)
    p_nk: matrix
         responsibilty matrix
    """

    samples, dim = np.shape(X)
    clusters_found = False
    while clusters_found==False and max_tries>0:
        max_tries -= 1
        # Initialized clusters
        center_list, cov_list, p_k = gmm_init(X, K, **init_kw)
        # Now perform the EM-steps:
        try:
            center_list, cov_list, p_k, logL, p_nk = \
                gmm_em_continue(X, center_list, cov_list, p_k,
                        max_iter=max_iter, verbose=verbose,
                        iter_call=iter_call,
                        delta_stop=delta_stop,
                        diag_add=diag_add)
            clusters_found = True
        except Cov_problem:
            if verbose:
                print "Problems with the co-variance matrix, tries left ", max_tries

    if clusters_found:
        return center_list, cov_list, p_k, logL, p_nk
    else:
        raise Cov_problem()


def gmm_em_continue(X, center_list, cov_list, p_k,
                    max_iter = 50, verbose = False, \
                    iter_call = None,\
                    delta_stop = 1e-6,\
                    diag_add = 1e-3,\
                    delta_stop_count_end=10):
    """
    """
    delta_stop_count = 0
    samples, dim = np.shape(X)
    K = len(center_list) # We should do some input checking
    if diag_add!=0:
        feature_var = np.var(X, axis=0)
        diag_add_vec = diag_add * feature_var
    old_logL = np.NaN
    logL = np.NaN
    for i in xrange(max_iter):
        try:
##                    if diag_add != 0:
##                        for c in cov_list:
##                            c = c + np.diag(feature_var * diag_add )
##                            #c = c + np.diag(np.ones(c.shape[0]) * diag_add )
            center_list, cov_list, p_k, logL, p_nk = __em_gm_step(X, center_list,\
                cov_list, p_k, K, diag_add_vec)
        except np.linalg.linalg.LinAlgError: # Singular cov matrix
            raise Cov_problem()
        if iter_call is not None:
            iter_call(center_list, cov_list, p_k, i)
        # Check if we have problems with cluster sizes
        for i2 in range(len(center_list)):
            if np.any(np.isnan(cov_list[i2])):
                print "problem"
                raise Cov_problem()

        if old_logL != np.NaN:
            if verbose:
                print "iteration=", i, " delta log likelihood=", \
                    old_logL - logL
            if np.abs(logL - old_logL) < delta_stop: #* samples:
                delta_stop_count += 1
                if verbose: print "gmm_em_continue: delta_stop_count =", delta_stop_count
            else:
                delta_stop_count = 0
            if delta_stop_count>=delta_stop_count_end:
                break # Sufficient precision reached
        old_logL = logL
    try:
        gm_log_likelihood(X, center_list, cov_list, p_k)
    except np.linalg.linalg.LinAlgError: # Singular cov matrix
        raise Cov_problem()
    return center_list, cov_list, p_k, logL, p_nk


def __em_gm_step(X, center_list, cov_list, p_k, K, diag_add_vec):
    samples = X.shape[0]
    # E-step:
    # Instead we use the log-sum-exp formula
    #p_Xn = np.zeros(samples)
    #for k in range(K):
    #    p_Xn += mulnormpdf(X, center_list[k], cov_list[k]) * p_k[k]
    
##    log_p_Xn = np.zeros(samples)
##    for k in range(K):
##        p = logmulnormpdf(X, center_list[k], cov_list[k]) + np.log(p_k[k])
##        if k == 0:
##            log_p_Xn = p
##        else:
##            pmax = np.max(np.concatenate((np.c_[log_p_Xn], np.c_[p]), axis=1), axis=1)
##            log_p_Xn = pmax + np.log( np.exp( log_p_Xn - pmax) + np.exp( p-pmax))
##    logL = np.sum(log_p_Xn)
    
    # New way of calculating the log likelihood:
    log_p_Xn_mat = np.zeros((samples, K))
    for k in range(K):
        log_p_Xn_mat[:,k] = logmulnormpdf(X, center_list[k], cov_list[k]) + np.log(p_k[k])
    pmax = np.max(log_p_Xn_mat, axis=1)
    log_p_Xn = pmax + np.log( np.sum( np.exp(log_p_Xn_mat.T - pmax), axis=0).T) # Maybe move this down
    logL = np.sum(log_p_Xn)

    # Instead we use log-sum-exp formula
    #p_nk = np.zeros((samples, K))
    #for k in range(K):
    #    p_nk[:,k] = mulnormpdf(X, center_list[k], cov_list[k]) * p_k[k] / p_Xn
    
    log_p_nk = np.zeros((samples, K))
    for k in range(K):
        #log_p_nk[:,k] = logmulnormpdf(X, center_list[k], cov_list[k]) + np.log(p_k[k]) - log_p_Xn
        log_p_nk[:,k] = log_p_Xn_mat[:,k] - log_p_Xn

    p_Xn = np.e**log_p_Xn
    p_nk = np.e**log_p_nk
    #print "p_nk=%s" %p_nk
    #print len(p_nk)
    # M-step:    
    for k in range(K):
        ck = np.sum(p_nk[:,k] * X.T, axis = 1) / np.sum(p_nk[:,k])
        center_list[k] = ck
        cov_list[k] = np.dot(p_nk[:,k] * ((X - ck).T), (X - ck)) / sum(p_nk[:,k])\
            + np.diag(diag_add_vec)
        p_k[k] = np.sum(p_nk[:,k]) / samples

    return (center_list, cov_list, p_k, logL, p_nk)

def gm_log_likelihood(X, center_list, cov_list, p_k):
    """Finds the likelihood for a set of samples belongin to a Gaussian mixture
    model.
    
    Return log likelighood
    """
    samples = X.shape[0]
    K =  len(center_list)
    log_p_Xn = np.zeros(samples)
    for k in range(K):
        p = logmulnormpdf(X, center_list[k], cov_list[k]) + np.log(p_k[k])
        if k == 0:
            log_p_Xn = p
        else:
            pmax = np.max(np.concatenate((np.c_[log_p_Xn], np.c_[p]), axis=1), axis=1)
            log_p_Xn = pmax + np.log( np.exp( log_p_Xn - pmax) + np.exp( p-pmax))
    logL = np.sum(log_p_Xn)
    return logL

def gm_assign_to_cluster(X, center_list, cov_list, p_k):
    """Assigns each sample to one of the Gaussian clusters given.
    
    Returns an array with numbers, 0 corresponding to the first cluster in the
    cluster list.
    """
    # Reused code from E-step, should be unified somehow:
    samples = X.shape[0]
    K = len(center_list)
    log_p_Xn_mat = np.zeros((samples, K))
    for k in range(K):
        log_p_Xn_mat[:,k] = logmulnormpdf(X, center_list[k], cov_list[k]) + np.log(p_k[k])
    pmax = np.max(log_p_Xn_mat, axis=1)
    log_p_Xn = pmax + np.log( np.sum( np.exp(log_p_Xn_mat.T - pmax), axis=0).T)
    logL = np.sum(log_p_Xn)
    
    log_p_nk = np.zeros((samples, K))
    for k in range(K):
        #log_p_nk[:,k] = logmulnormpdf(X, center_list[k], cov_list[k]) + np.log(p_k[k]) - log_p_Xn
        log_p_nk[:,k] = log_p_Xn_mat[:,k] - log_p_Xn
    
    print log_p_nk
    #Assign to cluster:
    maxP_k = np.c_[np.max(log_p_nk, axis=1)] == log_p_nk
    #print np.max(log_p_nk, axis=1)
    maxP_k = maxP_k * (np.array(range(K))+1)
    return np.sum(maxP_k, axis=1) - 1


class Cov_problem(Exception):
    """
    """
    pass

def cond_dist(Y, centroids, ccov, mc):
    """Finds the conditional distribution p(X|Y) for a GMM.

    Parameters
    ----------
    Y : D array
        An array of inputs. Inputs set to NaN are not set, and become inputs to
        the resulting distribution. Order is preserved.
    centroids : list
        List of cluster centers - [ [x1,y1,..],..,[xN, yN,..] ]
    ccov : list
        List of cluster co-variances DxD matrices
    mc : list
        Mixing cofficients for each cluster (must sum to one) by default equal
        for each cluster.

    Returns
    -------
    res : tuple
        A tuple containing a new set of (centroids, ccov, mc) for the
        conditional distribution.
    """
    not_set_idx = np.nonzero(np.isnan(Y))[0]
    set_idx = np.nonzero(True - np.isnan(Y))[0]
    new_idx = np.concatenate((not_set_idx, set_idx))
    y = Y[set_idx]
    # New centroids and covar matrices
    new_cen = []
    new_ccovs = []
    # Appendix A in C. E. Rasmussen & C. K. I. Williams, Gaussian Processes
    # for Machine Learning, the MIT Press, 2006
    fk = []
    for i in range(len(centroids)):
        # Make a new co-variance matrix with correct ordering
        new_ccov = copy.deepcopy(ccov[i])
        new_ccov = new_ccov[:,new_idx]
        new_ccov = new_ccov[new_idx,:]
        #ux = centroids[i][not_set_idx]
        #uy = centroids[i][set_idx]
        #A = new_ccov[0:len(not_set_idx), 0:len(not_set_idx)]
        #B = new_ccov[len(not_set_idx):, len(not_set_idx):]
        #C = new_ccov[0:len(not_set_idx), len(not_set_idx):]
        #cen = ux + np.dot(np.dot(C, np.linalg.inv(B)), (y - uy))
        #cov = A - np.dot(np.dot(C, np.linalg.inv(B)), C.transpose())
        ua = centroids[i][not_set_idx]
        ub = centroids[i][set_idx]
        Saa = new_ccov[0:len(not_set_idx), 0:len(not_set_idx)]
        Sbb = new_ccov[len(not_set_idx):, len(not_set_idx):]
        Sab = new_ccov[0:len(not_set_idx), len(not_set_idx):]
        L = np.linalg.inv(new_ccov)
        Laa = L[0:len(not_set_idx), 0:len(not_set_idx)]
        Lbb = L[len(not_set_idx):, len(not_set_idx):]
        Lab = L[0:len(not_set_idx), len(not_set_idx):]
        cen = ua - np.dot(np.dot(np.linalg.inv(Laa), Lab), (y-ub))
        cov = np.linalg.inv(Laa)
        new_cen.append(cen)
        new_ccovs.append(cov)
        #fk.append(mulnormpdf(Y[set_idx], uy, B)) # Used for normalizing the mc
        fk.append(mulnormpdf(Y[set_idx], ub, Sbb)) # Used for normalizing the mc
    # Normalize the mixing coef: p(X|Y) = p(Y,X) / p(Y) using the marginal dist.
    fk = np.array(fk).flatten()
    new_mc = (mc*fk)
    new_mc = new_mc / np.sum(new_mc)
    return (new_cen, new_ccovs, new_mc)

def cond_moments(X, centroids, ccov, mc):
    """EXPERIMENTAL CODE.

    Finds the conditional mean and variance evaluated at a point X.

    See "Active Learning with Statistical Models", David A. Cohn, et al.

    Parameters
    ----------
    X : D array
        An array of inputs. Inputs set to NaN are not set, and become inputs to
        the resulting distribution. Order is preserved.
    centroids : list
        List of cluster centers - [ [x1,y1,..],..,[xN, yN,..] ]
    ccov : list
        List of cluster co-variances DxD matrices
    mc : list
        Mixing cofficients for each cluster (must sum to one) by default equal
        for each cluster.

    Returns
    -------
    res : tuple
        A tuple containing a new set of (centroids, ccov, mc) for the
        conditional distribution.
    """
    not_set_idx = np.nonzero(np.isnan(X))[0]
    set_idx = np.nonzero(True - np.isnan(X))[0]
    new_idx = np.concatenate((not_set_idx, set_idx))

    x = X[set_idx]
    y_i = []
    sigma_yx_i = []
    px_i = []
    px_sum = 0
    N = 60
    for i in range(len(centroids)):
        new_ccov = copy.deepcopy(ccov[i])
        new_ccov = new_ccov[:,new_idx]
        new_ccov = new_ccov[new_idx,:]
        set_cov = new_ccov[len(not_set_idx):, len(not_set_idx):]
        set_u = centroids[i][set_idx]
        px_i.append(mulnormpdf(x, set_u, set_cov))
        #
        not_set_u = centroids[i][not_set_idx]
        inv_sigma_x_i = np.linalg.inv(set_cov)
        sigma_xy_i = new_ccov[len(not_set_idx):,:len(not_set_idx)]
        dx = (x-set_u)
        tmp = np.dot(sigma_xy_i, inv_sigma_x_i)
        y_i.append(not_set_u + np.dot(tmp, dx))
        #
        sigma_y_i = new_ccov[:len(not_set_idx),:len(not_set_idx)]
        sigma_yx_i.append(sigma_y_i - np.dot(sigma_xy_i.T, np.dot(inv_sigma_x_i, sigma_xy_i)))

    px_sum = np.sum(px_i)    
    h = [z/px_sum for z in px_i]

    y = np.sum(np.array(h) * np.array(y_i))
    sigma_y = np.zeros((len(not_set_idx), len(not_set_idx)))
    n_i = N * mc
    for i in range(len(centroids)):
        set_u = centroids[i][set_idx]
        inv_sigma_x_i = np.linalg.inv(set_cov)
        tmp = np.eye(len(set_idx)) + np.dot(np.dot((x-set_u).T, inv_sigma_x_i), x-set_u)
        sigma_y += (h[i]**2 / n_i[i]) * np.dot(sigma_yx_i[i], tmp)
    return y, sigma_y

    
def marg_dist(X_idx, centroids, ccov, mc):
    """Finds the marginal distribution p(X) for a GMM.

    Parameters
    ----------
    X_idx : list
        Indecies of dimensions to keep
    centroids : list
        List of cluster centers - [ [x1,y1,..],..,[xN, yN,..] ]
    ccov : list
        List of cluster co-variances DxD matrices
    mc : list
        Mixing cofficients for each cluster (must sum to one) by default equal
        for each cluster.

    Returns
    -------
    res : tuple
        A tuple containing a new set of (centroids, ccov, mc) for the
        marginal distribution.
    """
    new_cen = []
    new_cov = []
    for i in range(len(centroids)):
        new_cen.append(centroids[i][X_idx])
        new_cov.append(ccov[i][X_idx,:][:,X_idx])
    new_mc = mc
    return (new_cen, new_cov, new_mc)

def find_density_diff(center_list, cov_list, p_k=None, method='hellinger'):
    """Difference measures for each component of the GMM.

    Parameters
    ----------
    centroids : list
        List of cluster centers - [ [x1,y1,..],..,[xN, yN,..] ]
    ccov : list
        List of cluster co-variances DxD matrices
    p_k : list
        Mixing cofficients for each cluster (must sum to one) by default equal
        for each cluster.
    method : string, optional
        Select difference measure to use. Can be:

        - 'hellinger' :
        - 'hellinger_weighted' :
        - 'KL' : Kullback-Leibler divergence

    Returns
    -------
    diff : NxN np array
        The difference between the probability distribtions of the components
        pairwise. Only the upper triangular matrix is used.
    """
    N = len(center_list)
    if method=='hellinger':
        m = gauss_diff.hellinger
    elif method=='hellinger_weighted':
        m = gauss_diff.hellinger_weighted
    elif method=='KL':
        m = gauss_diff.KL
    elif isinstance(method, types.FunctionType):
        pass
    else:
        raise ValueError('Could not understand method option: '+str(method))
    diff = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            if method!='hellinger_weighted':
                dist = m(center_list[i], cov_list[i],\
                        center_list[j], cov_list[j])
            else:
                dist = m(center_list[i], cov_list[i], p_k[i],\
                        center_list[j], cov_list[j], p_k[j])
            diff[i, j] = dist
    return diff

def predict(X, centroids, ccov, mc):
    """Predict the entries in X, which contains NaNs.

    Parameters
    ----------
    X : np array
        2d np array containing the inputs. Target are specified with numpy NaNs.
        The NaNs will be replaced with the most probable result according to the
        GMM model provided.
    centroids : list
        List of cluster centers - [ [x1,y1,..],..,[xN, yN,..] ]
    ccov : list
        List of cluster co-variances DxD matrices
    mc : list
        Mixing cofficients for each cluster (must sum to one) by default equal
        for each cluster.

    Returns
    -------
    var : list
        List of variance
    """
    samples, D = X.shape
    variance_list = []
    for i in range(samples):
        row = X[i, :]
        targets = np.isnan(row)
        num_targets = np.sum(targets)
        cen_cond, cov_cond, mc_cond = cond_dist(row, centroids, ccov, mc)
        X[i, targets] = np.zeros(np.sum(targets))
        vara = np.zeros((num_targets, num_targets))
        varb = np.zeros((num_targets, num_targets))
        for j in range(len(cen_cond)):
            X[i,targets] = X[i,targets] + (cen_cond[j]*mc_cond[j])
            vara = vara + mc_cond[j] * \
                (np.dot(cen_cond[j], cen_cond[j]) + cov_cond[j])
            varb = varb + mc_cond[j] * cen_cond[j]
        variance_list.append(vara - np.dot(varb, varb))
    return variance_list

em = em_gm

