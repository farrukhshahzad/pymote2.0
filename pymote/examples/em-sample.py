# An example of using the Expectation Maximization (EM) algorithm to find the
# parameters for a mixture of gaussian given a set of samples.

from numpy import *
from matplotlib.pylab import *
from pymote import gmm

from em import expectation_maximization

def iter_plot(cen_lst, cov_lst, itr):
    # For plotting EM progress
    if itr % 2 == 0:
        for i in range(len(cen_lst)):
            x,y = gmm.gauss_ellipse_2d(cen_lst[i], cov_lst[i])
            plot(x, y, 'k', linewidth=0.5)

seed(1)
mc = [0.4, 0.4, 0.2] # Mixing coefficients
centroids = [ array([0,0]), array([3,3]), array([0,4]) ]
ccov = [ array([[1,0.4],[0.4,1]]), diag((1,2)), diag((0.4,0.1)) ]

# Generate samples from the gaussian mixture model
X = gmm.sample_gaussian_mixture(centroids, ccov, mc, samples=1000)
fig1 = figure()
plot(X[:,0], X[:,1], '.')

# Expectation-Maximization of Mixture of Gaussians
cen_lst, cov_lst, p_k, logL, p_nk = gmm.em_gm(X, K=5, max_iter=400, verbose=False)
print "Log likelihood (how well the data fits the model) = ", logL

print p_k
print cen_lst[0]
print cov_lst
print p_nk

#print expectation_maximization(X, nbclusters=5, nbiter=6, epsilon=3)

# Plot the cluster ellipses
for i in range(len(cen_lst)):
    x1,x2 = gmm.gauss_ellipse_2d(cen_lst[i], cov_lst[i])
    plot(x1, x2, 'k', linewidth=2)
title(""); xlabel(r'$x_1$'); ylabel(r'$x_2$')

# Now we will find the conditional distribution of x given y
fig2 = figure()
ax1 = subplot(111)
plot(X[:,0], X[:,1], ',')
y = -1.0
axhline(y)
x1plt = np.linspace(axis()[0], axis()[1], 200)
for i in range(len(cen_lst)):
    text(cen_lst[i][0], cen_lst[i][1], str(i+1), horizontalalignment='center',
        verticalalignment='center', size=32, color=(0.2,0,0))
    ex,ey = gmm.gauss_ellipse_2d(cen_lst[i], cov_lst[i])
    plot(ex, ey, 'k', linewidth=0.5)
ax2 = twinx()
(con_cen, con_cov, new_p_k) = gmm.cond_dist(np.array([np.nan, y]), \
        cen_lst, cov_lst, p_k)
x2plt = gmm.gmm_pdf(c_[x1plt], con_cen, con_cov, new_p_k)
ax2.plot(x1plt, x2plt,'r', linewidth=2,
     label='Cond. dist. of $x_1$ given $x_2='+str(y)+'$')
ax2.legend()
ax1.set_xlabel(r'$x_1$')
ax1.set_ylabel(r'$x_2$')
ax2.set_ylabel('Probability')

show()