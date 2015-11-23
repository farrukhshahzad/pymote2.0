__author__ = 'farrukh.shahzad'
from matplotlib.pylab import *
from pypr.clustering.kmeans import *
from pymote import gmm

# Make three clusters:
centroids = [ array([1,1]), array([3,3]), array([3,0]) ]
ccov = [ array([[1,0],[0,1]]), array([[1,0],[0,1]]), \
          array([[0.2, 0],[0, 0.2]]) ]
mc = [0.4, 0.4, 0.2] # Mixing coefficients

centroids=[array([100,100])]
ccov=[array([[100,0],[0,100]])]
#gmm.sample_gaussian_mixture(centroids, ccov, samples=samples)

X = gmm.sample_gaussian_mixture(centroids, ccov, mc=None, samples=100)
print len(X), X

figure(figsize=(10,5))
subplot(121)
title('Original unclustered data')
plot(X[:,0], X[:,1], '.')
xlabel('$x_1$'); ylabel('$x_2$')

subplot(122)
title('Clustered data')
m, cc = kmeans(X, 5)
print m
print cc[0][1]

plot(X[m==0, 0], X[m==0, 1], 'r.')
plot(X[m==1, 0], X[m==1, 1], 'b.')
plot(X[m==2, 0], X[m==2, 1], 'g.')
plot(X[m==3, 0], X[m==3, 1], 'y.')
plot(X[m==4, 0], X[m==4, 1], 'c.')
xlabel('$x_1$'); ylabel('$x_2$')

show()


